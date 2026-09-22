"""Fetch daily bars into `data/` as the CSV shape `core.data.sources` reads.

This runs where the network is open. The research container that Claude works in
has no route to any market-data host (verified: every vendor host is refused by
the egress proxy, only GitHub and PyPI answer), so the automated path is the
GitHub Actions runner -- see `.github/workflows/data-snapshot.yml` and ADR-0007.

Design rules, in order of how much they matter:

1. **Nothing is invented.** A response that is not the CSV we expected is an
   error, never a partial panel. A vendor that returns an HTML error page, a
   rate-limit notice or an empty body fails loudly here rather than becoming a
   short price history that a backtest then reports a Sharpe on.
2. **The vendor is recorded.** Every file gets a sibling `<symbol>.source.json`
   naming the source, the URL shape and the fetch time. The snapshot manifest
   digests the CSV; this says where those bytes came from.
3. **Stdlib only.** No new dependency for an HTTP GET. `urllib` is enough and
   keeps the CI job's install identical to the test job's.
4. **Prices never leave the runner.** This script writes into `data/`, which is
   git-ignored, and the workflow uploads no artifact containing prices. What gets
   committed is the manifest (hashes, dates, row counts) and gate verdicts --
   derived numbers, not the vendor's data. Licence reasoning is in ADR-0007.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

USER_AGENT = "quant-desk/0.1 (research; +https://github.com/InnbumBaek/quant-desk)"
EXPECTED_HEADER = "Date,Open,High,Low,Close,Volume"
#: Enough rows to survive the engine's in-sample / out-of-sample split and CV.
MIN_ROWS = 60


class FetchError(RuntimeError):
    """A source did not return usable data. Never swallowed into a short panel."""


@dataclass(frozen=True)
class Bar:
    day: str  # ISO date
    open: float
    high: float
    low: float
    close: float
    volume: float


def _get(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
            return response.read()
    except urllib.error.HTTPError as error:
        raise FetchError(f"HTTP {error.code} for {url}") from error
    except OSError as error:  # timeout, DNS, refused proxy CONNECT
        raise FetchError(f"{type(error).__name__} for {url}: {error}") from error


# --- sources ----------------------------------------------------------------


def fetch_stooq(symbol: str, start: date, end: date) -> list[Bar]:
    """Stooq's keyless daily CSV. Already in the column shape we want."""
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d&d1={start:%Y%m%d}&d2={end:%Y%m%d}"
    body = _get(url).decode("utf-8-sig", errors="replace").strip()
    first = body.splitlines()[0] if body else ""
    if first.strip() != EXPECTED_HEADER:
        # "Exceeded the daily hits limit", an HTML page, or "No data" all land here.
        raise FetchError(f"stooq returned {first[:120]!r} for {symbol}, not the daily CSV header")
    bars = []
    for line in body.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        bars.append(
            Bar(
                parts[0],
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
        )
    return bars


def fetch_yahoo(symbol: str, start: date, end: date) -> list[Bar]:
    """Yahoo's chart endpoint. JSON, so the shape has to be checked explicitly."""
    period1 = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp())
    period2 = int(datetime(end.year, end.month, end.day, tzinfo=UTC).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=div%2Csplit"
    )
    payload = json.loads(_get(url))
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise FetchError(f"yahoo error for {symbol}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise FetchError(f"yahoo returned no result block for {symbol}")
    result = results[0]
    stamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    if not stamps or len(closes) != len(stamps):
        raise FetchError(f"yahoo returned {len(stamps)} stamps and {len(closes)} closes for {symbol}")

    opens, highs, lows, volumes = (
        quote.get("open") or [],
        quote.get("high") or [],
        quote.get("low") or [],
        quote.get("volume") or [],
    )
    bars = []
    for i, stamp in enumerate(stamps):
        close = closes[i]
        if close is None:
            continue  # a holiday row Yahoo pads; dropping beats inventing
        day = datetime.fromtimestamp(stamp, tz=UTC).date().isoformat()
        bars.append(
            Bar(
                day,
                float(opens[i] if i < len(opens) and opens[i] is not None else close),
                float(highs[i] if i < len(highs) and highs[i] is not None else close),
                float(lows[i] if i < len(lows) and lows[i] is not None else close),
                float(close),
                float(volumes[i] if i < len(volumes) and volumes[i] is not None else 0.0),
            )
        )
    return bars


SOURCES: dict[str, Callable[[str, date, date], list[Bar]]] = {
    "stooq": fetch_stooq,
    "yahoo": fetch_yahoo,
}
#: Order tried by `--source auto`. Keyless first: a run that needs no secret is a
#: run the owner does not have to set up before it works.
AUTO_ORDER = ("stooq", "yahoo")


# --- writing ----------------------------------------------------------------


def write_bars(symbol: str, bars: list[Bar], directory: Path, source: str, url_shape: str) -> Path:
    if len(bars) < MIN_ROWS:
        raise FetchError(f"{symbol}: {len(bars)} rows from {source}, need at least {MIN_ROWS}")
    seen = {bar.day for bar in bars}
    if len(seen) != len(bars):
        raise FetchError(f"{symbol}: {source} returned a repeated date")
    if any(bar.close <= 0 for bar in bars):
        raise FetchError(f"{symbol}: {source} returned a non-positive close")

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{symbol.lower()}.csv"
    lines = [EXPECTED_HEADER]
    for bar in sorted(bars, key=lambda b: b.day):
        lines.append(
            f"{bar.day},{bar.open:.6f},{bar.high:.6f},{bar.low:.6f},{bar.close:.6f},{bar.volume:.0f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "symbol": symbol,
        "source": source,
        "url_shape": url_shape,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": len(bars),
        "first_date": min(seen),
        "last_date": max(seen),
    }
    (directory / f"{symbol.lower()}.source.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


URL_SHAPES = {
    "stooq": "https://stooq.com/q/d/l/?s={symbol}.us&i=d&d1=...&d2=...",
    "yahoo": "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d",
}


def fetch_all(
    symbols: list[str],
    directory: Path,
    source: str = "auto",
    years: int = 3,
    pause: float = 1.0,
) -> dict[str, object]:
    """Fetch every symbol, returning a report. A symbol that fails is named, not dropped silently."""
    end = datetime.now(UTC).date()
    start = end - timedelta(days=int(365.25 * years) + 5)

    order = AUTO_ORDER if source == "auto" else (source,)
    for name in order:
        if name not in SOURCES:
            raise FetchError(f"unknown source {name!r}; known: {sorted(SOURCES)}")

    attempts: list[dict[str, object]] = []
    for name in order:
        written: list[str] = []
        failures: dict[str, str] = {}
        for i, symbol in enumerate(symbols):
            try:
                bars = SOURCES[name](symbol, start, end)
                write_bars(symbol, bars, directory, name, URL_SHAPES[name])
                written.append(symbol)
            except FetchError as error:
                failures[symbol] = str(error)
            if pause and i + 1 < len(symbols):
                time.sleep(pause)
        attempts.append({"source": name, "written": written, "failures": failures})
        if len(written) == len(symbols):
            return {"source": name, "symbols": written, "attempts": attempts, "ok": True}

    # Nothing gave a complete panel. Say which source got how far, for every source tried.
    return {"source": None, "symbols": [], "attempts": attempts, "ok": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="SPY,QQQ,IWM,TLT,GLD")
    parser.add_argument("--source", default="auto", help="auto, stooq or yahoo")
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--out", default="data")
    parser.add_argument("--pause", type=float, default=1.0)
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    report = fetch_all(symbols, Path(args.out), args.source, args.years, args.pause)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        print("fetch_prices: no source returned a complete panel", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
