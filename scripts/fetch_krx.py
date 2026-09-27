"""Fetch Korean daily bars and membership from KRX's keyed Open API.

Run on the GitHub Actions runner, never here: this container reaches no market
host, and the key lives only in Actions secrets (ADR-0012, ADR-0022).

**Why this source and not the KRX website.** `data.krx.co.kr`, the endpoint the
website's own pages call, answers 403 from Actions even with the Referer it
documents. `data-dbg.krx.co.kr`, the keyed Open API, answers 401 -- which is
the sound of a host that is reachable and simply wants a key
(registry/probes/2026-09-27.json). They are different hosts and the first one's
refusal says nothing about the second.

**Why Korea gets a point-in-time universe on day one.** KRX returns the whole
market for one date in one request. A year of every listed name is 250
requests, so membership, prices and the delisting record all arrive together,
and `listed_on` is the first date a code appears rather than something derived
from a git history later (ADR-0017, ADR-0022). The US side has no equivalent.

**What is committed and what is not.** Prices go to `data/`, which is
git-ignored, exactly as ADR-0007 requires for the US panel: the manifest and
the derived numbers are ours to publish, the vendor's bytes are not. What is
committed is `registry/universe/kr.csv` (membership, which is a fact about
listings rather than a price) and, on the first run, the response's **field
names with no values**, so that the parser written from KRX's document can be
checked against KRX's bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from core.config import USER_AGENT, require_env
from core.data.krx import BLOCK, KrxBar, KrxNoSession, KrxShapeError, read_day_with_rejects, series
from core.data.universe import Listing, Universe, write_universe

BASE = "https://data-dbg.krx.co.kr/svc/apis/sto"

#: Endpoint per venue. One request per venue per date; three venues is three
#: requests for the entire Korean market that day.
ENDPOINTS: dict[str, str] = {
    "KOSPI": f"{BASE}/stk_bydd_trd",
    "KOSDAQ": f"{BASE}/ksq_bydd_trd",
    "KONEX": f"{BASE}/knx_bydd_trd",
}

#: KRX documents the key as a request header, not a query parameter. That is
#: also the only shape that keeps the key out of logs, proxies and this repo's
#: probe records, which is why a query-parameter fallback is not offered.
KEY_HEADER = "AUTH_KEY"
KEY_ENV = "KRX_API_KEY"

#: Seconds between requests. A backfill of a year is 750 of them, so this is
#: what keeps a backfill from looking like a scrape.
PAUSE_SECONDS = 1.0
RETRYABLE = (429, 500, 502, 503, 504)
BACKOFF_SECONDS = (10.0, 30.0, 90.0)

#: Minimum rows a venue's day must carry to be believed. KOSPI alone lists
#: around 950 names; a day that comes back with twenty is a partial answer, and
#: a partial answer written to disk is indistinguishable from a market where
#: most names stopped trading.
MIN_ROWS_PER_DAY = 100

EXPECTED_HEADER = "Date,Open,High,Low,Close,Volume"


class FetchError(RuntimeError):
    """The vendor did not give us a day. Never swallowed into a short series."""


def _explain(error: urllib.error.HTTPError) -> str:
    """What the server actually said, trimmed, and never the key we sent."""
    try:
        body = error.read()
    except OSError:
        return "no response body"
    text = " ".join(body.decode("utf-8", errors="replace").split())
    return text[:300] or "empty response body"


def day_url(venue: str, day: date) -> str:
    if venue not in ENDPOINTS:
        raise FetchError(f"{venue} is not a KRX venue this script knows")
    return f"{ENDPOINTS[venue]}?{urllib.parse.urlencode({'basDd': day.strftime('%Y%m%d')})}"


def _get(url: str, key: str, timeout: float = 30.0, sleep: Callable[[float], None] = time.sleep) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            KEY_HEADER: key,
        },
    )
    last = ""
    for attempt in range(len(BACKOFF_SECONDS) + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed host
                return response.read()
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code}: {_explain(error)}"
            if error.code not in RETRYABLE:
                raise FetchError(last) from error
        except OSError as error:
            last = f"{type(error).__name__}: {error}"
        if attempt < len(BACKOFF_SECONDS):
            sleep(BACKOFF_SECONDS[attempt])
    raise FetchError(f"KRX did not answer after {len(BACKOFF_SECONDS) + 1} attempts; last: {last}")


def schema_of(payload: dict[str, object]) -> dict[str, object]:
    """The response's field names and row count, with no values in it.

    This is the one artifact that closes the gap this parser was written with:
    the field names come from KRX's document, and nobody here has ever held the
    bytes. Committing the names alone checks the document without publishing a
    single price.
    """
    rows = payload.get(BLOCK)
    first = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}
    return {
        "top_level_keys": sorted(str(key) for key in payload),
        "row_keys": sorted(str(key) for key in first),
        "rows": len(rows) if isinstance(rows, list) else None,
    }


def fetch_day(
    venue: str,
    day: date,
    key: str,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[tuple[KrxBar, ...], tuple[str, ...], dict[str, object]]:
    """One venue's one day: the bars, the rows that were not bars, and the shape."""
    raw = _get(day_url(venue, day), key, sleep=sleep)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        head = " ".join(raw.decode("utf-8", errors="replace").split())[:200]
        raise FetchError(f"{venue} {day}: the reply is not JSON: {head!r}") from error
    if not isinstance(payload, dict):
        raise FetchError(f"{venue} {day}: the reply is a {type(payload).__name__}, not an object")
    shape = schema_of(payload)
    try:
        bars, rejected = read_day_with_rejects(payload)
    except KrxNoSession:
        # A holiday, not a failure. It travels as itself so the caller can
        # record the date instead of stopping the backfill on it.
        raise
    except KrxShapeError as error:
        raise FetchError(f"{venue} {day}: {error}") from error
    if len(bars) < MIN_ROWS_PER_DAY:
        raise FetchError(
            f"{venue} {day}: only {len(bars)} readable rows (need {MIN_ROWS_PER_DAY}). "
            "A partial day on disk cannot be told from a market that stopped trading"
        )
    return bars, rejected, shape


def write_series(bars: Iterable[KrxBar], directory: Path) -> int:
    """Per-symbol CSVs in the shape `core.data.sources` already reads."""
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for symbol, rows in series(bars).items():
        lines = [EXPECTED_HEADER]
        for bar in rows:
            lines.append(
                f"{bar.day},{bar.open:.6f},{bar.high:.6f},{bar.low:.6f},{bar.close:.6f},{bar.volume:.0f}"
            )
        (directory / f"{symbol}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (directory / f"{symbol}.source.json").write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "name": rows[-1].name,
                    "venue": rows[-1].venue,
                    "source": "krx-open-api",
                    "url_shape": f"{BASE}/<venue>_bydd_trd?basDd=YYYYMMDD",
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "rows": len(rows),
                    "first_date": rows[0].day.isoformat(),
                    "last_date": rows[-1].day.isoformat(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        written += 1
    return written


def membership(bars: Iterable[KrxBar], as_of: date, window_start: date) -> Universe:
    """Listings dated from the window, honest about what the window can prove.

    A code's first appearance inside the window is its `listed_on` only if the
    window starts before it. A name already trading on the first day fetched
    listed at some earlier date this run cannot see, so its `listed_on` stays
    empty and the universe is not point-in-time for it. Writing the window's
    first day as a listing date would invent a listing event on whatever date
    the backfill happened to start.
    """
    grouped = series(bars)
    listings = []
    for symbol, rows in sorted(grouped.items()):
        first, last = rows[0].day, rows[-1].day
        listings.append(
            Listing(
                symbol=symbol,
                market=rows[-1].market,
                name=rows[-1].name,
                listed_on=first if first > window_start else None,
                # A name that stopped appearing before the window ended is gone,
                # and the day after its last bar is the first it did not trade.
                delisted_on=last + timedelta(days=1) if last < as_of else None,
                source=f"krx-open-api:{rows[-1].venue}",
            )
        )
    dated = all(listing.listed_on for listing in listings)
    return Universe(
        listings=tuple(listings),
        as_of=as_of,
        source="krx-open-api",
        point_in_time=bool(listings) and dated,
    )


def weekdays(start: date, end: date) -> list[date]:
    """Every Monday-to-Friday in the window, inclusive of both ends.

    There is no Korean holiday table here, and there should not be one. KRX
    itself is the calendar: a weekday whose venue answers with an empty block
    did not trade, and that is a measurement rather than a declaration. Lunar
    holidays move every year, so a table written today is wrong next year and
    wrong silently -- the same reason `core.data.markets` measures sessions per
    year instead of declaring 246 (ADR-0013).

    Weekends are skipped because no exchange has ever opened on one and asking
    would double the requests to learn nothing.
    """
    day, out = start, []
    while day <= end:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def fetch(
    directory: Path,
    days: int = 30,
    as_of: date | None = None,
    venues: Iterable[str] = ("KOSPI", "KOSDAQ"),
    key: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    universe_path: Path | None = None,
) -> dict[str, object]:
    """Backfill `days` sessions of every listed Korean name."""
    end = as_of or datetime.now(UTC).date()
    start = end - timedelta(days=max(days, 1))
    secret = key or require_env(KEY_ENV)

    bars: list[KrxBar] = []
    rejected: list[str] = []
    shapes: dict[str, object] = {}
    closed: list[str] = []
    asked = 0
    for day in weekdays(start, end):
        traded = False
        for venue in venues:
            if asked:
                sleep(PAUSE_SECONDS)
            asked += 1
            try:
                got, bad, shape = fetch_day(venue, day, secret, sleep=sleep)
            except KrxNoSession:
                continue
            traded = True
            bars.extend(got)
            rejected.extend(bad)
            shapes.setdefault(venue, shape)
        if not traded:
            closed.append(day.isoformat())

    if not bars:
        raise FetchError(
            f"every weekday between {start} and {end} came back empty. That is not a run of "
            "holidays; it is a key that is accepted and a query that matches nothing"
        )

    written = write_series(bars, directory)
    universe = membership(bars, as_of=end, window_start=start)
    target = universe_path or Path("registry/universe/kr.csv")
    write_universe(
        target,
        universe,
        extra={
            "dataset": "KRX Open API daily trading, one request per venue per session",
            "urls": [f"{ENDPOINTS[venue]}?basDd=YYYYMMDD" for venue in venues],
            "licence": "KRX Open API; prices are not committed (ADR-0007), membership is",
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "window": {"from": start.isoformat(), "to": end.isoformat()},
            "requests": asked,
            "symbols": written,
            # Measured, not declared: these weekdays had no session on any
            # venue we asked. This is the Korean holiday calendar, and it is a
            # fact about the window rather than a table that rots.
            "weekdays_closed": closed,
            "rows_rejected": len(rejected),
            "rejected_examples": rejected[:5],
            "response_shape": shapes,
        },
    )
    return {
        "requests": asked,
        "symbols": written,
        "weekdays_closed": closed,
        "bars": len(bars),
        "rows_rejected": len(rejected),
        "universe": str(target),
        "response_shape": shapes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--out", default="data/kr")
    parser.add_argument("--universe", default="registry/universe/kr.csv")
    parser.add_argument("--venues", default="KOSPI,KOSDAQ")
    args = parser.parse_args(argv)

    try:
        report = fetch(
            Path(args.out),
            days=args.days,
            venues=tuple(v.strip().upper() for v in args.venues.split(",") if v.strip()),
            universe_path=Path(args.universe),
        )
    except (FetchError, RuntimeError) as error:
        print(f"KRX fetch failed: {error}", file=sys.stderr)
        return 1

    print(f"## KRX {report['symbols']} symbols, {report['bars']} bars in {report['requests']} requests")
    print(f"- rows rejected: {report['rows_rejected']}")
    print(f"- response shape: `{json.dumps(report['response_shape'], ensure_ascii=False)}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
