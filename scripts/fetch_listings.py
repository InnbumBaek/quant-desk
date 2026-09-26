"""Fetch the US exchange-listed universe from SEC filings into `registry/universe/`.

`core/data/universe.py` defines what a universe is; this fills it for the United
States. Two public-domain SEC sources, joined on CIK:

1. `company_tickers_exchange.json` -- every ticker the SEC knows, with its CIK,
   registered name and exchange. This is the membership list.
2. The DERA financial-statement data sets (`<year>q<n>.zip`, member `sub.txt`)
   -- one row per filing, carrying the filer's CIK and its SIC code. This is
   where the concentration bucket comes from, via `core/data/sic.py`.

The rules are `scripts/fetch_prices.py`'s, for the same reasons, with one that
is specific to this file.

1. **Nothing is invented.** A response that is not the JSON or the archive we
   expected is an error, never a partial universe. A listings file that silently
   lost half its names is a universe that silently stopped investing in them.
2. **The source is recorded.** The sidecar `write_universe` writes carries the
   quarters read, how many names were dropped and why, and the share left
   unclassified. A coverage number nobody can see is a coverage number nobody
   checks.
3. **Stdlib only.** `urllib`, `json`, `zipfile`, `csv`.
4. **This file IS committed, unlike prices**, which is why it lands in
   `registry/` rather than the git-ignored `data/`. Both sources are works of
   the US government and carry no redistribution restriction, and the reason to
   commit it is stronger than the absence of a reason not to: the SEC publishes who is
   listed *today* and never who left. A name in last week's committed file and
   absent from this week's has been delisted, so the git history of this one
   file is the delisting history no free source will sell us. It takes time to
   accumulate and it starts accumulating the first time this runs.

What this cannot do: the SEC does not publish a listing date with the ticker
file, so every listing comes back undated and the universe is written
`point_in_time=False`. It therefore refuses past-dated membership queries
(ADR-0017), which is correct today and is what deriving dates from this file's
own history will eventually fix.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from core.data.sic import bucket_for_sic
from core.data.universe import Listing, Universe, write_universe

#: The SEC refuses what it calls an undeclared automated tool, and asks for a
#: name and a way to reach whoever is running it. The repository's issue tracker
#: is that contact: a person's email address does not belong in a public file,
#: and a URL somebody can actually reach us through serves the same purpose.
USER_AGENT = "quant-desk research (InnbumBaek; https://github.com/InnbumBaek/quant-desk/issues)"
TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
DERA_BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"

#: Exchanges whose names trade on the US session calendar with a real closing
#: auction. OTC is deliberately out: those names have no auction close, and the
#: liquidity floor would drop essentially all of them anyway, so carrying twelve
#: thousand of them would be a longer file and not a wider universe. Including
#: them is a decision to take deliberately, not one to arrive at by default.
EXCHANGES = {"nasdaq": "US", "nyse": "US", "nyse american": "US", "cboe": "US", "cboe bzx": "US"}

#: How many completed quarters of filings to read for SIC codes. A company that
#: has not filed in a year is either newly listed or dark; four quarters covers
#: the ordinary filing cycle without downloading the whole archive every week.
DEFAULT_QUARTERS = 4
#: Below this the join failed rather than the data being thin, and a universe
#: that cannot classify most of itself is not worth writing over the last one.
MIN_CLASSIFIED_SHARE = 0.50


class FetchError(RuntimeError):
    """A source did not return usable data. Never swallowed into a short universe."""


@dataclass(frozen=True)
class TickerRow:
    cik: int
    name: str
    ticker: str
    exchange: str


#: HTTP statuses worth trying again. The SEC's rate limiter answers 403 with a
#: "Request Rate Threshold Exceeded" page rather than 429, so 403 is retried
#: here even though it usually means "never" -- the second runner attempt got
#: exactly that page, and a refusal we can wait out is not a refusal to report.
RETRYABLE = (403, 429, 500, 502, 503, 504)
#: A GitHub Actions runner shares its address with everyone else on that pool,
#: so a threshold breach is often somebody else's traffic.
#:
#: The waits are long because the block is long. Measured, not assumed: four
#: attempts spaced 5s/20s/60s were all refused with the same page (run
#: 36270027620), which puts the block past 85 seconds; the SEC documents holding
#: an offending address for about ten minutes. So the schedule now spans that,
#: and a weekly job can afford the sixteen minutes. Short retries against a
#: ten-minute block are not resilience, they are four ways to fail at once.
ATTEMPTS = 4
BACKOFF_SECONDS = (60.0, 300.0, 600.0)


def _get(
    url: str,
    timeout: float = 120.0,
    attempts: int = ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """GET with the headers the SEC asks of automated clients, and a backoff.

    Two runner attempts taught this function what it knows. The first got a bare
    403 and cost a run to diagnose, so the response body is now carried into the
    error. The second got the body, and it said "Request Rate Threshold
    Exceeded" -- a shared runner address, not a rejected client -- so a
    retryable status now waits instead of failing the week.
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "*/*",
            "Host": urllib.parse.urlsplit(url).netloc,
        },
    )
    last = ""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed host
                payload = response.read()
                if response.headers.get("Content-Encoding", "").lower() == "gzip":
                    payload = gzip.decompress(payload)
                return payload
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code} for {url}: {_explain(error)}"
            if error.code not in RETRYABLE:
                raise FetchError(last) from error
        except OSError as error:
            last = f"{type(error).__name__} for {url}: {error}"
        if attempt < attempts - 1:
            delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            print(f"retrying in {delay:.0f}s -- {last}", file=sys.stderr)
            sleep(delay)
    raise FetchError(f"{attempts} attempts failed; last: {last}")


def _explain(error: urllib.error.HTTPError) -> str:
    """The first line of what the server actually said, so a refusal is diagnosable."""
    try:
        body = error.read()
    except OSError:
        return "no response body"
    if error.headers.get("Content-Encoding", "").lower() == "gzip":
        try:
            body = gzip.decompress(body)
        except (OSError, EOFError):
            pass
    text = " ".join(body.decode("utf-8", errors="replace").split())
    return text[:300] or "empty response body"


# --- the membership list ----------------------------------------------------


def parse_company_tickers(raw: bytes) -> tuple[TickerRow, ...]:
    """Read `company_tickers_exchange.json`: a `fields` header and a `data` matrix.

    Columns are read by name rather than by position. The SEC has reordered and
    added fields in this file before, and a positional read would keep working
    while silently putting the exchange in the name column.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FetchError(f"the ticker file is not JSON: {error}") from error
    if not isinstance(payload, dict) or "fields" not in payload or "data" not in payload:
        raise FetchError("the ticker file has no 'fields'/'data' pair; the format changed")

    fields = [str(name).strip().lower() for name in payload["fields"]]
    required = ("cik", "name", "ticker", "exchange")
    missing = [name for name in required if name not in fields]
    if missing:
        raise FetchError(f"the ticker file is missing column(s): {', '.join(missing)}")
    index = {name: fields.index(name) for name in required}

    rows: list[TickerRow] = []
    for record in payload["data"]:
        if len(record) != len(fields):
            raise FetchError(f"a ticker row has {len(record)} cells for {len(fields)} columns")
        ticker = str(record[index["ticker"]] or "").strip().upper()
        cik = record[index["cik"]]
        if not ticker or cik in (None, ""):
            continue  # a filer with no ticker is not a listing
        rows.append(
            TickerRow(
                cik=int(cik),
                name=str(record[index["name"]] or "").strip(),
                ticker=ticker,
                exchange=str(record[index["exchange"]] or "").strip(),
            )
        )
    if not rows:
        raise FetchError("the ticker file parsed to zero listings")
    return tuple(rows)


# --- the SIC codes ----------------------------------------------------------


def completed_quarters(today: date, count: int = DEFAULT_QUARTERS) -> tuple[str, ...]:
    """The `count` most recent completed quarters, newest first, as `2026q2`.

    The current quarter is skipped because its data set does not exist yet, and
    the most recent completed one may not either -- DERA publishes about six
    weeks after the quarter ends. A missing archive is skipped by the caller
    rather than guessed at, which is why this returns candidates and not a
    promise.
    """
    if count < 1:
        raise ValueError(f"count {count} is not a number of quarters")
    year, quarter = today.year, (today.month - 1) // 3 + 1
    out: list[str] = []
    for _ in range(count):
        quarter -= 1
        if quarter == 0:
            year, quarter = year - 1, 4
        out.append(f"{year}q{quarter}")
    return tuple(out)


def parse_dera_sub(text: str) -> dict[int, int]:
    """CIK -> SIC from a DERA `sub.txt`, which is tab-separated with a header.

    A filer appears once per filing in the quarter, and the SIC is the same on
    each, so the last row simply wins. A row whose SIC is blank or unreadable is
    skipped: an absent code is `bucket_for_sic(None)`, which is already the
    "loaded but not orderable" path, and writing a zero here would invent one.
    """
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    header = [name.strip().lower() for name in (reader.fieldnames or [])]
    for column in ("cik", "sic"):
        if column not in header:
            raise FetchError(f"sub.txt has no {column!r} column; columns are {', '.join(header) or 'none'}")
    out: dict[int, int] = {}
    for row in reader:
        raw_cik = (row.get("cik") or row.get("CIK") or "").strip()
        raw_sic = (row.get("sic") or row.get("SIC") or "").strip()
        if not raw_cik or not raw_sic:
            continue
        try:
            out[int(raw_cik)] = int(float(raw_sic))
        except ValueError:
            continue
    if not out:
        raise FetchError("sub.txt parsed to zero CIK/SIC pairs")
    return out


def read_sub_member(archive: bytes, label: str) -> str:
    """Pull `sub.txt` out of a DERA quarterly zip."""
    try:
        bundle = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as error:
        raise FetchError(f"{label} is not a zip archive: {error}") from error
    names = [name for name in bundle.namelist() if name.lower().endswith("sub.txt")]
    if len(names) != 1:
        raise FetchError(f"{label} holds {len(names)} sub.txt members, expected exactly one")
    return bundle.read(names[0]).decode("utf-8", errors="replace")


def fetch_sic_codes(quarters: Iterable[str]) -> tuple[dict[int, int], tuple[str, ...]]:
    """Merge CIK -> SIC across quarters, oldest first so the newest filing wins."""
    merged: dict[int, int] = {}
    read: list[str] = []
    for label in sorted(quarters):
        url = f"{DERA_BASE}/{label}.zip"
        try:
            archive = _get(url)
        except FetchError as error:
            print(f"skipping {label}: {error}", file=sys.stderr)
            continue
        merged.update(parse_dera_sub(read_sub_member(archive, label)))
        read.append(label)
    if not read:
        raise FetchError("no DERA quarter could be read, so no listing can be classified")
    return merged, tuple(read)


# --- assembling the universe ------------------------------------------------


def build(
    tickers: Iterable[TickerRow],
    sic_by_cik: Mapping[int, int],
    as_of: date,
    source: str = "sec-edgar",
) -> tuple[Universe, dict[str, object]]:
    """Join the two sources into a `Universe`, and report what did not make it."""
    rows = tuple(tickers)
    listings: list[Listing] = []
    seen: set[str] = set()
    off_exchange = 0
    duplicates: list[str] = []
    no_filing = 0

    for row in sorted(rows, key=lambda r: r.ticker):
        market = EXCHANGES.get(row.exchange.strip().lower())
        if market is None:
            off_exchange += 1
            continue
        if row.ticker in seen:
            duplicates.append(row.ticker)
            continue
        seen.add(row.ticker)
        sic = sic_by_cik.get(row.cik)
        if sic is None:
            no_filing += 1
        listings.append(
            Listing(
                symbol=row.ticker,
                market=market,
                sector=bucket_for_sic(sic),
                name=row.name,
                source=source,
            )
        )

    universe = Universe(listings=tuple(listings), as_of=as_of, source=source, point_in_time=False)
    classified = len(universe) - len(universe.unclassified)
    coverage = {
        "tickers_in_source": len(rows),
        "dropped_off_exchange": off_exchange,
        "dropped_duplicate_ticker": duplicates,
        "no_recent_filing": no_filing,
        "classified": classified,
        "classified_share": round(classified / len(universe), 4) if len(universe) else 0.0,
    }
    return universe, coverage


def fetch(directory: Path, quarters: int = DEFAULT_QUARTERS, as_of: date | None = None) -> Path:
    tickers = parse_company_tickers(_get(TICKERS_URL))
    sic_by_cik, read = fetch_sic_codes(completed_quarters(as_of or datetime.now(UTC).date(), quarters))
    universe, coverage = build(tickers, sic_by_cik, as_of or datetime.now(UTC).date())

    share = float(coverage["classified_share"])  # type: ignore[arg-type]
    if share < MIN_CLASSIFIED_SHARE:
        raise FetchError(
            f"only {share:.1%} of {len(universe)} listings carry a sector bucket, under the "
            f"{MIN_CLASSIFIED_SHARE:.0%} floor. That is a failed join, not a thin quarter; the "
            "previous file is left in place."
        )
    extra = {
        "dataset": "SEC company tickers with exchange, joined to DERA filer SIC codes",
        "urls": [TICKERS_URL, f"{DERA_BASE}/<year>q<n>.zip"],
        "licence": "works of the US government; committed to the repository (ADR-0018)",
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dera_quarters_read": list(read),
        "coverage": coverage,
    }
    return write_universe(Path(directory) / "us.csv", universe, extra=extra)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="registry/universe", help="directory for the listings CSV")
    parser.add_argument("--quarters", type=int, default=DEFAULT_QUARTERS, help="DERA quarters to read")
    args = parser.parse_args(argv)

    path = fetch(Path(args.out), quarters=args.quarters)
    sidecar = json.loads((path.parent / "us.source.json").read_text(encoding="utf-8"))
    coverage = sidecar["coverage"]
    print(
        f"{path}: {sidecar['rows']} listings, {coverage['classified']} classified "
        f"({coverage['classified_share']:.1%}), {coverage['dropped_off_exchange']} off-exchange "
        f"dropped, quarters {', '.join(sidecar['dera_quarters_read'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
