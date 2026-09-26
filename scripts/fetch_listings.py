"""Fetch the US exchange-listed universe into `registry/universe/`.

`core/data/universe.py` defines what a universe is; this fills it for the United
States, from two sources with different jobs.

1. **Membership: the Nasdaq Trader symbol directory** (`nasdaqlisted.txt` and
   `otherlisted.txt`). Pipe-delimited text published for automated download,
   with no key and no rate limit, carrying the symbol, the venue, an ETF flag
   and a test-issue flag.
2. **Sector: the SEC**, via `company_tickers_exchange.json` for ticker -> CIK
   and the DERA financial-statement data sets for CIK -> SIC, mapped to a
   concentration bucket by `core/data/sic.py`.

**The two are not equally required, and that is the design.** Membership has no
fallback: no list, no run. Sectors have three sources in order -- this run's SEC
join, the sectors in the previously committed file, and nowhere -- because
`www.sec.gov` refused ten consecutive requests from the GitHub Actions address
range over twenty-five minutes, including four spread across sixteen minutes of
backoff (ADR-0018). A sector outage must not cost the week's snapshot of who was
listed, since that snapshot is the only delisting history this desk will ever
have and a missed week cannot be recovered. A name with no bucket loads and
cannot be ordered (ADR-0017), which is the failure closing where it belongs.

The rest of the rules are `scripts/fetch_prices.py`'s, for the same reasons.

1. **Nothing is invented.** A response that is not the file we expected is an
   error, never a partial universe. A listings file that silently lost half its
   names is a universe that silently stopped investing in them.
2. **The source is recorded.** The sidecar carries the quarters read, which
   source each sector came from, and what was dropped. A coverage number nobody
   can see is a coverage number nobody checks.
3. **Stdlib only.** `urllib`, `json`, `zipfile`, `csv`.
4. **This file IS committed, unlike prices**, which is why it lands in
   `registry/` rather than the git-ignored `data/`. Both sources permit it, and
   the reason to commit is stronger than the absence of a reason not to: nobody
   publishes who *left*, so a name in last week's committed file and absent from
   this week's has been delisted. The git history of this one file is the
   delisting history no free source sells.

What this cannot do: neither source publishes a listing date, so every listing
comes back undated and the universe is written `point_in_time=False`. It
therefore refuses past-dated membership queries (ADR-0017), which is correct
today and is what deriving dates from this file's own history will fix.
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

from core.data.classification import SECTORS
from core.data.sic import bucket_for_sic
from core.data.universe import (
    Listing,
    Universe,
    load_universe,
    sidecar_path,
    write_universe,
)

#: The SEC refuses what it calls an undeclared automated tool, and asks for a
#: name and a way to reach whoever is running it. The repository's issue tracker
#: is that contact: a person's email address does not belong in a public file,
#: and a URL somebody can actually reach us through serves the same purpose.
USER_AGENT = "quant-desk research (InnbumBaek; https://github.com/InnbumBaek/quant-desk/issues)"
TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
DERA_BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"

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
    """One line of a membership list: who is listed, and where."""

    ticker: str
    name: str
    exchange: str
    etf: bool = False
    cik: int | None = None


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
                ticker=ticker,
                name=str(record[index["name"]] or "").strip(),
                exchange=str(record[index["exchange"]] or "").strip(),
                cik=int(cik),
            )
        )
    if not rows:
        raise FetchError("the ticker file parsed to zero listings")
    return tuple(rows)


# --- the membership list that does not depend on the SEC --------------------

#: Nasdaq Trader publishes the exchange-listed symbol directory as pipe-delimited
#: text for exactly this purpose, with no rate limiting and no key. It became the
#: membership source because www.sec.gov refused ten consecutive requests from
#: the GitHub Actions address range over twenty-five minutes (ADR-0018), and the
#: delisting history this file accumulates only accumulates if it gets written.
NASDAQ_BASE = "https://www.nasdaqtrader.com/dynamic/SymDir"
NASDAQ_LISTED_URL = f"{NASDAQ_BASE}/nasdaqlisted.txt"
OTHER_LISTED_URL = f"{NASDAQ_BASE}/otherlisted.txt"

#: `otherlisted.txt` names the venue by a single letter. Only venues whose names
#: trade on the US session calendar with a real closing auction are taken. OTC is
#: deliberately out and is not in this file anyway: those names have no auction
#: close and the liquidity floor would drop essentially all of them, so carrying
#: twelve thousand of them would be a longer file and not a wider universe.
#: Including them is a decision to take deliberately, not to arrive at by default.
EXCHANGE_CODES = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}


def _pipe_rows(text: str, source: str) -> list[dict[str, str]]:
    """Parse the pipe-delimited directory, dropping the trailing timestamp line.

    Both files end with `File Creation Time: ...` as a single field. It is not a
    listing, and a parser that reads it as one produces a symbol named after a
    date -- which is the kind of thing that survives all the way into an order.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise FetchError(f"{source} is empty")
    header = [name.strip() for name in lines[0].split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if line.startswith("File Creation Time"):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) != len(header):
            raise FetchError(f"{source}: a row has {len(cells)} cells for {len(header)} columns")
        rows.append(dict(zip(header, cells, strict=True)))
    if not rows:
        raise FetchError(f"{source} parsed to zero listings")
    return rows


def _require(header: Iterable[str], columns: Iterable[str], source: str) -> None:
    missing = [name for name in columns if name not in set(header)]
    if missing:
        raise FetchError(f"{source} is missing column(s): {', '.join(missing)}")


def parse_nasdaq_listed(text: str) -> tuple[TickerRow, ...]:
    """`nasdaqlisted.txt`: every Nasdaq-listed security.

    Test issues are dropped. They are real rows in this file and they are not
    real securities -- they exist so that members can exercise their systems --
    so a universe that carries them would put an order into a test symbol.
    """
    rows = _pipe_rows(text, "nasdaqlisted.txt")
    _require(rows[0], ("Symbol", "Security Name", "Test Issue", "ETF"), "nasdaqlisted.txt")
    out = [
        TickerRow(
            ticker=row["Symbol"].upper(),
            name=row["Security Name"],
            exchange="Nasdaq",
            etf=row["ETF"].upper() == "Y",
        )
        for row in rows
        if row["Test Issue"].upper() != "Y" and row["Symbol"].strip()
    ]
    if not out:
        raise FetchError("nasdaqlisted.txt held only test issues")
    return tuple(out)


def parse_other_listed(text: str) -> tuple[TickerRow, ...]:
    """`otherlisted.txt`: NYSE, NYSE American, NYSE Arca, Cboe and IEX.

    The symbol taken is the ACT symbol, which is the one the price vendors use;
    the file's NASDAQ Symbol column is that venue's spelling of the same name
    and taking it would produce a ticker no price file has.
    """
    rows = _pipe_rows(text, "otherlisted.txt")
    _require(rows[0], ("ACT Symbol", "Security Name", "Exchange", "Test Issue", "ETF"), "otherlisted.txt")
    out = []
    for row in rows:
        if row["Test Issue"].upper() == "Y" or not row["ACT Symbol"].strip():
            continue
        venue = EXCHANGE_CODES.get(row["Exchange"].strip().upper())
        if venue is None:
            continue
        out.append(
            TickerRow(
                ticker=row["ACT Symbol"].upper(),
                name=row["Security Name"],
                exchange=venue,
                etf=row["ETF"].upper() == "Y",
            )
        )
    if not out:
        raise FetchError("otherlisted.txt held no listing on a venue we take")
    return tuple(out)


def fetch_membership() -> tuple[TickerRow, ...]:
    """The two Nasdaq Trader directories, together. Required: no list, no run."""
    rows = list(parse_nasdaq_listed(_get(NASDAQ_LISTED_URL).decode("utf-8", errors="replace")))
    rows.extend(parse_other_listed(_get(OTHER_LISTED_URL).decode("utf-8", errors="replace")))
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


def sectors_from_previous(path: Path) -> dict[str, str]:
    """Symbol -> bucket from the last committed file, for when the SEC is out.

    A sector is a slow-moving fact: a company that was a utility last Saturday
    is a utility this Saturday. Carrying it over is therefore honest, and the
    alternative is not "fresher data" but *no file at all* -- and no file means
    the delisting history this whole design rests on loses a week it can never
    get back. The sidecar records that the sectors were carried and from when.
    """
    if not path.exists() or not sidecar_path(path).exists():
        return {}
    try:
        return dict(load_universe(path).sector_map())
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"could not read {path} for carry-over: {error}", file=sys.stderr)
        return {}


def build(
    tickers: Iterable[TickerRow],
    as_of: date,
    cik_by_ticker: Mapping[str, int] | None = None,
    sic_by_cik: Mapping[int, int] | None = None,
    carried: Mapping[str, str] | None = None,
    source: str = "nasdaq-trader+sec-sic",
) -> tuple[Universe, dict[str, object]]:
    """Join membership to sectors, and report every name that did not make it.

    Sectors come from three places in order of preference: this run's SEC join,
    the previous file, and nowhere. An ETF never takes a SIC bucket -- its SIC
    is a trust code, which would file SPY under financials -- so it takes the
    declared bucket from `classification.SECTORS` or stays unclassified.
    """
    rows = tuple(tickers)
    cik_by_ticker = cik_by_ticker or {}
    sic_by_cik = sic_by_cik or {}
    carried = carried or {}

    listings: list[Listing] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    from_sic = from_previous = etf_declared = 0

    for row in sorted(rows, key=lambda r: r.ticker):
        if row.ticker in seen:
            duplicates.append(row.ticker)
            continue
        seen.add(row.ticker)

        sector: str | None = None
        if row.etf:
            sector = SECTORS.get(row.ticker)
            etf_declared += 1 if sector else 0
        else:
            sic = sic_by_cik.get(cik_by_ticker.get(row.ticker, row.cik or -1))
            sector = bucket_for_sic(sic)
            from_sic += 1 if sector else 0
        if sector is None and row.ticker in carried:
            sector = carried[row.ticker]
            from_previous += 1
        listings.append(Listing(symbol=row.ticker, market="US", sector=sector, name=row.name, source=source))

    universe = Universe(listings=tuple(listings), as_of=as_of, source=source, point_in_time=False)
    classified = len(universe) - len(universe.unclassified)
    coverage = {
        "rows_in_membership": len(rows),
        "dropped_duplicate_ticker": duplicates,
        "etfs": sum(1 for row in rows if row.etf),
        "classified": classified,
        "classified_from_sic": from_sic,
        "classified_from_previous_file": from_previous,
        "classified_etf_declared": etf_declared,
        "classified_share": round(classified / len(universe), 4) if len(universe) else 0.0,
    }
    return universe, coverage


def fetch_sector_inputs(
    quarters: int, today: date
) -> tuple[dict[str, int], dict[int, int], tuple[str, ...], str | None]:
    """Try the SEC for ticker -> CIK -> SIC. Returns why it failed rather than raising.

    Best effort on purpose. The membership list is what the run cannot do
    without; the sectors have a fallback, and a sector outage must not cost the
    week's snapshot of who was listed.
    """
    try:
        rows = parse_company_tickers(_get(TICKERS_URL))
    except FetchError as error:
        return {}, {}, (), f"ticker file unavailable: {error}"
    cik_by_ticker = {row.ticker: row.cik for row in rows if row.cik is not None}
    try:
        sic_by_cik, read = fetch_sic_codes(completed_quarters(today, quarters))
    except FetchError as error:
        return cik_by_ticker, {}, (), f"filer SIC codes unavailable: {error}"
    return cik_by_ticker, sic_by_cik, read, None


def fetch(directory: Path, quarters: int = DEFAULT_QUARTERS, as_of: date | None = None) -> Path:
    today = as_of or datetime.now(UTC).date()
    path = Path(directory) / "us.csv"

    membership = fetch_membership()
    cik_by_ticker, sic_by_cik, read, sector_error = fetch_sector_inputs(quarters, today)
    carried = sectors_from_previous(path) if sector_error else {}
    universe, coverage = build(membership, today, cik_by_ticker, sic_by_cik, carried)

    share = float(coverage["classified_share"])  # type: ignore[arg-type]
    if sector_error is None and share < MIN_CLASSIFIED_SHARE:
        raise FetchError(
            f"only {share:.1%} of {len(universe)} listings carry a sector bucket, under the "
            f"{MIN_CLASSIFIED_SHARE:.0%} floor. The SEC answered, so that is a failed join rather "
            "than an outage; the previous file is left in place."
        )
    extra = {
        "dataset": "Nasdaq Trader symbol directory, with sectors from SEC filer SIC codes",
        "urls": [NASDAQ_LISTED_URL, OTHER_LISTED_URL, TICKERS_URL, f"{DERA_BASE}/<year>q<n>.zip"],
        "licence": "Nasdaq Trader symbol directory and SEC filings; committed (ADR-0018)",
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dera_quarters_read": list(read),
        "sector_source": "sec_sic" if sector_error is None else "carried_over_or_absent",
        "sector_error": sector_error,
        "coverage": coverage,
    }
    return write_universe(path, universe, extra=extra)


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
        f"({coverage['classified_share']:.1%}) from sector source {sidecar['sector_source']}"
    )
    if sidecar["sector_error"]:
        print(f"sectors degraded: {sidecar['sector_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
