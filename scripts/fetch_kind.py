"""Fetch KRX KIND's listed-company list, for the Korean industry label.

Run on the runner. This container reaches no market host; KIND answered 200 in
1.2 seconds from Actions with no key at all (`registry/probes/2026-09-27.json`),
which is why this source was chosen over DART's per-name `company.json`.

**What this writes, and what it deliberately does not.** It writes
`registry/universe/kr_industry.csv` -- ticker, the vendor's own industry label,
the company name and the listing date -- and nothing else. In particular it does
**not** write `registry/universe/kr.csv`: `scripts/fetch_krx.py` owns that file,
builds it point-in-time from daily membership, and two writers for one artifact
is how a file ends up with no one who can safely change it. The join happens when
the bucket table exists.

**No bucket is assigned.** `sector_max` is a limit, and a limit fed by a
classification somebody typed from memory is worse than one that is visibly
missing. The labels are KSIC sub-class free text and nobody here has seen the
real set, so this run's job is to *produce* that set: the sidecar carries every
distinct label with its count, and the table that maps them is written from that
census and reviewed as a table.

**Nothing here is a price.** A listing, a company name and an industry label are
facts about who is listed, which is what `registry/universe/` is for (ADR-0018).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from core.config import USER_AGENT
from core.data.kind import ENCODING, EXPECTED_HEADER, USED, Census, KindShapeError, read_listings
from core.data.universe import Listing

#: `searchType=13` is the vendor's own code for the full listed-company list.
ENDPOINT = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"

DEFAULT_OUT = Path("registry/universe")
OUTPUT_NAME = "kr_industry.csv"
COLUMNS = ("symbol", "name", "venue", "industry", "listed_on", "source")
SOURCE = "krx-kind"

RETRYABLE = (429, 500, 502, 503, 504)
BACKOFF_SECONDS = (10.0, 30.0, 90.0)

#: Below this, the file is not the Korean market. KOSPI and KOSDAQ together have
#: carried well over two thousand names for years, so a few hundred rows means the
#: vendor served something else -- a filtered page, an error, a maintenance stub --
#: and writing it would replace the universe with that.
MIN_ROWS = 1500


class FetchError(RuntimeError):
    """The host did not give us the list."""


def _explain(error: urllib.error.HTTPError) -> str:
    """The refusal's own words. Guessing at a 403 cost four runner cycles once."""
    try:
        body = error.read()[:300]
    except Exception:  # noqa: BLE001 - a body we cannot read is still a refusal
        return "empty response body"
    try:
        return body.decode(ENCODING).strip() or "empty response body"
    except UnicodeDecodeError:
        return repr(body)


def _get(
    url: str = ENDPOINT,
    timeout: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """One request, with backoff. The User-Agent names this desk and never a browser."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html, application/vnd.ms-excel;q=0.9, */*;q=0.8",
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
    raise FetchError(f"KIND did not answer after {len(BACKOFF_SECONDS) + 1} attempts; last: {last}")


def write_rows(listings: tuple[Listing, ...], census: Census, directory: Path) -> Path:
    """The ticker-to-label file. One row per listing, sorted by ticker."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OUTPUT_NAME
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for listing in sorted(listings, key=lambda x: x.symbol):
            writer.writerow(
                [
                    listing.symbol,
                    listing.name,
                    census.venue_by_symbol.get(listing.symbol, ""),
                    census.by_symbol.get(listing.symbol, ""),
                    listing.listed_on.isoformat() if listing.listed_on else "",
                    listing.source,
                ]
            )
    return path


def write_sidecar(path: Path, census: Census, as_of: date, url: str) -> Path:
    """The census, which is the point of the first run.

    Every distinct label with its count, so the bucket table is written from the
    vendor's actual vocabulary rather than from anybody's recollection of it.
    """
    sidecar = path.with_suffix(".source.json")
    sidecar.write_text(
        json.dumps(
            {
                "as_of": as_of.isoformat(),
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "dataset": "KRX KIND listed-company list (searchType=13)",
                "url": url,
                "encoding": ENCODING,
                "columns_expected": list(EXPECTED_HEADER),
                "columns_used": list(USED),
                "rows_in_table": census.rows,
                "rows_written": census.rows - len(census.dropped),
                "dropped": census.dropped,
                "undated": list(census.undated),
                "distinct_industries": census.distinct_industries,
                "industry_counts": census.industries,
                "venue_counts": census.venues,
                # A code class we accept without yet knowing what it is.
                "nonnumeric_codes": census.nonnumeric,
                # Said plainly so nobody reads this file as a classification.
                "sector_buckets_assigned": 0,
                "why_no_buckets": (
                    "the label -> bucket table is written from this census and reviewed "
                    "as a table; a mapping typed from memory would sit behind sector_max"
                ),
                "point_in_time": False,
                "licence": "KRX KIND public listed-company list; membership facts committed (ADR-0018)",
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return sidecar


def fetch(
    directory: Path = DEFAULT_OUT,
    url: str = ENDPOINT,
    sleep: Callable[[float], None] = time.sleep,
    as_of: date | None = None,
) -> tuple[Path, Census]:
    """Fetch, parse, write. Raises rather than writing a file it does not believe."""
    body = _get(url, sleep=sleep)
    listings, census = read_listings(body, source=SOURCE)
    if len(listings) < MIN_ROWS:
        raise FetchError(
            f"KIND returned {len(listings)} listing(s), below the {MIN_ROWS} floor. "
            f"The Korean market is larger than that, so this is not the list. "
            f"Dropped: {list(census.dropped.items())[:5]}"
        )
    path = write_rows(listings, census, directory)
    write_sidecar(path, census, as_of or datetime.now(UTC).date(), url)
    return path, census


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--url", default=ENDPOINT)
    args = parser.parse_args(argv)

    try:
        path, census = fetch(Path(args.out), url=args.url)
    except (FetchError, KindShapeError) as error:
        print(f"KIND: {error}", file=sys.stderr)
        return 1

    written = census.rows - len(census.dropped)
    print(f"{written} of {census.rows} row(s) written -> {path}")
    print(f"- {census.distinct_industries} distinct industry label(s); no bucket assigned yet")
    print(f"- venues: {census.venues}")
    if census.nonnumeric:
        sample = list(census.nonnumeric.items())[:5]
        print(f"- {len(census.nonnumeric)} code(s) are not all digits, e.g. {sample}")
    top = list(census.industries.items())[:10]
    for label, count in top:
        print(f"  {count:>5}  {label}")
    if census.dropped:
        print(f"- {len(census.dropped)} row(s) dropped by name: {list(census.dropped)[:5]}")
    if census.undated:
        print(f"- {len(census.undated)} listing(s) carry no listing date: {list(census.undated)[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
