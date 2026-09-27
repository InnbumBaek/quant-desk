"""Fetch KRX KIND's listed-company list, for the Korean industry label.

Run on the runner. This container reaches no market host; KIND answered 200 in
1.2 seconds from Actions with no key at all (`registry/probes/2026-09-27.json`),
which is why this source was chosen over DART's per-name `company.json`.

**What this writes, and what it deliberately does not.** It writes
`registry/universe/kr_industry.csv` -- ticker, market, concentration bucket, the
vendor's own industry label, venue, company name and listing date -- and nothing
else. In particular it does **not** write `registry/universe/kr.csv`:
`scripts/fetch_krx.py` owns that file, builds it point-in-time from daily
membership, and two writers for one artifact is how a file ends up with no one
who can safely change it.

**The file this writes is a universe.** `core/data/universe.load_universe`
needs `symbol` and `market` and takes `sector`, `name` and `listed_on` when they
are there, so the Korean names became loadable and orderable the moment the
bucket table existed -- the KRX key this desk is waiting on buys prices, not
membership.

**The bucket comes from `core/data/ksic.py`, not from here.** The first two runs
of this script produced the census that table was written from: 158 KSIC
sub-class labels across 2,756 names, counted, rather than anybody's recollection
of what KRX publishes (ADR-0029). A label the table has not seen gets no bucket,
its names load and stay unorderable, and this run prints it by name.

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
from core.data.ksic import bucket_counts, bucket_for_label, coverage, unmapped
from core.data.universe import Listing

#: `searchType=13` is the vendor's own code for the full listed-company list.
ENDPOINT = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"

DEFAULT_OUT = Path("registry/universe")
OUTPUT_NAME = "kr_industry.csv"
#: `market` and `sector` are here so `core/data/universe.load_universe` can
#: read this file directly -- it requires `symbol` and `market` and takes the rest
#: when present. That makes the Korean universe loadable now: membership and
#: classification come from KIND, and the KRX key this desk is waiting on is for
#: prices, not for who is listed.
#:
#: The columns were reordered once, on 2026-09-29, to match `us.csv` where the two
#: overlap. That makes one commit's diff cover the whole file; the delisting
#: history ADR-0018 relies on is the appearance and disappearance of rows, which
#: later diffs still show.
COLUMNS = ("symbol", "name", "market", "sector", "industry", "venue", "listed_on", "source")
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
                    listing.market,
                    # The bucket and the label that produced it sit side by side,
                    # so the mapping can be reviewed from the artifact itself.
                    listing.sector or "",
                    census.by_symbol.get(listing.symbol, ""),
                    census.venue_by_symbol.get(listing.symbol, ""),
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
                # `core/data/universe.load_universe` requires as_of, point_in_time
                # and source, and refuses the file without them. This key was
                # missing until 2026-09-29, so the file had looked loadable and was
                # not -- found by asserting the round trip rather than by reading
                # the writer (ADR-0029).
                "source": SOURCE,
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
                # What the bucket table did with this census. A table that
                # quietly sends half a market into one bucket is worse than no
                # table, and this is where that shows (ADR-0029).
                "sector_buckets_assigned": sum(
                    1 for label in census.by_symbol.values() if bucket_for_label(label)
                ),
                "sector_bucket_counts": bucket_counts(census.by_symbol),
                "sector_coverage": round(coverage(census.by_symbol), 4),
                "unmapped_industries": list(unmapped(census.by_symbol.values())),
                "bucket_table": "core/data/ksic.py (ADR-0029)",
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
    listings, census = read_listings(body, source=SOURCE, classify=bucket_for_label)
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
    placed = sum(1 for label in census.by_symbol.values() if bucket_for_label(label))
    print(
        f"- {census.distinct_industries} distinct industry label(s); "
        f"{placed} of {len(census.by_symbol)} name(s) placed in a bucket "
        f"({coverage(census.by_symbol):.1%})"
    )
    print(f"- buckets: {bucket_counts(census.by_symbol)}")
    missing = unmapped(census.by_symbol.values())
    if missing:
        # A new label is the expected way this file changes, and it is not a
        # reason to refuse the fetch: the names under it load and stay
        # unorderable until a line is added to `core/data/ksic.py`.
        print(f"- {len(missing)} label(s) have no bucket, so their names are not orderable: {missing}")
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
