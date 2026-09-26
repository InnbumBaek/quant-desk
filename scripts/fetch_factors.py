"""Fetch the daily Fama-French 5 factors plus momentum into `data/factors/`.

G4 measures a strategy's residual alpha against a factor model. Until now no
factor returns were supplied, so the engine fell back to an equal-weight panel
proxy and labelled every verdict `FactorSource.PANEL_PROXY` -- which is honest
but means the t-statistic answers "is this different from the average of these
five ETFs" rather than "is this different from known risk premia". This script
supplies the real thing for the US market.

Same rules as `scripts/fetch_prices.py`, for the same reasons:

1. **Nothing is invented.** A response that is not the archive we expected is an
   error. The parser refuses a file it cannot recognise rather than returning the
   handful of rows it managed to read, because a factor matrix with a silent hole
   moves a t-statistic without moving anything visible.
2. **The source is recorded.** A sibling `.source.json` names the URL shape, the
   fetch time, the row count and the dates where the vendor marked a factor
   missing. The snapshot digest says which bytes; this says where from.
3. **Stdlib only.** `urllib` and `zipfile`.
4. **The data never leaves the runner.** `data/` is git-ignored and no artifact
   carries it. Ken French's library is free to use for research and says nothing
   that licenses us to redistribute it from a public repository, so the treatment
   is the same as for prices (ADR-0007): the bytes stay on the runner, the digest
   and the derived verdicts get committed.

Percent to decimal happens here, once. The library publishes returns in percent
(`0.11` means 11 basis points) and the engine works in decimals; converting at
read time in two places is how the two copies eventually disagree.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

USER_AGENT = "quant-desk/0.1 (research; +https://github.com/InnbumBaek/quant-desk)"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF5_URL = f"{BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = f"{BASE}/F-F_Momentum_Factor_daily_CSV.zip"

#: The library writes a missing observation as -99.99 or -999. Anything at or
#: below this is not a return, and it becomes a hole we refuse to fill.
MISSING_BELOW = -99.0
DATE_ROW = re.compile(r"^\s*(\d{8})\s*$")
#: Enough rows that a three-year panel can be covered end to end.
MIN_ROWS = 500
OUTPUT_NAME = "ff5_mom_daily.csv"


class FetchError(RuntimeError):
    """The vendor did not return a file we recognise. Never a partial matrix."""


@dataclass(frozen=True)
class FactorFile:
    """One parsed French CSV: the day each return was earned, and the returns."""

    names: tuple[str, ...]
    rows: dict[str, tuple[float | None, ...]]
    missing: dict[str, list[str]]


def _get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https host
            return response.read()
    except urllib.error.HTTPError as error:
        raise FetchError(f"HTTP {error.code} for {url}") from error
    except OSError as error:  # timeout, DNS, refused proxy CONNECT
        raise FetchError(f"{type(error).__name__} for {url}: {error}") from error


def unzip_single_csv(body: bytes, url: str) -> str:
    """The archive holds one CSV. Two would mean the vendor changed the layout."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile as error:
        raise FetchError(f"{url} did not return a zip archive: {body[:120]!r}") from error
    members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if len(members) != 1:
        raise FetchError(f"{url} holds {len(members)} CSV members ({archive.namelist()}), expected 1")
    return archive.read(members[0]).decode("utf-8-sig", errors="replace")


def parse_french_csv(text: str) -> FactorFile:
    """Read a Ken French daily CSV: prose header, one data block, percent units.

    The layout is found rather than assumed. The header is the last line before
    the data whose first field is empty, and the data block is the run of rows
    whose first field is an eight-digit date. Reading it that way survives the
    copyright notice growing a line and the column set changing order, and it
    stops at the annual block some files append instead of folding annual returns
    in among the daily ones.
    """
    names: tuple[str, ...] = ()
    rows: dict[str, tuple[float | None, ...]] = {}
    missing: dict[str, list[str]] = {}
    for line in csv.reader(io.StringIO(text)):
        if not line:
            continue
        head = line[0].strip()
        if not head and len(line) > 1 and not rows:
            candidate = tuple(field.strip() for field in line[1:] if field.strip())
            if candidate and not any(DATE_ROW.match(field) for field in candidate):
                names = candidate
            continue
        if not DATE_ROW.match(head):
            if rows:
                break  # the daily block ended; whatever follows is a different table
            continue
        if not names:
            raise FetchError("found data rows before any column header; layout not recognised")
        fields = [field.strip() for field in line[1 : len(names) + 1]]
        if len(fields) != len(names):
            raise FetchError(f"row {head} has {len(fields)} values for {len(names)} columns")
        try:
            day = datetime.strptime(head, "%Y%m%d").date().isoformat()
        except ValueError as error:
            # An eight-digit field is not a date. Keying a row by 20220132 would
            # push the failure into whatever reads the file next.
            raise FetchError(f"{head} is not a calendar date") from error
        if day in rows:
            raise FetchError(f"the file repeats {day}")
        values: list[float | None] = []
        for name, field in zip(names, fields, strict=True):
            try:
                number = float(field)
            except ValueError as error:
                raise FetchError(f"{day} {name}: {field!r} is not a number") from error
            if number <= MISSING_BELOW:
                missing.setdefault(name, []).append(day)
                values.append(None)
            else:
                values.append(number / 100.0)  # percent to decimal, once
        rows[day] = tuple(values)
    if not names:
        raise FetchError(f"no column header found; first 200 characters: {text[:200]!r}")
    if len(rows) < MIN_ROWS:
        raise FetchError(f"parsed {len(rows)} rows, need at least {MIN_ROWS}")
    return FactorFile(names=names, rows=rows, missing=missing)


Rows = dict[str, tuple[float | None, ...]]


def merge(ff5: FactorFile, mom: FactorFile) -> tuple[tuple[str, ...], Rows, int]:
    """Join the two files on the dates both cover, and say how many were dropped.

    The momentum file is published separately and is often a day or two behind the
    five-factor file. The intersection is the only join that does not invent a
    momentum return for a day it was not published for; the count of dropped days
    goes into the sidecar so a short tail is visible rather than inferred.
    """
    if len(mom.names) != 1:
        raise FetchError(f"expected one momentum column, got {mom.names}")
    shared = sorted(set(ff5.rows) & set(mom.rows))
    if not shared:
        raise FetchError("the five-factor and momentum files share no date")
    names = ff5.names + mom.names
    merged = {day: ff5.rows[day] + mom.rows[day] for day in shared}
    dropped = len(set(ff5.rows) | set(mom.rows)) - len(shared)
    return names, merged, dropped


def write_factors(
    names: tuple[str, ...],
    rows: Rows,
    directory: Path,
    dropped: int,
    missing: dict[str, list[str]],
) -> Path:
    """Write the normalised CSV and its sidecar. A missing value stays empty.

    An empty field is not a zero. `core.data.factors` reads an empty field as a
    hole and refuses to build a matrix over it, which is the behaviour a
    regression needs: a zero factor return on a day the factor was not published
    is a fabricated observation.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OUTPUT_NAME
    lines = ["Date," + ",".join(names)]
    for day in sorted(rows):
        cells = ["" if value is None else f"{value:.8f}" for value in rows[day]]
        lines.append(f"{day}," + ",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    days = sorted(rows)
    provenance = {
        "dataset": "Fama-French 5 factors (2x3, daily) + momentum",
        "source": "ken-french-data-library",
        "url_shape": f"{BASE}/F-F_*_daily_CSV.zip",
        "units": "decimal returns; the library publishes percent and this file divides by 100",
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "columns": list(names),
        "rows": len(rows),
        "first_date": days[0],
        "last_date": days[-1],
        "dates_dropped_joining_momentum": dropped,
        "vendor_missing_markers": {name: sorted(dates) for name, dates in sorted(missing.items())},
    }
    (directory / f"{Path(OUTPUT_NAME).stem}.source.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def fetch(directory: Path) -> Path:
    ff5 = parse_french_csv(unzip_single_csv(_get(FF5_URL), FF5_URL))
    mom = parse_french_csv(unzip_single_csv(_get(MOM_URL), MOM_URL))
    names, rows, dropped = merge(ff5, mom)
    missing = {**ff5.missing, **mom.missing}
    return write_factors(names, rows, directory, dropped, missing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/factors", help="directory for the normalised CSV")
    args = parser.parse_args(argv)

    path = fetch(Path(args.out))
    sidecar = json.loads((path.parent / f"{Path(OUTPUT_NAME).stem}.source.json").read_text(encoding="utf-8"))
    print(
        f"{path}: {sidecar['rows']} rows {sidecar['first_date']}..{sidecar['last_date']}, "
        f"columns {', '.join(sidecar['columns'])}, "
        f"{sidecar['dates_dropped_joining_momentum']} date(s) dropped joining momentum"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
