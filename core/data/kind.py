"""KRX KIND's listed-company list: the Korean universe, with an industry label.

`sector_max` caps a bucket at 20% of NAV, and on 2026-09-27 the desk could not
check it for a single Korean name: `classification.SECTORS` carries exactly two
KRX tickers by hand, and the universe this agent trades is every listed name.
The US side has the same wound for a different reason -- SEC's ticker file
answers `Request Rate Threshold Exceeded`, measured, not guessed
(`registry/probes/2026-09-27.json`).

**Why this source.** The same probe asked four candidates. DART's `company.json`
and `corpCode.xml` bounce a keyless request to the OpenDART portal page, so they
teach nothing until a key is registered, and they cost one request per name.
KIND's list answered 200 in 1.2s with the whole market in one response, needs no
key, and **carries the industry label directly**. One request instead of 2,800.

**It is an HTML table wearing an Excel content type.** The vendor serves
`application/vnd.ms-excel; charset=EUC-KR` and the body is `<table>` markup in
EUC-KR. That is the vendor's choice, not a mistake to correct, so the parser
reads what is actually sent. The encoding is declared twice (header and meta) and
this module refuses a body it cannot decode rather than dropping the rows it
managed to read.

**The header is asserted, never assumed.** Positions are how a parser silently
starts reading 주요제품 as an industry when the vendor adds a column. The column
names are checked against what arrived and a mismatch raises naming both.

**No industry label is mapped to a bucket here, on purpose.** The labels are
free text at a KSIC sub-class level and nobody on this desk has seen the real
set. Inventing a mapping from memory would put a made-up classification behind a
concentration limit, which is the one thing the limit cannot survive. So this
module preserves the raw label and counts the distinct ones; the table that maps
them is built from that census, reviewed as a table, and lands separately.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser

from core.data.universe import Listing

#: What the vendor sends. Declared in the Content-Type and again in a meta tag.
ENCODING = "euc-kr"

#: The market code every row of this file belongs to (`core/data/markets.py`).
MARKET = "KR"

#: The columns as KIND serves them, in order. A mismatch raises: the whole point
#: of reading by name is that a new column must not shift the industry label.
EXPECTED_HEADER: tuple[str, ...] = (
    "회사명",
    # Measured on the first real run (2026-09-27): the vendor serves a market
    # segment column here that this parser's documented header did not have.
    # Reading by position would have taken it as the ticker and the ticker as the
    # industry, which is exactly what `check_header` exists to stop.
    "시장구분",
    "종목코드",
    "업종",
    "주요제품",
    "상장일",
    "결산월",
    "대표자명",
    "홈페이지",
    "지역",
)

#: The fields this module actually uses. The rest are read and discarded, which
#: is recorded here so a reader knows they were seen and not missed.
#:
#: 시장구분 is worth carrying: it names the venue (KOSPI/KOSDAQ/KONEX) that
#: `scripts/fetch_krx.py` otherwise has to infer from which of three endpoints
#: answered for a code.
USED = ("회사명", "시장구분", "종목코드", "업종", "상장일")

TICKER = re.compile(r"^\d{6}$")
_DATE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")


class KindShapeError(ValueError):
    """The body is not the table this module knows how to read."""


@dataclass(frozen=True)
class Census:
    """What the file held, so the next decision is made from data and not memory.

    `by_symbol` carries the label per ticker and `industries` the counts. Both
    come out of the one pass over the table: reading the body twice is how the
    written file and the counted census drift apart.
    """

    rows: int
    industries: dict[str, int]
    by_symbol: dict[str, str]
    venue_by_symbol: dict[str, str]
    venues: dict[str, int]
    undated: tuple[str, ...]
    dropped: dict[str, str]

    @property
    def distinct_industries(self) -> int:
        return len(self.industries)


class _TableReader(HTMLParser):
    """Rows of cell text. Deliberately small: this is one vendor's one table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ARG002 - the API's shape
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def decode(body: bytes) -> str:
    """The body as text, or a refusal.

    A body that will not decode is not a partial answer: EUC-KR either reads or
    it does not, and returning the rows that happened to be ASCII would produce a
    universe missing exactly the names with Korean characters in them.
    """
    try:
        return body.decode(ENCODING)
    except UnicodeDecodeError as error:
        head = body[:120]
        raise KindShapeError(
            f"the body does not decode as {ENCODING} ({error}); it starts {head!r}. "
            "A partial decode would drop exactly the rows with Korean names in them."
        ) from error


def read_table(text: str) -> list[list[str]]:
    """Every `<tr>` as a list of cell strings, header row first."""
    reader = _TableReader()
    reader.feed(text)
    if not reader.rows:
        raise KindShapeError(f"no table row in a {len(text)} character body; it starts {text[:160]!r}")
    return reader.rows


def check_header(header: list[str]) -> None:
    """Refuse a table whose columns are not the ones this parser reads."""
    if tuple(header) != EXPECTED_HEADER:
        raise KindShapeError(
            f"the table's columns are {header}, not {list(EXPECTED_HEADER)}. "
            "Reading by position instead would start taking a different column as the industry."
        )


def _day(raw: str) -> date | None:
    match = _DATE.match(raw.strip())
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def read_listings(body: bytes, source: str = "krx-kind") -> tuple[tuple[Listing, ...], Census]:
    """The file as listings, plus a census of what it held.

    A row this module cannot read is **dropped by name with a reason** rather
    than skipped: a universe that is quietly short of a hundred names still looks
    like a universe. The reasons land in the sidecar.

    `sector` is left empty on every listing. The industry label is real and it is
    counted, but a bucket it has not been mapped to yet would be a fabricated
    classification behind a concentration limit.
    """
    rows = read_table(decode(body))
    check_header(rows[0])
    index = {name: position for position, name in enumerate(EXPECTED_HEADER)}

    listings: list[Listing] = []
    industries: dict[str, int] = {}
    by_symbol: dict[str, str] = {}
    venue_by_symbol: dict[str, str] = {}
    venues: dict[str, int] = {}
    undated: list[str] = []
    dropped: dict[str, str] = {}
    seen: dict[str, str] = {}

    for row in rows[1:]:
        if len(row) != len(EXPECTED_HEADER):
            dropped[f"row:{len(dropped)}"] = f"{len(row)} cell(s), not {len(EXPECTED_HEADER)}: {row[:3]}"
            continue
        ticker = row[index["종목코드"]].strip()
        name = row[index["회사명"]].strip()
        if not TICKER.match(ticker):
            # KIND pads to six digits. Anything else is a row shape we do not
            # know, and guessing at it would put a wrong symbol in the universe.
            dropped[ticker or f"(blank):{name}"] = f"{ticker!r} is not a six-digit KRX code"
            continue
        if ticker in seen:
            dropped[ticker] = f"appears twice; kept the first ({seen[ticker]}), dropped {name!r}"
            continue
        seen[ticker] = name

        venue = row[index["시장구분"]].strip()
        venue_by_symbol[ticker] = venue
        if venue:
            venues[venue] = venues.get(venue, 0) + 1

        industry = row[index["업종"]].strip()
        by_symbol[ticker] = industry
        if industry:
            industries[industry] = industries.get(industry, 0) + 1
        listed_on = _day(row[index["상장일"]])
        if listed_on is None:
            # Recorded, not refused: `Listing.status_on` already raises for an
            # undated name, so the failure stays closed where it belongs.
            undated.append(ticker)
        listings.append(
            Listing(
                symbol=ticker,
                market=MARKET,
                sector=None,
                name=name,
                listed_on=listed_on,
                source=source,
            )
        )

    if not listings:
        raise KindShapeError(
            f"the table had {len(rows) - 1} data row(s) and none of them was a listing; "
            f"dropped: {list(dropped.items())[:5]}"
        )
    census = Census(
        rows=len(rows) - 1,
        industries=dict(sorted(industries.items(), key=lambda kv: (-kv[1], kv[0]))),
        by_symbol=by_symbol,
        venue_by_symbol=venue_by_symbol,
        venues=dict(sorted(venues.items(), key=lambda kv: (-kv[1], kv[0]))),
        undated=tuple(sorted(undated)),
        dropped=dropped,
    )
    return tuple(listings), census


def industry_labels(body: bytes) -> tuple[str, ...]:
    """Just the distinct industry labels, for building the bucket table from data."""
    _listings, census = read_listings(body)
    return tuple(census.industries)


def unmapped(labels: Iterable[str], mapping: Mapping[str, str]) -> tuple[str, ...]:
    """Labels the bucket table does not cover, so the gap is reportable before it bites."""
    return tuple(sorted({label for label in labels if label not in mapping}))
