"""Reading KRX's daily trading endpoint, and refusing a shape we do not know.

KRX gives the whole market for one day in one request, which is the opposite of
every other source here. Yahoo answers one symbol at a time, so a year of a
thousand names is a thousand requests; KRX answers one date at a time, so the
same year is 250 requests and the thousand names come free. That is why the
Korean side can have a point-in-time universe from its first run while the US
side is still backfilling sectors a budget at a time (ADR-0017).

**The shape is asserted, not assumed.** This container cannot reach KRX and the
probe only ever got 401 (no key), so the field names below come from KRX's
published contract rather than from bytes we have held. A parser written on a
document and never checked is the failure this desk keeps meeting, so it fails
loudly instead: a payload missing a declared field raises with **the field
names it did see**. The first keyed run on the runner then tells us the real
schema in one line of a job log, rather than producing a panel of zeros.

**No number is invented.** KRX sends numbers as strings, sometimes with
thousands separators and sometimes empty for a halted name. An empty price is
not a zero price: it raises, and the caller decides whether a halted day is
dropped or is the end of the series. `0` reached by parsing is a measurement;
`0` reached by failing to parse is a lie that reaches a limit report.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

#: The response block. KRX names it per endpoint but has used `OutBlock_1`
#: throughout the data API; it is a parameter so a rename is a one-line fix.
BLOCK = "OutBlock_1"

#: The fields this desk reads, and the attribute each becomes. Anything else in
#: the payload is ignored rather than stored: a field we record and never read
#: rots, and this module is where a Korean price becomes a US-shaped bar.
FIELDS: dict[str, str] = {
    "BAS_DD": "day",
    "ISU_CD": "symbol",
    "ISU_NM": "name",
    "MKT_NM": "venue",
    "TDD_OPNPRC": "open",
    "TDD_HGPRC": "high",
    "TDD_LWPRC": "low",
    "TDD_CLSPRC": "close",
    "ACC_TRDVOL": "volume",
    "ACC_TRDVAL": "value",
    "MKTCAP": "market_cap",
    "LIST_SHRS": "shares",
}

#: KRX's venue names to the one market code they share. KOSPI and KOSDAQ keep
#: the same session calendar and currency, so `core.data.markets` models them
#: as one market and the venue stays here as a label (ADR-0013).
VENUES: dict[str, str] = {
    "KOSPI": "KR",
    "KOSDAQ": "KR",
    "KONEX": "KR",
}


class KrxShapeError(ValueError):
    """The payload is not what the contract says. Never swallowed into an empty day."""


class KrxNoSession(KrxShapeError):
    """The venue answered, and the answer is that nothing traded that date.

    A separate type because the caller has to tell two things apart that look
    identical from a status code: a holiday, which is a fact worth recording,
    and a day we failed to read, which is a gap. KRX is the only Korean
    trading calendar this desk has -- there is no declared holiday table and
    `core.data.markets` deliberately measures rather than declares -- so a
    weekday that comes back empty **is** the measurement of a holiday.
    """


@dataclass(frozen=True)
class KrxBar:
    """One symbol's one day, in the terms the rest of the desk uses."""

    day: date
    symbol: str
    name: str
    venue: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    value: float
    market_cap: float
    shares: int

    @property
    def market(self) -> str:
        return VENUES[self.venue]


def _number(raw: object, field: str, symbol: str) -> float:
    text = str(raw if raw is not None else "").strip().replace(",", "")
    if text in ("", "-", "null", "None"):
        raise KrxShapeError(
            f"{symbol}: {field} is empty. A halted or unlisted day has no price, "
            "and reading one as zero would put a fabricated bar in a panel"
        )
    try:
        return float(text)
    except ValueError as error:
        raise KrxShapeError(f"{symbol}: {field} is {text!r}, which is not a number") from error


def _day(raw: object) -> date:
    text = str(raw or "").strip()
    if len(text) != 8 or not text.isdigit():
        raise KrxShapeError(f"BAS_DD is {text!r}, expected eight digits like 20260925")
    return date(int(text[:4]), int(text[4:6]), int(text[6:]))


def read_row(row: Mapping[str, object]) -> KrxBar:
    """One record to one bar, or an error naming what was missing."""
    missing = [key for key in FIELDS if key not in row]
    if missing:
        raise KrxShapeError(
            f"the row is missing {', '.join(missing)}. What it does carry: "
            f"{', '.join(sorted(str(key) for key in row)) or '(nothing)'}"
        )
    symbol = str(row["ISU_CD"] or "").strip()
    if not symbol:
        raise KrxShapeError("a row carries no ISU_CD, so there is nothing to file it under")
    venue = str(row["MKT_NM"] or "").strip().upper()
    if venue not in VENUES:
        raise KrxShapeError(
            f"{symbol}: MKT_NM is {venue!r}, which is not a venue this desk knows "
            f"({', '.join(sorted(VENUES))}). A symbol with no market cannot be ordered"
        )
    return KrxBar(
        day=_day(row["BAS_DD"]),
        symbol=symbol,
        name=str(row["ISU_NM"] or "").strip(),
        venue=venue,
        open=_number(row["TDD_OPNPRC"], "TDD_OPNPRC", symbol),
        high=_number(row["TDD_HGPRC"], "TDD_HGPRC", symbol),
        low=_number(row["TDD_LWPRC"], "TDD_LWPRC", symbol),
        close=_number(row["TDD_CLSPRC"], "TDD_CLSPRC", symbol),
        volume=int(_number(row["ACC_TRDVOL"], "ACC_TRDVOL", symbol)),
        value=_number(row["ACC_TRDVAL"], "ACC_TRDVAL", symbol),
        market_cap=_number(row["MKTCAP"], "MKTCAP", symbol),
        shares=int(_number(row["LIST_SHRS"], "LIST_SHRS", symbol)),
    )


def read_day(payload: Mapping[str, object], block: str = BLOCK) -> tuple[KrxBar, ...]:
    """Every tradable row of one day's response.

    A row that cannot be read is dropped **by name**, not silently: the caller
    gets the bars and the reasons separately through `read_day_with_rejects`.
    This wrapper is for the case where a single bad row should fail the day,
    which is the right default for a backfill.
    """
    bars, rejected = read_day_with_rejects(payload, block)
    if rejected:
        first = rejected[0]
        raise KrxShapeError(f"{len(rejected)} of {len(bars) + len(rejected)} rows unreadable; first: {first}")
    return bars


def read_day_with_rejects(
    payload: Mapping[str, object], block: str = BLOCK
) -> tuple[tuple[KrxBar, ...], tuple[str, ...]]:
    """The readable bars, and one sentence per row that was not.

    A halted name has no close, and on a market of 2,600 names there is always
    one. Failing the whole day for it would mean never storing a day; storing
    it as zero would mean a fabricated bar. So the day is kept and the rejects
    are counted in the manifest, where a number that climbs is visible.
    """
    if not isinstance(payload, Mapping):
        raise KrxShapeError(f"the response is a {type(payload).__name__}, not an object")
    if block not in payload:
        raise KrxShapeError(
            f"the response has no {block!r}. Its keys: "
            f"{', '.join(sorted(str(key) for key in payload)) or '(none)'}"
        )
    rows = payload[block]
    if not isinstance(rows, list):
        raise KrxShapeError(f"{block} is a {type(rows).__name__}, not a list of rows")
    if not rows:
        raise KrxNoSession(f"{block} is empty: the venue did not trade that date")

    bars: list[KrxBar] = []
    rejected: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            rejected.append(f"a {type(row).__name__} where a row was expected")
            continue
        try:
            bars.append(read_row(row))
        except KrxShapeError as error:
            rejected.append(str(error))
    return tuple(bars), tuple(rejected)


def series(bars: Iterable[KrxBar]) -> dict[str, list[KrxBar]]:
    """Bars regrouped by symbol and sorted by day, which is what a CSV needs.

    KRX answers one day of every symbol; a panel wants one symbol of every day.
    The transpose lives here rather than in the fetcher so it can be tested
    without a network, and so a second Korean source can reuse it.
    """
    out: dict[str, list[KrxBar]] = {}
    for bar in bars:
        out.setdefault(bar.symbol, []).append(bar)
    for rows in out.values():
        rows.sort(key=lambda bar: bar.day)
    return out
