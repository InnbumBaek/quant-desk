"""Macro series from FRED and ECOS, read as of when they were first published.

Two sources, one contract. FRED is the US series the trend-macro pod and the
regime work need; ECOS is the Bank of Korea's equivalent. They answer the same
kind of question, so they land in the same shape and a caller does not need to
know which one a series came from.

**The revision problem, which is the whole reason this module is careful.**
A macro series is not a price. GDP for 2015Q1 as you read it today is not the
number anybody could have read in 2015: it has been revised, sometimes by more
than the signal a strategy trades. A backtest that reads today's revised value
at a 2015 timestamp is reading the future, and it is the most flattering kind
of look-ahead because nothing about it looks wrong -- the dates all line up.

So the fetcher asks FRED for the **initial release** rather than the current
value, and this module keeps two dates per number: the period it is about, and
the day it was first published. `MacroSeries.as_of` reads the series through
the second one, and refuses outright when a series has no publication dates
rather than approximating them. A series whose vintage is unknown is marked as
such instead of being quietly treated as point-in-time (ADR-0024).

**A missing observation is missing, never zero.** FRED sends `"."` for a
period with no value and ECOS simply omits it. Both become a gap here. A zero
in an interest-rate series is a policy event; a zero that came from a parser is
a fabricated one.

**Both sources hide their real outcome behind HTTP 200.** ECOS answers a bad
key, a bad series code and a genuinely empty window with a `RESULT` block and a
200, exactly as DART does (ADR-0023). The codes are sorted here so the caller
can tell "nothing to report" from "we were refused".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

#: How a series' values relate to what was knowable at the time.
VINTAGE_INITIAL = "initial_release"
VINTAGE_CURRENT = "current_revision"
VINTAGE_UNKNOWN = "unknown"
VINTAGES = (VINTAGE_INITIAL, VINTAGE_CURRENT, VINTAGE_UNKNOWN)

#: What FRED sends instead of a number when a period has no value.
FRED_MISSING = "."
#: FRED's "no bound" date. As a publication date it means nothing was published.
FRED_OPEN_ENDED = date(9999, 12, 31)

#: ECOS puts its outcome in a RESULT block. INFO-200 is "no data", which is an
#: answer; everything else beginning with ERROR is a refusal.
ECOS_NO_DATA = "INFO-200"
ECOS_ROOT = "StatisticSearch"


class MacroShapeError(ValueError):
    """The payload is not a series response we can read."""


class MacroRefused(RuntimeError):
    """The source answered, and the answer is no."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message or 'no message'}")


@dataclass(frozen=True)
class Observation:
    """One period's value, and when that value became knowable.

    `day` is the period the number is *about*; `published_on` is when anybody
    could first have read it. For CPI those are five weeks apart, and using the
    first where the second belongs is the look-ahead this module exists to stop.
    `published_on` is None when the source did not say -- which is a fact about
    the source, not a licence to assume same-day publication.
    """

    day: date
    value: float
    published_on: date | None = None


@dataclass(frozen=True)
class MacroSeries:
    """One series, and what is knowable about when its numbers were knowable."""

    series_id: str
    source: str
    observations: tuple[Observation, ...]
    vintage: str = VINTAGE_UNKNOWN
    units: str = ""
    name: str = ""

    def __post_init__(self) -> None:
        if self.vintage not in VINTAGES:
            raise MacroShapeError(f"{self.series_id}: vintage {self.vintage!r} is not one of {VINTAGES}")

    @property
    def point_in_time(self) -> bool:
        """Whether this series may be read at a past timestamp without look-ahead."""
        return self.vintage == VINTAGE_INITIAL

    @property
    def span(self) -> tuple[date, date] | None:
        if not self.observations:
            return None
        return self.observations[0].day, self.observations[-1].day

    def as_of(self, day: date) -> tuple[Observation, ...]:
        """What a reader on `day` could actually have seen.

        A flag saying "this series is revised" only documents the hazard; this
        is what removes it. Refused rather than approximated when the series
        cannot support the question: an observation with no publication date
        has no place in time, and guessing one (the period end, say) would
        reintroduce exactly the look-ahead the caller came here to avoid.
        """
        undated = [o.day.isoformat() for o in self.observations if o.published_on is None]
        if undated:
            raise MacroShapeError(
                f"{self.series_id}: {len(undated)} observation(s) carry no publication date "
                f"(first: {undated[0]}), so this series cannot be read as of a past day. "
                f"Its vintage is {self.vintage!r}"
            )
        return tuple(o for o in self.observations if o.published_on is not None and o.published_on <= day)


def _iso_day(raw: object, field: str) -> date:
    text = str(raw or "").strip()
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise MacroShapeError(f"{field} is {text!r}, expected a date like 2026-09-25") from error


def _maybe_day(raw: object) -> date | None:
    """A date the source may simply not have sent. Absent stays absent."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        day = date.fromisoformat(text)
    except ValueError:
        return None
    # FRED writes 9999-12-31 for "no bound", and a number published at the end
    # of time was never published. None, so `as_of` refuses the series rather
    # than letting an unbounded stamp pass every date comparison.
    return None if day == FRED_OPEN_ENDED else day


def _ecos_day(raw: object) -> date:
    """ECOS stamps a period as YYYY, YYYYMM, YYYYMMDD or YYYYQn.

    Every one becomes the **first day of the period**, which is the only
    reading that does not pretend to know more than the source said. A monthly
    figure stamped 202609 is about September; calling it the 30th would imply
    the source dated it to a day it never named.
    """
    text = str(raw or "").strip().upper()
    if len(text) == 4 and text.isdigit():
        return date(int(text), 1, 1)
    if len(text) == 6 and text[4] == "Q" and text[5] in "1234":
        return date(int(text[:4]), (int(text[5]) - 1) * 3 + 1, 1)
    if len(text) == 6 and text.isdigit():
        return date(int(text[:4]), int(text[4:]), 1)
    if len(text) == 8 and text.isdigit():
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    raise MacroShapeError(f"TIME is {text!r}, which is not a period ECOS documents")


def read_fred(
    payload: Mapping[str, object],
    series_id: str,
    vintage: str = VINTAGE_UNKNOWN,
    units: str = "",
) -> MacroSeries:
    """FRED's `series/observations` reply. `"."` is a gap, not a zero."""
    if not isinstance(payload, Mapping):
        raise MacroShapeError(f"the response is a {type(payload).__name__}, not an object")
    if "error_message" in payload:
        raise MacroRefused(str(payload.get("error_code", "fred")), str(payload["error_message"]))
    rows = payload.get("observations")
    if not isinstance(rows, list):
        raise MacroShapeError(
            f"{series_id}: the response has no `observations` list. Its keys: "
            f"{', '.join(sorted(str(key) for key in payload)) or '(none)'}"
        )

    out: list[Observation] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise MacroShapeError(f"{series_id}: a {type(row).__name__} where an observation was expected")
        for field in ("date", "value"):
            if field not in row:
                raise MacroShapeError(
                    f"{series_id}: an observation is missing {field}. It carries: "
                    f"{', '.join(sorted(str(key) for key in row))}"
                )
        text = str(row["value"]).strip()
        if text == FRED_MISSING or not text:
            # A period FRED has no number for. Dropped, never zeroed.
            continue
        try:
            value = float(text)
        except ValueError as error:
            raise MacroShapeError(f"{series_id}: value {text!r} is not a number") from error
        out.append(
            Observation(
                day=_iso_day(row["date"], "date"),
                value=value,
                # With `output_type=4` FRED's realtime_start is the day this
                # number was first published, which is the only stamp that lets
                # a backtest place it in time.
                published_on=_maybe_day(row.get("realtime_start")),
            )
        )

    return MacroSeries(
        series_id=series_id,
        source="fred",
        observations=tuple(sorted(out, key=lambda o: o.day)),
        vintage=vintage,
        units=units,
    )


def read_ecos(payload: Mapping[str, object], series_id: str) -> MacroSeries:
    """ECOS's `StatisticSearch` reply, where the outcome hides behind HTTP 200."""
    if not isinstance(payload, Mapping):
        raise MacroShapeError(f"the response is a {type(payload).__name__}, not an object")

    result = payload.get("RESULT")
    if isinstance(result, Mapping):
        code = str(result.get("CODE", "")).strip()
        message = str(result.get("MESSAGE", "")).strip()
        if code == ECOS_NO_DATA:
            # A real answer: the window held nothing. An empty series, so a
            # quiet quarter does not turn a scheduled job red.
            return MacroSeries(series_id, "ecos", (), vintage=VINTAGE_UNKNOWN)
        raise MacroRefused(code or "ecos", message)

    root = payload.get(ECOS_ROOT)
    if not isinstance(root, Mapping):
        raise MacroShapeError(
            f"{series_id}: the response has no {ECOS_ROOT!r} block and no RESULT. Its keys: "
            f"{', '.join(sorted(str(key) for key in payload)) or '(none)'}"
        )
    rows = root.get("row")
    if not isinstance(rows, list):
        raise MacroShapeError(f"{series_id}: {ECOS_ROOT}.row is a {type(rows).__name__}, not a list")

    out: list[Observation] = []
    units = ""
    name = ""
    for row in rows:
        if not isinstance(row, Mapping):
            raise MacroShapeError(f"{series_id}: a {type(row).__name__} where a row was expected")
        for field in ("TIME", "DATA_VALUE"):
            if field not in row:
                raise MacroShapeError(
                    f"{series_id}: a row is missing {field}. It carries: "
                    f"{', '.join(sorted(str(key) for key in row))}"
                )
        units = units or str(row.get("UNIT_NAME", "")).strip()
        name = name or str(row.get("STAT_NAME", "")).strip()
        text = str(row["DATA_VALUE"]).strip().replace(",", "")
        if not text or text == "-":
            continue
        try:
            value = float(text)
        except ValueError as error:
            raise MacroShapeError(f"{series_id}: DATA_VALUE {text!r} is not a number") from error
        out.append(Observation(_ecos_day(row["TIME"]), value))

    return MacroSeries(
        series_id=series_id,
        source="ecos",
        observations=tuple(sorted(out, key=lambda o: o.day)),
        # ECOS publishes revisions in place and exposes no vintage parameter,
        # so nothing here can claim to be the initial release. Saying "unknown"
        # is what keeps a revised number out of a point-in-time backtest.
        vintage=VINTAGE_UNKNOWN,
        units=units,
        name=name,
    )


def to_rows(series: MacroSeries) -> list[Sequence[str]]:
    """The series as CSV rows, header first, for a file a research run can read.

    `published_on` is a column and not a footnote: a file that carries only the
    period is a file whose reader cannot avoid look-ahead even if it wants to.
    """
    return [
        ("date", "value", "published_on"),
        *(
            (o.day.isoformat(), repr(o.value), o.published_on.isoformat() if o.published_on else "")
            for o in series.observations
        ),
    ]
