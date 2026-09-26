"""Who is in the universe on a given day, and who is not tradable that day.

This desk invests across every listed name rather than a shortlist somebody
picked, so membership stops being a hand-written table and becomes a measurement
with a date attached. `core/data/markets.py` and `core/data/classification.py`
hold about thirty declared symbols each; those tables were right for a seed and
are wrong for a universe of thousands, because the interesting names in an
all-listings universe are precisely the ones nobody would have typed in: the
company that listed in March, the one that was halted for a week, the one that
delisted in 2019 and is therefore missing from every free listings file.

Five decisions, and all five exist because of what an all-symbol universe breaks.

**A universe has a date.** `members(on)` answers "who was listed then", not "who
is listed now". Backtesting today's roster over ten years of history is the
single largest free lunch in quantitative finance and it is entirely fictional,
because the roster already excludes everything that failed.

**A source with no delisting history is not a source with no delistings.** The
free listings files carry live names only. Such a universe is marked
`point_in_time=False` and *refuses* a past-dated membership query unless the
caller opts in by name (`allow_survivorship_bias=True`). The gap it leaves is
reported as unmeasured, never as zero -- the same rule as ADR-0015 and ADR-0016
one level further upstream. `survivorship_adjustment` in
`core/data/quality.REQUIRED_CHECKS` is what that verdict feeds; until now that
check was a string in a tuple with nothing behind it.

**Not tradable is not the same as not listed.** A halted name is still a
position: it cannot be traded and it cannot be marked away either. So the state
is four-valued (`PRE_LISTING`, `LISTED`, `HALTED`, `DELISTED`) and the exclusion
of a name from today's order list always carries the reason it was excluded.

**A missing sector excludes a name, not the file.** Once the universe is every
listed name, an unclassified ticker is a daily event rather than an emergency:
the free listings sources carry a symbol, a name and a venue, and nothing that
maps to a concentration bucket. Refusing to load the file over one new ticker
would stop the desk instead of protecting it, so such a listing loads and is
then excluded from the order list by name, with the reason recorded.

**The liquidity floor is derived from the declared limits, not invented here.**
The tail of an all-listings universe is mostly names that cannot absorb a
position at all. `limits.yaml` already says how that is decided --
`liquidation_days_max` at the pod participation share -- so `floor_from_limits`
computes the floor from those numbers rather than adding a second liquidity
constant that could disagree with the first.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from core.backtest.engine import PricePanel
from core.data.classification import SECTORS
from core.data.markets import MARKETS, UNIVERSE
from core.risk.exposure import UNWIND_PARTICIPATION
from core.risk.limits import load_limits

#: A price that has not moved for a working week is a halt or a dead feed. Either
#: way the name is not tradable, and both readings point the same direction.
DEFAULT_STALE_RUN = 5
#: Median dollar volume is taken over a month of sessions: long enough that one
#: block trade cannot qualify a name, short enough to notice a name going quiet.
DEFAULT_LIQUIDITY_WINDOW = 21


class ListingStatus(StrEnum):
    """What a symbol was on a given date. Derived from dates, never stored."""

    PRE_LISTING = "pre_listing"
    LISTED = "listed"
    HALTED = "halted"
    DELISTED = "delisted"


class SurvivorshipBias(RuntimeError):
    """A past-dated question put to a universe that only knows today's survivors."""


class UnknownListing(KeyError):
    """A symbol the universe does not carry. It has no market, sector or dates."""


def as_date(value: date | str | np.datetime64) -> date:
    """Accept the three date shapes this codebase passes around, reject the rest."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]").astype(date)
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"{value!r} is not a date, an ISO string or a numpy datetime64")


def _opt_date(value: date | str | np.datetime64 | None) -> date | None:
    if value is None or value == "":
        return None
    return as_date(value)


@dataclass(frozen=True)
class Halt:
    """A suspension of trading. `end` is None while the name is still halted."""

    start: date
    end: date | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", as_date(self.start))
        object.__setattr__(self, "end", _opt_date(self.end))
        if self.end is not None and self.end < self.start:
            raise ValueError(f"a halt cannot end ({self.end}) before it starts ({self.start})")

    def covers(self, day: date) -> bool:
        """`end` is the first session trading resumed, so the range is half-open."""
        return self.start <= day and (self.end is None or day < self.end)


@dataclass(frozen=True)
class Listing:
    """One symbol's life: when it started trading, when it stopped, when it paused.

    `listed_on` is the first session the name traded and `delisted_on` the first
    session it no longer did, so both are half-open and a one-day listing is
    expressible. `listed_on` may be None only in a universe that is not
    point-in-time; asking such a listing for a past status raises rather than
    assuming it has always existed.
    """

    symbol: str
    market: str
    sector: str | None = None
    name: str = ""
    listed_on: date | None = None
    delisted_on: date | None = None
    halts: tuple[Halt, ...] = ()
    source: str = "declared"

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("a listing needs a symbol")
        if self.market not in MARKETS:
            raise ValueError(
                f"{self.symbol!r} is declared in market {self.market!r}; "
                f"known markets: {', '.join(sorted(MARKETS))}"
            )
        object.__setattr__(self, "sector", self.sector or None)
        object.__setattr__(self, "listed_on", _opt_date(self.listed_on))
        object.__setattr__(self, "delisted_on", _opt_date(self.delisted_on))
        object.__setattr__(self, "halts", tuple(self.halts))
        if self.listed_on and self.delisted_on and self.delisted_on < self.listed_on:
            raise ValueError(
                f"{self.symbol!r} delisted on {self.delisted_on} before it listed on {self.listed_on}"
            )
        for halt in self.halts:
            if self.listed_on and halt.start < self.listed_on:
                raise ValueError(f"{self.symbol!r} is halted on {halt.start}, before it listed")
            if self.delisted_on and halt.start >= self.delisted_on:
                raise ValueError(f"{self.symbol!r} is halted on {halt.start}, after it delisted")

    @property
    def dated(self) -> bool:
        """Whether this listing can answer a question about the past at all."""
        return self.listed_on is not None

    @property
    def classified(self) -> bool:
        """Whether the concentration limit can be checked for a book holding this name.

        A sector is missing far more often than it is wrong once the universe is
        every listed name: the free listings sources carry a ticker, a name and a
        venue, and nothing that maps to a bucket. So an unclassified listing
        loads -- refusing the whole file over one new ticker would stop the desk
        rather than protect it -- and is then excluded from the order list by
        name, which keeps the failure closed where it belongs.
        """
        return self.sector is not None

    def status_on(self, day: date | str | np.datetime64) -> ListingStatus:
        """The symbol's state on `day`, or a refusal when the listing date is unknown."""
        when = as_date(day)
        if self.listed_on is None:
            raise ValueError(
                f"{self.symbol!r} has no listing date, so its status on {when} is unknown. "
                "A universe without listing dates cannot be queried in the past."
            )
        if when < self.listed_on:
            return ListingStatus.PRE_LISTING
        if self.delisted_on is not None and when >= self.delisted_on:
            return ListingStatus.DELISTED
        if any(halt.covers(when) for halt in self.halts):
            return ListingStatus.HALTED
        return ListingStatus.LISTED


@dataclass(frozen=True)
class Universe:
    """Every symbol the desk may hold, with the dates that make membership checkable.

    `as_of` is the date the source was read. A query for any earlier date is a
    point-in-time query, and a universe built from a source with no delisting
    history cannot answer one honestly -- so it refuses instead.
    """

    listings: tuple[Listing, ...]
    as_of: date
    source: str = "declared"
    point_in_time: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "listings", tuple(self.listings))
        object.__setattr__(self, "as_of", as_date(self.as_of))
        seen = [listing.symbol for listing in self.listings]
        duplicates = sorted({symbol for symbol in seen if seen.count(symbol) > 1})
        if duplicates:
            raise ValueError(f"the universe carries a symbol twice: {', '.join(duplicates)}")
        if self.point_in_time:
            undated = sorted(listing.symbol for listing in self.listings if not listing.dated)
            if undated:
                raise ValueError(
                    f"a point-in-time universe needs a listing date for every name; missing for "
                    f"{len(undated)}: {', '.join(undated[:5])}"
                )

    def __len__(self) -> int:
        return len(self.listings)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(listing.symbol for listing in self.listings))

    def listing(self, symbol: str) -> Listing:
        for candidate in self.listings:
            if candidate.symbol == symbol:
                return candidate
        raise UnknownListing(f"{symbol!r} is not in the universe read from {self.source!r}")

    def market_map(self) -> dict[str, str]:
        """Symbol -> market code, the mapping `markets.group_by_market` takes."""
        return {listing.symbol: listing.market for listing in self.listings}

    def sector_map(self) -> dict[str, str]:
        """Symbol -> bucket, the mapping `classification.sector_weights` takes.

        Unclassified names are left out rather than given a placeholder bucket.
        `sector_weights` then raises on a book that holds one, `pod_snapshot`
        records the sector weights as unmeasured, and the limit engine blocks --
        which is the path a missing bucket should take (ADR-0015).
        """
        return {x.symbol: x.sector for x in self.listings if x.sector is not None}

    @property
    def unclassified(self) -> tuple[str, ...]:
        """Names with no sector bucket. They load, and they cannot be ordered."""
        return tuple(sorted(x.symbol for x in self.listings if not x.classified))

    def effective_date(
        self, on: date | str | np.datetime64 | None, allow_survivorship_bias: bool
    ) -> date | None:
        """The date to answer for, or None meaning "the roster as read, undated".

        A source that is not point-in-time can only ever answer with the roster
        it was read as, whatever date is asked. For the as-of date and later that
        is the honest answer -- the file says who is listed now. For an earlier
        date it is survivorship bias, so it takes an opt-in.
        """
        if self.point_in_time:
            return self.as_of if on is None else as_date(on)
        if on is None:
            return None
        when = as_date(on)
        if when >= self.as_of or allow_survivorship_bias:
            return None
        raise SurvivorshipBias(
            f"{self.source!r} was read on {self.as_of} and carries no delisting history, so it "
            f"cannot say who was listed on {when}: everything that failed before {self.as_of} is "
            "already missing from it. Pass allow_survivorship_bias=True to use today's roster "
            "anyway and have the bias recorded, or load a point-in-time source."
        )

    def members(
        self,
        on: date | str | np.datetime64 | None = None,
        allow_survivorship_bias: bool = False,
    ) -> tuple[str, ...]:
        """Symbols in the market on `on` -- listed or halted, both of which can be held."""
        when = self.effective_date(on, allow_survivorship_bias)
        if when is None:
            return self.symbols
        holdable = {ListingStatus.LISTED, ListingStatus.HALTED}
        return tuple(sorted(x.symbol for x in self.listings if x.status_on(when) in holdable))

    def tradable(
        self,
        on: date | str | np.datetime64 | None = None,
        allow_survivorship_bias: bool = False,
    ) -> tuple[str, ...]:
        """Symbols that can actually be traded on `on`: listed and not halted."""
        when = self.effective_date(on, allow_survivorship_bias)
        if when is None:
            return self.symbols
        return tuple(sorted(x.symbol for x in self.listings if x.status_on(when) is ListingStatus.LISTED))

    def counts(self, on: date | str | np.datetime64) -> dict[str, int]:
        """How many names sit in each state on `on`, for a report that has to add up."""
        when = as_date(on)
        out = {status.value: 0 for status in ListingStatus}
        for listing in self.listings:
            out[listing.status_on(when).value] += 1
        return out

    def delisted_between(
        self, start: date | str | np.datetime64, end: date | str | np.datetime64
    ) -> tuple[str, ...]:
        """Names that left the market inside [start, end]."""
        first, last = as_date(start), as_date(end)
        return tuple(
            sorted(
                x.symbol
                for x in self.listings
                if x.delisted_on is not None and first <= x.delisted_on <= last
            )
        )


# --- the seed table, kept only as a fallback --------------------------------


def seed_universe(as_of: date | str | np.datetime64) -> Universe:
    """The hand-declared ETF table as a `Universe`, explicitly not point-in-time.

    These thirty names are what `markets.UNIVERSE` and `classification.SECTORS`
    hold, and they carry no listing dates, so this universe answers "who is in
    it" and refuses "who was in it in 2019". That refusal is the point: the seed
    is a fallback for a smoke run, not a basis for a backtest.
    """
    listings = tuple(
        Listing(symbol=symbol, market=market, sector=SECTORS.get(symbol), source="declared-seed")
        for symbol, market in sorted(UNIVERSE.items())
    )
    return Universe(listings=listings, as_of=as_of, source="declared-seed", point_in_time=False)


# --- reading and writing a listings file ------------------------------------

COLUMNS = ("symbol", "name", "market", "sector", "listed_on", "delisted_on", "halts", "source")


def _parse_halts(raw: str) -> tuple[Halt, ...]:
    """`2024-03-01:2024-03-05;2025-01-02:` -- ranges separated by `;`, open end allowed."""
    out: list[Halt] = []
    for chunk in (part.strip() for part in raw.split(";")):
        if not chunk:
            continue
        start, _, end = chunk.partition(":")
        out.append(Halt(start=as_date(start.strip()), end=_opt_date(end.strip())))
    return tuple(out)


def _format_halts(halts: Sequence[Halt]) -> str:
    return ";".join(f"{halt.start.isoformat()}:{halt.end.isoformat() if halt.end else ''}" for halt in halts)


def sidecar_path(path: Path | str) -> Path:
    """`listings.csv` -> `listings.source.json`, the convention `fetch_prices.py` uses."""
    target = Path(path)
    return target.with_suffix(".source.json")


def write_universe(path: Path | str, universe: Universe, extra: Mapping[str, Any] | None = None) -> Path:
    """Write the listings CSV and the sidecar that says what kind of source it is.

    `extra` adds fetch-side facts to the sidecar -- how many rows the source
    held, what was dropped and why. `load_universe` ignores keys it does not
    know, so a fetcher can record whatever its own reader would want later.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for listing in sorted(universe.listings, key=lambda x: x.symbol):
            writer.writerow(
                [
                    listing.symbol,
                    listing.name,
                    listing.market,
                    listing.sector or "",
                    listing.listed_on.isoformat() if listing.listed_on else "",
                    listing.delisted_on.isoformat() if listing.delisted_on else "",
                    _format_halts(listing.halts),
                    listing.source,
                ]
            )
    sidecar: dict[str, Any] = dict(extra or {})
    sidecar.update(
        {
            "as_of": universe.as_of.isoformat(),
            "source": universe.source,
            "point_in_time": universe.point_in_time,
            "rows": len(universe.listings),
            "unclassified": len(universe.unclassified),
        }
    )
    sidecar_path(target).write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_universe(path: Path | str) -> Universe:
    """Read a listings CSV plus its sidecar. A missing sidecar is an error.

    The sidecar carries `as_of` and `point_in_time`, and neither is inferable
    from the rows: a file of live names looks exactly like a point-in-time file
    whose window happens to contain no delistings. Defaulting `point_in_time` to
    true would turn a missing sidecar into a silent survivorship bias, and
    defaulting it to false would silently downgrade a good source, so it is read
    or the load fails.
    """
    target = Path(path)
    meta_file = sidecar_path(target)
    if not meta_file.exists():
        raise FileNotFoundError(
            f"{target} has no sidecar at {meta_file}; a listings file without an as-of date and a "
            "point-in-time flag cannot be used, because whether it is survivorship-free is not "
            "visible in the rows."
        )
    meta: dict[str, Any] = json.loads(meta_file.read_text(encoding="utf-8"))
    for key in ("as_of", "point_in_time", "source"):
        if key not in meta:
            raise ValueError(f"{meta_file} is missing {key!r}")
    if not isinstance(meta["point_in_time"], bool):
        raise ValueError(f"{meta_file}: point_in_time must be true or false, got {meta['point_in_time']!r}")

    listings: list[Listing] = []
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        missing_columns = [name for name in ("symbol", "market") if name not in header]
        if missing_columns:
            raise ValueError(f"{target} is missing column(s): {', '.join(missing_columns)}")
        for row in reader:
            listings.append(
                Listing(
                    symbol=(row.get("symbol") or "").strip(),
                    market=(row.get("market") or "").strip(),
                    sector=(row.get("sector") or "").strip(),
                    name=(row.get("name") or "").strip(),
                    listed_on=_opt_date((row.get("listed_on") or "").strip()),
                    delisted_on=_opt_date((row.get("delisted_on") or "").strip()),
                    halts=_parse_halts(row.get("halts") or ""),
                    source=(row.get("source") or meta["source"]).strip(),
                )
            )
    return Universe(
        listings=tuple(listings),
        as_of=as_date(meta["as_of"]),
        source=str(meta["source"]),
        point_in_time=bool(meta["point_in_time"]),
    )


# --- the illiquid tail ------------------------------------------------------


def floor_from_limits(
    capital: float,
    min_weight: float,
    participation: float = UNWIND_PARTICIPATION,
    limits: Mapping[str, Any] | None = None,
) -> float:
    """The dollar volume below which the smallest position we take cannot be unwound.

    Derived, not declared. `liquidation_days_max` and the unwind participation it
    is written against already answer this question, so the floor is
    `min_weight * capital / (participation * liquidation_days_max)`: a name with
    less ADV than that would breach the liquidation limit on the day it was
    bought. Writing a second liquidity constant here instead would let the screen
    admit names the limit engine then rejects, which is a book that cannot be
    built from a universe that says it can.

    `min_weight` is the smallest position the book bothers to take. That is a
    construction choice rather than a risk limit, so it is the caller's number.
    """
    if capital <= 0:
        raise ValueError(f"capital {capital} is not a positive amount")
    if not 0.0 < min_weight <= 1.0:
        raise ValueError(f"min_weight {min_weight} is not a share of NAV in (0, 1]")
    if not 0.0 < participation <= 1.0:
        raise ValueError(f"participation {participation} is not a share of volume")
    days = float(dict(limits or load_limits())["pod"]["liquidation_days_max"])
    if days <= 0.0:
        raise ValueError(f"liquidation_days_max is {days}, so no position could ever be unwound")
    return float(min_weight) * float(capital) / (participation * days)


@dataclass(frozen=True)
class Screen:
    """What survived a screen and, for everything else, the reason it did not."""

    kept: tuple[str, ...]
    excluded: dict[str, str] = field(default_factory=dict)

    @property
    def dropped(self) -> tuple[str, ...]:
        return tuple(sorted(self.excluded))

    def reason(self, symbol: str) -> str | None:
        return self.excluded.get(symbol)


def median_dollar_volume(
    panel: PricePanel, window: int = DEFAULT_LIQUIDITY_WINDOW
) -> dict[str, float] | None:
    """Trailing median dollar volume per symbol, or None when the panel carries none.

    The median rather than the mean: one block trade in a dead name lifts a mean
    over any floor, and the question here is whether the name trades every day.
    """
    if panel.dollar_volume is None:
        return None
    if window < 1:
        raise ValueError(f"window {window} is not a number of sessions")
    recent = np.asarray(panel.dollar_volume, dtype=float)[-window:]
    return {symbol: float(np.median(recent[:, i])) for i, symbol in enumerate(panel.symbols)}


def liquidity_screen(
    panel: PricePanel,
    min_dollar_volume: float,
    symbols: Iterable[str] | None = None,
    window: int = DEFAULT_LIQUIDITY_WINDOW,
) -> Screen:
    """Drop the names that cannot absorb a position, and say so for each of them.

    A panel with no volume column drops *everything*: an all-listings universe
    without volume is mostly names that cannot be traded, and reading a missing
    measurement as "liquid enough" is the failure mode ADR-0015 exists to stop.
    """
    if min_dollar_volume <= 0:
        raise ValueError(f"min_dollar_volume {min_dollar_volume} is not a positive floor")
    wanted = tuple(symbols) if symbols is not None else tuple(panel.symbols)
    volumes = median_dollar_volume(panel, window)
    if volumes is None:
        reason = "the panel carries no dollar volume, so liquidity is unmeasured"
        return Screen(kept=(), excluded=dict.fromkeys(sorted(wanted), reason))

    kept: list[str] = []
    excluded: dict[str, str] = {}
    for symbol in sorted(wanted):
        if symbol not in volumes:
            excluded[symbol] = "no price history in the panel"
            continue
        traded = volumes[symbol]
        if traded < min_dollar_volume:
            excluded[symbol] = (
                f"median dollar volume {traded:,.0f} over {window} sessions is below the "
                f"{min_dollar_volume:,.0f} floor"
            )
            continue
        kept.append(symbol)
    return Screen(kept=tuple(kept), excluded=excluded)


def stale_price_runs(panel: PricePanel, min_run: int = DEFAULT_STALE_RUN) -> dict[str, int]:
    """Symbols whose close has not moved for `min_run` sessions, with the run length.

    This is a *proxy* for a halt, not a halt feed. A vendor that keeps printing
    the last close through a suspension is indistinguishable here from a name
    that genuinely did not move, and a genuinely motionless name is not one to
    put a position on either -- so both readings exclude it, which is why the
    proxy is usable at all. A real halt calendar, when a source provides one,
    belongs in `Listing.halts` and takes precedence over this.
    """
    if min_run < 2:
        raise ValueError(f"min_run {min_run} is shorter than a repeat; a single print is not a run")
    close = np.asarray(panel.close, dtype=float)
    out: dict[str, int] = {}
    for index, symbol in enumerate(panel.symbols):
        column = close[:, index]
        run = 1
        for position in range(column.size - 1, 0, -1):
            if column[position] != column[position - 1]:
                break
            run += 1
        if run >= min_run:
            out[symbol] = run
    return out


@dataclass(frozen=True)
class TradableSet:
    """Today's order-eligible names and, for every other name, why it is not."""

    on: date | None
    kept: tuple[str, ...]
    excluded: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        return {"kept": len(self.kept), "excluded": len(self.excluded)}


def tradable_set(
    universe: Universe,
    panel: PricePanel,
    min_dollar_volume: float,
    on: date | str | np.datetime64 | None = None,
    window: int = DEFAULT_LIQUIDITY_WINDOW,
    stale_run: int = DEFAULT_STALE_RUN,
    allow_survivorship_bias: bool = False,
) -> TradableSet:
    """The names that may be ordered on `on`, and one recorded reason per exclusion.

    The order of the screens is the order of severity, and the first reason wins,
    so a delisted name is reported as delisted rather than as illiquid. Every
    symbol in the universe appears exactly once across `kept` and `excluded`:
    a name that silently disappears between the universe and the order list is
    the bug this whole structure exists to make impossible.
    """
    requested = as_date(on) if on is not None else None
    when = universe.effective_date(on, allow_survivorship_bias)
    excluded: dict[str, str] = {}
    notes: list[str] = []

    members = set(universe.members(on, allow_survivorship_bias=allow_survivorship_bias))
    open_for_trading = set(universe.tradable(on, allow_survivorship_bias=allow_survivorship_bias))
    if requested is not None and requested < universe.as_of and not universe.point_in_time:
        notes.append(
            f"{universe.source!r} is not point-in-time; the roster of {universe.as_of} was applied "
            f"to {requested} at the caller's request and the result is survivorship-biased"
        )

    for symbol in universe.symbols:
        if symbol in open_for_trading:
            continue
        if symbol in members:
            excluded[symbol] = "halted"
        elif when is None:
            excluded[symbol] = "not a member of the universe"
        else:
            excluded[symbol] = universe.listing(symbol).status_on(when).value

    unclassified = set(universe.unclassified)
    candidates = []
    in_panel = set(panel.symbols)
    for symbol in sorted(open_for_trading):
        if symbol in unclassified:
            excluded[symbol] = "no sector bucket, so the concentration limit cannot be checked"
        elif symbol not in in_panel:
            excluded[symbol] = "no price history in the panel"
        else:
            candidates.append(symbol)
    if unclassified:
        notes.append(
            f"{len(unclassified)} name(s) carry no sector bucket and are therefore not orderable; "
            "a listings source without a classification leaves them there until one is added"
        )

    screen = liquidity_screen(panel, min_dollar_volume, symbols=candidates, window=window)
    excluded.update(screen.excluded)

    stale = stale_price_runs(panel, stale_run)
    kept: list[str] = []
    for symbol in screen.kept:
        if symbol in stale:
            excluded[symbol] = f"price unchanged for {stale[symbol]} sessions, which reads as a halt"
        else:
            kept.append(symbol)

    return TradableSet(on=requested, kept=tuple(kept), excluded=excluded, notes=tuple(notes))


# --- survivorship -----------------------------------------------------------


@dataclass(frozen=True)
class SurvivorshipReport:
    """How much of the universe's history is missing, or that the answer is unknown."""

    point_in_time: bool
    survivors: int
    delisted: int | None
    gap_pct: float | None
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Unknown is not a pass. A gap that was never measured is not a gap of zero."""
        return self.point_in_time and self.gap_pct is not None


def survivorship_gap(
    universe: Universe,
    start: date | str | np.datetime64,
    end: date | str | np.datetime64,
) -> SurvivorshipReport:
    """The share of the universe over [start, end] that has since left the market.

    On a source with no delisting history the count is `None`, not `0`. Those two
    look identical in a report and mean opposite things: one says nothing failed,
    the other says we would not know if it had.
    """
    first, last = as_date(start), as_date(end)
    if last < first:
        raise ValueError(f"the window ends ({last}) before it starts ({first})")
    survivors = len(universe.tradable(last)) if universe.point_in_time else len(universe.listings)
    if not universe.point_in_time:
        return SurvivorshipReport(
            point_in_time=False,
            survivors=survivors,
            delisted=None,
            gap_pct=None,
            notes=(
                f"{universe.source!r} carries no delisting history, so the number of names that "
                f"left between {first} and {last} is unmeasured rather than zero. Any backtest "
                "over this window is survivorship-biased by an unknown amount.",
            ),
        )
    gone = len(universe.delisted_between(first, last))
    total = survivors + gone
    return SurvivorshipReport(
        point_in_time=True,
        survivors=survivors,
        delisted=gone,
        gap_pct=(100.0 * gone / total) if total else 0.0,
        notes=(f"{gone} of {total} names present in the window had left by {last}.",),
    )


def survivorship_adjustment(
    universe: Universe,
    start: date | str | np.datetime64,
    end: date | str | np.datetime64,
) -> bool:
    """The `survivorship_adjustment` entry of `quality.REQUIRED_CHECKS`.

    That check has been in the required list since P0 with nothing computing it,
    which meant a health report either omitted it -- caught, because a missing
    check fails -- or carried a hand-written `True`. This is the measurement
    behind it: the check passes only when the universe can actually say who left.
    """
    return survivorship_gap(universe, start, end).ok
