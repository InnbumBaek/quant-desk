"""Which market a symbol trades on, declared rather than inferred.

A panel is rectangular in one calendar. `load_csv_panel` intersects dates across
symbols, which is right inside one market -- every US name shares the US session
calendar, so the intersection only removes genuine holes -- and wrong across two.
Korea and the United States keep different holidays, so a mixed panel loses every
Korean holiday and every US holiday at once and the surviving dates are the
trading calendar of neither market. Until now the loader had no way to tell the
two cases apart: the only thing between a mixed panel and a quietly truncated
backtest was `min_coverage` refusing it for the wrong reason.

Three decisions worth stating.

**A symbol's market is declared in `UNIVERSE`, never guessed from the ticker.** A
six-digit code looks like a KRX symbol and a three-letter code looks like a US
one, right until a vendor zero-pads a ticker or a foreign listing turns up. A
wrong guess picks the wrong currency and the wrong session close, and neither
surfaces as an error -- it surfaces as an alpha. An undeclared symbol raises.

**Markets are modelled by calendar, not by listing venue.** NYSE, Nasdaq and NYSE
Arca share one session calendar; KOSPI and KOSDAQ share another. A venue field
would be a fact we record and never read, and an unread fact rots. If a
venue-level difference ever starts to matter, it earns its own code then.

**Sessions per year is measured from the data, not declared per market.** An
annualisation constant that disagrees with the panel scales every Sharpe in the
report, and holding two independent numbers for one quantity is how a book passes
a limit it is breaching. Lunar holidays move every year, so a hardcoded 246 would
be wrong in some of them. See `sessions_per_year`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


class UnknownSymbol(KeyError):
    """A symbol whose market is not declared. Guessing it picks a currency for you."""


@dataclass(frozen=True)
class Market:
    """A set of instruments that share one session calendar and one currency."""

    code: str
    name: str
    timezone: str
    currency: str
    session_close_local: str
    notes: tuple[str, ...] = ()


MARKETS: dict[str, Market] = {
    "US": Market(
        code="US",
        name="United States equities and ETFs",
        timezone="America/New_York",
        currency="USD",
        session_close_local="16:00",
        notes=(
            "NYSE, Nasdaq and NYSE Arca share this calendar, so the venue is not modelled.",
            "Early closes at 13:00 exist and do not change a daily-close panel.",
        ),
    ),
    "KR": Market(
        code="KR",
        name="Korea Exchange (KOSPI and KOSDAQ)",
        timezone="Asia/Seoul",
        currency="KRW",
        session_close_local="15:30",
        notes=(
            "KOSPI and KOSDAQ share one calendar.",
            "Lunar holidays move each year, so the calendar is read from the data, never hardcoded.",
        ),
    ),
}


#: Symbol -> market code. Every symbol a run touches has to appear here first.
#: The list is short on purpose: it holds what has actually been fetched plus the
#: liquid US ETFs a dispatch is likely to ask for. Adding a symbol is a one-line
#: diff, which is the point -- the declaration is reviewable, a heuristic is not.
UNIVERSE: dict[str, str] = {
    # US ETFs. The five defaults of scripts/fetch_prices.py come first.
    "SPY": "US",
    "QQQ": "US",
    "IWM": "US",
    "TLT": "US",
    "GLD": "US",
    "DIA": "US",
    "EFA": "US",
    "EEM": "US",
    "VTI": "US",
    "VOO": "US",
    "AGG": "US",
    "IEF": "US",
    "SHY": "US",
    "TIP": "US",
    "LQD": "US",
    "HYG": "US",
    "VNQ": "US",
    "SLV": "US",
    "USO": "US",
    "XLB": "US",
    "XLE": "US",
    "XLF": "US",
    "XLI": "US",
    "XLK": "US",
    "XLP": "US",
    "XLU": "US",
    "XLV": "US",
    "XLY": "US",
    # KRX. Two seeds, enough to exercise the Korean branch end to end; the real
    # universe arrives with the KRX adapter and its own ADR.
    "005930": "KR",  # 삼성전자
    "000660": "KR",  # SK하이닉스
}


def market_for(symbol: str, mapping: Mapping[str, str] | None = None) -> Market:
    """The market a symbol trades on, or `UnknownSymbol`.

    Raising is the whole design. A symbol we cannot place has no currency and no
    session close we can name, and a panel built on an unnamed calendar answers a
    question nobody asked.
    """
    table = UNIVERSE if mapping is None else mapping
    try:
        code = table[symbol]
    except KeyError as err:
        raise UnknownSymbol(
            f"{symbol!r} is not in the universe; add a line to core/data/markets.py UNIVERSE "
            f"naming its market (known markets: {', '.join(sorted(MARKETS))}). "
            "A symbol's market is not inferable from its ticker."
        ) from err
    if code not in MARKETS:
        raise ValueError(f"{symbol!r} is declared in market {code!r}, which is not in MARKETS")
    return MARKETS[code]


def group_by_market(
    symbols: Iterable[str], mapping: Mapping[str, str] | None = None
) -> dict[str, tuple[str, ...]]:
    """Split symbols into one group per market, both keys and members sorted.

    Sorted because the grouping feeds a snapshot id: the same set of symbols has
    to produce the same id whatever order it arrived in.
    """
    groups: dict[str, list[str]] = {}
    for symbol in symbols:
        groups.setdefault(market_for(symbol, mapping).code, []).append(symbol)
    return {code: tuple(sorted(members)) for code, members in sorted(groups.items())}


def currencies(codes: Iterable[str]) -> dict[str, str]:
    """Market code -> currency, for callers deciding whether two panels can be added up."""
    out: dict[str, str] = {}
    for code in codes:
        if code not in MARKETS:
            raise ValueError(f"unknown market {code!r}; known: {', '.join(sorted(MARKETS))}")
        out[code] = MARKETS[code].currency
    return out


def sessions_per_year(dates: np.ndarray, min_span_days: int = 350) -> float:
    """Sessions per year measured from the observed dates.

    The estimator counts intervals, not dates: `n - 1` gaps spanning `s` calendar
    days is `(n - 1) * 365.25 / s` sessions a year. Counting dates instead would
    read a full year of 252 sessions as 252.9, because the first date opens the
    span without closing a gap.

    A span under `min_span_days` raises rather than returning a noisy number. An
    annualisation constant guessed from two months is a constant that will be
    wrong in the Sharpe of every report built on it, and a caller who genuinely
    knows the right value can pass it explicitly instead. The default sits at 350
    rather than 365 because a calendar year of sessions spans 363 to 366 days
    depending on where its weekends fall, and refusing a genuine year would push
    callers into passing a constant by hand -- which is the habit this replaces.
    """
    if dates.ndim != 1:
        raise ValueError(f"dates must be one-dimensional, got shape {dates.shape}")
    if dates.shape[0] < 2:
        raise ValueError("a rate needs at least two dates")
    days = dates.astype("datetime64[D]").astype("int64")
    gaps = np.diff(days)
    if np.any(gaps == 0):
        raise ValueError("dates repeat; a calendar cannot have the same session twice")
    if np.any(gaps < 0):
        raise ValueError("dates are not sorted ascending")
    span = int(days[-1] - days[0])
    if span < min_span_days:
        raise ValueError(
            f"the panel spans {span} days, under the {min_span_days} required to measure a yearly "
            "rate; pass the rate explicitly rather than annualising from a short sample"
        )
    return float((dates.shape[0] - 1) * 365.25 / span)
