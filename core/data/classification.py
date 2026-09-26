"""Which bucket a symbol counts against for the concentration limit.

`limits.yaml` caps `sector_max` at 20% of NAV. Until now nothing computed a
sector weight, so `check_pod` read an absent `sector_weights` as an empty mapping
and the limit passed every book ever submitted. This is the declared table that
makes it checkable, and `core/risk/exposure.py` is what sums it.

**These are asset-class buckets, not GICS sectors.** For a single stock the two
coincide closely enough; for an ETF they do not, because SPY holds every sector
and a true decomposition needs the fund's holdings, which we do not fetch. So
`SPY` counts as broad US equity rather than being split across eleven sectors.
That serves what the limit is for -- do not put a fifth of the fund in one place
-- and it is deliberately not called a GICS sector anywhere, because a look
through to holdings is a separate piece of work with its own data source.

Declared, never inferred, for the same reason as a symbol's market
(`core/data/markets.py`): a ticker does not say what a fund holds, and a guess
here moves a concentration limit without moving anything visible.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from core.data.markets import MARKETS


class UnclassifiedSymbol(KeyError):
    """A symbol with no bucket. The concentration limit cannot be checked without one."""


#: Symbol -> bucket. Add a line when a symbol enters the universe; the buckets
#: themselves are coarse on purpose, because a bucket that splits finely enough
#: to never bind is a limit that never binds.
SECTORS: dict[str, str] = {
    # US equity, by breadth rather than by industry
    "SPY": "us_equity_broad",
    "VOO": "us_equity_broad",
    "VTI": "us_equity_broad",
    "DIA": "us_equity_broad",
    "IWM": "us_equity_small",
    "QQQ": "us_equity_growth",
    # US equity, by industry (these funds are the industry)
    "XLB": "materials",
    "XLE": "energy",
    "XLF": "financials",
    "XLI": "industrials",
    "XLK": "technology",
    "XLP": "consumer_staples",
    "XLU": "utilities",
    "XLV": "health_care",
    "XLY": "consumer_discretionary",
    "VNQ": "real_estate",
    # Non-US equity
    "EFA": "developed_ex_us_equity",
    "EEM": "emerging_equity",
    # Rates and credit, bucketed by what drives them
    "TLT": "rates_long",
    "IEF": "rates_intermediate",
    "SHY": "rates_short",
    "TIP": "rates_inflation",
    "AGG": "rates_aggregate",
    "LQD": "credit_investment_grade",
    "HYG": "credit_high_yield",
    # Commodities
    "GLD": "commodity_gold",
    "SLV": "commodity_silver",
    "USO": "commodity_oil",
    # KRX
    "005930": "technology",
    "000660": "technology",
}


def bare(symbol: str) -> str:
    """`US:SPY` -> `SPY`. A cross-market panel qualifies its symbols (ADR-0013)."""
    code, _, rest = symbol.partition(":")
    return rest if rest and code in MARKETS else symbol


def sector_for(symbol: str, mapping: Mapping[str, str] | None = None) -> str:
    """The bucket a symbol counts against, or `UnclassifiedSymbol`."""
    table = SECTORS if mapping is None else mapping
    try:
        return table[bare(symbol)]
    except KeyError as err:
        raise UnclassifiedSymbol(
            f"{symbol!r} has no sector bucket; add a line to core/data/classification.py SECTORS. "
            "Without one the sector concentration limit cannot be checked for this book."
        ) from err


def sector_weights(
    weights: Mapping[str, float], mapping: Mapping[str, str] | None = None
) -> dict[str, float]:
    """Net weight per bucket.

    Net, not gross: a long and a short inside one bucket are a spread, and the
    limit is about having a fifth of the fund riding on one thing. The gross
    leverage limit is what catches a book that is large on both sides.
    """
    out: dict[str, float] = {}
    for symbol, weight in weights.items():
        bucket = sector_for(symbol, mapping)
        out[bucket] = out.get(bucket, 0.0) + float(weight)
    return out


def unclassified(symbols: Iterable[str], mapping: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The symbols with no bucket, for a caller that wants to report before it fails."""
    table = SECTORS if mapping is None else mapping
    return tuple(symbol for symbol in symbols if bare(symbol) not in table)
