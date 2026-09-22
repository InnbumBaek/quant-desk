"""The strategy families the large systematic funds actually run, as editable code.

What is here is what the published record says those funds trade and what our
data can honestly support: daily closes for a handful of liquid instruments. The
families are the ones with decades of out-of-sample evidence and named papers
behind them, because a strategy whose only evidence is our own backtest is
exactly what the gates exist to reject.

- **Time-series momentum / trend** is the managed-futures core (the AHL, Winton,
  Aspect lineage). Moskowitz, Ooi & Pedersen (2012); Hurst, Ooi & Pedersen (2017).
- **Cross-sectional momentum** is the oldest equity factor still standing.
  Jegadeesh & Titman (1993); Asness, Moskowitz & Pedersen (2013).
- **Short-horizon reversal** is the family the highest-turnover statistical funds
  are publicly associated with. Lehmann (1990); Lo & MacKinlay (1990).
- **Betting against beta** is the defensive/low-risk premium. Frazzini &
  Pedersen (2014).
- **Blending many weak, decorrelated signals** is the structural insight those
  funds share, and it beats sharpening any one of them. Grinold & Kahn's
  fundamental law: IR is roughly IC times the square root of breadth.

What is deliberately **not** here: value, quality and carry. Each needs data this
repository does not have (fundamentals from DART/EDGAR, yield curves), and a
value factor reconstructed from price history is not a value factor. They are
registered as unavailable with the reason, so `names()` tells the truth about
what can be run today.

Every function is point-in-time: row `t` is computed from `close[:t+1]` only, and
`tests/strategies/test_library.py` re-derives each one on truncated panels
through the same G0 scan the gates use rather than trusting this sentence.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from core.strategies.base import (
    DEFAULT_GROSS,
    Strategy,
    bar_returns,
    blend,
    cross_sectional_rank,
    dollar_neutral,
    register,
    scale_to_gross,
    trailing_return,
)


def ts_momentum(close: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Long what has risen over `lookback`, short what has fallen, equal weight.

    The sign of the trailing return, not its size: the evidence for trend is that
    the direction persists, and sizing by the magnitude turns it into a bet on
    the last move being repeated.
    """
    lookback = int(params["lookback"])
    past = trailing_return(close, lookback)
    signs = np.where(np.isfinite(past), np.sign(past), 0.0)
    return scale_to_gross(signs, float(params["gross"]))


def xs_momentum(close: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Cross-sectional momentum, dollar neutral, with the most recent bars skipped.

    The skip is not decoration. The last few days of a momentum window carry
    short-horizon reversal, which is the opposite signal, so 12-1 rather than
    12-0 is the convention the literature settled on.
    """
    lookback, skip = int(params["lookback"]), int(params["skip"])
    scores = np.full_like(close, np.nan)
    start = lookback + skip
    if start < close.shape[0]:
        # close[t - skip] / close[t - skip - lookback] - 1, for every t >= start.
        recent = close[start - skip : close.shape[0] - skip]
        older = close[: close.shape[0] - start]
        scores[start:] = recent / older - 1.0
    return dollar_neutral(cross_sectional_rank(scores), float(params["gross"]))


def short_term_reversal(close: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Short the recent winners over a few days, buy the recent losers.

    The same quantity as `xs_momentum` with the sign flipped and a short window.
    That both are real at different horizons is why horizon diversification is a
    risk budget line in the limits table and not a preference.
    """
    lookback = int(params["lookback"])
    scores = -trailing_return(close, lookback)
    return dollar_neutral(cross_sectional_rank(scores), float(params["gross"]))


def rolling_beta(close: np.ndarray, lookback: int) -> np.ndarray:
    """Beta of each instrument against the equal-weight panel, window ending at `t`."""
    returns = bar_returns(close)
    market = returns.mean(axis=1)
    out = np.full_like(close, np.nan)
    for t in range(lookback, close.shape[0]):
        window = returns[t - lookback + 1 : t + 1]
        bench = market[t - lookback + 1 : t + 1]
        if not np.all(np.isfinite(window)) or not np.all(np.isfinite(bench)):
            continue
        variance = bench.var(ddof=1)
        if variance <= 0:
            continue
        centred_bench = bench - bench.mean()
        out[t] = (centred_bench @ (window - window.mean(axis=0))) / ((lookback - 1) * variance)
    return out


def betting_against_beta(close: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Long the low-beta instruments, short the high-beta ones, dollar neutral.

    The premium is for taking the unpopular side of leverage aversion: investors
    who cannot lever buy high-beta assets instead, and overpay for them. Our own
    book cannot lever either (the gross limit is 1.0), which is worth noticing
    rather than hiding -- we are taking the same premium under the same
    constraint that creates it.
    """
    beta = rolling_beta(close, int(params["lookback"]))
    return dollar_neutral(cross_sectional_rank(-beta), float(params["gross"]))


def trend_breakout(close: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Donchian channel breakout: flip long on a new high, short on a new low, else hold.

    Holding between breakouts is the whole mechanism -- it is what makes turnover
    low and lets a trend run. The state only ever depends on bars at or before
    `t`, so recomputing on a truncated panel reproduces the same path.
    """
    lookback, gross = int(params["lookback"]), float(params["gross"])
    n_rows, n_cols = close.shape
    signs = np.zeros_like(close)
    if lookback < 1:
        return signs
    state = np.zeros(n_cols)
    for t in range(lookback, n_rows):
        window = close[t - lookback : t]
        state = np.where(close[t] >= window.max(axis=0), 1.0, state)
        state = np.where(close[t] <= window.min(axis=0), -1.0, state)
        signs[t] = state
    return scale_to_gross(signs, gross)


def multi_signal(close: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Equal blend of four decorrelated families at different horizons.

    This is the structural idea, not a fifth signal: breadth raises the
    information ratio faster than sharpening any single forecast does. The
    horizons are deliberately spread (days to a year) because that is what makes
    the sleeves decorrelated, and a single-horizon book is capped at 60% of the
    risk budget by the limits table.
    """
    gross = float(params["gross"])
    parts = [
        ts_momentum(close, {"lookback": params["trend_lookback"], "gross": 1.0}),
        xs_momentum(
            close,
            {"lookback": params["xs_lookback"], "skip": params["xs_skip"], "gross": 1.0},
        ),
        short_term_reversal(close, {"lookback": params["reversal_lookback"], "gross": 1.0}),
        betting_against_beta(close, {"lookback": params["beta_lookback"], "gross": 1.0}),
    ]
    return blend(parts, gross)


# --- registry ---------------------------------------------------------------

register(
    Strategy(
        name="ts_momentum",
        family="trend",
        fn=ts_momentum,
        defaults={"lookback": 60, "gross": DEFAULT_GROSS},
        grid=(
            {"lookback": 20},
            {"lookback": 40},
            {"lookback": 60},
            {"lookback": 120},
            {"lookback": 250},
        ),
        citation="Moskowitz, Ooi & Pedersen (2012), 'Time Series Momentum', JFE 104(2)",
        notes=("Managed-futures core: AHL, Winton, Aspect lineage.",),
    )
)

register(
    Strategy(
        name="xs_momentum",
        family="momentum",
        fn=xs_momentum,
        defaults={"lookback": 250, "skip": 21, "gross": DEFAULT_GROSS},
        grid=(
            {"lookback": 120, "skip": 21},
            {"lookback": 180, "skip": 21},
            {"lookback": 250, "skip": 21},
            {"lookback": 250, "skip": 5},
        ),
        citation="Jegadeesh & Titman (1993), JF 48(1); Asness, Moskowitz & Pedersen (2013), JF 68(3)",
    )
)

register(
    Strategy(
        name="short_term_reversal",
        family="reversal",
        fn=short_term_reversal,
        defaults={"lookback": 5, "gross": DEFAULT_GROSS},
        grid=({"lookback": 2}, {"lookback": 3}, {"lookback": 5}, {"lookback": 10}),
        citation="Lehmann (1990), QJE 105(1); Lo & MacKinlay (1990), RFS 3(2)",
        notes=("The high-turnover statistical family. Costs decide it, not the signal.",),
    )
)

register(
    Strategy(
        name="betting_against_beta",
        family="defensive",
        fn=betting_against_beta,
        defaults={"lookback": 120, "gross": DEFAULT_GROSS},
        grid=({"lookback": 60}, {"lookback": 120}, {"lookback": 250}),
        citation="Frazzini & Pedersen (2014), 'Betting Against Beta', JFE 111(1)",
    )
)

register(
    Strategy(
        name="trend_breakout",
        family="trend",
        fn=trend_breakout,
        defaults={"lookback": 100, "gross": DEFAULT_GROSS},
        grid=({"lookback": 50}, {"lookback": 100}, {"lookback": 200}),
        citation="Hurst, Ooi & Pedersen (2017), 'A Century of Evidence on Trend-Following'",
    )
)

register(
    Strategy(
        name="multi_signal",
        family="ensemble",
        fn=multi_signal,
        defaults={
            "trend_lookback": 60,
            "xs_lookback": 250,
            "xs_skip": 21,
            "reversal_lookback": 5,
            "beta_lookback": 120,
            "gross": DEFAULT_GROSS,
        },
        grid=(
            {"trend_lookback": 40, "reversal_lookback": 3},
            {"trend_lookback": 60, "reversal_lookback": 5},
            {"trend_lookback": 120, "reversal_lookback": 10},
        ),
        citation="Grinold & Kahn (1999), Active Portfolio Management, ch. 6 (fundamental law)",
        notes=(
            "Equal blend, not risk weighted: risk weighting needs each sleeve's return "
            "series and belongs in core/portfolio/allocate.py.",
        ),
    )
)

# --- families this repository cannot honestly run yet ------------------------

for _name, _family, _needs, _why, _cite in (
    (
        "value",
        "value",
        ("close", "fundamentals"),
        "book value, earnings and cash flow are not in this repository; a value "
        "factor rebuilt from price history is a reversal signal wearing its name",
        "Fama & French (1992), JF 47(2); Asness, Frazzini & Pedersen (2019), RFS 32(1)",
    ),
    (
        "quality",
        "quality",
        ("close", "fundamentals"),
        "profitability, growth and leverage need filings (DART, SEC EDGAR)",
        "Asness, Frazzini & Pedersen (2019), 'Quality Minus Junk', RAST 24(1)",
    ),
    (
        "carry",
        "carry",
        ("close", "yields"),
        "carry is a yield difference, and no yield or futures curve is loaded",
        "Koijen, Moskowitz, Pedersen & Vrugt (2018), 'Carry', JFE 127(2)",
    ),
):
    register(
        Strategy(
            name=_name,
            family=_family,
            fn=lambda close, params: np.zeros_like(close),
            defaults={"gross": DEFAULT_GROSS},
            grid=(),
            citation=_cite,
            requires=_needs,
            available=False,
            unavailable_because=_why,
        )
    )
