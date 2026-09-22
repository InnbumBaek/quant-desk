"""Volatility targeting, as an overlay that only ever reduces.

Nearly every large systematic fund sizes to a volatility target rather than to a
notional, and the limits table now carries the owner's band (`pod.target_volatility`,
ADR-0009). This is the piece that connects the two: it measures what the book's
volatility actually was and scales the book down when that is above target.

**It never scales up.** CLAUDE.md rule 5 says an overlay may only reduce a
position, and the arithmetic reason is the same as the rule: scaling up to reach
a volatility target is borrowing, and the pod gross limit is 1.0. A quiet market
is not a reason to lever, so undershooting the band is reported and left alone --
the same asymmetry the limits engine applies, where above the band is a breach
and below it is not.

**An unmeasured volatility scales to zero, not to one.** During the warm-up the
overlay has no estimate, and the consistent choice with `volatility_breaches()`
blocking an unmeasured book is to hold no position rather than a full one.
Treating "not yet known" as "fine" is how a fail-closed rule quietly becomes a
fail-open one.

Point-in-time by construction: the estimate for row `t` uses the weights held up
to `t-1` and the returns realised up to and including bar `t`, both of which are
known when the row `t` decision is made.
"""

from __future__ import annotations

import math

import numpy as np

from core.strategies.base import bar_returns


def realised_volatility(
    weights: np.ndarray,
    close: np.ndarray,
    lookback: int = 60,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Annualised realised volatility of the book, as known at each row.

    `NaN` until a full window of realised returns exists. That absence is
    deliberate and is what the caller must not read as zero.
    """
    weights = np.asarray(weights, dtype=float)
    close = np.asarray(close, dtype=float)
    if weights.shape != close.shape:
        raise ValueError(f"weights are {weights.shape} and close is {close.shape}")
    if lookback < 2:
        raise ValueError(f"lookback {lookback} is too short to estimate a volatility")

    returns = bar_returns(close)
    # The return earned into bar t by the book held into it. Row 0 has no prior
    # holding, so it earns nothing and stays absent.
    realised_pnl = np.full(close.shape[0], np.nan)
    realised_pnl[1:] = np.sum(weights[:-1] * returns[1:], axis=1)

    out = np.full(close.shape[0], np.nan)
    scale = math.sqrt(periods_per_year)
    for t in range(lookback, close.shape[0]):
        window = realised_pnl[t - lookback + 1 : t + 1]
        if np.all(np.isfinite(window)):
            out[t] = float(window.std(ddof=1)) * scale
    return out


def vol_target(
    weights: np.ndarray,
    close: np.ndarray,
    target: float,
    lookback: int = 60,
    periods_per_year: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale the book down toward `target` annualised volatility.

    Returns the scaled weights and the realised-volatility series that produced
    them. The series is the caller's `realised_volatility` for the risk
    snapshot, so the number the overlay acted on is the same number
    `core/risk/limits.py` checks -- two independent estimates of the same
    quantity is how a book passes the limit while breaching it.
    """
    if target <= 0:
        raise ValueError(f"target volatility {target} is not a target")
    weights = np.asarray(weights, dtype=float)
    measured = realised_volatility(weights, close, lookback, periods_per_year)

    scale = np.zeros(weights.shape[0])
    known = np.isfinite(measured)
    moving = known & (measured > 0)
    # Reduce only: a book quieter than target keeps its size, it does not grow.
    scale[moving] = np.minimum(1.0, target / measured[moving])
    # A measured zero means the book was flat, not that it is unmeasured.
    scale[known & (measured == 0)] = 1.0
    return weights * scale[:, None], measured
