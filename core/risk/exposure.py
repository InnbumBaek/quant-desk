"""Measuring the pod snapshot the limit engine checks.

`core/risk/limits.py` reads a snapshot -- gross, net, weights, sector weights,
style betas, liquidation days, realised volatility, drawdown -- and returns
breaches. Nothing computed those fields, so a caller assembled them by hand and
whatever it left out went unchecked. This module measures them from one object,
the pod's weight history, so every field in a snapshot comes from the same book.

Three decisions worth stating.

**One weight history, not a row.** Exposure, sector weights and liquidation days
describe the book being put on, so they come from the last row. Volatility and
style betas describe how the book behaves, so they come from the history. Taking
them from different objects is how a book ends up with two realised volatilities
and passes the limit on one of them.

**Volatility is not re-implemented here.** It comes from
`core/strategies/overlay.realised_volatility`, which is what the volatility
overlay scales on. Two estimates of one quantity let a book breach the band on
one and clear it on the other.

**An unmeasurable field is `None`, never a zero.** A missing dollar volume, a
factor file that does not reach today, a history shorter than the window: each
returns `None` with the reason in `notes`, and `check_pod` reads `None` as a
breach (ADR-0015). A zero would read as "measured, and fine".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.backtest.engine import PricePanel
from core.data.classification import sector_weights, unclassified
from core.data.factors import FactorPanel
from core.strategies.overlay import realised_volatility


@dataclass(frozen=True)
class PodExposure:
    """The snapshot, and what could not be measured for it."""

    snapshot: dict[str, Any]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def unmeasured(self) -> tuple[str, ...]:
        return tuple(sorted(key for key, value in self.snapshot.items() if value is None))


def pod_returns(weights: np.ndarray, panel: PricePanel) -> np.ndarray:
    """The book's gross return series: row `t` is the weights of `t` earning bar `t`.

    The same convention as `engine.simulate`, minus costs. Costs belong in a P&L;
    a beta and a volatility are about exposure, and charging them here would make
    this series disagree with the one the overlay scales on.
    """
    returns = panel.bar_returns
    if weights.shape != panel.close.shape:
        raise ValueError(f"weights {weights.shape} must match the close panel {panel.close.shape}")
    return np.sum(weights[:-1] * returns, axis=1)


def style_betas(
    book_returns: np.ndarray,
    factor_matrix: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, float]:
    """OLS betas of the book on the factors, with an intercept that is discarded.

    The intercept is G4's business (residual alpha). What a limit cares about is
    the slopes: a market-neutral mandate that is carrying a market beta of 0.8 is
    running a different strategy from the one it was approved for.
    """
    if book_returns.size != factor_matrix.shape[0]:
        raise ValueError(f"{book_returns.size} book returns against {factor_matrix.shape[0]} factor rows")
    design = np.column_stack([np.ones(book_returns.size), factor_matrix])
    beta, *_ = np.linalg.lstsq(design, book_returns, rcond=None)
    return {name: float(value) for name, value in zip(names, beta[1:], strict=True)}


def liquidation_days(
    current: np.ndarray,
    panel: PricePanel,
    capital: float,
    participation: float,
    window: int,
) -> float:
    """Days to unwind the largest position at `participation` of its recent ADV.

    The worst name decides. An average would let one illiquid position hide
    behind four liquid ones, and it is the worst name that is still on the book
    when the fund needs to be flat.
    """
    if panel.dollar_volume is None:
        raise ValueError("the panel carries no dollar volume, so a liquidation horizon is unmeasurable")
    if not 0.0 < participation <= 1.0:
        raise ValueError(f"participation {participation} is not a share of volume")
    recent = panel.dollar_volume[-window:, :]
    adv = np.median(recent, axis=0)
    if not np.all(np.isfinite(adv)) or np.any(adv <= 0.0):
        raise ValueError("a symbol has no positive median dollar volume in the window")
    notional = np.abs(current) * capital
    return float(np.max(notional / (participation * adv)))


def pod_snapshot(
    weights: np.ndarray,
    panel: PricePanel,
    factors: FactorPanel | None = None,
    capital: float = 1_000_000.0,
    participation: float = 0.20,
    window: int = 60,
    periods_per_year: int = 252,
    backtest_drawdowns: np.ndarray | None = None,
) -> PodExposure:
    """Measure everything `check_pod` reads, from the pod's own weight history.

    `participation` is the share of ADV the unwind assumes, matching the note on
    `liquidation_days_max` in `limits.yaml` (20% of ADV). `backtest_drawdowns` is
    the drawdown distribution the pod's backtest produced; without it the
    percentile cannot be computed and the drawdown check has nothing to compare
    against, which is a breach rather than a pass.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError("weights must be a 2-D (dates x symbols) history")
    if weights.shape[0] < 2:
        raise ValueError("a weight history needs at least two rows to have a return")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contain a non-finite value")

    notes: list[str] = []
    current = weights[-1]
    book = {symbol: float(value) for symbol, value in zip(panel.symbols, current, strict=True)}

    snapshot: dict[str, Any] = {
        "gross": float(np.sum(np.abs(current))),
        "net": float(np.sum(current)),
        "weights": book,
        "sector_weights": None,
        "style_betas": None,
        "liquidation_days": None,
        "realised_volatility": None,
        "drawdown": None,
        "backtest_dd_pct": None,
    }

    missing_buckets = unclassified(panel.symbols)
    if missing_buckets:
        notes.append(
            f"sector weights unmeasured: no bucket for {', '.join(missing_buckets)}; "
            "add them to core/data/classification.py"
        )
    else:
        snapshot["sector_weights"] = sector_weights(book)

    series = pod_returns(weights, panel)

    measured = realised_volatility(
        weights, np.asarray(panel.close, dtype=float), lookback=window, periods_per_year=periods_per_year
    )
    if np.isfinite(measured[-1]):
        snapshot["realised_volatility"] = float(measured[-1])
    else:
        notes.append(
            f"realised volatility unmeasured: the history holds {weights.shape[0]} rows, "
            f"short of the {window}-bar window"
        )

    if series.size >= window:
        equity = np.cumprod(1.0 + series)
        peak = np.maximum.accumulate(equity)
        snapshot["drawdown"] = float(equity[-1] / peak[-1] - 1.0)
        if backtest_drawdowns is None:
            notes.append(
                "backtest drawdown percentile unmeasured: no backtest drawdown distribution was "
                "supplied, so the live drawdown has nothing to be compared against"
            )
        else:
            reference = np.asarray(backtest_drawdowns, dtype=float)
            if reference.size == 0 or not np.all(np.isfinite(reference)):
                notes.append("backtest drawdown distribution is empty or not finite")
            else:
                # The reference holds drawdowns as positive depths; the live one is
                # negative. Compare like with like.
                depth = -snapshot["drawdown"]
                snapshot["backtest_dd_pct"] = float(100.0 * np.mean(reference <= depth))
    else:
        notes.append(f"drawdown unmeasured: {series.size} return(s) is short of the {window}-bar window")

    try:
        snapshot["liquidation_days"] = liquidation_days(current, panel, capital, participation, window)
    except ValueError as error:
        notes.append(f"liquidation days unmeasured: {error}")

    if factors is None:
        notes.append("style betas unmeasured: no factor file was supplied")
    else:
        try:
            matrix = factors.align_to_bars(panel.dates)
            if series.size < window:
                raise ValueError(f"{series.size} return(s) is short of the {window}-bar window")
            snapshot["style_betas"] = style_betas(series[-window:], matrix[-window:], factors.names)
        except ValueError as error:
            notes.append(f"style betas unmeasured: {error}")

    return PodExposure(snapshot=snapshot, notes=tuple(notes))
