"""The reporting metric set: CAGR, volatility, Sharpe, Sortino, MDD, Calmar,
turnover, and beta/alpha against a benchmark.

These are *reporting* numbers, not gate numbers. `core/backtest/stats.py` holds
the statistics a gate compares against `core/risk/limits.yaml`; nothing here is
compared against a threshold, and nothing here may become a pass condition.
Keeping them apart is the point: a reporting metric that starts gating would be
a gate criterion with no ADR behind it.

The one rule that carries over from the gate side is that absence is recorded as
absence. A Sortino with no losing day, a Calmar with no drawdown and a beta with
no benchmark are all `None` with a note saying why, never 0.0 -- a zero here
would read as a measurement and get quoted as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.backtest import stats


@dataclass(frozen=True)
class Performance:
    """One strategy's reported performance. `None` means not measurable."""

    periods: int
    periods_per_year: int
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    sortino: float | None = None
    calmar: float | None = None
    turnover_annual: float | None = None
    beta: float | None = None
    alpha_annual: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "periods": self.periods,
            "periods_per_year": self.periods_per_year,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "turnover_annual": self.turnover_annual,
            "beta": self.beta,
            "alpha_annual": self.alpha_annual,
            "notes": list(self.notes),
        }


def _cagr(returns: np.ndarray, periods_per_year: int) -> tuple[float, str | None]:
    equity = float(np.prod(1.0 + returns))
    if equity <= 0.0:
        # A book that compounds to zero has no geometric mean return. Reporting
        # -100% is the truthful summary; reporting a root of a negative number
        # is not.
        return -1.0, "equity compounded to zero or below; CAGR reported as -100%"
    return equity ** (periods_per_year / len(returns)) - 1.0, None


def _sortino(returns: np.ndarray, periods_per_year: int) -> tuple[float | None, str | None]:
    downside = returns[returns < 0.0]
    if downside.size == 0:
        return None, "no losing period in the sample, so Sortino is undefined rather than infinite"
    # Downside deviation against a zero target, the usual convention: the mean
    # square is taken over the whole sample, not over the losing days only.
    deviation = math.sqrt(float(np.sum(downside**2)) / len(returns))
    if deviation == 0.0:
        return None, "downside deviation is zero, so Sortino is undefined"
    return float(returns.mean() / deviation * math.sqrt(periods_per_year)), None


def _beta_alpha(
    returns: np.ndarray, benchmark: np.ndarray, periods_per_year: int
) -> tuple[float | None, float | None, str | None]:
    variance = float(benchmark.var(ddof=1))
    if variance == 0.0:
        return None, None, "the benchmark does not move in this sample, so beta is undefined"
    covariance = float(np.cov(returns, benchmark, ddof=1)[0, 1])
    beta = covariance / variance
    alpha = (float(returns.mean()) - beta * float(benchmark.mean())) * periods_per_year
    return beta, alpha, None


def performance_summary(
    returns: np.ndarray,
    turnover: np.ndarray | None = None,
    benchmark: np.ndarray | None = None,
    periods_per_year: int = 252,
) -> Performance:
    """Summarise a net return series.

    `turnover[t]` is the fraction of the book traded into bar `t`, as the engine
    reports it; the annualised figure is its mean scaled by the period count.
    `benchmark` must be the same length as `returns` -- a shorter benchmark is a
    different sample, so it raises rather than being padded or truncated.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError(f"returns must be one series, got shape {returns.shape}")
    if len(returns) < 2:
        raise ValueError(f"{len(returns)} periods is not enough to report on")
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns contain a non-finite value; the simulation is the place to fix that")

    notes: list[str] = []
    cagr, note = _cagr(returns, periods_per_year)
    if note:
        notes.append(note)

    volatility = float(returns.std(ddof=1) * math.sqrt(periods_per_year))
    sharpe = stats.sharpe_ratio(returns, periods_per_year)
    max_dd = stats.max_drawdown(returns)

    sortino, note = _sortino(returns, periods_per_year)
    if note:
        notes.append(note)

    if max_dd > 0.0:
        calmar = cagr / max_dd
    else:
        calmar = None
        notes.append("no drawdown in the sample, so Calmar is undefined rather than infinite")

    turnover_annual: float | None = None
    if turnover is None:
        notes.append("no turnover series supplied: turnover is unreported, not zero")
    else:
        turnover = np.asarray(turnover, dtype=float)
        if turnover.shape != returns.shape:
            raise ValueError(
                f"turnover is {turnover.shape} and returns are {returns.shape}; "
                "they must describe the same bars"
            )
        turnover_annual = float(turnover.mean() * periods_per_year)

    beta = alpha = None
    if benchmark is None:
        notes.append("no benchmark supplied: beta and alpha are unreported, not zero")
    else:
        benchmark = np.asarray(benchmark, dtype=float)
        if benchmark.shape != returns.shape:
            raise ValueError(
                f"benchmark is {benchmark.shape} and returns are {returns.shape}; "
                "a benchmark of a different length is a different sample"
            )
        beta, alpha, note = _beta_alpha(returns, benchmark, periods_per_year)
        if note:
            notes.append(note)

    return Performance(
        periods=len(returns),
        periods_per_year=periods_per_year,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        max_drawdown=max_dd,
        sortino=sortino,
        calmar=calmar,
        turnover_annual=turnover_annual,
        beta=beta,
        alpha_annual=alpha,
        notes=tuple(notes),
    )
