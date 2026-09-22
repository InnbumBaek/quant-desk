"""The runner that turns market data into a `Submission` the gates can judge.

`core/backtest/gates.py` decides; this module is what feeds it. Until now a
submission packet had to be hand-assembled, which means the honest parts of the
protocol -- the trial count, the point-in-time convention, the cost drag -- were
whatever the caller chose to put in. Here they are properties of the code:

- A weight decided on bar `t` earns the return of bar `t -> t+1`, never bar
  `t-1 -> t`. This is the single most common way a backtest invents a Sharpe,
  so the alignment lives in one function (`simulate`) and is tested directly.
- Every point of the parameter grid is run and kept, so `Submission.trial_returns`
  carries the true N. The deflated Sharpe and PBO are only as honest as that
  count; a caller cannot quietly submit the winner of 5,000 trials as N=1.
- Costs are charged on turnover at the traded bar, and the double-cost series
  G5 needs is produced by the same code path rather than a second model.
- Gross leverage is checked against `core/risk/limits.yaml` before anything is
  simulated. A strategy that wants more leverage than the pod limit is rejected
  here, not silently rescaled into something that was never tested.

What this module does not do: it does not fetch data, and it does not invent a
factor model. If the caller has no factor returns, the residual-alpha regression
in G4 runs against an equal-weight return of the panel itself -- a market proxy,
not FF5+MOM -- and `FactorSource.PANEL_PROXY` says so in the submission's
provenance so nobody reads that t-statistic as the real thing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from core.backtest.cv import purged_kfold
from core.backtest.gates import Submission
from core.backtest.leakage import LeakReport, lookahead_scan
from core.risk.limits import load_limits

BPS = 1e-4

#: A strategy takes the close panel it is allowed to see and the parameters, and
#: returns one target weight per symbol per row. Row `t` may only use rows `<= t`.
StrategyFn = Callable[[np.ndarray, Mapping[str, float]], np.ndarray]


class FactorSource(StrEnum):
    """Where G4's factor regression got its factors. Recorded, never inferred."""

    SUPPLIED = "supplied"
    PANEL_PROXY = "panel_proxy"


@dataclass(frozen=True)
class PricePanel:
    """A rectangular close (and optional dollar-volume) panel, validated on construction.

    Validation is not politeness. A panel with a zero price, a repeated date or a
    hole produces a return series that no gate can interpret, and the failure
    shows up much later as a number nobody can reproduce.
    """

    dates: np.ndarray
    symbols: tuple[str, ...]
    close: np.ndarray
    dollar_volume: np.ndarray | None = None

    def __post_init__(self) -> None:
        close = np.asarray(self.close, dtype=float)
        if close.ndim != 2:
            raise ValueError("close must be a 2-D (dates x symbols) panel")
        n_rows, n_cols = close.shape
        if n_rows < 3:
            raise ValueError("close needs at least 3 rows to produce returns")
        if len(self.symbols) != n_cols:
            raise ValueError(f"{len(self.symbols)} symbols for {n_cols} columns")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("duplicate symbol in panel")
        if np.asarray(self.dates).shape[0] != n_rows:
            raise ValueError("dates and close disagree on the number of rows")
        order = np.argsort(self.dates, kind="stable")
        if not np.array_equal(order, np.arange(n_rows)):
            raise ValueError("dates must be ascending")
        if len(np.unique(self.dates)) != n_rows:
            raise ValueError("duplicate date in panel")
        if not np.all(np.isfinite(close)):
            raise ValueError("close contains a non-finite value; fill or drop it upstream")
        if np.any(close <= 0):
            raise ValueError("close contains a non-positive price")
        if self.dollar_volume is not None:
            volume = np.asarray(self.dollar_volume, dtype=float)
            if volume.shape != close.shape:
                raise ValueError("dollar_volume must match the close panel shape")
            if not np.all(np.isfinite(volume)) or np.any(volume <= 0):
                raise ValueError("dollar_volume must be finite and positive")

    @property
    def n_rows(self) -> int:
        return int(np.asarray(self.close).shape[0])

    @property
    def bar_returns(self) -> np.ndarray:
        """`r[i]` is the return of bar `i -> i+1`, so `r` has one row fewer than the panel."""
        close = np.asarray(self.close, dtype=float)
        return close[1:] / close[:-1] - 1.0


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = 5.0
    oos_fraction: float = 0.30
    n_splits: int = 5
    label_horizon: int = 1
    embargo_pct: float = 0.01
    periods_per_year: int = 252
    param_perturbation: float = 0.20
    capital: float = 1_000_000.0
    #: Reported ADV participation percentile. The mean hides the days that move
    #: the market; the maximum is one bad print. G6 reads this number.
    participation_percentile: float = 95.0

    def __post_init__(self) -> None:
        if not 0.0 < self.oos_fraction < 1.0:
            raise ValueError("oos_fraction must be in (0, 1)")
        if self.cost_bps < 0:
            raise ValueError("cost_bps may not be negative")
        if not 0.0 < self.param_perturbation < 1.0:
            raise ValueError("param_perturbation must be in (0, 1)")


@dataclass(frozen=True)
class SimResult:
    net: np.ndarray
    gross: np.ndarray
    turnover: np.ndarray
    traded_notional: np.ndarray


@dataclass
class RunReport:
    """Provenance for a submission. Every number a gate reports traces back to this."""

    alpha_id: str
    n_trials: int
    chosen_params: dict[str, float]
    factor_source: FactorSource
    gross_leverage_max: float
    cost_bps: float
    is_rows: int
    oos_rows: int
    adv_participation: float
    notes: list[str] = field(default_factory=list)


def simulate(
    panel: PricePanel,
    weights: np.ndarray,
    cost_bps: float,
    capital: float = 1_000_000.0,
) -> SimResult:
    """Net returns from target weights, with the point-in-time alignment fixed here.

    `weights[t]` is the position held into bar `t -> t+1`, so it earns
    `panel.bar_returns[t]`. The last row of `weights` is therefore never paid: it
    is a decision about a bar the panel does not contain. Turnover is charged
    when the position changes, and the first bar pays for building the book.
    """
    weights = np.asarray(weights, dtype=float)
    close = np.asarray(panel.close, dtype=float)
    if weights.shape != close.shape:
        raise ValueError(f"weights {weights.shape} must match the close panel {close.shape}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contain a non-finite value")

    held = weights[:-1]
    returns = panel.bar_returns
    gross = np.sum(held * returns, axis=1)

    previous = np.vstack([np.zeros((1, held.shape[1])), held[:-1]])
    turnover = np.sum(np.abs(held - previous), axis=1)
    traded_notional = np.abs(held - previous) * capital

    net = gross - turnover * cost_bps * BPS
    return SimResult(net=net, gross=gross, turnover=turnover, traded_notional=traded_notional)


def adv_participation(
    panel: PricePanel,
    traded_notional: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Traded dollars as a share of that bar's dollar volume, at the given percentile.

    The volume used is the bar the decision was made on, never the bar the trade
    prints in: using tomorrow's volume to justify today's size is look-ahead in
    the capacity estimate. With no volume panel this returns 0.0 and the caller
    is told, because a silent zero would read as "no capacity problem".
    """
    if panel.dollar_volume is None:
        return 0.0
    volume = np.asarray(panel.dollar_volume, dtype=float)[:-1]
    shares = traded_notional / volume
    traded = shares[traded_notional > 0]
    if traded.size == 0:
        return 0.0
    return float(np.percentile(traded, percentile))


def _perturbed_grid(params: Mapping[str, float], fraction: float) -> list[dict[str, float]]:
    """The +/- `fraction` neighbourhood of one numeric parameter at a time.

    G5 asks whether the strategy sits on a plateau or on a spike. Perturbing one
    parameter at a time is what distinguishes the two; perturbing all of them
    together only says the corner of the box is worse.
    """
    grid: list[dict[str, float]] = []
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        for direction in (1.0 - fraction, 1.0 + fraction):
            candidate = dict(params)
            moved = value * direction
            candidate[key] = type(value)(round(moved)) if isinstance(value, int) else moved
            if candidate[key] != value:
                grid.append(candidate)
    return grid


def _check_gross(weights: np.ndarray, gross_max: float, label: str) -> None:
    worst = float(np.max(np.sum(np.abs(weights), axis=1)))
    if worst > gross_max + 1e-12:
        raise ValueError(
            f"{label} reaches gross leverage {worst:.2f}, above the pod limit {gross_max:.2f}. "
            "Scale the strategy before submitting; the engine will not rescale it for you."
        )


def _panel_proxy_factors(panel: PricePanel) -> np.ndarray:
    """Equal-weight panel return as a one-factor market proxy. Not FF5+MOM."""
    return panel.bar_returns.mean(axis=1).reshape(-1, 1)


def run(
    alpha_id: str,
    panel: PricePanel,
    strategy: StrategyFn,
    grid: Sequence[Mapping[str, float]],
    chosen: int = 0,
    config: BacktestConfig | None = None,
    factor_returns: np.ndarray | None = None,
    book_returns: np.ndarray | None = None,
    limits: dict[str, Any] | None = None,
    leak_probes: int = 24,
) -> tuple[Submission, LeakReport, RunReport]:
    """Run the whole grid, return the submission for `grid[chosen]` plus its provenance.

    Every grid point is simulated and its net returns become a column of
    `trial_returns`, which is what the deflated Sharpe and PBO read. The trial
    count is therefore the size of the grid that was actually run -- the caller
    cannot search wide and submit narrow.
    """
    cfg = config or BacktestConfig()
    if not grid:
        raise ValueError("grid is empty; pre-register at least one parameter set (G1)")
    if not 0 <= chosen < len(grid):
        raise ValueError(f"chosen index {chosen} is outside a grid of {len(grid)}")

    lim = limits or load_limits()
    gross_max = float(lim["pod"]["gross_leverage_max"])
    notes: list[str] = []

    trial_columns: list[np.ndarray] = []
    for i, params in enumerate(grid):
        weights = np.asarray(strategy(np.asarray(panel.close, dtype=float), params), dtype=float)
        _check_gross(weights, gross_max, f"grid point {i}")
        trial_columns.append(simulate(panel, weights, cfg.cost_bps, cfg.capital).net)

    chosen_params = dict(grid[chosen])
    chosen_weights = np.asarray(strategy(np.asarray(panel.close, dtype=float), chosen_params), dtype=float)
    result = simulate(panel, chosen_weights, cfg.cost_bps, cfg.capital)
    doubled = simulate(panel, chosen_weights, cfg.cost_bps * 2.0, cfg.capital).net

    # G0: recompute the chosen strategy point-in-time. This is the only gate the
    # engine cannot make true by construction, so it is measured, not asserted.
    leak_report = lookahead_scan(
        lambda data: np.asarray(strategy(data, chosen_params), dtype=float),
        np.asarray(panel.close, dtype=float),
        n_probes=leak_probes,
    )

    net = result.net
    n_returns = net.shape[0]
    split = n_returns - int(round(n_returns * cfg.oos_fraction))
    if split < 2 or n_returns - split < 2:
        raise ValueError("panel too short to hold an in-sample and an out-of-sample block")
    in_sample, out_of_sample = net[:split], net[split:]

    splits = purged_kfold(
        n_returns,
        n_splits=cfg.n_splits,
        label_horizon=cfg.label_horizon,
        embargo_pct=cfg.embargo_pct,
    )
    fold_returns = [net[test] for _, test in splits]

    if factor_returns is None:
        factors = _panel_proxy_factors(panel)
        factor_source = FactorSource.PANEL_PROXY
        notes.append(
            "No factor returns supplied: G4's residual alpha is measured against an "
            "equal-weight panel proxy, not FF5+MOM. Read that t-statistic as provisional."
        )
    else:
        factors = np.asarray(factor_returns, dtype=float)
        if factors.ndim == 1:
            factors = factors.reshape(-1, 1)
        if factors.shape[0] != n_returns:
            raise ValueError(f"factor_returns has {factors.shape[0]} rows, need {n_returns}")
        factor_source = FactorSource.SUPPLIED

    if book_returns is None:
        book_correlation = 0.0
        notes.append("No existing book supplied: G6 book correlation is 0.0 by absence, not by measurement.")
    else:
        book = np.asarray(book_returns, dtype=float)
        if book.shape[0] != n_returns:
            raise ValueError(f"book_returns has {book.shape[0]} rows, need {n_returns}")
        book_correlation = float(np.corrcoef(net, book)[0, 1]) if np.std(book) > 0 else 0.0

    participation = adv_participation(panel, result.traded_notional, cfg.participation_percentile)
    if panel.dollar_volume is None:
        notes.append("No dollar-volume panel: G6 ADV participation is 0.0 by absence, not by measurement.")

    # G5 perturbs the chosen parameters, and a strategy sitting at the gross limit
    # has neighbours that breach it. Those neighbours are dropped rather than
    # simulated: a robustness number measured on a book the pod could not run is
    # worse than a missing one. Dropping is recorded, because G5 then describes a
    # smaller neighbourhood than the caller asked for.
    perturbed: list[np.ndarray] = []
    infeasible = 0
    for candidate in _perturbed_grid(chosen_params, cfg.param_perturbation):
        weights = np.asarray(strategy(np.asarray(panel.close, dtype=float), candidate), dtype=float)
        if float(np.max(np.sum(np.abs(weights), axis=1))) > gross_max + 1e-12:
            infeasible += 1
            continue
        perturbed.append(simulate(panel, weights, cfg.cost_bps, cfg.capital).net)
    if infeasible:
        notes.append(
            f"{infeasible} perturbed parameter set(s) would breach the pod gross limit "
            f"{gross_max:.2f} and were not simulated; G5 read the feasible neighbours only."
        )
    if not perturbed:
        notes.append("No feasible perturbed parameters: G5 could not test a parameter plateau.")

    submission = Submission(
        alpha_id=alpha_id,
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        fold_returns=fold_returns,
        trial_returns=np.column_stack(trial_columns),
        factor_returns=factors,
        adv_participation=participation,
        book_correlation=book_correlation,
        cost_doubled=doubled,
        param_perturbed=perturbed,
        periods_per_year=cfg.periods_per_year,
    )
    report = RunReport(
        alpha_id=alpha_id,
        n_trials=len(grid),
        chosen_params=chosen_params,
        factor_source=factor_source,
        gross_leverage_max=float(np.max(np.sum(np.abs(chosen_weights), axis=1))),
        cost_bps=cfg.cost_bps,
        is_rows=int(in_sample.shape[0]),
        oos_rows=int(out_of_sample.shape[0]),
        adv_participation=participation,
        notes=notes,
    )
    return submission, leak_report, report
