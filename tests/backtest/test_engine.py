"""The engine's job is to make the honest accounting structural, so that is what is tested.

Each test here pins one property a caller could otherwise get wrong quietly:
the point-in-time alignment, the trial count, the gross-leverage ceiling, and the
fact that a look-ahead strategy still looks profitable and is caught only by G0.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.backtest import gates
from core.backtest.engine import (
    BacktestConfig,
    FactorSource,
    PricePanel,
    RunReport,
    _perturbed_grid,
    adv_participation,
    run,
    simulate,
)


def panel(n_rows: int = 260, n_cols: int = 4, seed: int = 7, with_volume: bool = True) -> PricePanel:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.012, size=(n_rows, n_cols))
    close = 100.0 * np.exp(np.cumsum(steps, axis=0))
    volume = np.full((n_rows, n_cols), 5e7) if with_volume else None
    return PricePanel(
        dates=np.datetime64("2020-01-01") + np.arange(n_rows),
        symbols=tuple(f"S{i}" for i in range(n_cols)),
        close=close,
        dollar_volume=volume,
    )


def flat_long(close: np.ndarray, params) -> np.ndarray:
    """Equal-weight, always on. No parameter reaches the weights, so it is a plateau."""
    weight = float(params["weight"])
    return np.full_like(close, weight / close.shape[1])


def lookahead(close: np.ndarray, params) -> np.ndarray:
    """Tomorrow's sign, today. Profitable and fraudulent; only G0 can see it."""
    scale = float(params["scale"])
    weights = np.zeros_like(close)
    forward = close[1:] / close[:-1] - 1.0
    weights[:-1] = np.sign(forward) * scale / close.shape[1]
    return weights


def momentum(close: np.ndarray, params) -> np.ndarray:
    """Cross-sectional momentum over `lookback` bars, dollar-neutral, point-in-time."""
    lookback = int(params["lookback"])
    gross = float(params["gross"])
    weights = np.zeros_like(close)
    for t in range(lookback, close.shape[0]):
        past = close[t] / close[t - lookback] - 1.0
        centred = past - past.mean()
        scale = np.sum(np.abs(centred))
        if scale > 0:
            weights[t] = centred / scale * gross
    return weights


# --- the alignment, which is the whole point ---------------------------------


def test_weight_earns_the_following_bar_not_the_preceding_one():
    prices = PricePanel(
        dates=np.datetime64("2020-01-01") + np.arange(5),
        symbols=("A",),
        close=np.array([[100.0], [100.0], [110.0], [110.0], [110.0]]),
    )
    weights = np.zeros((5, 1))
    weights[1, 0] = 1.0  # decided on row 1, so it earns the row 1 -> 2 return

    result = simulate(prices, weights, cost_bps=5.0)

    assert result.gross.shape == (4,)
    assert result.gross[1] == pytest.approx(0.10)
    assert result.gross[0] == pytest.approx(0.0)
    # Entering costs 5bps of one unit, and leaving on the next bar costs it again.
    assert result.net[1] == pytest.approx(0.10 - 5e-4)
    assert result.net[2] == pytest.approx(-5e-4)


def test_the_last_weight_row_is_never_paid():
    prices = PricePanel(
        dates=np.datetime64("2020-01-01") + np.arange(4),
        symbols=("A",),
        close=np.array([[100.0], [100.0], [100.0], [200.0]]),
    )
    only_last = np.zeros((4, 1))
    only_last[3, 0] = 1.0
    assert simulate(prices, only_last, cost_bps=0.0).net.sum() == pytest.approx(0.0)


def test_cost_is_charged_on_turnover_only():
    prices = panel(n_rows=40, n_cols=2)
    held = np.full_like(np.asarray(prices.close), 0.5)  # never trades after day one
    result = simulate(prices, held, cost_bps=10.0)
    assert result.turnover[0] == pytest.approx(1.0)
    assert np.allclose(result.turnover[1:], 0.0)
    assert result.net[5] == pytest.approx(result.gross[5])


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"close": np.array([[100.0], [0.0], [100.0]])}, "non-positive"),
        ({"close": np.array([[100.0], [np.nan], [100.0]])}, "non-finite"),
        ({"dates": np.array(["2020-01-03", "2020-01-02", "2020-01-01"], dtype="datetime64[D]")}, "ascending"),
        (
            {"dates": np.array(["2020-01-01", "2020-01-01", "2020-01-02"], dtype="datetime64[D]")},
            "duplicate date",
        ),
        ({"symbols": ("A", "B")}, "columns"),
    ],
)
def test_panel_rejects_data_no_gate_could_interpret(kwargs, message):
    base = {
        "dates": np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        "symbols": ("A",),
        "close": np.array([[100.0], [101.0], [102.0]]),
    }
    with pytest.raises(ValueError, match=message):
        PricePanel(**(base | kwargs))


def test_engine_refuses_a_strategy_above_the_pod_gross_limit():
    with pytest.raises(ValueError, match="above the pod limit"):
        run("too-levered", panel(n_rows=60), flat_long, [{"weight": 2.0}])


def test_engine_will_not_rescale_instead_of_refusing():
    """Rescaling would submit a strategy nobody tested. The refusal is the feature."""
    submission, _, report = run(
        "at-the-limit", panel(n_rows=60), flat_long, [{"weight": 1.5}, {"weight": 1.0}]
    )
    assert report.gross_leverage_max == pytest.approx(1.5)
    assert submission.in_sample.size > 0


def test_a_strategy_at_the_limit_drops_infeasible_neighbours_and_says_so():
    """+20% of a book already at 1.5x gross is not a book; G5 reads the rest."""
    submission, _, report = run(
        "at-the-limit", panel(n_rows=60), flat_long, [{"weight": 1.5}, {"weight": 1.0}]
    )
    assert any("would breach the pod gross limit" in note for note in report.notes)
    assert all(np.isfinite(series).all() for series in submission.param_perturbed), (
        "a dropped neighbour must be absent, not NaN"
    )


# --- honest trial counting ---------------------------------------------------


def test_every_grid_point_becomes_a_trial_column():
    grid = [{"lookback": lb, "gross": 1.0} for lb in (5, 10, 20, 40, 60)]
    submission, _, report = run("momo", panel(), momentum, grid, chosen=2)

    assert submission.trial_returns.shape[1] == len(grid)
    assert report.n_trials == len(grid)
    assert report.chosen_params == {"lookback": 20, "gross": 1.0}
    # The submitted series must be the chosen column, not the best one.
    chosen_column = submission.trial_returns[:, 2]
    assert np.allclose(submission.full, chosen_column)


def test_perturbed_grid_moves_one_parameter_at_a_time():
    moved = _perturbed_grid({"lookback": 20, "gross": 1.0}, 0.20)
    assert {"lookback": 16, "gross": 1.0} in moved
    assert {"lookback": 24, "gross": 1.0} in moved
    assert {"lookback": 20, "gross": 0.8} in moved
    assert all(sum(1 for k in m if m[k] != {"lookback": 20, "gross": 1.0}[k]) == 1 for m in moved)


def test_double_cost_series_is_worse_by_exactly_the_cost():
    grid = [{"lookback": 20, "gross": 1.0}]
    submission, _, _ = run("momo", panel(), momentum, grid)
    extra = submission.full - submission.cost_doubled
    assert np.all(extra >= -1e-12)
    assert extra.sum() > 0


# --- capacity and provenance -------------------------------------------------


def test_adv_participation_uses_the_decision_bar_volume():
    prices = PricePanel(
        dates=np.datetime64("2020-01-01") + np.arange(3),
        symbols=("A",),
        close=np.array([[100.0], [100.0], [100.0]]),
        dollar_volume=np.array([[1_000.0], [9_999_999.0], [1.0]]),
    )
    weights = np.array([[1.0], [0.0], [0.0]])
    result = simulate(prices, weights, cost_bps=0.0, capital=100.0)
    # Two trades of 100 dollars: in against volume 1,000 and out against 9,999,999.
    assert adv_participation(prices, result.traded_notional, percentile=100.0) == pytest.approx(0.1)


def test_absent_inputs_are_recorded_as_absent_not_as_zero_risk():
    grid = [{"lookback": 20, "gross": 1.0}]
    _, _, report = run("momo", panel(with_volume=False), momentum, grid)
    assert report.factor_source is FactorSource.PANEL_PROXY
    assert report.adv_participation == 0.0
    joined = " ".join(report.notes)
    assert "not FF5+MOM" in joined
    assert "ADV participation is 0.0 by absence" in joined
    assert "book correlation is 0.0 by absence" in joined


def test_supplied_factors_are_marked_as_supplied():
    grid = [{"lookback": 20, "gross": 1.0}]
    prices = panel()
    factors = prices.bar_returns.mean(axis=1)
    _, _, report = run("momo", prices, momentum, grid, factor_returns=factors)
    assert report.factor_source is FactorSource.SUPPLIED
    assert not any("not FF5+MOM" in note for note in report.notes)


# --- the whole path: engine output feeds the gates ---------------------------


def test_lookahead_strategy_is_profitable_and_only_g0_catches_it():
    """The engine must not launder leakage into a failing Sharpe. G0 is the defence."""
    grid = [{"scale": 1.0}, {"scale": 0.75}, {"scale": 0.5}]
    submission, leak_report, _ = run("leaky", panel(), lookahead, grid)

    assert not leak_report.ok, "the scan must see the future the strategy used"
    assert leak_report.max_deviation > 0.1
    assert float(np.mean(submission.full)) > 0, "a leaking strategy does make money"

    verdicts = gates.evaluate(submission, leak_report)
    assert not gates.approved(verdicts)
    assert "G0_data" in gates.failed_gates(verdicts)


def test_a_clean_strategy_passes_the_leak_scan():
    grid = [{"lookback": 20, "gross": 1.0}]
    _, leak_report, _ = run("momo", panel(), momentum, grid)
    assert leak_report.ok
    assert leak_report.probes > 0


def test_a_random_strategy_gets_rejected_end_to_end(tmp_path):
    def noise(close: np.ndarray, params) -> np.ndarray:
        rng = np.random.default_rng(int(params["seed"]))
        raw = rng.normal(size=close.shape)
        centred = raw - raw.mean(axis=1, keepdims=True)
        scale = np.sum(np.abs(centred), axis=1, keepdims=True)
        return centred / scale

    grid = [{"seed": s} for s in range(6)]
    submission, leak_report, _ = run("noise", panel(), noise, grid, chosen=0)
    verdicts = gates.evaluate(submission, leak_report, audit_path=tmp_path / "audit.log")

    assert not gates.approved(verdicts)
    assert {"G2_in_sample", "G4_statistics"} & set(gates.failed_gates(verdicts))


def test_run_report_is_a_dataclass_of_provenance_not_prose():
    grid = [{"lookback": 20, "gross": 1.0}]
    _, _, report = run("momo", panel(), momentum, grid, config=BacktestConfig(cost_bps=7.5))
    assert isinstance(report, RunReport)
    assert report.cost_bps == 7.5
    assert report.is_rows + report.oos_rows == 259
    assert report.oos_rows == pytest.approx(259 * 0.30, abs=1)


def test_short_panel_is_refused_rather_than_split_into_nothing():
    with pytest.raises(ValueError, match="too short"):
        run("momo", panel(n_rows=5), momentum, [{"lookback": 2, "gross": 1.0}])


def test_empty_grid_is_refused_because_n_would_be_a_lie():
    with pytest.raises(ValueError, match="pre-register"):
        run("momo", panel(), momentum, [])


def test_a_single_configuration_fails_g4_instead_of_skipping_pbo():
    """One pre-registered parameter set is exactly the case PBO exists to catch."""
    submission, leak_report, _ = run("momo", panel(), momentum, [{"lookback": 20, "gross": 1.0}])
    verdicts = gates.evaluate(submission, leak_report)
    g4 = next(v for v in verdicts if v.gate == "G4_statistics")
    assert not g4.passed
    assert "single configuration" in g4.reason


# --- the reporting metric set (core/report/metrics.py) ----------------------


def test_the_run_report_carries_the_reporting_metrics():
    prices = panel(n_rows=300)
    _, _, report = run("alpha", prices, momentum, [{"lookback": 10, "gross": 1.0}])
    performance = report.performance
    assert {"cagr", "volatility", "sharpe", "max_drawdown", "turnover_annual"} <= performance.keys()
    # Turnover comes from the engine's own series, so it is measured, not absent.
    assert performance["turnover_annual"] is not None
    # No benchmark was supplied, so beta must be absent rather than zero.
    assert performance["beta"] is None


def test_a_supplied_benchmark_is_reported_as_beta_and_alpha():
    prices = panel(n_rows=300)
    benchmark = prices.bar_returns.mean(axis=1)
    _, _, report = run(
        "alpha",
        prices,
        momentum,
        [{"lookback": 10, "gross": 1.0}],
        benchmark_returns=benchmark,
    )
    assert report.performance["beta"] is not None
    assert report.performance["alpha_annual"] is not None


def test_a_benchmark_of_the_wrong_length_is_refused():
    prices = panel(n_rows=300)
    with pytest.raises(ValueError, match="benchmark_returns has"):
        run(
            "alpha",
            prices,
            momentum,
            [{"lookback": 10, "gross": 1.0}],
            benchmark_returns=np.zeros(17),
        )
