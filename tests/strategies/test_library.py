"""Every registered strategy is put through the gate's own look-ahead scan.

The docstrings claim each function is point-in-time. That claim is worth nothing
unless it is re-derived, so these tests recompute each strategy on truncated
panels with `core.backtest.leakage.lookahead_scan` -- the same scan G0 runs -- and
also check the shaping invariants the engine relies on: gross within the pod
limit, dollar neutrality where a strategy claims it, and no NaN reaching a weight.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import strategies
from core.backtest.leakage import lookahead_scan
from core.strategies import base

POD_GROSS_MAX = 1.0
AVAILABLE = strategies.names(available_only=True)
DOLLAR_NEUTRAL = ("xs_momentum", "short_term_reversal", "betting_against_beta")


def close_panel(n_rows: int = 500, n_cols: int = 5, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0003, 0.011, size=(n_rows, n_cols))
    return 100.0 * np.exp(np.cumsum(steps, axis=0))


# --- the claim that matters --------------------------------------------------


@pytest.mark.parametrize("name", AVAILABLE)
def test_no_strategy_looks_ahead(name):
    signal = strategies.get(name).build()
    report = lookahead_scan(signal, close_panel(), n_probes=24)
    assert report.ok, f"{name} leaked at {len(report.leaks)} of {report.probes} probes"


@pytest.mark.parametrize("name", AVAILABLE)
def test_no_strategy_exceeds_the_pod_gross_limit(name):
    weights = strategies.get(name).build()(close_panel())
    worst = float(np.max(np.sum(np.abs(weights), axis=1)))
    assert worst <= POD_GROSS_MAX + 1e-12, f"{name} reaches {worst:.2f}x"


@pytest.mark.parametrize("name", AVAILABLE)
def test_no_strategy_emits_a_non_finite_weight(name):
    weights = strategies.get(name).build()(close_panel())
    assert np.all(np.isfinite(weights)), f"{name} emitted a NaN or inf weight"


@pytest.mark.parametrize("name", AVAILABLE)
def test_the_warm_up_holds_nothing_rather_than_something(name):
    """Before there is history there is no signal, and no signal means no position."""
    weights = strategies.get(name).build()(close_panel())
    assert np.all(weights[0] == 0.0), f"{name} took a position on the first bar"


@pytest.mark.parametrize("name", DOLLAR_NEUTRAL)
def test_a_cross_sectional_strategy_is_dollar_neutral(name):
    weights = strategies.get(name).build()(close_panel())
    assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-12), name


# --- editability ------------------------------------------------------------


def test_a_parameter_override_actually_changes_the_book():
    prices = close_panel()
    fast = strategies.get("ts_momentum").build(lookback=20)(prices)
    slow = strategies.get("ts_momentum").build(lookback=250)(prices)
    assert not np.allclose(fast, slow), "the lookback override did nothing"


def test_a_misspelled_parameter_raises_instead_of_running_the_default():
    """The silent version of this bug attributes a result to a parameter nobody used."""
    with pytest.raises(strategies.UnknownParameter, match="lookbak"):
        strategies.get("ts_momentum").build(lookbak=20)


def test_the_error_names_the_parameters_the_strategy_does_have():
    with pytest.raises(strategies.UnknownParameter, match="gross"):
        strategies.get("ts_momentum").build(horizon=20)


def test_gross_is_a_parameter_like_any_other():
    weights = strategies.get("ts_momentum").build(gross=0.4)(close_panel())
    live = np.sum(np.abs(weights), axis=1)
    assert np.isclose(live.max(), 0.4)


def test_the_default_gross_leaves_room_for_the_robustness_gate():
    """G5 perturbs +/-20%; a default at the limit makes every neighbour infeasible."""
    assert base.DEFAULT_GROSS * 1.2 <= POD_GROSS_MAX + 1e-12


# --- the registry -----------------------------------------------------------


def test_every_registered_strategy_carries_a_citation():
    for name, strategy in strategies.REGISTRY.items():
        assert strategy.citation.strip(), name


def test_the_pre_registered_grid_is_what_g1_counts():
    trend = strategies.get("ts_momentum")
    grid = trend.search_grid()
    assert len(grid) == len(trend.grid)
    assert all("gross" in point and "lookback" in point for point in grid)


def test_overriding_a_fixed_parameter_narrows_the_grid_without_widening_the_count():
    trend = strategies.get("ts_momentum")
    grid = trend.search_grid(gross=0.5)
    assert len(grid) == len(trend.grid)
    assert {point["gross"] for point in grid} == {0.5}


def test_a_family_we_lack_data_for_is_unavailable_rather_than_approximated():
    for name in ("value", "quality", "carry"):
        strategy = strategies.get(name)
        assert not strategy.available
        assert strategy.unavailable_because
        with pytest.raises(strategies.MissingData, match=name):
            strategy.build()


def test_unavailable_families_are_excluded_from_the_runnable_list():
    assert "value" not in AVAILABLE
    assert "ts_momentum" in AVAILABLE


def test_registering_the_same_name_twice_is_refused():
    existing = strategies.get("ts_momentum")
    with pytest.raises(ValueError, match="already registered"):
        strategies.register(existing)


def test_a_grid_point_naming_an_unknown_parameter_is_refused():
    bad = strategies.Strategy(
        name="typo-grid",
        family="test",
        fn=lambda close, params: np.zeros_like(close),
        defaults={"lookback": 10},
        grid=({"lookbak": 20},),
        citation="none",
    )
    with pytest.raises(ValueError, match="unknown parameter"):
        strategies.register(bad)


def test_an_available_strategy_without_a_grid_is_refused():
    bad = strategies.Strategy(
        name="no-grid",
        family="test",
        fn=lambda close, params: np.zeros_like(close),
        defaults={"lookback": 10},
        grid=(),
        citation="none",
    )
    with pytest.raises(ValueError, match="G1 needs one"):
        strategies.register(bad)


# --- the shape the families are supposed to have ----------------------------


def test_trend_follows_the_direction_of_a_trending_market():
    """A monotonically rising panel must end up long, or it is not trend following."""
    rising = np.cumprod(np.full((300, 3), 1.002), axis=0) * 100.0
    weights = strategies.get("ts_momentum").build(lookback=60)(rising)
    assert np.all(weights[-1] > 0)


def test_reversal_and_momentum_disagree_on_the_same_cross_section():
    """They are the same quantity at different horizons, so the signs oppose."""
    prices = close_panel()
    reversal = strategies.get("short_term_reversal").build(lookback=5)(prices)
    momentum = strategies.get("xs_momentum").build(lookback=5, skip=0)(prices)
    overlap = ~np.all(reversal == 0, axis=1) & ~np.all(momentum == 0, axis=1)
    assert np.all(np.sign(reversal[overlap]) == -np.sign(momentum[overlap]))


def test_breakout_holds_its_position_between_breakouts():
    """Holding is the mechanism; a breakout rule that re-decides daily is not one."""
    prices = close_panel()
    weights = strategies.get("trend_breakout").build(lookback=100)(prices)
    live = weights[150:]
    unchanged = np.mean(np.all(np.isclose(live[1:], live[:-1]), axis=1))
    assert unchanged > 0.9, f"only {unchanged:.0%} of bars held"


def test_betting_against_beta_is_long_the_quiet_name():
    """One instrument built with a third of the panel's beta must end up long."""
    rng = np.random.default_rng(3)
    market = rng.normal(0.0004, 0.011, 400)
    quiet = 0.3 * market + rng.normal(0.0, 0.0005, 400)
    loud = 1.6 * market + rng.normal(0.0, 0.0005, 400)
    panel = 100.0 * np.exp(np.cumsum(np.column_stack([quiet, market, loud]), axis=0))
    weights = strategies.get("betting_against_beta").build(lookback=120)(panel)
    assert weights[-1][0] > 0 > weights[-1][2]


def test_the_ensemble_is_not_any_one_of_its_sleeves():
    prices = close_panel()
    blended = strategies.get("multi_signal").build()(prices)
    for name in ("ts_momentum", "xs_momentum", "short_term_reversal"):
        assert not np.allclose(blended, strategies.get(name).build()(prices)), name


def test_blending_mismatched_shapes_is_refused():
    with pytest.raises(ValueError, match="shapes"):
        strategies.blend([np.zeros((5, 3)), np.zeros((5, 2))], 1.0)
