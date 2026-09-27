"""The inverted gates, checked against the gates themselves.

`core/backtest/power.py` is algebra, and algebra about a threshold is the kind of
thing that is wrong quietly. So the closed forms are checked by running the real
`core.backtest.stats` functions on synthetic series at the Sharpe the algebra
says is the minimum, and the "these are lower bounds" claim is checked by
breaking the normality assumption and watching the requirement get harder.
"""

from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import pytest

from core.backtest import power
from core.backtest.stats import block_bootstrap_pvalue, deflated_sharpe_ratio
from core.risk.limits import load_limits

YEAR = 252


@pytest.fixture
def limits():
    return copy.deepcopy(load_limits())


# --- the algebra agrees with the gate it inverts -------------------------------


@pytest.mark.parametrize("n_obs", [735, 1260, 2520])
def test_a_series_at_the_stated_minimum_lands_on_the_floor(n_obs, limits):
    """The check that matters: feed the real `deflated_sharpe_ratio` a series at
    exactly the Sharpe the closed form demands, and the floor is where it lands."""
    floor = float(limits["gates"]["deflated_sharpe_probability_min"])
    annualised = power.sharpe_for_deflated_probability(n_obs, 6, floor)
    assert annualised is not None
    draws = power.monte_carlo_deflated_probability(annualised / math.sqrt(YEAR), n_obs, 6, draws=300, seed=7)
    # Median rather than mean: DSR is a bounded probability and the realised
    # Sharpe of any one draw scatters, so the left tail pulls the mean down while
    # the typical draw sits on the floor.
    assert abs(float(np.median(draws)) - floor) < 0.04


def test_a_series_below_the_stated_minimum_typically_fails_the_gate(limits):
    floor = float(limits["gates"]["deflated_sharpe_probability_min"])
    annualised = power.sharpe_for_deflated_probability(735, 6, floor)
    draws = power.monte_carlo_deflated_probability(
        0.6 * annualised / math.sqrt(YEAR), 735, 6, draws=200, seed=3
    )
    assert float(np.median(draws)) < floor


def _unit(series: np.ndarray, per_period: float) -> np.ndarray:
    """Centre, scale to unit sd, then shift to the per-period Sharpe under test."""
    centred = series - float(np.mean(series))
    return centred / float(np.std(centred, ddof=1)) + per_period


def _median_dsr(make, seed: int, n_obs: int = 735, n_trials: int = 6, draws: int = 300) -> float:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        returns = make(rng)
        trials = rng.normal(0.0, 1.0, size=(n_obs, n_trials))
        trial_sharpes = np.array(
            [float(np.mean(trials[:, j]) / np.std(trials[:, j], ddof=1)) for j in range(n_trials)]
        )
        out.append(deflated_sharpe_ratio(returns, trial_sharpes))
    return float(np.median(out))


def _per_period(limits) -> float:
    floor = float(limits["gates"]["deflated_sharpe_probability_min"])
    return power.sharpe_for_deflated_probability(735, 6, floor) / math.sqrt(YEAR)


def test_fat_tails_move_the_criterion_by_almost_nothing(limits):
    """Measured, not assumed. Kurtosis enters multiplied by the squared per-period
    Sharpe, which is about 0.01 at daily frequency, so the term is a rounding
    error -- and a module claiming fat tails make the gate much harder would be
    wrong about its own gate."""
    per = _per_period(limits)
    normal = _median_dsr(lambda r: r.normal(per, 1.0, size=735), 7)
    heavy = _median_dsr(lambda r: _unit(r.standard_t(df=4, size=735), per), 11)
    assert abs(heavy - normal) < 0.02


def test_positive_skew_clears_the_criterion_more_easily_than_normal(limits):
    """The non-conservatism worth recording: a lottery-shaped series passes under
    the Sharpe the algebra names, so the deflated-Sharpe figure is not a universal
    lower bound (ADR-0033). Asserted so a future change to `stats` that removes
    the skew term shows up here as a failure."""
    per = _per_period(limits)
    normal = _median_dsr(lambda r: r.normal(per, 1.0, size=735), 7)
    positive = _median_dsr(lambda r: _unit(r.lognormal(0.0, 1.0, size=735), per), 13)
    negative = _median_dsr(lambda r: _unit(-r.lognormal(0.0, 1.0, size=735), per), 13)
    assert positive > normal > negative


def test_the_bootstrap_floor_is_a_real_floor_on_autocorrelated_returns(limits):
    """The one closed form that understates by a lot. At the stated minimum an iid
    series clears p = 0.05 and an AR(1) series does not come close."""
    pmax = float(limits["gates"]["bootstrap_pvalue_max"])
    per = power.sharpe_for_bootstrap_pvalue(735, pmax) / math.sqrt(YEAR)
    rng = np.random.default_rng(5)

    def ar1(rho: float) -> np.ndarray:
        noise = rng.normal(0.0, 1.0, size=735)
        out = np.empty(735)
        out[0] = noise[0]
        for i in range(1, 735):
            out[i] = rho * out[i - 1] + noise[i]
        return _unit(out, per)

    iid = float(np.median([block_bootstrap_pvalue(rng.normal(per, 1.0, size=735)) for _ in range(40)]))
    dependent = float(np.median([block_bootstrap_pvalue(ar1(0.3)) for _ in range(40)]))
    assert iid <= pmax * 1.2, "the iid case should sit at the floor the algebra names"
    assert dependent > pmax * 2, "autocorrelation must make the bootstrap harder, not easier"


# --- the shape of the answer ---------------------------------------------------


def test_the_requirement_falls_as_the_sample_grows(limits):
    floor = float(limits["gates"]["deflated_sharpe_probability_min"])
    series = [power.sharpe_for_deflated_probability(n, 6, floor) for n in (500, 1000, 2000, 4000)]
    assert all(a > b for a, b in itertools.pairwise(series))


def test_the_requirement_rises_with_the_number_of_trials(limits):
    """More configurations tried means a higher Sharpe is the price of luck."""
    floor = float(limits["gates"]["deflated_sharpe_probability_min"])
    few = power.sharpe_for_deflated_probability(1000, 2, floor)
    many = power.sharpe_for_deflated_probability(1000, 200, floor)
    assert many > few


def test_the_in_sample_floor_does_not_move_with_the_sample(limits):
    """It is the one criterion a longer history does not help with."""
    short = next(r for r in power.requirements(500, 6, limits) if "is_sharpe_min" in r.criterion)
    long = next(r for r in power.requirements(9000, 6, limits) if "is_sharpe_min" in r.criterion)
    assert short.annualised_sharpe_min == long.annualised_sharpe_min
    assert not short.depends_on_sample


def test_a_short_sample_is_bound_by_the_statistics_gate_and_a_long_one_is_not(limits):
    """The finding this module exists to state (ADR-0033)."""
    short = power.binding(power.requirements(735, 6, limits))
    long = power.binding(power.requirements(252 * 20, 6, limits))
    assert "deflated_sharpe_probability_min" in short.criterion
    assert "is_sharpe_min" in long.criterion
    assert short.annualised_sharpe_min > long.annualised_sharpe_min


# --- the inverse question ------------------------------------------------------


def test_a_target_below_the_immovable_floor_is_never_reachable(limits):
    """G2 asks for an annualised 1.0. No sample size makes 0.5 acceptable, and
    returning a number here would say the opposite."""
    assert float(limits["gates"]["is_sharpe_min"]) == 1.0
    assert power.observations_for(0.5, 6, limits) is None
    assert power.observations_for(0.99, 6, limits) is None


def test_a_better_strategy_needs_less_history(limits):
    modest = power.observations_for(1.2, 6, limits)
    strong = power.observations_for(2.5, 6, limits)
    assert modest is not None and strong is not None
    assert strong < modest


def test_the_returned_length_actually_clears_the_battery(limits):
    """The binary search must land on a length that passes, not next to one."""
    target = 1.5
    n = power.observations_for(target, 6, limits)
    assert n is not None
    worst = power.binding(power.requirements(n, 6, limits))
    assert worst.annualised_sharpe_min <= target
    tighter = power.binding(power.requirements(n - 1, 6, limits))
    assert tighter.annualised_sharpe_min > target, "n is not the smallest length that clears"


def test_a_nonpositive_target_is_not_a_target(limits):
    assert power.observations_for(0.0, 6, limits) is None
    assert power.observations_for(-1.0, 6, limits) is None


# --- refusals ------------------------------------------------------------------


def test_too_few_observations_is_unmeasured_rather_than_zero():
    assert power.sharpe_for_deflated_probability(2, 6, 0.95) is None
    assert power.sharpe_for_bootstrap_pvalue(2, 0.05) is None


def test_one_trial_buys_no_free_sharpe():
    """SR0 is the expected *maximum* of N. With one trial there is no selection."""
    assert power.expected_max_sharpe(1000, 1) == 0.0


def test_pbo_and_the_residual_t_are_not_reported(limits):
    """Both depend on an assumed strategy rather than on the gate, so a number
    for either would be a number about nothing (CLAUDE.md 2)."""
    names = [r.criterion for r in power.requirements(735, 6, limits)]
    assert not any("pbo" in n for n in names)
    assert not any("residual_alpha" in n for n in names)
