"""Fund-level VaR, expected shortfall, pod correlation and the halt."""

from __future__ import annotations

import numpy as np
import pytest

from core.risk.fund import (
    average_pod_correlation,
    check_fund,
    expected_shortfall,
    fund_returns,
    fund_snapshot,
    historical_var,
    required_actions,
)

CLEAN = {
    "pod_count": 3,
    "var95_1d": 0.015,
    "es975_1d": 0.022,
    "pod_avg_correlation": 0.10,
    "drawdown": -0.02,
}


def series(n: int = 300, mean: float = 0.0004, sigma: float = 0.006, seed: int = 4) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, sigma, n)


# --- the statistics ---------------------------------------------------------


def test_the_var_is_a_positive_loss():
    losses = np.array([-0.05, -0.04, -0.03, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    assert historical_var(losses, 0.90) > 0.0


def test_the_var_is_the_quantile_of_the_worst_days():
    ordered = np.linspace(-0.10, 0.10, 101)
    # The 5th percentile of this series is -0.09, so the VaR95 is 9%.
    assert historical_var(ordered, 0.95) == pytest.approx(0.09, abs=1e-9)


def test_the_expected_shortfall_is_worse_than_the_var():
    """ES averages the tail, so it sits beyond the quantile that opens it."""
    sample = series()
    assert expected_shortfall(sample, 0.95)[0] > historical_var(sample, 0.95)


def test_the_expected_shortfall_reports_how_many_days_it_averaged():
    shortfall, tail = expected_shortfall(series(200), 0.975)
    assert shortfall > 0.0
    assert tail == 5, "2.5% of 200 days"


def test_a_level_that_is_not_a_tail_probability_is_refused():
    with pytest.raises(ValueError, match="not a tail probability"):
        historical_var(series(), 0.4)
    with pytest.raises(ValueError, match="not a tail probability"):
        expected_shortfall(series(), 1.0)


def test_a_non_finite_series_is_refused():
    with pytest.raises(ValueError, match="finite return series"):
        historical_var(np.array([0.01, np.nan, 0.02]))


# --- pod correlation --------------------------------------------------------


def test_identical_pods_correlate_at_one():
    sample = series(100)
    assert average_pod_correlation({"a": sample, "b": sample.copy()}) == pytest.approx(1.0)


def test_opposite_pods_correlate_at_minus_one():
    sample = series(100)
    assert average_pod_correlation({"a": sample, "b": -sample}) == pytest.approx(-1.0)


def test_the_average_is_over_pairs_not_over_the_matrix():
    """Three pods give three pairs; the diagonal is not one of them."""
    a, b = series(200, seed=1), series(200, seed=2)
    value = average_pod_correlation({"a": a, "b": b, "c": (a + b) / 2})
    assert 0.0 < value < 1.0


def test_one_pod_has_no_pairwise_correlation():
    with pytest.raises(ValueError, match="at least two pods"):
        average_pod_correlation({"a": series(50)})


def test_a_flat_pod_has_no_correlation_rather_than_a_zero_one():
    with pytest.raises(ValueError, match="no variance"):
        average_pod_correlation({"a": series(50), "b": np.zeros(50)})


# --- the fund's own return series -------------------------------------------


def test_the_fund_return_is_the_allocation_weighted_sum():
    a, b = np.array([0.01, 0.02, -0.01]), np.array([-0.01, 0.00, 0.03])
    combined = fund_returns({"a": a, "b": b}, {"a": 0.5, "b": 0.25})
    assert combined.tolist() == pytest.approx([0.5 * 0.01 - 0.25 * 0.01, 0.01, 0.5 * -0.01 + 0.25 * 0.03])


def test_an_allocation_is_required_for_every_pod():
    """Equal weighting would invent an allocation, and a VaR on an invented fund is fiction."""
    with pytest.raises(ValueError, match="missing: \\['b'\\]"):
        fund_returns({"a": series(10), "b": series(10)}, {"a": 0.5})


def test_an_allocation_for_a_pod_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="unknown: \\['ghost'\\]"):
        fund_returns({"a": series(10)}, {"a": 0.5, "ghost": 0.2})


def test_allocations_above_the_funds_capital_are_refused():
    with pytest.raises(ValueError, match="more than the fund's capital"):
        fund_returns({"a": series(10), "b": series(10)}, {"a": 0.7, "b": 0.7})


def test_a_negative_allocation_is_refused():
    with pytest.raises(ValueError, match="negative allocation"):
        fund_returns({"a": series(10)}, {"a": -0.2})


def test_pods_of_different_lengths_are_refused():
    with pytest.raises(ValueError, match="different lengths"):
        fund_returns({"a": series(10), "b": series(11)}, {"a": 0.4, "b": 0.4})


# --- measuring the snapshot -------------------------------------------------


def test_a_year_of_returns_measures_everything():
    pods = {"a": series(300, seed=1), "b": series(300, seed=2)}
    exposure = fund_snapshot(pods, {"a": 0.4, "b": 0.4})

    assert exposure.unmeasured == ()
    assert exposure.snapshot["pod_count"] == 2
    assert exposure.snapshot["returns_used"] == 250
    assert exposure.snapshot["var95_1d"] > 0.0
    assert exposure.snapshot["es975_1d"] > exposure.snapshot["var95_1d"]


def test_a_short_history_leaves_the_tail_unmeasured():
    pods = {"a": series(30, seed=1), "b": series(30, seed=2)}
    exposure = fund_snapshot(pods, {"a": 0.4, "b": 0.4})
    assert exposure.snapshot["var95_1d"] is None
    assert exposure.snapshot["es975_1d"] is None
    assert any("short of the 60" in note for note in exposure.notes)


def test_the_window_cap_is_recorded_beside_the_number():
    """A historical VaR cannot exceed the worst day in its window; say so."""
    pods = {"a": series(300)}
    exposure = fund_snapshot(pods, {"a": 0.5})
    assert any("cannot exceed the worst of them" in note for note in exposure.notes)


def test_a_single_pod_is_exempt_from_the_correlation_limit_and_says_why():
    exposure = fund_snapshot({"a": series(300)}, {"a": 0.5})
    assert exposure.snapshot["pod_avg_correlation"] is None
    assert exposure.snapshot["pod_count"] == 1
    assert any("does not exist with a single pod" in note for note in exposure.notes)
    assert check_fund(exposure.snapshot) == []


def test_a_flat_pod_leaves_the_correlation_unmeasured_and_blocks():
    pods = {"a": series(300), "b": np.zeros(300)}
    exposure = fund_snapshot(pods, {"a": 0.4, "b": 0.4})
    assert exposure.snapshot["pod_avg_correlation"] is None
    assert "POD_CORRELATION_UNMEASURED" in {b.code for b in check_fund(exposure.snapshot)}


def test_the_drawdown_is_measured_from_the_funds_own_equity_curve():
    falling = np.full(300, -0.001)
    exposure = fund_snapshot({"a": falling}, {"a": 1.0})
    assert exposure.snapshot["drawdown"] < -0.1


# --- the limits -------------------------------------------------------------


def test_a_clean_fund_has_no_breach():
    assert check_fund(dict(CLEAN)) == []


@pytest.mark.parametrize(
    "patch, code",
    [
        ({"var95_1d": 0.025}, "FUND_VAR95"),
        ({"es975_1d": 0.04}, "FUND_ES975"),
        ({"pod_avg_correlation": 0.45}, "POD_CORRELATION"),
        ({"drawdown": -0.11}, "FUND_DD_REDUCE"),
        ({"drawdown": -0.16}, "FUND_DD_HALT"),
    ],
)
def test_each_fund_limit_trips(patch, code):
    assert code in {b.code for b in check_fund({**CLEAN, **patch})}


@pytest.mark.parametrize(
    "key, code",
    [
        ("var95_1d", "FUND_VAR_UNMEASURED"),
        ("es975_1d", "FUND_ES_UNMEASURED"),
        ("pod_avg_correlation", "POD_CORRELATION_UNMEASURED"),
        ("drawdown", "FUND_DD_UNMEASURED"),
        ("pod_count", "POD_COUNT_UNMEASURED"),
    ],
)
def test_a_missing_fund_measurement_blocks(key, code):
    snapshot = {name: value for name, value in CLEAN.items() if name != key}
    assert code in {b.code for b in check_fund(snapshot)}


def test_the_halt_tier_wins_over_the_reduce_tier():
    """Both conditions hold at -16%; only the more severe action is returned."""
    breaches = check_fund({**CLEAN, "drawdown": -0.20})
    assert {b.code for b in breaches} == {"FUND_DD_HALT"}
    assert required_actions(breaches) == ("halt_all",)


def test_the_reduce_tier_names_its_action():
    breaches = check_fund({**CLEAN, "drawdown": -0.12})
    assert required_actions(breaches) == ("cut_gross_half",)


def test_a_fund_inside_its_tiers_needs_no_action():
    assert required_actions(check_fund(dict(CLEAN))) == ()


def test_zero_is_a_measurement_here_too():
    flat = {**CLEAN, "var95_1d": 0.0, "es975_1d": 0.0, "pod_avg_correlation": 0.0, "drawdown": 0.0}
    assert check_fund(flat) == []


def test_a_measured_snapshot_is_one_check_fund_can_read():
    pods = {"a": series(300, seed=1), "b": series(300, seed=2), "c": series(300, seed=3)}
    exposure = fund_snapshot(pods, {"a": 0.3, "b": 0.3, "c": 0.3})
    codes = {breach.code for breach in check_fund(exposure.snapshot)}
    assert not {code for code in codes if code.endswith("UNMEASURED")}
