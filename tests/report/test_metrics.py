"""The reporting metrics are tested for what they refuse to invent.

A Sortino with no losing day, a Calmar with no drawdown and a beta with no
benchmark are all reported as absent. The tests that matter here are the ones
that would catch a 0.0 being quoted as a measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.report.metrics import performance_summary

RNG = np.random.default_rng(11)


def test_a_flat_positive_series_reports_its_geometric_return():
    daily = np.full(252, 0.001)
    perf = performance_summary(daily)
    assert perf.cagr == pytest.approx(1.001**252 - 1.0, rel=1e-9)
    assert perf.volatility == pytest.approx(0.0)
    assert perf.max_drawdown == pytest.approx(0.0)


def test_no_losing_day_leaves_sortino_absent_not_infinite():
    perf = performance_summary(np.full(252, 0.001))
    assert perf.sortino is None
    assert any("undefined rather than infinite" in note for note in perf.notes)


def test_no_drawdown_leaves_calmar_absent_not_infinite():
    perf = performance_summary(np.full(252, 0.001))
    assert perf.calmar is None
    assert any("Calmar is undefined" in note for note in perf.notes)


def test_sortino_exceeds_sharpe_when_the_downside_is_the_thin_tail():
    """A series whose losses are small and frequent and whose gains are large."""
    daily = np.where(np.arange(504) % 4 == 0, 0.02, -0.004)
    perf = performance_summary(daily)
    assert perf.sortino is not None
    assert perf.sortino > perf.sharpe


def test_a_wipeout_reports_minus_one_hundred_percent_not_a_complex_root():
    daily = np.concatenate([np.full(10, 0.01), [-1.5], np.full(10, 0.01)])
    perf = performance_summary(daily)
    assert perf.cagr == pytest.approx(-1.0)
    assert any("compounded to zero or below" in note for note in perf.notes)


def test_calmar_is_cagr_over_drawdown():
    daily = RNG.normal(0.0004, 0.01, 756)
    perf = performance_summary(daily)
    assert perf.calmar == pytest.approx(perf.cagr / perf.max_drawdown)


def test_turnover_is_annualised_from_the_engines_per_bar_series():
    daily = RNG.normal(0.0002, 0.008, 252)
    perf = performance_summary(daily, turnover=np.full(252, 0.05))
    assert perf.turnover_annual == pytest.approx(0.05 * 252)


def test_absent_turnover_is_absent_not_zero():
    perf = performance_summary(RNG.normal(0.0, 0.01, 60))
    assert perf.turnover_annual is None
    assert any("turnover is unreported, not zero" in note for note in perf.notes)


def test_absent_benchmark_is_absent_not_zero_beta():
    perf = performance_summary(RNG.normal(0.0, 0.01, 60))
    assert perf.beta is None and perf.alpha_annual is None
    assert any("beta and alpha are unreported, not zero" in note for note in perf.notes)


def test_beta_recovers_a_known_exposure():
    benchmark = RNG.normal(0.0003, 0.011, 2000)
    returns = 0.4 * benchmark + 0.0002
    perf = performance_summary(returns, benchmark=benchmark)
    assert perf.beta == pytest.approx(0.4, abs=1e-9)
    assert perf.alpha_annual == pytest.approx(0.0002 * 252, abs=1e-9)


def test_a_motionless_benchmark_leaves_beta_absent():
    perf = performance_summary(RNG.normal(0.0, 0.01, 60), benchmark=np.zeros(60))
    assert perf.beta is None
    assert any("does not move" in note for note in perf.notes)


def test_a_benchmark_of_a_different_length_is_refused():
    with pytest.raises(ValueError, match="a different sample"):
        performance_summary(RNG.normal(0.0, 0.01, 60), benchmark=RNG.normal(0.0, 0.01, 59))


def test_a_turnover_series_of_a_different_length_is_refused():
    with pytest.raises(ValueError, match="the same bars"):
        performance_summary(RNG.normal(0.0, 0.01, 60), turnover=np.full(59, 0.05))


def test_a_non_finite_return_is_refused():
    daily = RNG.normal(0.0, 0.01, 60)
    daily[7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        performance_summary(daily)


def test_one_period_is_not_enough_to_report_on():
    with pytest.raises(ValueError, match="not enough"):
        performance_summary(np.array([0.01]))


def test_a_matrix_is_not_a_return_series():
    with pytest.raises(ValueError, match="one series"):
        performance_summary(RNG.normal(0.0, 0.01, (60, 2)))
