"""Gates read limits.yaml. A gate with its threshold in code cannot be governed."""

import copy

import numpy as np
import pytest

from core.backtest import gates
from core.backtest.leakage import LeakReport
from core.risk.limits import load_limits
from tests.canaries.alphas import canary_factor, canary_lookahead


@pytest.fixture
def limits():
    return copy.deepcopy(load_limits())


def _submission():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0012, 0.01, size=1000)
    trials = rng.normal(0.0, 0.01, size=(1000, 10))
    trials[:, 0] = returns
    return gates.Submission(
        alpha_id="unit",
        in_sample=returns[:600],
        out_of_sample=returns[600:],
        fold_returns=list(np.array_split(returns[600:], 5)),
        trial_returns=trials,
        factor_returns=rng.normal(0.0, 0.01, size=(1000, 6)),
    )


def test_raising_the_limit_flips_the_verdict(limits):
    submission = _submission()
    assert gates.g2_in_sample(submission, limits).passed

    limits["gates"]["is_sharpe_min"] = 99.0
    assert not gates.g2_in_sample(submission, limits).passed


def test_every_enforced_threshold_comes_from_the_limits_table(limits):
    """Loosen the table to nothing and every enforced criterion must pass."""
    submission, _ = canary_factor()
    limits["gates"].update(
        {
            "is_sharpe_min": -99.0,
            "oos_to_is_sharpe_ratio_min": -99.0,
            "fold_sign_stability_min": 0.0,
            "deflated_sharpe_probability_min": 0.0,
            "bootstrap_pvalue_max": 1.01,
            "dd_to_return_ratio_max": 1e9,
            "pbo_max": 1.01,
            "residual_alpha_tstat_min": -99.0,
            "adv_participation_max": 1.0,
            "book_correlation_abs_max": 1.0,
        }
    )
    verdicts = [
        gates.g2_in_sample(submission, limits),
        gates.g3_oos(submission, limits),
        gates.g4_statistics(submission, limits),
        gates.g6_capacity(submission, limits),
    ]
    assert gates.approved(verdicts)


def test_capacity_reads_participation_and_book_correlation(limits):
    submission = _submission()
    submission.adv_participation = 0.10
    submission.book_correlation = 0.9
    verdict = gates.g6_capacity(submission, limits)
    assert not verdict.passed
    assert "ADV participation" in verdict.reason
    assert "book correlation" in verdict.reason


def test_robustness_fails_on_a_parameter_cliff():
    submission = _submission()
    # Same volatility, most of the edge gone: a Sharpe cliff, not a rescaling.
    submission.param_perturbed = [submission.full - submission.full.mean() * 0.7]
    verdict = gates.g5_robustness(submission)
    assert not verdict.passed
    assert "decays" in verdict.reason


def test_robustness_fails_when_double_cost_eats_the_edge():
    submission = _submission()
    submission.cost_doubled = submission.full - 0.002
    verdict = gates.g5_robustness(submission)
    assert not verdict.passed
    assert "double cost" in verdict.reason


def test_a_clean_scan_passes_g0():
    assert gates.g0_data(LeakReport(probes=24)).passed


def test_g4_enforces_every_documented_criterion(limits):
    """All five G4 criteria now come from limits.yaml (ADR-0002)."""
    submission, _ = canary_lookahead()  # profitable, so no criterion trivially fails
    metrics = gates.g4_statistics(submission, limits).metrics
    for key in (
        "deflated_sharpe_probability",
        "pbo",
        "residual_alpha_tstat",
        "bootstrap_pvalue",
        "dd_to_return",
    ):
        assert key in metrics


def test_loosening_one_g4_key_still_leaves_the_others_binding(limits):
    """canary_factor fails several G4 criteria; relaxing only the residual t must not pass it."""
    submission, _ = canary_factor()
    limits["gates"]["residual_alpha_tstat_min"] = -99.0
    assert not gates.g4_statistics(submission, limits).passed
