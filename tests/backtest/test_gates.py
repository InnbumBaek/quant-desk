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


# --- G7: the gate before live capital, which had no code (ADR-0032) -----------


def _paper(days: int | None):
    """A submission carrying a paper-trading record, or carrying none."""
    submission = _submission()
    if days is not None:
        submission.paper_trading_days = days
        submission.paper_trading_measured = True
    return submission


def test_a_submission_that_never_paper_traded_is_refused(limits):
    """The default. 0 sessions with no record must not read as "not applicable"."""
    verdict = gates.g7_paper_trading(_paper(None), limits)
    assert not verdict.passed
    assert "no paper-trading record" in verdict.reason


def test_zero_recorded_sessions_is_still_a_refusal(limits):
    """A record saying zero is a measurement, and it is below any floor."""
    verdict = gates.g7_paper_trading(_paper(0), limits)
    assert not verdict.passed
    assert "0 paper session(s)" in verdict.reason


def test_a_record_one_session_short_is_refused(limits):
    floor = int(limits["gates"]["paper_trading_days_min"])
    assert not gates.g7_paper_trading(_paper(floor - 1), limits).passed


def test_a_record_at_the_floor_passes(limits):
    floor = int(limits["gates"]["paper_trading_days_min"])
    verdict = gates.g7_paper_trading(_paper(floor), limits)
    assert verdict.passed
    assert verdict.metrics["paper_trading_days_min"] == float(floor)


def test_the_floor_comes_from_the_table_and_not_from_the_gate(limits):
    """The point of the gate existing at all: the owner moves the number."""
    passing = _paper(63)
    assert gates.g7_paper_trading(passing, limits).passed
    limits["gates"]["paper_trading_days_min"] = 252
    assert not gates.g7_paper_trading(passing, limits).passed


def test_a_truthy_non_boolean_does_not_count_as_a_record(limits):
    """`is not True` and not truthiness: "yes" must not clear a gate."""
    submission = _paper(None)
    submission.paper_trading_measured = "yes"  # type: ignore[assignment]
    submission.paper_trading_days = 999
    assert not gates.g7_paper_trading(submission, limits).passed


def test_a_malformed_session_count_is_a_refusal_and_not_an_exception(limits):
    """A gate that raises stops the evaluation instead of recording a refusal, so
    the audit log ends up with no verdict for the gate that objected."""
    submission = _paper(None)
    submission.paper_trading_measured = True
    submission.paper_trading_days = "sixty three"  # type: ignore[assignment]
    verdict = gates.g7_paper_trading(submission, limits)
    assert not verdict.passed
    assert "not a session count" in verdict.reason


def test_a_boolean_day_count_is_not_a_count(limits):
    """True is 1 to Python, and 1 session is not 63 -- but it must not read as a
    count at all, on the same reasoning as `as_measurement` (ADR-0015)."""
    submission = _paper(None)
    submission.paper_trading_measured = True
    submission.paper_trading_days = True  # type: ignore[assignment]
    assert not gates.g7_paper_trading(submission, limits).passed


# --- live capital is never granted by code -----------------------------------


def test_live_blockers_is_never_empty_even_when_everything_passes(limits):
    """A function that could return no blockers would be a function that
    approves live trading."""
    submission = _paper(63)
    clean = [gates.Verdict("G0_data", True), gates.Verdict("G2_in_sample", True)]
    blockers = gates.live_blockers(clean, submission, limits)
    assert blockers == [
        "G8_live is a human approval: the owner authorises live capital and no code path grants it"
    ]


def test_live_blockers_names_the_failed_research_gates_too(limits):
    submission = _paper(None)
    verdicts = [gates.Verdict("G4_statistics", False, reason="PBO 0.4 >= 0.05")]
    blockers = gates.live_blockers(verdicts, submission, limits)
    assert any("G4_statistics" in b for b in blockers)
    assert any("G7_paper" in b for b in blockers)
    assert any("G8_live" in b for b in blockers)


def test_research_approval_does_not_run_the_paper_gate(limits):
    """`approved()` must keep meaning "the research stands up". Folding G7 in
    would silently turn every existing caller's verdict into a deployment claim."""
    submission = _paper(None)
    verdicts = gates.evaluate(submission, LeakReport(probes=1, leaks=()), limits)
    assert [v.gate for v in verdicts] == [
        "G0_data",
        "G2_in_sample",
        "G3_oos",
        "G4_statistics",
        "G5_robustness",
        "G6_capacity",
    ]
    assert "G7_paper" not in [v.gate for v in verdicts]
