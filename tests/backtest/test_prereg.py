"""G1: was N declared before the search, and did the run stay inside it?

G4's deflated Sharpe deflates by N, so N chosen after seeing the answer is the
cheapest way to turn a failed search into a passing one: drop the configurations
that lost and the deflated Sharpe rises with nothing about the strategy having
changed. `core/backtest/power.py` measures how much N moves the bar, which is why
this is a gate and not a convention.

These tests are almost all refusals, for the same reason the limits tests are.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
import yaml

from core.backtest import gates, prereg

GRID = {"lookback": [5, 10, 20, 40, 60, 120], "gross": [1.0]}


def submission(trials: int = 6):
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, size=400)
    matrix = rng.normal(0.0, 0.01, size=(400, trials))
    return gates.Submission(
        alpha_id="statarb-001",
        in_sample=returns[:300],
        out_of_sample=returns[300:],
        fold_returns=list(np.array_split(returns[300:], 4)),
        trial_returns=matrix,
        factor_returns=rng.normal(0.0, 0.01, size=(400, 6)),
    )


def record(**overrides):
    body = {
        "alpha_id": "statarb-001",
        "economic_rationale": "a reason",
        "universe": "five US ETFs",
        "horizon": "weeks",
        "parameters_declared": GRID,
        "committed": True,
        "path": "registry/alphas/statarb-001.yaml",
    }
    body.update(overrides)
    return gates.Preregistration(**body)


# --- the count -----------------------------------------------------------------


def test_the_declared_grid_is_the_product_of_its_axes():
    assert record().declared_trials == 6
    assert record(parameters_declared={"a": [1, 2], "b": [1, 2, 3]}).declared_trials == 6


def test_an_empty_grid_declares_no_number():
    assert record(parameters_declared={}).declared_trials == 0
    assert record(parameters_declared={"a": []}).declared_trials == 0
    assert record(parameters_declared={"a": 5}).declared_trials == 0


# --- refusals ------------------------------------------------------------------


def test_no_declaration_at_all_fails_and_says_why_the_number_is_not_a_claim():
    verdict = gates.g1_preregistration(submission(), None)
    assert not verdict.passed
    assert "counted after the search" in verdict.reason


def test_a_form_with_empty_fields_is_not_a_hypothesis():
    verdict = gates.g1_preregistration(submission(), record(economic_rationale="  "))
    assert not verdict.passed
    assert "economic_rationale" in verdict.reason


def test_a_declaration_with_no_grid_fails():
    verdict = gates.g1_preregistration(submission(), record(parameters_declared={}))
    assert not verdict.passed
    assert "N is whatever the run reports" in verdict.reason


def test_searching_more_configurations_than_were_declared_fails():
    """The failure this gate exists for, in the direction that flatters a result."""
    verdict = gates.g1_preregistration(submission(trials=20), record())
    assert not verdict.passed
    assert "more than the 6 declared" in verdict.reason
    assert verdict.metrics["trials_run"] == 20.0


def test_searching_fewer_than_declared_passes():
    """Declaring a wide grid and running part of it deflates by the wider N, which
    errs against the strategy. That is the safe direction."""
    assert gates.g1_preregistration(submission(trials=3), record()).passed


def test_an_uncommitted_declaration_fails_because_it_could_postdate_the_result():
    verdict = gates.g1_preregistration(submission(), record(committed=False))
    assert not verdict.passed
    assert "uncommitted or modified" in verdict.reason


def test_a_complete_committed_declaration_passes():
    verdict = gates.g1_preregistration(submission(), record())
    assert verdict.passed
    assert verdict.metrics["declared_trials"] == 6.0


# --- loading from registry/alphas/ ---------------------------------------------


def write(directory, alpha_id, **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "id": alpha_id,
        "hypothesis": {
            "economic_rationale": "a reason",
            "universe": "five US ETFs",
            "horizon": "weeks",
            "parameters_declared": GRID,
        },
    }
    document.update(overrides)
    path = directory / f"{alpha_id}.yaml"
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    return path


def test_a_missing_file_is_absence_and_not_an_error(tmp_path):
    assert prereg.load("statarb-001", tmp_path) is None


def test_the_template_is_never_loaded_as_a_declaration(tmp_path):
    """It is a form with an empty hypothesis; reading it would give every alpha the
    same declaration and a grid of nothing."""
    write(tmp_path, "_template")
    assert prereg.load("_template", tmp_path) is None


def test_a_file_that_is_not_a_declaration_raises_rather_than_reading_as_absence(tmp_path):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "statarb-001.yaml").write_text("just a string\n", encoding="utf-8")
    with pytest.raises(prereg.PreregistrationError, match="not a mapping"):
        prereg.load("statarb-001", tmp_path)


def test_a_declaration_naming_another_alpha_is_refused(tmp_path):
    write(tmp_path, "statarb-001", id="statarb-999")
    with pytest.raises(prereg.PreregistrationError, match="not this alpha"):
        prereg.load("statarb-001", tmp_path)


def test_a_grid_that_is_not_a_mapping_is_refused(tmp_path):
    write(
        tmp_path,
        "statarb-001",
        hypothesis={
            "economic_rationale": "a reason",
            "universe": "u",
            "horizon": "h",
            "parameters_declared": [1, 2, 3],
        },
    )
    with pytest.raises(prereg.PreregistrationError, match="no grid to count"):
        prereg.load("statarb-001", tmp_path)


def test_an_untracked_file_loads_but_is_not_committed(tmp_path):
    """The loader reports the state; the gate decides what it means."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    alphas = tmp_path / "registry" / "alphas"
    write(alphas, "statarb-001")
    loaded = prereg.load("statarb-001", alphas, repo=tmp_path)
    assert loaded is not None
    assert loaded.committed is False
    assert not gates.g1_preregistration(submission(), loaded).passed


def test_a_committed_and_unmodified_file_is_committed(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    alphas = tmp_path / "registry" / "alphas"
    path = write(alphas, "statarb-001")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "declare"], cwd=tmp_path, check=True)
    loaded = prereg.load("statarb-001", alphas, repo=tmp_path)
    assert loaded is not None and loaded.committed is True
    assert gates.g1_preregistration(submission(), loaded).passed

    # Editing it after the fact takes the claim away again.
    path.write_text(path.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8")
    again = prereg.load("statarb-001", alphas, repo=tmp_path)
    assert again is not None and again.committed is False


# --- where it binds ------------------------------------------------------------


def test_research_approval_does_not_run_the_preregistration_gate():
    """A gate every submission fails would make `tests/canaries/` pass for the
    wrong reason: a canary has to be rejected for its own statistical defect."""
    from core.backtest.leakage import LeakReport

    verdicts = gates.evaluate(submission(), LeakReport(probes=1, leaks=()))
    assert "G1_preregistration" not in [v.gate for v in verdicts]


def test_live_capital_is_blocked_by_a_missing_preregistration():
    blockers = gates.live_blockers([gates.Verdict("G0_data", True)], submission(), prereg=None)
    assert any("G1_preregistration" in b for b in blockers)


def test_a_good_declaration_removes_that_blocker_but_not_the_others():
    blockers = gates.live_blockers([gates.Verdict("G0_data", True)], submission(), prereg=record())
    assert not any("G1_preregistration" in b for b in blockers)
    assert any("G7_paper" in b for b in blockers)
    assert any("G8_live" in b for b in blockers)
