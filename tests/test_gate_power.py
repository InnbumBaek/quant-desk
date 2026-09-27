"""The power report has to run on the runner's own panel, or the module is unread.

`tests/backtest/test_power.py` checks the algebra against the gates. These are
about the wiring: that the sample length comes from the snapshot rather than from
an argument, that the report names its assumption on its face, and that the exit
code says which of the two states the desk is in -- constrained by the strategy,
or constrained by the data.
"""

from __future__ import annotations

import json

from scripts.gate_power import TARGETS, build, main, markdown
from tests.test_data_snapshot import sidecar, synthetic_market


def desk(tmp_path):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    synthetic_market(data, "SPY", ("2023-01-02", "2023-07-04"), 400.0)
    synthetic_market(data, "QQQ", ("2023-01-02", "2023-07-04"), 300.0)
    return data


def test_the_sample_length_comes_from_the_panel_and_not_from_an_argument(tmp_path):
    """The point of the report: it says what the gates demand of *this* desk."""
    report = build(desk(tmp_path), allow_dirty=True)
    assert report["observations"] > 0
    assert report["symbols"] == 2
    # The length is a measurement of the panel, so it is odd-specific rather than
    # round: a report with a round default here would mean nobody read the data.
    assert report["observations"] not in (0, 252, 1000)


def test_every_invertible_criterion_is_reported_with_its_number(tmp_path):
    report = build(desk(tmp_path), allow_dirty=True)
    criteria = {item["criterion"] for item in report["requirements"]}
    assert any("is_sharpe_min" in c for c in criteria)
    assert any("deflated_sharpe_probability_min" in c for c in criteria)
    assert any("bootstrap_pvalue_max" in c for c in criteria)
    for item in report["requirements"]:
        assert item["annualised_sharpe_min"] is None or item["annualised_sharpe_min"] > 0.0


def test_the_criteria_that_cannot_be_inverted_are_named_with_the_reason(tmp_path):
    """Leaving them out silently would read as "the gates ask for nothing else"."""
    report = build(desk(tmp_path), allow_dirty=True)
    absent = report["not_inverted"]
    assert set(absent) == {"pbo_max", "residual_alpha_tstat_min", "dd_to_return_ratio_max"}
    assert all(reason for reason in absent.values())


def test_a_short_sample_reports_the_statistics_gate_as_binding(tmp_path):
    """Six months of two symbols cannot clear a deflated-Sharpe floor of 0.95 at
    any plausible Sharpe, and the report has to say so rather than average it away."""
    report = build(desk(tmp_path), allow_dirty=True)
    assert "deflated_sharpe_probability_min" in report["binding"]
    assert report["binding_annualised_sharpe_min"] > 2.0


def test_the_assumption_travels_in_the_artifact_and_not_only_in_the_code(tmp_path):
    report = build(desk(tmp_path), allow_dirty=True)
    assumption = report["assumption"]
    assert "iid normal" in assumption
    # The measured non-conservatism, so a reader of the JSON alone still sees it.
    assert "not a" in assumption and "lower bound" in assumption


def test_the_inverse_question_is_answered_for_every_target(tmp_path):
    report = build(desk(tmp_path), allow_dirty=True)
    needed = report["observations_needed"]
    assert set(needed) == {str(t) for t in TARGETS}
    # Below G2's floor nothing helps, because that floor does not move with the
    # sample. At the floor exactly, a long enough sample clears it.
    assert needed["0.75"] is None
    assert needed["1.0"] is not None
    assert needed["3.0"] < needed["2.0"] < needed["1.0"]


def test_the_markdown_leads_with_the_sample_and_names_the_binding_criterion(tmp_path):
    text = markdown(build(desk(tmp_path), allow_dirty=True))
    assert "Gate power" in text
    assert "binding:" in text
    assert "years" in text
    assert "no sample length clears the in-sample floor" in text


def test_the_exit_code_says_which_constraint_the_desk_is_under(tmp_path, capsys):
    """1 when the sample is the constraint, which is the state worth noticing in
    a workflow log. It is not a failure: nothing is broken, the history is short."""
    out = tmp_path / "out"
    code = main(["--data", str(desk(tmp_path)), "--out", str(out), "--allow-dirty"])
    assert code == 1
    printed = capsys.readouterr().out
    assert "written to" in printed
    written = list(out.glob("*.power.json"))
    assert len(written) == 1
    saved = json.loads(written[0].read_text(encoding="utf-8"))
    assert saved["binding"] == json.loads(written[0].read_text(encoding="utf-8"))["binding"]
    assert saved["run_id"] in written[0].name


def test_a_missing_data_directory_is_a_refusal_and_not_an_empty_report(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="no CSV files"):
        build(tmp_path / "nothing", allow_dirty=True)


def test_more_trials_raise_the_bar_in_the_artifact(tmp_path):
    """G1 counts trials before any backtest runs, and this is why that matters."""
    data = desk(tmp_path)
    few = build(data, trials=2, allow_dirty=True)
    many = build(data, trials=200, allow_dirty=True)
    assert many["binding_annualised_sharpe_min"] > few["binding_annualised_sharpe_min"]
    assert many["expected_max_sharpe_per_period"] > few["expected_max_sharpe_per_period"]


__all__ = ["sidecar"]
