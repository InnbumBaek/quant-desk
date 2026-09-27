"""The status of an alpha is derived, and the derivation has one direction rule.

Measurements raise, the recorded human decision lowers. Most of these tests are
the refusals that keep that asymmetry true, because the failure mode is not a
wrong table -- it is a table that says `gated` because somebody wrote `gated` in
a file.
"""

from __future__ import annotations

import json

import pytest
import yaml

from core.alphas import lifecycle
from scripts import alpha_status

GATES_PASSED = [
    {"gate": name, "passed": True, "reason": "", "metrics": {}}
    for name in (
        "G0_data",
        "G2_in_sample",
        "G3_oos",
        "G4_statistics",
        "G5_robustness",
        "G6_capacity",
        "G1_preregistration",
    )
]


def write_declaration(alphas, alpha_id="tsmom-001", status="proposed"):
    alphas.mkdir(parents=True, exist_ok=True)
    (alphas / f"{alpha_id}.yaml").write_text(
        yaml.safe_dump({"id": alpha_id, "status": status}, allow_unicode=True), encoding="utf-8"
    )


def write_submission(submissions, alpha_id="tsmom-001", run_id="aaaa", *, approved, verdicts, failed=()):
    submissions.mkdir(parents=True, exist_ok=True)
    path = submissions / f"{run_id}.{alpha_id}.json"
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "alpha_id": alpha_id,
                "approved": approved,
                "failed_gates": list(failed),
                "verdicts": verdicts,
                "declaration": {"committed_and_unmodified": True},
            }
        ),
        encoding="utf-8",
    )
    return path


def write_ledger(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"entries": entries}, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def desk(tmp_path):
    alphas, submissions = tmp_path / "alphas", tmp_path / "submissions"
    write_declaration(alphas)
    ledger = write_ledger(alphas / "lifecycle.yaml", [])
    return alphas, submissions, ledger


# --- measurements raise -------------------------------------------------------


def test_a_declared_alpha_with_no_submission_is_proposed(desk):
    state = lifecycle.derive("tsmom-001", *desk)
    assert state.status is lifecycle.Status.PROPOSED
    assert "never submitted" in state.why


def test_a_rejected_submission_leaves_it_proposed_and_names_the_gates(desk):
    alphas, submissions, ledger = desk
    write_submission(
        submissions,
        approved=False,
        failed=["G4_statistics", "G5_robustness"],
        verdicts=[{"gate": "G0_data", "passed": True, "reason": "", "metrics": {}}],
    )
    state = lifecycle.derive("tsmom-001", alphas, submissions, ledger)
    assert state.status is lifecycle.Status.PROPOSED
    assert "G4_statistics" in state.why


def test_clearing_the_research_gates_raises_it_to_gated(desk):
    alphas, submissions, ledger = desk
    write_submission(submissions, approved=True, verdicts=GATES_PASSED)
    state = lifecycle.derive("tsmom-001", alphas, submissions, ledger)
    assert state.status is lifecycle.Status.GATED


def test_approved_without_g1_is_not_gated_because_n_was_never_declared(desk):
    alphas, submissions, ledger = desk
    without_g1 = [v for v in GATES_PASSED if v["gate"] != "G1_preregistration"]
    write_submission(submissions, approved=True, verdicts=without_g1)
    state = lifecycle.derive("tsmom-001", alphas, submissions, ledger)
    assert state.status is lifecycle.Status.PROPOSED


def test_a_backtest_can_never_reach_paper_on_its_own(desk):
    """`paper` needs G7, and G7 needs a paper-trading record no backtest produces."""
    alphas, submissions, ledger = desk
    write_submission(submissions, approved=True, verdicts=GATES_PASSED)
    assert lifecycle.derive("tsmom-001", alphas, submissions, ledger).status is lifecycle.Status.GATED

    with_g7 = [*GATES_PASSED, {"gate": "G7_paper", "passed": True, "reason": "", "metrics": {}}]
    write_submission(submissions, run_id="bbbb", approved=True, verdicts=with_g7)
    assert lifecycle.derive("tsmom-001", alphas, submissions, ledger).status is lifecycle.Status.PAPER


# --- the ledger only lowers ---------------------------------------------------


def test_the_ledger_can_stop_a_gated_alpha(desk):
    alphas, submissions, ledger = desk
    write_submission(submissions, approved=True, verdicts=GATES_PASSED)
    write_ledger(
        ledger,
        [{"alpha_id": "tsmom-001", "to": "stopped", "at": "2026-09-27", "by": "owner", "why": "decay"}],
    )
    state = lifecycle.derive("tsmom-001", alphas, submissions, ledger)
    assert state.status is lifecycle.Status.STOPPED
    assert state.ledger[0].applied
    assert "decay" in state.why


@pytest.mark.parametrize("target", ["gated", "paper", "live"])
def test_writing_a_promotion_in_the_ledger_is_refused_and_recorded(desk, target):
    alphas, submissions, ledger = desk
    write_ledger(
        ledger,
        [{"alpha_id": "tsmom-001", "to": target, "at": "2026-09-27", "by": "someone", "why": "looks good"}],
    )
    state = lifecycle.derive("tsmom-001", alphas, submissions, ledger)
    assert state.status is lifecycle.Status.PROPOSED
    assert state.ledger[0].applied is False
    assert state.ledger[0].refused_because


def test_a_ledger_entry_that_does_not_lower_is_refused_even_when_allowed(desk):
    alphas, submissions, ledger = desk
    write_ledger(
        ledger,
        [
            {"alpha_id": "tsmom-001", "to": "retired", "at": "2026-09-27", "by": "o", "why": "done"},
            {"alpha_id": "tsmom-001", "to": "stopped", "at": "2026-09-28", "by": "o", "why": "again"},
        ],
    )
    state = lifecycle.derive("tsmom-001", alphas, submissions, ledger)
    assert state.status is lifecycle.Status.RETIRED
    assert "does not lower" in state.ledger[1].refused_because


def test_a_ledger_entry_for_another_alpha_is_not_applied_here(desk):
    alphas, submissions, ledger = desk
    write_ledger(
        ledger,
        [{"alpha_id": "other-001", "to": "retired", "at": "2026-09-27", "by": "o", "why": "n/a"}],
    )
    state = lifecycle.derive("tsmom-001", alphas, submissions, ledger)
    assert state.status is lifecycle.Status.PROPOSED
    assert state.ledger == ()


def test_a_word_that_is_not_a_status_is_refused_not_ignored(desk):
    alphas, submissions, ledger = desk
    write_ledger(
        ledger,
        [{"alpha_id": "tsmom-001", "to": "paused", "at": "2026-09-27", "by": "o", "why": "typo"}],
    )
    state = lifecycle.derive("tsmom-001", alphas, submissions, ledger)
    assert "is not a status" in state.ledger[0].refused_because


# --- absence is not an answer -------------------------------------------------


def test_a_missing_ledger_is_not_an_empty_ledger(tmp_path):
    alphas = tmp_path / "alphas"
    write_declaration(alphas)
    state = lifecycle.derive("tsmom-001", alphas, tmp_path / "submissions", alphas / "lifecycle.yaml")
    assert any("not set up" in note for note in state.notes)
    assert lifecycle.load_ledger(alphas / "lifecycle.yaml") is None


def test_a_declared_empty_ledger_says_so_without_a_note(desk):
    alphas, submissions, ledger = desk
    assert lifecycle.load_ledger(ledger) == []
    assert lifecycle.derive("tsmom-001", alphas, submissions, ledger).notes == ()


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("", "does not declare an empty ledger"),
        ("- a\n- b\n", "not a mapping"),
        ("other: 1\n", "no `entries` key"),
        ("entries: 3\n", "not a list"),
        ("entries: [3]\n", "entry 0 is not a mapping"),
        ("entries:\n  - {alpha_id: a, to: stopped, at: '', by: o, why: w}\n", "leaves ['at'] empty"),
    ],
)
def test_a_ledger_that_is_not_a_ledger_raises_rather_than_reads_as_empty(tmp_path, text, fragment):
    path = tmp_path / "lifecycle.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(lifecycle.LedgerError, match=fragment.replace("[", r"\[").replace("]", r"\]")):
        lifecycle.load_ledger(path)


def test_an_unparseable_submission_record_is_not_counted_as_evidence(desk):
    alphas, submissions, ledger = desk
    submissions.mkdir(parents=True, exist_ok=True)
    (submissions / "cccc.tsmom-001.json").write_text("{not json", encoding="utf-8")
    state = lifecycle.derive("tsmom-001", alphas, submissions, ledger)
    assert state.submissions == ()
    assert state.status is lifecycle.Status.PROPOSED


# --- no status is a permission ------------------------------------------------


@pytest.mark.parametrize("status", list(lifecycle.Status))
def test_no_status_grants_capital(status):
    state = lifecycle.Lifecycle(alpha_id="tsmom-001", status=status, why="constructed")
    assert state.may_take_capital is False
    assert state.as_dict()["may_take_capital"] is False


def test_the_order_and_allocation_path_does_not_read_the_status():
    """A status a file edit can change must not be able to move capital.

    CLAUDE.md rule 1 keeps judgement out of the order path; this keeps a
    *derived label* out of it too, which is the same risk wearing a different
    name. Permission is `live_blockers` being empty, and it never is.
    """
    from pathlib import Path as _Path

    governance = (
        "core/risk/limits.py",
        "core/execution/orders.py",
        "core/pipeline.py",
        "core/portfolio/center_book.py",
        ".claude/hooks/pretrade_gate.py",
    )
    checked = 0
    for name in governance:
        path = _Path(name)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert "core.alphas" not in text, f"{name} reads the derived alpha status"
        assert "lifecycle" not in text, f"{name} reads the derived alpha status"
        checked += 1
    # A loop that checked nothing is not a check: the modules are named in
    # CLAUDE.md rules 1 and 5, so all five must be here to be read.
    assert checked == len(governance), f"only {checked} of {len(governance)} governance modules found"


def test_the_ledger_may_only_record_the_two_lowering_states():
    assert lifecycle.LEDGER_ALLOWED == {lifecycle.Status.STOPPED, lifecycle.Status.RETIRED}
    assert lifecycle.Status.LIVE not in lifecycle.LEDGER_ALLOWED


def test_the_declared_status_is_reported_separately_from_the_current_one(desk):
    alphas, submissions, ledger = desk
    write_submission(submissions, approved=True, verdicts=GATES_PASSED)
    state = lifecycle.derive("tsmom-001", alphas, submissions, ledger)
    assert state.declared_status == "proposed"
    assert state.status is lifecycle.Status.GATED


# --- the listing --------------------------------------------------------------


def test_the_template_and_the_ledger_are_not_alphas(tmp_path):
    alphas = tmp_path / "alphas"
    write_declaration(alphas, "tsmom-001")
    write_declaration(alphas, "_template")
    write_ledger(alphas / "lifecycle.yaml", [])
    assert lifecycle.declared_ids(alphas) == ["tsmom-001"]


# --- the script ---------------------------------------------------------------


def test_the_table_is_written_and_names_the_capital_rule(desk, tmp_path):
    alphas, submissions, ledger = desk
    out = tmp_path / "STATUS.md"
    code = alpha_status.main(
        [
            "--alphas",
            str(alphas),
            "--submissions",
            str(submissions),
            "--ledger",
            str(ledger),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "tsmom-001" in text
    assert "live_blockers" in text


def test_an_unreadable_ledger_stops_the_table_rather_than_reading_as_empty(desk, tmp_path, capsys):
    alphas, submissions, ledger = desk
    ledger.write_text("entries: 3\n", encoding="utf-8")
    code = alpha_status.main(
        ["--alphas", str(alphas), "--submissions", str(submissions), "--ledger", str(ledger)]
    )
    assert code == 1
    assert "unreadable" in capsys.readouterr().out


def test_the_json_form_carries_the_refusals(desk, capsys):
    alphas, submissions, ledger = desk
    write_ledger(ledger, [{"alpha_id": "tsmom-001", "to": "live", "at": "x", "by": "y", "why": "z"}])
    assert (
        alpha_status.main(
            ["--alphas", str(alphas), "--submissions", str(submissions), "--ledger", str(ledger), "--json"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["ledger"][0]["applied"] is False
    assert payload[0]["may_take_capital"] is False


# --- the repository's own state ------------------------------------------------


def test_this_repository_has_a_declared_ledger_and_one_alpha():
    assert lifecycle.load_ledger() is not None, "the ledger must exist; absence is not an empty ledger"
    assert "tsmom-001" in lifecycle.declared_ids()
