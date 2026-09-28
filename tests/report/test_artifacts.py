"""Resolving a report's artifact references.

CLAUDE.md 2항 has had half a reader since P0: a hook checked that a numeric claim
carried a reference, and nothing checked that the reference pointed anywhere. These
tests pin the other half, and they pin the refusals hardest -- a resolver that
guesses is worse than no resolver, because it launders a made-up number into a
verified one (ADR-0045).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.report.artifacts import (
    UnresolvedReference,
    artifacts_for,
    default_run,
    references,
    resolve,
    walk,
    written_as,
)

RUN = "17121f69010340d5"


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    (root / "universe").mkdir(parents=True)
    (root / "universe" / f"{RUN}.frontier.json").write_text(
        json.dumps(
            {
                "baseline": {"effective_bets": 3.038659454253419, "observations": 5031},
                "frontier": [{"effective_bets": 6.387126230769066, "count": 26}],
                "unreadable": {},
            }
        ),
        encoding="utf-8",
    )
    (root / "submissions").mkdir()
    for alpha, participation in (("blend-001", 0.000256), ("tsmom-001", 0.000756)):
        (root / "submissions" / f"cafe1234.{alpha}.json").write_text(
            json.dumps({"adv_participation": participation}), encoding="utf-8"
        )
    return root


def only(text: str, run: str | None = None):
    found = references(text, run)
    assert len(found) == 1
    return found[0]


def test_a_reference_resolves_to_the_value_the_run_wrote(registry: Path):
    reference = only(f"{{{{artifact:{RUN}/baseline.effective_bets}}}}")
    assert resolve(reference, registry) == pytest.approx(3.038659454253419)


def test_a_list_index_resolves(registry: Path):
    reference = only(f"{{{{artifact:{RUN}/frontier[0].count}}}}")
    assert resolve(reference, registry) == 26


def test_a_metric_no_artifact_holds_is_refused(registry: Path):
    with pytest.raises(UnresolvedReference, match="is in none of"):
        resolve(only(f"{{{{artifact:{RUN}/baseline.sharpe}}}}"), registry)


def test_a_run_that_wrote_nothing_is_refused(registry: Path):
    with pytest.raises(UnresolvedReference, match="no committed artifact"):
        resolve(only("{{artifact:deadbeef/baseline.effective_bets}}"), registry)


def test_an_index_past_the_end_is_refused(registry: Path):
    """The aggregate message names the reference, because the reason is per file."""
    with pytest.raises(UnresolvedReference, match="is in none of 1 artifact"):
        resolve(only(f"{{{{artifact:{RUN}/frontier[3].count}}}}"), registry)


def test_walk_says_which_step_failed():
    with pytest.raises(UnresolvedReference, match=r"\[3\] is not in this list"):
        walk({"frontier": [1, 2]}, "frontier[3]")
    with pytest.raises(UnresolvedReference, match="'sharpe' is not a key here"):
        walk({"baseline": {}}, "baseline.sharpe")


def test_a_path_meaning_two_things_in_one_run_is_refused(registry: Path):
    """The reference would mean six things and the report would be right by luck."""
    with pytest.raises(UnresolvedReference, match="means different things"):
        resolve(only("{{artifact:cafe1234/adv_participation}}"), registry)


def test_naming_the_file_resolves_what_the_run_id_alone_could_not(registry: Path):
    reference = only("{{artifact:cafe1234.blend-001/adv_participation}}")
    assert resolve(reference, registry) == pytest.approx(0.000256)


def test_a_short_reference_uses_the_run_the_file_declared(registry: Path):
    text = f"<!-- artifact-run: {RUN} -->\n3.04 {{{{artifact:/baseline.effective_bets}}}}"
    reference = only("3.04 {{artifact:/baseline.effective_bets}}", default_run(text))
    assert reference.run_id == RUN
    assert resolve(reference, registry) == pytest.approx(3.038659454253419)


def test_a_short_reference_with_no_declared_run_is_refused(registry: Path):
    with pytest.raises(UnresolvedReference, match="names no run"):
        resolve(only("{{artifact:/baseline.effective_bets}}", None), registry)


def test_a_run_id_that_is_a_path_is_refused(registry: Path):
    with pytest.raises(UnresolvedReference):
        resolve(only("{{artifact:../../etc/passwd/baseline}}"), registry)


def test_derived_is_read_off_the_marker():
    assert only("{{artifact:x/y|derived}}").derived
    assert not only("{{artifact:x/y}}").derived


def test_the_empty_path_is_refused():
    with pytest.raises(UnresolvedReference, match="empty metric path"):
        walk({"a": 1}, "  ")


def test_digits_are_compared_at_the_precision_they_were_written():
    assert written_as(3.038659454253419, "유효 베팅 3.04")
    assert written_as(3.038659454253419, "유효 베팅 3.0387")
    assert not written_as(3.038659454253419, "유효 베팅 3.05")


def test_a_transposed_digit_fails():
    assert not written_as(0.832458019914136, "요구 DSR 0.823")


def test_a_thousands_separator_is_a_number():
    assert written_as(5031, "관측 5,031")


def test_a_nonnumeric_value_is_matched_as_text():
    assert written_as("US", "market US")
    assert not written_as("US", "market KR")


def test_every_artifact_of_a_run_is_searched(registry: Path):
    assert len(artifacts_for("cafe1234", registry)) == 2
