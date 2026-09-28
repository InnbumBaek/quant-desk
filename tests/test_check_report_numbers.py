"""The reader for CLAUDE.md 2항: which reports it judges, and what it refuses.

The boundary matters as much as the check. Twelve decision records predate the
reader and are out of scope on purpose -- retrofitting references into them would
mean editing decision records after the fact, and a record edited later is a
record nobody can trust. So the scope is a number, not a growing list of
filenames, and these tests count it in both directions (ADR-0045).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_report_numbers import LEGACY_BEFORE, in_scope, main, problems

RUN = "17121f69010340d5"


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    (root / "universe").mkdir(parents=True)
    (root / "universe" / f"{RUN}.frontier.json").write_text(
        json.dumps({"baseline": {"effective_bets": 3.038659454253419}}), encoding="utf-8"
    )
    return root


def report(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "ADR-9999-probe.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_claim_with_a_resolving_reference_passes(tmp_path: Path, registry: Path):
    path = report(tmp_path, f"유효 베팅이 3.04배 늘었다 {{{{artifact:{RUN}/baseline.effective_bets}}}}")
    assert problems(path, registry) == []


def test_a_claim_with_no_reference_is_reported(tmp_path: Path, registry: Path):
    path = report(tmp_path, "유효 베팅이 2.1배 늘었다.")
    assert any("no artifact reference" in line for line in problems(path, registry))


def test_a_reference_that_points_nowhere_is_reported(tmp_path: Path, registry: Path):
    path = report(tmp_path, f"3.04배 {{{{artifact:{RUN}/baseline.sharpe}}}}")
    assert any("does not resolve" in line for line in problems(path, registry))


def test_digits_that_disagree_with_the_artifact_are_reported(tmp_path: Path, registry: Path):
    """The point of the whole convention: a mistyped number must not pass."""
    path = report(tmp_path, f"3.05배 {{{{artifact:{RUN}/baseline.effective_bets}}}}")
    assert any("not written on this line" in line for line in problems(path, registry))


def test_a_derived_number_needs_the_marker_and_then_passes(tmp_path: Path, registry: Path):
    unmarked = report(tmp_path, f"69% {{{{artifact:{RUN}/baseline.effective_bets}}}}")
    assert any("not written on this line" in line for line in problems(unmarked, registry))
    marked = report(tmp_path, f"69% {{{{artifact:{RUN}/baseline.effective_bets|derived}}}}")
    assert problems(marked, registry) == []


def test_a_derived_marker_does_not_excuse_a_reference_to_nothing(tmp_path: Path, registry: Path):
    path = report(tmp_path, f"69% {{{{artifact:{RUN}/nope|derived}}}}")
    assert any("does not resolve" in line for line in problems(path, registry))


def test_a_table_row_is_checked_like_prose(tmp_path: Path, registry: Path):
    """The P0 hook skipped table rows, which is where the measured numbers are."""
    path = report(tmp_path, "| 26 | 2.1배 |")
    assert any("no artifact reference" in line for line in problems(path, registry))


def test_a_bullet_is_checked_like_prose(tmp_path: Path, registry: Path):
    path = report(tmp_path, "- 유효 베팅이 2.1배 늘었다")
    assert any("no artifact reference" in line for line in problems(path, registry))


def test_a_heading_is_not_a_claim(tmp_path: Path, registry: Path):
    assert problems(report(tmp_path, "## 21종목은 2.1배다"), registry) == []


def test_inline_code_is_quoting_code_not_citing_a_number(tmp_path: Path, registry: Path):
    """A document about this convention has to be able to write the syntax down."""
    path = report(tmp_path, "형식은 `{{artifact:<run_id>/<metric>}}`이고 상한은 `0.7`이다")
    assert problems(path, registry) == []


def test_a_fenced_block_is_not_a_claim(tmp_path: Path, registry: Path):
    path = report(tmp_path, "```python\nMAX = 0.7  # 70% 상한\n```\n")
    assert problems(path, registry) == []


def test_the_scope_starts_at_the_first_record_written_after_the_reader():
    names = [path.name for path in in_scope()]
    assert names, "the in-scope list may not be empty"
    assert all(int(name[4:8]) >= LEGACY_BEFORE for name in names)
    assert f"ADR-{LEGACY_BEFORE:04d}" in names[0]


def test_every_record_is_either_in_scope_or_older_than_the_boundary():
    """Counted in both directions, so the exemption cannot grow by one file."""
    every = sorted(Path("registry/decisions").glob("ADR-*.md"))
    scoped = set(in_scope())
    exempt = [p for p in every if p not in scoped]
    assert len(scoped) + len(exempt) == len(every)
    assert all(int(p.name[4:8]) < LEGACY_BEFORE for p in exempt)


def test_the_repository_s_own_in_scope_records_pass():
    """Not a self-test: this is the check CI runs, run here so a push cannot skip it."""
    assert main([]) == 0


def test_a_blocked_report_exits_two(tmp_path: Path, registry: Path):
    path = report(tmp_path, "유효 베팅이 2.1배 늘었다.")
    assert main([str(path), "--registry", str(registry)]) == 2
