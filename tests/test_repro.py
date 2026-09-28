"""The pin exists to refuse, so the refusals are what is tested."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core import audit
from core.repro import NotReproducible, ReproPin, gate_input_ok, head_sha, is_dirty, pin_current


def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for args in (
        ["init", "-q"],
        ["config", "user.email", "test@example.com"],
        ["config", "user.name", "test"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    (root / "alpha.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "first"], cwd=root, check=True, capture_output=True)
    return root


def test_a_pin_needs_all_three_legs():
    with pytest.raises(NotReproducible, match="git_sha"):
        ReproPin(git_sha="", snapshot_id="snap", seed=1)
    with pytest.raises(NotReproducible, match="snapshot_id"):
        ReproPin(git_sha="abc", snapshot_id="", seed=1)


def test_run_id_is_stable_and_depends_on_every_leg():
    base = ReproPin("abc123", "snap1", 7)
    assert base.run_id == ReproPin("abc123", "snap1", 7).run_id
    assert base.run_id != ReproPin("abc124", "snap1", 7).run_id
    assert base.run_id != ReproPin("abc123", "snap2", 7).run_id
    assert base.run_id != ReproPin("abc123", "snap1", 8).run_id
    assert len(base.run_id) == 16


def test_a_clean_tree_pins_to_head(tmp_path):
    root = git_repo(tmp_path)
    log = tmp_path / "audit.log"
    pin = pin_current("snap-abc", seed=42, repo=root, audit_path=log)

    assert pin.git_sha == head_sha(root)
    assert not pin.dirty
    assert gate_input_ok(pin) == (True, "")

    records = list(audit.read(log))
    assert records[0]["event"] == "repro.pin"
    assert records[0]["data"]["run_id"] == pin.run_id


def test_a_dirty_tree_is_refused(tmp_path):
    root = git_repo(tmp_path)
    (root / "alpha.py").write_text("x = 2\n", encoding="utf-8")
    assert is_dirty(root)

    with pytest.raises(NotReproducible, match="uncommitted changes"):
        pin_current("snap-abc", seed=42, repo=root)


def test_a_scratch_run_may_pin_dirty_but_is_not_gate_input(tmp_path):
    root = git_repo(tmp_path)
    (root / "alpha.py").write_text("x = 2\n", encoding="utf-8")

    pin = pin_current("snap-abc", seed=42, repo=root, allow_dirty=True)
    assert pin.dirty
    ok, reason = gate_input_ok(pin)
    assert not ok
    assert "dirty working tree" in reason


# --- one code check, several panels ------------------------------------------


def test_the_same_code_can_seal_a_second_panel(tmp_path):
    """A run that judges two universes checks the code once and reseals per panel."""
    root = git_repo(tmp_path)
    log = tmp_path / "audit.log"
    first = pin_current("snap-one", seed=42, repo=root, audit_path=log)
    second = first.for_snapshot("snap-two", audit_path=log)

    assert second.git_sha == first.git_sha
    assert second.seed == first.seed
    assert second.dirty == first.dirty
    assert second.snapshot_id == "snap-two"
    assert second.run_id != first.run_id

    events = [r["data"]["snapshot_id"] for r in audit.read(log)]
    assert events == ["snap-one", "snap-two"]


def test_resealing_the_same_panel_is_the_same_pin(tmp_path):
    root = git_repo(tmp_path)
    pin = pin_current("snap-one", seed=42, repo=root)
    assert pin.for_snapshot("snap-one") is pin


def test_resealing_does_not_launder_a_scratch_run(tmp_path):
    """A scratch run stays a scratch run for every universe in it."""
    root = git_repo(tmp_path)
    (root / "alpha.py").write_text("x = 2\n", encoding="utf-8")
    pin = pin_current("snap-one", seed=42, repo=root, allow_dirty=True)

    resealed = pin.for_snapshot("snap-two")
    assert resealed.dirty
    assert not gate_input_ok(resealed)[0]


def test_resealing_needs_a_snapshot_id(tmp_path):
    root = git_repo(tmp_path)
    pin = pin_current("snap-one", seed=42, repo=root)
    with pytest.raises(NotReproducible, match="snapshot_id is empty"):
        pin.for_snapshot("")


def test_an_untracked_file_also_counts_as_dirty(tmp_path):
    """An untracked strategy file is exactly the code a pin would fail to describe."""
    root = git_repo(tmp_path)
    (root / "new_alpha.py").write_text("y = 1\n", encoding="utf-8")
    with pytest.raises(NotReproducible, match="uncommitted changes"):
        pin_current("snap-abc", seed=1, repo=root)


def test_outside_a_repository_the_failure_is_explicit(tmp_path):
    with pytest.raises(NotReproducible, match="failed"):
        pin_current("snap-abc", seed=1, repo=tmp_path)
