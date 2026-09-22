"""The reproducibility pin: (git SHA, data snapshot id, seed) and the run id over them.

`.claude/skills/backtest-protocol/SKILL.md` says every artefact is sealed with
those three, that a re-run must hash the same, and that **an artefact missing any
one of the three may not be used as gate input**. That last sentence is the whole
point of this module: the rule needs somewhere it can be enforced rather than
remembered, so `ReproPin` cannot be constructed with a blank leg, and
`pin_current` refuses a dirty working tree.

A pin taken on uncommitted code names a commit that does not describe the code
that ran, which is worse than no pin at all: it looks reproducible and is not.
`allow_dirty=True` exists for a scratch run and records the fact in the pin, so a
reader can see why the numbers cannot be reproduced.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core import audit


class NotReproducible(RuntimeError):
    """Raised when a pin cannot honestly be taken."""


@dataclass(frozen=True)
class ReproPin:
    git_sha: str
    snapshot_id: str
    seed: int
    dirty: bool = False

    def __post_init__(self) -> None:
        if not self.git_sha:
            raise NotReproducible("git_sha is empty; an artefact without it is not gate input")
        if not self.snapshot_id:
            raise NotReproducible("snapshot_id is empty; name the data or do not use the result")
        if self.seed is None:
            raise NotReproducible("seed is empty")

    @property
    def run_id(self) -> str:
        """A short, stable name for this exact combination of code, data and seed."""
        material = f"{self.git_sha}:{self.snapshot_id}:{self.seed}"
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "snapshot_id": self.snapshot_id,
            "seed": self.seed,
            "dirty": self.dirty,
        }


def _git(args: list[str], repo: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise NotReproducible(f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}")
    return result.stdout.strip()


def head_sha(repo: Path | None = None) -> str:
    return _git(["rev-parse", "HEAD"], repo or Path.cwd())


def is_dirty(repo: Path | None = None) -> bool:
    return bool(_git(["status", "--porcelain"], repo or Path.cwd()))


def pin_current(
    snapshot_id: str,
    seed: int,
    repo: Path | None = None,
    allow_dirty: bool = False,
    audit_path: Path | None = None,
) -> ReproPin:
    """Take the pin for a run starting now, and record it.

    The dirty-tree refusal is the part that earns its keep. Everything else here
    is bookkeeping; this is the check that stops a number nobody can reproduce
    from entering the gate pipeline wearing a commit hash.
    """
    root = repo or Path.cwd()
    dirty = is_dirty(root)
    if dirty and not allow_dirty:
        raise NotReproducible(
            "the working tree has uncommitted changes, so HEAD does not describe the code "
            "that would run. Commit first, or pass allow_dirty=True for a scratch run that "
            "may not be used as gate input."
        )
    pin = ReproPin(git_sha=head_sha(root), snapshot_id=snapshot_id, seed=seed, dirty=dirty)
    audit.append("repro.pin", pin.as_dict(), path=audit_path)
    return pin


def gate_input_ok(pin: ReproPin) -> tuple[bool, str]:
    """May an artefact sealed with this pin be used as gate input?

    Returns the verdict and the reason, rather than a bare bool, because the
    reason is what a rejection report has to carry.
    """
    if pin.dirty:
        return False, "pinned on a dirty working tree; the commit does not describe the code that ran"
    return True, ""
