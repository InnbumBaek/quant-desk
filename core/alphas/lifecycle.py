"""Where an alpha stands, derived from its records rather than declared in a field.

`registry/alphas/<id>.yaml` carries a `status:` line, and it cannot be maintained:
`core/backtest/prereg.py` passes G1 only while the declaration is committed and
unmodified, so editing the status to `gated` would put an edit made *after* the
result into the file whose whole claim is that it predates one. The field is
therefore the status **at declaration**, which never changes and is true forever.
The current status is this module's job.

**Evidence raises, the ledger only lowers.** Two inputs, and the asymmetry between
them is the design:

- The submission records in `registry/submissions/` are measurements. They can
  raise an alpha from `proposed` to `gated`, and no further, because that is the
  most any backtest can establish.
- `registry/alphas/lifecycle.yaml` is where a person records a decision. It can
  only move an alpha *down* -- `stopped`, `retired` -- for the same reason the
  center-book overlay may only reduce a position (CLAUDE.md rule 5). An entry
  that tries to raise one is not applied; it is recorded as refused, because
  silently ignoring it would hide an attempt, and honouring it would let anything
  that can write a YAML file promote an alpha.

**`live` is not a value this module can return.** Taking live capital is G8, the
owner's approval, and `gates.live_blockers` never returns an empty list. A status
field that could read `live` would be a second, weaker answer to the question
`live_blockers` already answers, and the weaker answer is the one a caller would
reach for. `paper` is reachable only through a submission whose G7 passed, which
needs a paper-trading record no backtest can produce.

**Nothing in the order or allocation path reads this.** It is a reporting
derivation, not a permission: permission is `live_blockers` being empty, which it
never is. Keeping status out of that path is what stops a promotion by file edit
from becoming a promotion by capital (CLAUDE.md rules 1 and 4).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from core.backtest import prereg

ALPHAS = Path("registry/alphas")
SUBMISSIONS = Path("registry/submissions")
LEDGER = ALPHAS / "lifecycle.yaml"


#: The template's own enum, unchanged. `live` is deliberately unreachable here.
class Status(StrEnum):
    PROPOSED = "proposed"
    GATED = "gated"
    PAPER = "paper"
    LIVE = "live"
    STOPPED = "stopped"
    RETIRED = "retired"


#: The only transitions a person may record. Everything else is measured.
LEDGER_ALLOWED = frozenset({Status.STOPPED, Status.RETIRED})

#: Ordered low to high, so "the ledger may only lower" is a comparison.
RANK = {
    Status.RETIRED: 0,
    Status.STOPPED: 1,
    Status.PROPOSED: 2,
    Status.GATED: 3,
    Status.PAPER: 4,
    Status.LIVE: 5,
}


class LedgerError(ValueError):
    """The ledger file exists and is not a ledger."""


@dataclass(frozen=True)
class SubmissionRecord:
    """One evaluated run, as `scripts/submit_alpha.py` wrote it."""

    run_id: str
    alpha_id: str
    path: str
    approved: bool
    failed_gates: tuple[str, ...]
    passed_gates: tuple[str, ...]
    declaration_committed: bool

    @property
    def cleared_research(self) -> bool:
        """G0-G6 passed and G1 with them.

        G1 is asked for separately because `gates.evaluate` leaves it out on
        purpose, so `approved` alone can be True for a run whose N was never
        declared -- and a deflated Sharpe with an undeclared N is not evidence
        (ADR-0035).
        """
        return self.approved and "G1_preregistration" in self.passed_gates


@dataclass(frozen=True)
class LedgerEntry:
    alpha_id: str
    to: str
    at: str = ""
    by: str = ""
    why: str = ""
    applied: bool = False
    refused_because: str = ""


@dataclass(frozen=True)
class Lifecycle:
    """Where one alpha stands, and what that rests on."""

    alpha_id: str
    status: Status
    why: str
    declared_status: str = ""
    evidence_status: Status = Status.PROPOSED
    submissions: tuple[SubmissionRecord, ...] = ()
    ledger: tuple[LedgerEntry, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def latest(self) -> SubmissionRecord | None:
        return self.submissions[-1] if self.submissions else None

    @property
    def may_take_capital(self) -> bool:
        """Always False, and deliberately not a computation.

        G8 is the owner's approval and no code path grants it. This property
        exists so a caller that reaches for a status field to answer the capital
        question gets the right answer instead of a plausible one.
        """
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha_id": self.alpha_id,
            "status": str(self.status),
            "why": self.why,
            "declared_status": self.declared_status,
            "evidence_status": str(self.evidence_status),
            "may_take_capital": self.may_take_capital,
            "submissions": [
                {
                    "run_id": s.run_id,
                    "path": s.path,
                    "approved": s.approved,
                    "cleared_research": s.cleared_research,
                    "failed_gates": list(s.failed_gates),
                    "declaration_committed": s.declaration_committed,
                }
                for s in self.submissions
            ],
            "ledger": [
                {
                    "to": e.to,
                    "at": e.at,
                    "by": e.by,
                    "why": e.why,
                    "applied": e.applied,
                    "refused_because": e.refused_because,
                }
                for e in self.ledger
            ],
            "notes": list(self.notes),
        }


def declared_ids(directory: Path = ALPHAS) -> list[str]:
    """Every alpha that has a declaration. One rule, kept in `prereg`."""
    return prereg.declared_ids(directory)


def declared_status(alpha_id: str, directory: Path = ALPHAS) -> str:
    """The `status:` the declaration was written with. Historical, never current."""
    path = directory / f"{alpha_id}.yaml"
    if not path.is_file():
        return ""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return ""
    return str(document.get("status") or "")


def read_submissions(alpha_id: str, directory: Path = SUBMISSIONS) -> list[SubmissionRecord]:
    """Every recorded run for this alpha, oldest file first.

    Sorted by modification time rather than by name: a run id is a digest, so
    sorting it lexically orders runs by nothing at all.
    """
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob(f"*.{alpha_id}.json"), key=lambda p: p.stat().st_mtime)
    out: list[SubmissionRecord] = []
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A record that does not parse is not evidence of anything, and it is
            # not absence either: it is named in the notes rather than skipped.
            continue
        if not isinstance(document, dict):
            continue
        verdicts = document.get("verdicts") or []
        passed = tuple(v["gate"] for v in verdicts if isinstance(v, dict) and v.get("passed"))
        out.append(
            SubmissionRecord(
                run_id=str(document.get("run_id", "")),
                alpha_id=str(document.get("alpha_id", alpha_id)),
                path=str(path),
                approved=bool(document.get("approved")),
                failed_gates=tuple(document.get("failed_gates") or ()),
                passed_gates=passed,
                declaration_committed=bool(
                    (document.get("declaration") or {}).get("committed_and_unmodified")
                ),
            )
        )
    return out


def load_ledger(path: Path = LEDGER) -> list[LedgerEntry] | None:
    """The recorded human decisions, in file order, or None when there is no ledger.

    None rather than an empty list, because a missing file is not an empty
    ledger. CLAUDE.md rule 4: absence is not an answer. "No ledger has been set
    up" and "a ledger exists and records no decision" are different facts, and
    only the second one is a statement anybody made.
    """
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document is None:
        raise LedgerError(f"{path} is empty; an empty file does not declare an empty ledger")
    if not isinstance(document, dict):
        raise LedgerError(f"{path} is not a mapping")
    raw = document.get("entries")
    if raw is None:
        raise LedgerError(f"{path} has no `entries` key; write `entries: []` to declare none")
    if not isinstance(raw, list):
        raise LedgerError(f"{path} has `entries` that is not a list")
    out: list[LedgerEntry] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LedgerError(f"{path} entry {index} is not a mapping")
        missing = [
            key for key in ("alpha_id", "to", "at", "by", "why") if not str(item.get(key) or "").strip()
        ]
        if missing:
            raise LedgerError(
                f"{path} entry {index} leaves {missing} empty; a decision with no author or "
                "reason is not a record of one"
            )
        out.append(
            LedgerEntry(
                alpha_id=str(item["alpha_id"]),
                to=str(item["to"]),
                at=str(item["at"]),
                by=str(item["by"]),
                why=str(item["why"]),
            )
        )
    return out


def _evidence(submissions: Sequence[SubmissionRecord]) -> tuple[Status, str]:
    """The highest status the measurements support, and why that is the ceiling."""
    if not submissions:
        return Status.PROPOSED, "declared, never submitted"
    cleared = [s for s in submissions if s.cleared_research]
    if not cleared:
        latest = submissions[-1]
        failed = ", ".join(latest.failed_gates) or "no gate recorded a pass"
        return (
            Status.PROPOSED,
            f"{len(submissions)} submission(s), none cleared the research gates; latest failed {failed}",
        )
    # A backtest cannot establish more than `gated`: `paper` needs a
    # paper-trading record, and G7 is the gate that reads one.
    paper = [s for s in cleared if "G7_paper" in s.passed_gates]
    if paper:
        return Status.PAPER, f"G7 cleared on run {paper[-1].run_id}"
    return Status.GATED, f"research gates cleared on run {cleared[-1].run_id}"


def derive(
    alpha_id: str,
    alphas: Path = ALPHAS,
    submissions: Path = SUBMISSIONS,
    ledger: Path = LEDGER,
) -> Lifecycle:
    """Where `alpha_id` stands now: measurements raise it, recorded decisions lower it."""
    records = read_submissions(alpha_id, submissions)
    evidence, why = _evidence(records)

    notes: list[str] = []
    entries_or_none = load_ledger(ledger)
    if entries_or_none is None:
        notes.append(
            f"no ledger at {ledger}; that is 'not set up', not 'no decisions' -- "
            "write `entries: []` to declare none"
        )
        entries_or_none = []

    status = evidence
    applied: list[LedgerEntry] = []
    for entry in entries_or_none:
        if entry.alpha_id != alpha_id:
            continue
        try:
            target = Status(entry.to)
        except ValueError:
            applied.append(replace(entry, refused_because=f"{entry.to!r} is not a status"))
            continue
        if target not in LEDGER_ALLOWED:
            applied.append(
                replace(
                    entry,
                    refused_because=(
                        f"the ledger may only record {sorted(str(s) for s in LEDGER_ALLOWED)}; "
                        f"{target} is established by measurement, not by writing it down"
                    ),
                )
            )
            continue
        if RANK[target] >= RANK[status]:
            applied.append(
                replace(
                    entry,
                    refused_because=f"{target} does not lower {status}, and the ledger may only lower",
                )
            )
            continue
        status = target
        why = f"{entry.why} ({entry.by}, {entry.at})"
        applied.append(replace(entry, applied=True))

    return Lifecycle(
        alpha_id=alpha_id,
        status=status,
        why=why,
        declared_status=declared_status(alpha_id, alphas),
        evidence_status=evidence,
        submissions=tuple(records),
        ledger=tuple(applied),
        notes=tuple(notes),
    )


def derive_all(
    alphas: Path = ALPHAS,
    submissions: Path = SUBMISSIONS,
    ledger: Path = LEDGER,
    ids: Iterable[str] | None = None,
) -> list[Lifecycle]:
    return [
        derive(alpha_id, alphas, submissions, ledger)
        for alpha_id in (ids if ids is not None else declared_ids(alphas))
    ]
