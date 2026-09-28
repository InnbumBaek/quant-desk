"""Resolving `{{artifact:<run_id>/<metric>}}` references against committed artifacts.

CLAUDE.md 2항: every number in a report carries a reference to the artifact that
produced it, and a number typed directly is void. `.claude/hooks/numbers_from_artifacts.py`
has checked the first half since P0 -- that a numeric claim has a reference on its
line. Nothing checked the second half: that the reference **points at a real
value**, and that the digits written beside it are that value.

A reference nobody resolves is a typed number with extra steps. It is the same
shape as every other finding in this repository (ADR-0015, 0016, 0026, 0032,
0035, 0041): the rule was written, the reader was not.

A reference resolves against the artifact files a run wrote:
`registry/**/<run_id>.json` and `registry/**/<run_id>.<name>.json`. The path
after the slash is dotted, with `[i]` for a list index:

    {{artifact:17121f69010340d5/frontier[0].effective_bets}}

A run that wrote several files (one submission record per alpha) holds the same
path in each with a different value. That is an error, not a first-hit win: the
reference would mean six things and the report would be right by luck. Name the
file instead, by appending its part to the run id:

    {{artifact:f3c9d40604a56e2e.blend-001/adv_participation}}

A report that cites one run over and over declares it once and drops the run id
from each reference, because a row of eight-character hashes is a row nobody
reads and an unread record is the failure this whole convention exists to
prevent:

    <!-- artifact-run: 17121f69010340d5 -->
    | 26 | 5,031 {{artifact:/frontier[0].observations}} | ...

`|derived` marks a number computed *from* artifact values rather than copied from
one (a ratio of two metrics, a percentage of a floor). Such a reference is still
resolved -- it must point somewhere real -- but the digits are not compared,
because no artifact holds them. The marker is in the text, where a reader sees
it, rather than in an exemption list where it would go quiet.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: `{{artifact:run_id/dotted.path[0]}}`, optionally `|derived`. A leading slash
#: and no run id means the run the file declares in `ARTIFACT_RUN`.
REFERENCE = re.compile(r"\{\{artifact:(?P<run>[^}/|]*)/(?P<path>[^}|]+?)(?P<flags>\|[a-z]+)*\}\}")
#: `<!-- artifact-run: 17121f69010340d5 -->`, the run a report cites throughout.
ARTIFACT_RUN = re.compile(r"<!--\s*artifact-run:\s*(?P<run>[\w.-]+)\s*-->")
_STEP = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
#: A number as a reader would write one: 0.832, 1.45, 5,031, -0.0140.
LITERAL = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?", re.IGNORECASE)

DEFAULT_ROOT = Path("registry")


class UnresolvedReference(LookupError):
    """A reference names a run or a metric no committed artifact holds."""


@dataclass(frozen=True)
class Reference:
    run_id: str
    path: str
    derived: bool
    text: str

    @property
    def label(self) -> str:
        return f"{self.run_id or '<no run declared>'}/{self.path}"


def default_run(text: str) -> str | None:
    """The run id a report declares for its short references, if it declares one."""
    found = ARTIFACT_RUN.search(text)
    return found.group("run") if found else None


def references(text: str, run: str | None = None) -> list[Reference]:
    """Every artifact reference in `text`, in the order written.

    `run` supplies the run id for short references. A short reference with no run
    to fall back on is returned with an empty run id, so the caller reports it as
    unresolved rather than silently skipping it.
    """
    out = []
    for match in REFERENCE.finditer(text):
        flags = match.group("flags") or ""
        out.append(
            Reference(
                run_id=match.group("run").strip() or (run or ""),
                path=match.group("path").strip(),
                derived="derived" in flags,
                text=match.group(0),
            )
        )
    return out


def artifacts_for(run_id: str, root: Path = DEFAULT_ROOT) -> list[Path]:
    """The committed artifact files a run wrote, sorted for a stable search order."""
    if not run_id:
        raise UnresolvedReference(
            "this reference names no run and the file declares none (add `<!-- artifact-run: ... -->`)"
        )
    if "/" in run_id or ".." in run_id:
        raise UnresolvedReference(f"{run_id!r} is not a run id")
    found = {path for path in root.rglob(f"{run_id}.json")}
    found |= {path for path in root.rglob(f"{run_id}.*.json")}
    return sorted(found)


def walk(value: Any, path: str) -> Any:
    """`frontier[0].effective_bets` against a loaded artifact. Raises on a miss."""
    cursor = value
    for step in _steps(path):
        if isinstance(step, int):
            if not isinstance(cursor, list) or not -len(cursor) <= step < len(cursor):
                raise UnresolvedReference(f"[{step}] is not in this list")
            cursor = cursor[step]
        else:
            if not isinstance(cursor, dict) or step not in cursor:
                raise UnresolvedReference(f"{step!r} is not a key here")
            cursor = cursor[step]
    return cursor


def _steps(path: str) -> Iterator[str | int]:
    if not path.strip():
        raise UnresolvedReference("an empty metric path resolves to the whole artifact")
    for match in _STEP.finditer(path):
        name, index = match.group(1), match.group(2)
        yield name.strip() if name is not None else int(index)


def resolve(reference: Reference, root: Path = DEFAULT_ROOT) -> Any:
    """The value a reference names.

    Every artifact of the run is searched, because a run writes several files and
    the reference does not say which. Two files holding the same path with
    *different* values is an error rather than a first-hit win: the reference
    would then mean two things and the report would be reproducible by luck.
    """
    files = artifacts_for(reference.run_id, root)
    if not files:
        raise UnresolvedReference(f"no committed artifact for run {reference.run_id}")
    hits: dict[Any, list[Path]] = {}
    for path in files:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UnresolvedReference(f"{path} is not readable JSON: {error}") from error
        try:
            value = walk(loaded, reference.path)
        except UnresolvedReference:
            continue
        key = json.dumps(value, sort_keys=True)
        hits.setdefault(key, []).append(path)
    if not hits:
        raise UnresolvedReference(f"{reference.label} is in none of {len(files)} artifact(s) of that run")
    if len(hits) > 1:
        where = "; ".join(f"{files[0].name}: {value}" for value, files in sorted(hits.items()))
        raise UnresolvedReference(f"{reference.label} means different things in the same run ({where})")
    return json.loads(next(iter(hits)))


def written_as(value: Any, line: str) -> bool:
    """Does `line` contain `value` written out, at whatever precision it was written?

    The digits are compared at the precision the author chose: a report saying
    0.83 is not wrong because the artifact holds 0.832458. Rounding the artifact
    to the written precision is what a reader does by eye, and it is what makes a
    transposed digit fail.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value) in line
    for literal in LITERAL.finditer(line):
        text = literal.group(0).replace(",", "")
        try:
            written = float(text)
        except ValueError:  # pragma: no cover - the pattern only matches numbers
            continue
        decimals = len(text.partition(".")[2].partition("e")[0])
        if round(float(value), decimals) == round(written, decimals):
            return True
    return False
