"""A rule that names a file which does not exist governs nothing.

CLAUDE.md's first absolute rule lists the four modules where no LLM judgement may
sit, and its fifth names the center book. Those rules are enforced by people and
by agents reading them, not by an import, so a rename anywhere in that list would
quietly narrow the rule's scope with nothing failing. The same is true of the 33
paths the 26 agent definitions point their agents at: an agent told to use a
module that has been renamed will improvise, and improvising on the order path is
the one thing rule 1 forbids.

This is the cheapest possible guard for that -- it checks existence, nothing else
-- and it is the same finding as ADR-0032 one level up: a rule with no mechanism
is a wish, and the mechanism can be five lines.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: A repo path in backticks. Only the roots that are ours, so a backticked
#: `numpy/random` or a shell fragment is not read as a promise about this tree.
REFERENCE = re.compile(r"`((?:core|scripts|tests|registry|docs|\.claude)/[A-Za-z0-9_./-]+)`")

#: The modules CLAUDE.md rule 1 puts off limits to LLM judgement, plus the center
#: book from rule 5. Spelled out rather than parsed: the point is that this list
#: cannot lose a member by accident, and a parser reading the rule would lose the
#: member exactly when the rule's wording changed.
GOVERNANCE_PATH = (
    "core/risk/limits.py",
    "core/execution/orders.py",
    "core/pipeline.py",
    ".claude/hooks/pretrade_gate.py",
    "core/portfolio/center_book.py",
)

DOCUMENTS = ("CLAUDE.md", "README.md")


def referenced(text: str) -> list[str]:
    return sorted(set(REFERENCE.findall(text)))


def cases() -> list[tuple[str, str]]:
    files = [Path(d) for d in DOCUMENTS] + sorted(Path(".claude/agents").glob("*.md"))
    return [(str(f), ref) for f in files for ref in referenced(f.read_text(encoding="utf-8"))]


@pytest.mark.parametrize("document, reference", cases(), ids=lambda x: x)
def test_a_document_that_names_a_path_names_one_that_exists(document, reference):
    assert Path(reference).exists(), (
        f"{document} points at {reference}, which is not in the tree. "
        "Either the path moved and the document was not followed, or the document "
        "is describing something that was never built."
    )


@pytest.mark.parametrize("path", GOVERNANCE_PATH)
def test_every_module_the_absolute_rules_name_is_still_there(path):
    """Rule 1 and rule 5 are scoped by filename, so a rename is a scope change."""
    assert Path(path).exists(), f"CLAUDE.md scopes an absolute rule to {path}, which is gone"


def test_the_rules_still_name_every_module_this_test_guards():
    """The other direction: the list above must stay the rule's list, not a copy
    of it that stopped matching."""
    rules = Path("CLAUDE.md").read_text(encoding="utf-8")
    for path in GOVERNANCE_PATH:
        assert path in rules, (
            f"{path} is guarded here but CLAUDE.md no longer mentions it. "
            "If the rule dropped it, drop it here too, with a record of why."
        )


def test_the_canary_directory_the_rules_point_at_holds_canaries():
    """Rule 8 says a gate-threshold change must pass `tests/canaries/` again,
    which is only a rule while that directory has tests in it."""
    canaries = sorted(Path("tests/canaries").glob("*.py"))
    assert [p.name for p in canaries if p.name.startswith("test_")], "no canary tests"
