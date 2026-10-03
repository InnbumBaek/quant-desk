"""Every key in the limits table must have code that reads it.

ADR-0015 (pod), ADR-0016 (fund) and ADR-0026 (capacity) were all the same
finding: **a limit written down is not a limit.** Each time, a key sat in
`core/risk/limits.yaml` with nothing reading it, so a book could breach it and
nothing would happen. Each time it was found by a hand grep, which is why it kept
coming back.

`tests/limits/test_limits.py` asserts the *values*. That is a different thing
from a reader, and the difference is exactly how `gates.paper_trading_days_min`
survived: a test asserting it equals 63 made it look attended to while no gate
consulted it. **A test is not a reader.**

So this is the mechanical version of that grep, and it works in both directions:

1. a key with no reader fails unless it is in `NO_READER_YET` with a reason, and
2. **an entry in `NO_READER_YET` that has since gained a reader fails too**, so
   the list cannot quietly rot into a list of keys nobody rechecked.

The check is a floor, not a proof. It looks for the key's name as a string
literal anywhere under `core/`, `scripts/` and `.claude/hooks/`, so it catches a
key nobody reads but cannot tell whether the reader that mentions it enforces it
correctly. The per-limit tests are what say that.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

LIMITS = Path("core/risk/limits.yaml")
SEARCHED = ("core", "scripts", ".claude/hooks")

#: Segments that name a shape rather than a limit, so the segment above them is
#: what a reader would mention. `pod.drawdown.warn.abs` is read as
#: `rule["abs"]` after the tier was selected by name, so "warn" is the evidence.
STRUCTURAL = frozenset({"abs", "action"})

#: Not a limit: the table's own schema version.
NOT_A_LIMIT = frozenset({"version"})

#: Keys with no reader, and why. **Every line here is a limit that does not
#: currently exist**, whatever the table says. Delete a line when its reader
#: lands; the test fails if you forget.
#:
#: Empty since the allocator landed (ADR-0010): the three keys that used to be
#: excused here -- `pod.kelly_fraction`, `horizon.risk_budget_share_max` and
#: `allocation.lock_months` -- are read by `core/portfolio/allocate.py`, so their
#: lines had to go or the parametrised test below would fail on each of them.
#: An empty list is the goal state, not a reason to relax: the test still runs in
#: both directions, so a key that loses its reader fails here rather than going
#: quiet.
NO_READER_YET: dict[str, str] = {}


def leaves(node: object, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if isinstance(node, dict):
        out: list[tuple[str, ...]] = []
        for key, value in node.items():
            out += leaves(value, path + (str(key),))
        return out
    return [path]


def probe_for(path: tuple[str, ...]) -> str:
    """The name a reader of this key would spell out."""
    leaf = path[-1]
    if leaf in STRUCTURAL and len(path) > 1:
        return path[-2]
    return leaf


def sources() -> dict[Path, str]:
    found: dict[Path, str] = {}
    for root in SEARCHED:
        for file in Path(root).rglob("*.py"):
            found[file] = file.read_text(encoding="utf-8")
    return found


def readers(probe: str, text: dict[Path, str]) -> list[str]:
    pattern = re.compile(rf"""["']{re.escape(probe)}["']""")
    return sorted(str(path) for path, body in text.items() if pattern.search(body))


def limit_paths() -> list[tuple[str, ...]]:
    table = yaml.safe_load(LIMITS.read_text(encoding="utf-8"))
    return [p for p in leaves(table) if p[-1] not in NOT_A_LIMIT]


@pytest.mark.parametrize("path", limit_paths(), ids=lambda p: ".".join(p))
def test_the_key_is_read_by_code_or_is_recorded_as_not_being_a_limit_yet(path):
    dotted = ".".join(path)
    found = readers(probe_for(path), sources())
    if dotted in NO_READER_YET:
        assert not found, (
            f"{dotted} is listed in NO_READER_YET but {found} now mentions it. "
            "If the reader landed, delete the line -- a stale exception list is how "
            "an unenforced limit hides."
        )
        return
    assert found, (
        f"{dotted} is in {LIMITS} with no code reading it, so it is not a limit. "
        "Add the reader, or add the key to NO_READER_YET with why it cannot be read yet."
    )


def test_the_exception_list_only_names_keys_the_table_actually_has():
    """A renamed key would otherwise keep its excuse and lose its reader at once."""
    declared = {".".join(p) for p in limit_paths()}
    missing = sorted(set(NO_READER_YET) - declared)
    assert not missing, f"{missing} are excused but no longer in {LIMITS}"


def test_a_value_assertion_is_not_counted_as_a_reader():
    """The bug this file exists for: `tests/` is not searched.

    `gates.paper_trading_days_min` was asserted in tests/limits/test_limits.py and
    read by nothing, and that made it look attended to. ADR-0032 gave it a reader.

    The demonstration used to borrow whichever key was still unread, which tied
    this property to the state of the table: `allocation.lock_months` was the
    example until the allocator started reading it, and then this test failed for
    a reason that had nothing to do with the property it checks. So the probe is
    defined here instead. This file lives under `tests/`, which is not searched,
    so a string that appears only in it must come back with no readers.
    """
    assert "tests" not in SEARCHED
    # Appears nowhere but the next line, and this file is not one of the sources.
    probe = "a_limit_key_spelled_only_inside_this_test"
    assert readers(probe, sources()) == []
    # The same probe is found once `tests/` is searched, which is what makes the
    # exclusion load-bearing rather than incidental.
    here = {Path(__file__): Path(__file__).read_text(encoding="utf-8")}
    assert readers(probe, here) == [str(Path(__file__))]
