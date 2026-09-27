"""The broker read adapter is not connected to the order path, and cannot be
connected by accident.

CLAUDE.md 1항 keeps judgement out of the order path and 5항 keeps orders behind
the center book. `core/data/kis.py` holds a credential that can place orders, so
"it is only used for reads" has to be a fact about the import graph rather than
an intention. These tests read the source and say so.

A test here failing does not mean a bug in the adapter. It means somebody wired
a broker credential into the path that sends orders, which is a decision for the
owner and the `cio` gate, not a refactor.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.config import ROOT
from core.data.kis import ORDER_TR

#: The modules CLAUDE.md 1항 names, plus the center book 5항 names.
ORDER_PATH = (
    "core/risk/limits.py",
    "core/execution/orders.py",
    "core/pipeline.py",
    "core/portfolio/center_book.py",
    ".claude/hooks/pretrade_gate.py",
)

#: Where the broker adapter is allowed to be named at all.
BROKER_MODULES = {"core/data/kis.py", "scripts/fetch_kis.py"}


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("relative", ORDER_PATH)
def test_the_order_path_does_not_import_the_broker_adapter(relative: str):
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} does not exist yet")
    offending = {name for name in imports_of(path) if "kis" in name.lower()}
    assert offending == set(), (
        f"{relative} imports {offending}. A broker credential in the order path is a cio decision "
        "(CLAUDE.md 1항·5항, ADR-0025), not a refactor"
    )


def test_nothing_outside_the_adapter_names_the_broker_module():
    """One reader and one fetcher. Anything else is a second, ungated door."""
    offenders: list[str] = []
    for path in sorted((ROOT / "core").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in BROKER_MODULES:
            continue
        if any("kis" in name.lower() for name in imports_of(path)):
            offenders.append(relative)
    assert offenders == [], f"these import the broker adapter: {', '.join(offenders)}"


def test_no_order_tr_id_appears_anywhere_it_could_be_sent():
    """The order `tr_id` values exist in exactly one place: the list that refuses them."""
    allowed = {"core/data/kis.py", "tests/data/test_kis.py", "registry/decisions"}
    found: dict[str, list[str]] = {}
    for path in sorted(ROOT.rglob("*.py")) + sorted((ROOT / "registry").rglob("*.md")):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(tuple(allowed)) or "/.venv/" in relative or relative.startswith(".venv"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [tr_id for tr_id in ORDER_TR if tr_id in text]
        if hits:
            found[relative] = hits
    assert found == {}, f"order tr_id values outside the refusal list: {found}"
