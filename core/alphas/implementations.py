"""Which strategy function a declared alpha is judged by.

A declaration is a hypothesis; a `Strategy` in `core/strategies/library.py` is the
code that tests it. Something has to join the two, and it cannot be a field inside
the declaration: `core/backtest/prereg.py` accepts a declaration only while git has
it unmodified, so adding the mapping to an alpha that has already been submitted
would invalidate its own earlier verdicts (ADR-0040). The join therefore lives in
`registry/alphas/_implementations.yaml`, a committed file the run's git SHA pins
like any other input, and one `scripts/check_limits_change.py` controls -- changing
which code answers a hypothesis is a decision, not an edit.

**Why a missing entry is a refusal.** Submitting every declared alpha is how the
desk's declared search and its measured search are kept the same thing. An alpha
that is declared but silently not run makes the desk's trial count larger than
what anyone ever looked at: the deflation stays honest, but nobody finds out that
a hypothesis was never tested. So `for_all` names the gaps and the caller stops.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.backtest import prereg

DEFAULT_PATH = Path("registry/alphas/_implementations.yaml")


class ImplementationError(ValueError):
    """The map is there and does not say what runs what."""


def load(path: Path | None = None) -> dict[str, str]:
    """The alpha -> strategy map, as written."""
    target = path or DEFAULT_PATH
    if not target.is_file():
        raise ImplementationError(
            f"{target} is missing, so no declaration can be matched to the code that tests it"
        )
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ImplementationError(f"{target} is not a mapping")
    table = document.get("implementations")
    if not isinstance(table, dict) or not table:
        raise ImplementationError(f"{target} declares no `implementations` mapping")
    out: dict[str, str] = {}
    for alpha_id, strategy in table.items():
        if not isinstance(strategy, str) or not strategy.strip():
            raise ImplementationError(f"{target}: {alpha_id} names no strategy")
        out[str(alpha_id)] = strategy.strip()
    return out


def for_alpha(alpha_id: str, path: Path | None = None) -> str:
    """The strategy that implements `alpha_id`."""
    table = load(path)
    if alpha_id not in table:
        raise ImplementationError(
            f"{alpha_id} has no entry in {path or DEFAULT_PATH}, so nothing says which code "
            "tests its hypothesis"
        )
    return table[alpha_id]


def for_all(
    directory: Path | None = None,
    path: Path | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Every declared alpha with its strategy, and the reasons the map is incomplete.

    Both directions are checked. A declaration with no entry would go unrun; an
    entry with no declaration means the map names an alpha the desk's trial count
    does not include, which is the same drift seen from the other side.
    """
    alphas = directory or prereg.DEFAULT_DIRECTORY
    table = load(path)
    declared = prereg.declared_ids(alphas)
    problems = [
        f"{alpha_id} is declared in {alphas} and has no implementation, so it would never be run"
        for alpha_id in declared
        if alpha_id not in table
    ]
    problems += [
        f"{alpha_id} has an implementation and no declaration in {alphas}"
        for alpha_id in sorted(table)
        if alpha_id not in declared
    ]
    return {alpha_id: table[alpha_id] for alpha_id in declared if alpha_id in table}, problems
