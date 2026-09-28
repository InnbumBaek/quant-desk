"""Read a pre-registration out of `registry/alphas/` and say whether it counts.

`core/backtest/gates.g1_preregistration` judges a `Preregistration`; this builds
one from a file. The split is deliberate: the gate stays a pure function of its
input so it can be tested without a repository, and every way a declaration can
fail to be a declaration -- missing, unparseable, still the template, uncommitted
-- is decided here and handed over as a reason rather than an exception.

**Why `committed` is part of it.** The gate's whole job is to establish that N was
fixed *before* the search. A file sitting modified in the working tree carries no
such claim: it could have been written after the result, and nothing in its
contents would show that. So a declaration counts only when git has it and the
working copy matches. That is not a proof of ordering -- a commit can be made at
any time -- but it puts the declaration in the history where a reviewer can see
when it arrived, which is the part that was missing entirely.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from core.backtest.gates import Preregistration

DEFAULT_DIRECTORY = Path("registry/alphas")

#: The template is a form, not a declaration. Loading it as one would give every
#: alpha the same empty hypothesis and a grid of nothing.
TEMPLATE = "_template.yaml"

#: The lifecycle ledger shares this directory and is not an alpha (ADR-0037).
LEDGER_FILE = "lifecycle.yaml"


class PreregistrationError(ValueError):
    """The file is there and is not a pre-registration."""


def _tracked_and_clean(path: Path, repo: Path | None = None) -> bool:
    """True when git has this file and the working copy matches it."""
    root = repo or Path.cwd()
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if listed.returncode != 0:
            return False
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if changed.returncode != 0:  # pragma: no cover - git present but refusing
            return False
        return not changed.stdout.strip()
    except OSError:  # pragma: no cover - no git binary at all
        return False


def declared_ids(directory: Path = DEFAULT_DIRECTORY) -> list[str]:
    """Every alpha with a declaration in `directory`, sorted.

    The template is excluded because it is a form, and so is any file starting
    with an underscore, which is what marks one. `lifecycle.yaml` lives in the
    same directory and is a ledger of decisions, not an alpha.
    """
    if not directory.is_dir():
        return []
    return sorted(
        path.stem
        for path in directory.glob("*.yaml")
        if not path.name.startswith("_") and path.name != LEDGER_FILE
    )


def load(
    alpha_id: str,
    directory: Path = DEFAULT_DIRECTORY,
    repo: Path | None = None,
) -> Preregistration | None:
    """The declaration for `alpha_id`, or None when there is not one.

    None rather than a raise, because "no declaration" is the ordinary case the
    gate is built to refuse and a refusal belongs in a verdict, not a traceback.
    A file that exists but is not a declaration raises: that is a mistake someone
    made, and it must not be read as absence.
    """
    if not alpha_id or alpha_id == TEMPLATE.removesuffix(".yaml"):
        return None
    path = directory / f"{alpha_id}.yaml"
    if path.name == TEMPLATE or not path.is_file():
        return None

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise PreregistrationError(f"{path} does not parse as YAML: {error}") from error
    if not isinstance(document, dict):
        raise PreregistrationError(f"{path} is not a mapping, so it declares nothing")
    if str(document.get("id", "")) != alpha_id:
        raise PreregistrationError(
            f"{path} declares id {document.get('id')!r}, not {alpha_id!r}; a declaration that "
            "names another alpha is not this alpha's declaration"
        )

    hypothesis: Any = document.get("hypothesis") or {}
    if not isinstance(hypothesis, dict):
        raise PreregistrationError(f"{path} has a hypothesis that is not a mapping")
    grid: Any = hypothesis.get("parameters_declared") or {}
    if not isinstance(grid, dict):
        raise PreregistrationError(
            f"{path} declares parameters that are not a mapping of name to values, "
            "so there is no grid to count"
        )

    return Preregistration(
        alpha_id=alpha_id,
        economic_rationale=str(hypothesis.get("economic_rationale") or ""),
        universe=str(hypothesis.get("universe") or ""),
        horizon=str(hypothesis.get("horizon") or ""),
        parameters_declared=grid,
        committed=_tracked_and_clean(path, repo),
        path=str(path),
    )


def declared_universe(alpha_id: str, directory: Path = DEFAULT_DIRECTORY) -> tuple[str, ...] | None:
    """The symbols the declaration names as its universe, or None when it names none.

    Separate from `Preregistration` for the same reason `declared_chosen` is: G1
    does not judge it. The `universe:` prose is required and says *why* those
    instruments, which is what a reviewer reads; this is the machine-readable
    list, which is what the run is restricted to. Prose is never parsed -- a
    universe recovered from a sentence is a universe nobody declared.

    None rather than a raise for a declaration that has no list, because the six
    alphas declared before this field existed cannot gain it: `prereg` accepts a
    declaration only while git has it unmodified, so editing them would
    invalidate their own earlier verdicts. `core/alphas/universes.py` decides what
    to do about that; this function only reports what the file says (ADR-0041).
    """
    path = directory / f"{alpha_id}.yaml"
    if path.name == TEMPLATE or not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PreregistrationError(f"{path} is not a mapping, so it declares nothing")
    hypothesis = document.get("hypothesis") or {}
    if not isinstance(hypothesis, dict):
        raise PreregistrationError(f"{path} has a hypothesis that is not a mapping")
    declared = hypothesis.get("universe_symbols")
    if declared is None:
        return None
    if not isinstance(declared, list | tuple) or not declared:
        raise PreregistrationError(
            f"{path} declares `universe_symbols` that is not a non-empty list of symbols"
        )
    symbols = tuple(str(symbol).strip() for symbol in declared)
    if any(not symbol for symbol in symbols):
        raise PreregistrationError(f"{path} declares an empty symbol in `universe_symbols`")
    if len(set(symbols)) != len(symbols):
        raise PreregistrationError(
            f"{path} declares a repeated symbol in `universe_symbols`; a universe is a set"
        )
    return symbols


def declared_chosen(alpha_id: str, directory: Path = DEFAULT_DIRECTORY) -> dict[str, float] | None:
    """The configuration the declaration names as the one to report, or None.

    Separate from `Preregistration` because G1 does not judge it: searching the
    declared grid and reporting the declared point are different promises, and
    only the first is arithmetic the gate can check. It is read here so that the
    submitted configuration comes out of a committed file rather than out of the
    results, which is the whole reason the grid is declared at all. A caller that
    cannot find it must refuse to submit rather than fall back to a default --
    a default can be edited after the run, and then the choice is the result's.
    """
    path = directory / f"{alpha_id}.yaml"
    if path.name == TEMPLATE or not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PreregistrationError(f"{path} is not a mapping, so it declares nothing")
    hypothesis = document.get("hypothesis") or {}
    if not isinstance(hypothesis, dict):
        raise PreregistrationError(f"{path} has a hypothesis that is not a mapping")
    chosen = hypothesis.get("chosen_declared")
    if chosen is None:
        return None
    if not isinstance(chosen, dict) or not chosen:
        raise PreregistrationError(
            f"{path} declares a chosen configuration that is not a non-empty mapping of parameter to value"
        )
    return {str(key): float(value) for key, value in chosen.items()}
