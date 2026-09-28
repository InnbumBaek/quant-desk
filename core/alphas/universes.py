"""Which instruments a declared alpha is judged on.

A declaration's `universe:` prose says *why* those instruments, and a reviewer
reads it. Nothing read the *which*. G1 counts the declared trials and
`submit_alpha.grid_within_declaration` checks the declared parameter values, so
both halves of the search were pinned -- and the panel the search ran on was
whatever the day's fetch happened to hold. Widen the fetch and every old
declaration re-runs on a universe nobody pre-registered, with the result
improving and no gate noticing. That is the same loophole as counting trials
after the search (ADR-0041).

So a submission is restricted to a declared list of symbols, and the list comes
from one of two places, in this order:

1. `hypothesis.universe_symbols` in the declaration, which `prereg` accepts only
   while git has the file unmodified. This is where it belongs.
2. `legacy_universes` in `registry/alphas/_implementations.yaml`, for the six
   alphas declared before the field existed. They cannot gain it -- editing a
   declaration invalidates its own earlier verdicts -- and the prose is not
   parsed, so the five ETFs they were actually judged on are written down in the
   wiring file instead. That file is committed, pinned by the run's git SHA, and
   controlled by `scripts/check_limits_change.py`.

**Neither is a refusal.** An alpha with no declared universe cannot be submitted:
the alternative is judging a hypothesis on a panel chosen after the fact, which
is the thing this module exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.alphas.implementations import DEFAULT_PATH, ImplementationError
from core.backtest import prereg

#: The alphas whose universe lives in the wiring file rather than in their own
#: declaration. Closed by hand and counted by `tests/test_alpha_universes.py`: an
#: exemption list that can grow silently is how an unenforced rule hides.
LEGACY = (
    "bab-001",
    "blend-001",
    "breakout-001",
    "strev-001",
    "tsmom-001",
    "xsmom-001",
)


class UniverseError(ValueError):
    """The alpha's universe cannot be established, so it cannot be judged."""


def legacy_universes(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """The grandfathered universes, as the wiring file writes them."""
    target = path or DEFAULT_PATH
    if not target.is_file():
        raise ImplementationError(f"{target} is missing, so no legacy universe can be read")
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ImplementationError(f"{target} is not a mapping")
    table = document.get("legacy_universes") or {}
    if not isinstance(table, dict):
        raise ImplementationError(f"{target} has a `legacy_universes` that is not a mapping")
    out: dict[str, tuple[str, ...]] = {}
    for alpha_id, symbols in table.items():
        if not isinstance(symbols, list | tuple) or not symbols:
            raise ImplementationError(f"{target}: {alpha_id} names no universe")
        names = tuple(str(symbol).strip() for symbol in symbols)
        if any(not symbol for symbol in names) or len(set(names)) != len(names):
            raise ImplementationError(f"{target}: {alpha_id} names an empty or repeated symbol")
        out[str(alpha_id)] = names
    return out


def for_alpha(
    alpha_id: str,
    directory: Path | None = None,
    path: Path | None = None,
) -> tuple[tuple[str, ...], str]:
    """The alpha's universe and where it was declared.

    Returns `(symbols, source)` with source `"declaration"` or `"legacy"`, so the
    submission record can say which -- a universe read out of the wiring file is a
    weaker claim than one read out of an unmodified declaration, and the record
    should not flatten the two.
    """
    alphas = directory or prereg.DEFAULT_DIRECTORY
    declared = prereg.declared_universe(alpha_id, directory=alphas)
    if declared is not None:
        return declared, "declaration"
    legacy = legacy_universes(path)
    if alpha_id in legacy:
        return legacy[alpha_id], "legacy"
    raise UniverseError(
        f"{alpha_id} declares no `universe_symbols` and has no entry in the wiring file's "
        "`legacy_universes`, so the panel it would be judged on is whatever the fetch held "
        "rather than what anyone pre-registered (ADR-0041)"
    )
