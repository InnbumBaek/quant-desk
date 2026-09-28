#!/usr/bin/env python3
"""Require an ADR whenever a controlled file changes.

The plan says limit and gate thresholds may only move with human approval and a
record. A rule with no mechanism is a wish, so CI enforces the record half: a
commit range that touches a controlled file must also add or change a file under
`registry/decisions/`.

**The classification tables are controlled for the same reason the thresholds
are.** `core/data/sic.py` and `core/data/ksic.py` decide which bucket a position
counts against, so moving a line in either changes what `sector_max` measures
without changing the number it measures against. That is the harder change to
notice of the two, and ADR-0029 moved five ranges in `sic.py` at once, which is
what made the gap obvious.

**`core/backtest/gates.py` is controlled because not every gate threshold is in
the table.** `g5_robustness` carries two of its own -- the 30% Sharpe decay and
the double-cost sign test, both from the alpha-gate skill -- and the module
docstring claimed otherwise until ADR-0032. CLAUDE.md already required an ADR for
a change to a gate criterion; this is the mechanism for the half of that rule
that lives in code rather than in YAML.

**`registry/alphas/_implementations.yaml` is controlled because it decides which
code answers which hypothesis.** A declaration is judged by the strategy this file
names, so moving one line re-points a committed hypothesis at different code while
every gate still passes (ADR-0040). The declaration itself cannot carry the mapping:
`prereg` requires it unmodified, so editing it would invalidate its own earlier
verdicts.

**`registry/alphas/lifecycle.yaml` is controlled because it is the one file
where a person, rather than a measurement, changes an alpha's standing.** It can
only lower one (ADR-0037), so a bad entry cannot promote anything -- but stopping
a running alpha is still a decision, and a decision with no record did not
happen.

The approval half stays with the repository owner (branch protection / review);
this check only makes an unrecorded change impossible to merge quietly.

Usage: check_limits_change.py <base-ref> [head-ref]
"""

from __future__ import annotations

import subprocess
import sys

CONTROLLED = (
    "core/risk/limits.yaml",
    "core/data/sic.py",
    "core/data/ksic.py",
    "core/backtest/gates.py",
    "registry/alphas/lifecycle.yaml",
    "registry/alphas/_implementations.yaml",
)
RECORD_PREFIX = "registry/decisions/"


def violation(changed_files: list[str]) -> str | None:
    """Return an error message when a controlled file moved without a record."""
    touched = [f for f in changed_files if f in CONTROLLED]
    if not touched:
        return None
    if any(f.startswith(RECORD_PREFIX) and f.endswith(".md") for f in changed_files):
        return None
    return (
        f"{', '.join(touched)} changed with no ADR in {RECORD_PREFIX}. "
        "Record why the threshold moved, then commit both together."
    )


def changed_files(base: str, head: str = "HEAD") -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check_limits_change.py <base-ref> [head-ref]", file=sys.stderr)
        return 64
    files = changed_files(argv[0], argv[1] if len(argv) > 1 else "HEAD")
    if message := violation(files):
        print(f"check_limits_change: {message}", file=sys.stderr)
        return 2
    print("check_limits_change: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
