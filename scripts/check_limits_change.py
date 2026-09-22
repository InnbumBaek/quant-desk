#!/usr/bin/env python3
"""Require an ADR whenever a controlled file changes.

The plan says limit and gate thresholds may only move with human approval and a
record. A rule with no mechanism is a wish, so CI enforces the record half: a
commit range that touches `core/risk/limits.yaml` must also add or change a file
under `registry/decisions/`.

The approval half stays with the repository owner (branch protection / review);
this check only makes an unrecorded change impossible to merge quietly.

Usage: check_limits_change.py <base-ref> [head-ref]
"""

from __future__ import annotations

import subprocess
import sys

CONTROLLED = ("core/risk/limits.yaml",)
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
