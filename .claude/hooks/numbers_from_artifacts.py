#!/usr/bin/env python3
"""Numbers-from-artifacts check.

An agent that types a number into a report has invented it. Every figure must
reference the artifact that produced it, as {{artifact:<run_id>/<metric>}}.
This hook scans a report file and fails when a numeric claim carries no such
reference on its line.

Usage: numbers_from_artifacts.py <report.md> [...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ARTIFACT = re.compile(r"\{\{artifact:[^}]+\}\}")
# A claim is a number with a unit or comparison that a reader would act on.
CLAIM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?\s*(?:%|bps|x|배|σ)(?![\w])")
SKIP_PREFIX = ("|", ">", "#", "-", "*")  # tables and quoted spec text


def offending_lines(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for n, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(SKIP_PREFIX):
            continue
        if CLAIM.search(stripped) and not ARTIFACT.search(stripped):
            out.append((n, stripped))
    return out


def main(paths: list[str]) -> int:
    failed = False
    for path in paths:
        bad = offending_lines(Path(path).read_text(encoding="utf-8"))
        for n, line in bad:
            print(f"{path}:{n}: numeric claim without artifact reference: {line}", file=sys.stderr)
            failed = True
    if failed:
        print("numbers_from_artifacts: blocked", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
