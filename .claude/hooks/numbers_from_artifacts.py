#!/usr/bin/env python3
"""Numbers-from-artifacts check, as a hook.

An agent that types a number into a report has invented it. Every figure must
reference the artifact that produced it, and **the reference must resolve to that
value** -- a reference pointing nowhere is a typed number with extra steps.

This file is a thin front end. The rule lives in `scripts/check_report_numbers.py`
so the hook and CI cannot drift apart: two readers of one rule is two rules, and
the one that runs less often is the one that quietly stops agreeing (ADR-0045).

Usage: numbers_from_artifacts.py <report.md> [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_report_numbers import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
