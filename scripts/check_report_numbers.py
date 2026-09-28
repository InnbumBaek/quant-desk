"""CLAUDE.md 2항's reader: a report's numbers must come from artifacts, and the
references must point at real values.

Two checks, and the second one is new.

1. **A numeric claim carries a reference.** A number with a unit on a prose line
   needs `{{artifact:<run_id>/<metric>}}` on that line. This is what
   `.claude/hooks/numbers_from_artifacts.py` has done since P0.
2. **The reference resolves, and the digits match.** The run's committed
   artifacts must hold that metric, and a number written on the line must equal
   it at the precision it was written. A reference that points nowhere is a typed
   number with extra steps (ADR-0045). `|derived` marks a number computed from
   artifact values -- it is still resolved, but the digits are not compared.

Scope: decision records numbered `LEGACY_BEFORE` and above, plus any file named
on the command line. **The twelve earlier ADRs are out of scope on purpose.**
They were written before the rule had a reader, and retrofitting references into
them would mean editing decision records after the fact -- which is the one thing
this registry refuses to do, because a record edited later is a record nobody can
trust. The boundary is one number rather than a list of filenames, so it cannot
grow quietly by one name at a time; this file is CONTROLLED, so moving it needs
an ADR.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from core.report.artifacts import (
    DEFAULT_ROOT,
    UnresolvedReference,
    default_run,
    references,
    resolve,
    written_as,
)

#: ADR-0043 is the first decision record written after the reader existed.
LEGACY_BEFORE = 43
DECISIONS = Path("registry/decisions")
ADR_NUMBER = re.compile(r"ADR-(\d{4})")
#: A claim is a number with a unit or comparison that a reader would act on.
CLAIM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?\s*(?:%|bps|x|배|σ)(?![\w])")
#: Headings and quoted text only. The P0 hook also skipped table rows and bullet
#: items, which is precisely where a measured number lives, so the rule stopped
#: at the prose and let the data through (ADR-0045).
SKIP_PREFIX = ("#", ">")
#: Inline code is quoting the code, not citing a measurement: a document about
#: this convention has to be able to write `{{artifact:<run_id>/<metric>}}` down.
INLINE_CODE = re.compile(r"`[^`]*`")


def in_scope(directory: Path = DECISIONS, legacy_before: int = LEGACY_BEFORE) -> list[Path]:
    """Decision records the rule applies to, newest last."""
    out = []
    for path in sorted(directory.glob("ADR-*.md")):
        found = ADR_NUMBER.search(path.name)
        if found and int(found.group(1)) >= legacy_before:
            out.append(path)
    return out


def problems(path: Path, root: Path = DEFAULT_ROOT) -> list[str]:
    """Every way `path` fails the rule, as lines a reader can act on."""
    out: list[str] = []
    text = path.read_text(encoding="utf-8")
    run = default_run(text)
    fenced = False
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not stripped:
            continue
        prose = INLINE_CODE.sub("", stripped)
        cited = references(prose, run)
        if stripped.startswith(SKIP_PREFIX):
            pass  # tables and quoted spec text carry no prose claim
        elif CLAIM.search(prose) and not cited:
            out.append(f"{path}:{number}: numeric claim with no artifact reference: {stripped}")
        for reference in cited:
            try:
                value = resolve(reference, root)
            except UnresolvedReference as error:
                out.append(f"{path}:{number}: {reference.text} does not resolve: {error}")
                continue
            if reference.derived:
                continue
            if not written_as(value, line):
                out.append(
                    f"{path}:{number}: {reference.label} is {value!r}, "
                    f"which is not written on this line (mark it |derived if it was computed)"
                )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="reports to check; default is the in-scope ADRs")
    parser.add_argument("--decisions", default=str(DECISIONS))
    parser.add_argument("--registry", default=str(DEFAULT_ROOT))
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.paths] or in_scope(Path(args.decisions))
    if not paths:
        print("no reports in scope")
        return 0

    failed = False
    for path in paths:
        for line in problems(path, Path(args.registry)):
            print(line, file=sys.stderr)
            failed = True
    if failed:
        print("check_report_numbers: blocked", file=sys.stderr)
        return 2
    print(f"check_report_numbers: {len(paths)} report(s) clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
