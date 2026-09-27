"""Write `registry/alphas/STATUS.md`: where every declared alpha stands, and why.

The derivation is `core/alphas/lifecycle.py`; this is the reader that makes it an
artifact. Without a reader the derivation would be another thing the repository
knows and nobody can see, which is the failure ADR-0032 and ADR-0035 both closed.

The file is generated. Editing it changes nothing -- the next run overwrites it --
which is the point: the status is a function of the records, so the only way to
change it is to change a record.

Exit codes. 0 when the table was written. 1 when something is *set up* wrongly
rather than merely unfinished: a ledger that will not parse, or an alpha whose
declaration no longer loads. An alpha with no submissions is not an error; it is
the ordinary state of a newly declared alpha and the table says so.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from core.alphas import lifecycle

DEFAULT_OUT = lifecycle.ALPHAS / "STATUS.md"

HEADER = """<!-- 생성 파일. `uv run python -m scripts.alpha_status`가 덮어쓴다. 직접 고치지 말 것. -->
# 알파 현황

상태는 **선언이 아니라 기록에서 파생된다.** 선언 파일(`registry/alphas/<id>.yaml`)의
`status:`는 선언 시점의 값이고 갱신되지 않는다 — 고치면 G1이 기각한다(ADR-0036).

**측정은 올리고, 원장은 내린다.** `registry/submissions/`의 판정이 `proposed`를
`gated`까지 올릴 수 있고, 그 위는 G7(페이퍼 기록)이 올린다.
`registry/alphas/lifecycle.yaml`에 적은 사람의 결정은 `stopped`·`retired`로
**내리기만** 한다(ADR-0037).

**`live`는 여기 나오지 않는다.** 실전 자본은 G8이고 사람의 승인이다.
`gates.live_blockers`가 절대 빈 목록을 돌려주지 않으므로, 이 표의 어떤 값도
자본을 받아도 된다는 뜻이 아니다. 주문·배분 경로는 이 표를 읽지 않는다.
"""


def markdown(states: Sequence[lifecycle.Lifecycle]) -> str:
    lines = [HEADER, "", "| 알파 | 현재 | 선언 시점 | 근거 | 제출 |", "| --- | --- | --- | --- | --- |"]
    for state in states:
        why = state.why.replace("|", "/")
        lines.append(
            f"| `{state.alpha_id}` | **{state.status}** | {state.declared_status or '-'} "
            f"| {why} | {len(state.submissions)} |"
        )
    if not states:
        lines.append("| -- | -- | -- | 선언된 알파가 없다 | 0 |")

    for state in states:
        refused = [entry for entry in state.ledger if entry.refused_because]
        if not (refused or state.notes):
            continue
        lines += ["", f"### `{state.alpha_id}`", ""]
        lines += [f"- 원장 항목 거부: `{entry.to}` -- {entry.refused_because}" for entry in refused]
        lines += [f"- {note}" for note in state.notes]

    lines += [
        "",
        "---",
        "",
        "어느 줄도 자본 허가가 아니다. 허가는 `gates.live_blockers`가 비는 것이고, 비지 않는다.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default=str(lifecycle.ALPHAS))
    parser.add_argument("--submissions", default=str(lifecycle.SUBMISSIONS))
    parser.add_argument("--ledger", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--json", action="store_true", help="print the derivation instead of the table")
    args = parser.parse_args(argv)

    alphas = Path(args.alphas)
    ledger = Path(args.ledger) if args.ledger else alphas / lifecycle.LEDGER.name
    try:
        states = lifecycle.derive_all(alphas, Path(args.submissions), ledger)
    except lifecycle.LedgerError as error:
        # A ledger that will not parse is not an empty ledger: refusing here
        # keeps a malformed file from reading as "nobody has decided anything".
        print(f"the ledger is unreadable, so no status can be derived: {error}")
        return 1

    if args.json:
        print(json.dumps([state.as_dict() for state in states], indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    table = markdown(states)
    out = Path(args.out) if args.out else alphas / DEFAULT_OUT.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table, encoding="utf-8")
    print(table)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
