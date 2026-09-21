#!/usr/bin/env python3
"""Pre-trade gate — runs before any order file may be written.

Reads a target-position snapshot on stdin (JSON) and exits non-zero when the
deterministic limit engine reports a breach, when data health is not green, or
when an order from an earlier run is still unresolved. Fail-closed is the
default: a missing input is a block, not a pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.data.quality import HealthReport  # noqa: E402
from core.execution.orders import Order, State, blocking_orders  # noqa: E402
from core.risk.limits import check_pod  # noqa: E402


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"pretrade_gate: unreadable payload ({exc}) -> blocked", file=sys.stderr)
        return 2

    health = HealthReport(payload.get("health") or {})
    if not health.ok:
        print(
            "pretrade_gate: data health not green -> blocked "
            f"(failed={health.failed}, missing={health.missing})",
            file=sys.stderr,
        )
        return 2

    orders = [
        Order(
            trade_date=o["trade_date"],
            alpha_id=o["alpha_id"],
            symbol=o["symbol"],
            seq=o["seq"],
            quantity=o["quantity"],
            state=State(o.get("state", "intended")),
            filled=o.get("filled", 0),
        )
        for o in payload.get("open_orders") or []
    ]
    if stuck := blocking_orders(orders):
        print(
            f"pretrade_gate: unresolved orders -> blocked ({[o.client_id for o in stuck]})",
            file=sys.stderr,
        )
        return 2

    breaches = check_pod(payload.get("snapshot") or {})
    if breaches:
        for breach in breaches:
            print(f"pretrade_gate: {breach} -> blocked", file=sys.stderr)
        return 2

    print("pretrade_gate: clear")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
