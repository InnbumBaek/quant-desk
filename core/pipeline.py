"""The daily pipeline, wired end to end and failing closed.

P0 runs every stage as a no-op so the skeleton is exercised by CI. What is
already real here is the control flow: a failed data-health check or an
unresolved order stops new orders, and every stage decision is audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import audit
from core.data.quality import HealthReport
from core.execution.orders import Order, blocking_orders
from core.risk.limits import Breach, check_pod


@dataclass
class DayResult:
    trade_date: str
    orders_allowed: bool
    reasons: list[str] = field(default_factory=list)
    breaches: list[Breach] = field(default_factory=list)

    @property
    def liquidate_only(self) -> bool:
        return not self.orders_allowed


def run_day(
    trade_date: str,
    health: HealthReport,
    target_snapshot: dict[str, Any],
    open_orders: list[Order] | None = None,
    audit_path: Path | None = None,
) -> DayResult:
    result = DayResult(trade_date=trade_date, orders_allowed=True)

    if not health.ok:
        result.orders_allowed = False
        result.reasons.append(
            "data_health: " + ", ".join(health.failed + [f"missing:{m}" for m in health.missing])
        )

    stuck = blocking_orders(open_orders or [])
    if stuck:
        result.orders_allowed = False
        result.reasons.append(f"unresolved_orders: {[o.client_id for o in stuck]}")

    result.breaches = check_pod(target_snapshot)
    if result.breaches:
        result.orders_allowed = False
        result.reasons.extend(str(b) for b in result.breaches)

    audit.append(
        "pipeline.run_day",
        {
            "trade_date": trade_date,
            "orders_allowed": result.orders_allowed,
            "reasons": result.reasons,
        },
        path=audit_path,
    )
    return result
