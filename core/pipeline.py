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
from core.portfolio.center_book import NettingResult, net_orders
from core.risk.financing import check_financing
from core.risk.limits import Breach, check_pod, load_limits


@dataclass
class DayResult:
    trade_date: str
    orders_allowed: bool
    reasons: list[str] = field(default_factory=list)
    breaches: list[Breach] = field(default_factory=list)
    netting: NettingResult | None = None

    @property
    def liquidate_only(self) -> bool:
        return not self.orders_allowed


def run_day(
    trade_date: str,
    health: HealthReport,
    target_snapshot: dict[str, Any],
    open_orders: list[Order] | None = None,
    audit_path: Path | None = None,
    pod_targets: dict[str, dict[str, float]] | None = None,
    financing_snapshot: dict[str, Any] | None = None,
) -> DayResult:
    result = DayResult(trade_date=trade_date, orders_allowed=True)

    # Center book: net offsetting pod flows and trim crowded names before any
    # order exists, so the market never sees two pods cancelling each other out.
    if pod_targets is not None:
        limits = load_limits()
        result.netting = net_orders(pod_targets, single_name_max=limits["pod"]["single_name_max"])
        target_snapshot = {**target_snapshot, "weights": result.netting.net_targets}

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
    if financing_snapshot is not None:
        result.breaches += check_financing(financing_snapshot)
    if result.breaches:
        result.orders_allowed = False
        result.reasons.extend(str(b) for b in result.breaches)

    audit.append(
        "pipeline.run_day",
        {
            "trade_date": trade_date,
            "orders_allowed": result.orders_allowed,
            "reasons": result.reasons,
            "turnover_saved": (result.netting.turnover_saved if result.netting else None),
        },
        path=audit_path,
    )
    return result
