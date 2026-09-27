"""The daily pipeline, wired end to end and failing closed.

P0 runs every stage as a no-op so the skeleton is exercised by CI. What is
already real here is the control flow: a failed data-health check or an
unresolved order stops new orders, and every stage decision is audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from core import audit
from core.data.quality import HealthReport
from core.execution.orders import Order, blocking_orders
from core.portfolio.center_book import NettingResult, net_orders
from core.risk.capacity import check_capacity
from core.risk.financing import check_financing
from core.risk.fund import check_fund, required_actions
from core.risk.limits import Breach, check_pod, load_limits


@dataclass
class DayResult:
    trade_date: str
    orders_allowed: bool
    reasons: list[str] = field(default_factory=list)
    breaches: list[Breach] = field(default_factory=list)
    netting: NettingResult | None = None
    #: What the fund drawdown tiers call for, e.g. ("cut_gross_half",).
    actions: tuple[str, ...] = ()
    #: The factor the fund tier applies to every target. Reduce-only by
    #: construction: it is 1.0 or less, never more (CLAUDE.md 5).
    gross_multiplier: float = 1.0

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
    fund_snapshot: dict[str, Any] | None = None,
    capacity_snapshot: dict[str, Any] | None = None,
) -> DayResult:
    result = DayResult(trade_date=trade_date, orders_allowed=True)

    # Center book: net offsetting pod flows and trim crowded names before any
    # order exists, so the market never sees two pods cancelling each other out.
    # The fund tiers are read before the book is built, because "cut gross in
    # half" has to reach the targets rather than be reported next to them.
    #
    # A missing snapshot is a breach, not a skipped check. An optional fund halt
    # is not a fund halt: the day nobody passes the snapshot is exactly the day
    # it would have mattered (ADR-0016).
    if fund_snapshot is None:
        result.breaches.append(
            Breach("FUND_UNMEASURED", "no fund snapshot was supplied; the fund halt cannot be checked")
        )
    else:
        fund_breaches = check_fund(fund_snapshot)
        result.breaches += fund_breaches
        result.actions = required_actions(fund_breaches)
        if "cut_gross_half" in result.actions:
            result.gross_multiplier = 0.5

    if pod_targets is not None:
        limits = load_limits()
        result.netting = net_orders(pod_targets, single_name_max=limits["pod"]["single_name_max"])
        if result.gross_multiplier != 1.0:
            result.netting = replace(
                result.netting,
                net_targets={
                    symbol: weight * result.gross_multiplier
                    for symbol, weight in result.netting.net_targets.items()
                },
            )
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

    result.breaches += check_pod(target_snapshot)
    if financing_snapshot is None:
        result.breaches.append(
            Breach(
                "FINANCING_UNMEASURED",
                "no financing snapshot was supplied; margin and cash cannot be checked",
            )
        )
    else:
        result.breaches += check_financing(financing_snapshot)

    # Capacity is the layer that had a limit and no reader until ADR-0026. An
    # absent snapshot blocks for the same reason the fund one does: the day
    # nobody measures capacity is the day a pod is already over it.
    if capacity_snapshot is None:
        result.breaches.append(
            Breach(
                "CAPACITY_UNMEASURED",
                "no capacity snapshot was supplied; the 80% cap cannot be checked",
            )
        )
    else:
        result.breaches += check_capacity(capacity_snapshot)

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
            "actions": list(result.actions),
            "gross_multiplier": result.gross_multiplier,
        },
        path=audit_path,
    )
    return result
