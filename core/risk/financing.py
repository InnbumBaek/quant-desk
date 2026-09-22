"""Financing and margin limits.

Leverage is not a setting, it is something a prime broker extends. Treat margin
utilisation, broker concentration and financing cost as first-class limits: a
book that is fine on exposure can still be one margin call away from a forced
unwind.
"""

from __future__ import annotations

from typing import Any

from core.risk.limits import Breach, load_limits


def check_financing(snapshot: dict[str, Any], limits: dict[str, Any] | None = None) -> list[Breach]:
    """snapshot keys: margin_utilization, pb_shares {broker: share},
    financing_cost_bps, cash_buffer."""
    lim = (limits or load_limits())["financing"]
    out: list[Breach] = []

    util = snapshot.get("margin_utilization")
    if util is None or util > lim["margin_utilization_max"]:
        out.append(Breach("MARGIN_UTIL", f"{util} vs max {lim['margin_utilization_max']}"))

    for broker, share in (snapshot.get("pb_shares") or {}).items():
        if share > lim["pb_concentration_max"]:
            out.append(Breach("PB_CONCENTRATION", f"{broker} {share:.0%}"))

    cost = snapshot.get("financing_cost_bps")
    if cost is not None and cost > lim["financing_cost_bps_max"]:
        out.append(Breach("FINANCING_COST", f"{cost:.0f}bps"))

    buffer = snapshot.get("cash_buffer")
    if buffer is None or buffer < lim["overnight_cash_buffer_min"]:
        out.append(Breach("CASH_BUFFER", f"{buffer} vs min {lim['overnight_cash_buffer_min']}"))

    return out
