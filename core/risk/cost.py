"""Pass-through economics: a pod carries its own cost, and an uncharged cost blocks.

`limits.yaml` has carried a `cost_attribution:` block since the first commit --
`charge_pods: true` and `min_net_of_cost_ir: 0.0` -- and until this module
**nothing read it.** Those were the last two keys on the unread list that
`tests/limits/test_every_limit_has_a_reader.py` counts; the other three are on
the allocator branch (ADR-0010, PR #1).

**The excuse had drifted, and that is worth saying plainly.** ADR-0026 parked
these keys behind the allocator, ADR-0032 parked them behind realised fills
(`core/execution/tca.py`), and the table's own comment says something else
again: the cost a pod carries is *the data and compute it consumes*. Only the
table is normative -- it is the file the owner approved -- and read that way,
neither excuse holds. Data and compute cost needs no fills and no allocator; it
needs a measurement of what each pod consumed, which is what did not exist
(ADR-0038).

**Why zero cost is refused rather than used.** With nothing charged, the
net-of-cost IR equals the gross IR and a floor of 0.0 is cleared by anything
that made money at all. That is the same shape as a participation of zero
scaling to infinite capacity (ADR-0026) and a flat book reporting 0.0 exposure
(ADR-0031): the most forgiving number available, arrived at by not measuring.
So a pod whose cost is not a measurement is a breach, not a pod that costs
nothing.

**Why the drag is computed here rather than supplied.** A caller that hands over
a finished `net_of_cost_ir` can hand over a flattering one, and nothing in the
number shows whether a cost was ever subtracted. So this module takes the plain
measurements -- gross IR, annual cost, allocated capital, volatility -- and does
the subtraction itself:

    drag = (annual_cost / allocated_capital) / volatility
    net  = gross_ir - drag

The cost as a share of the capital carrying it is an annual return drag, and
dividing by volatility puts it in the same units as the IR it is subtracted
from.

**`charge_pods: false` is configuration, not a bypass, and that is a real
difference from `capacity.respect_hard_cap`.** CLAUDE.md 7항 makes the capacity
cap absolute, so switching it off is itself a breach. No rule makes cost
pass-through absolute: the owner may decide the fund carries its own overhead.
The flag lives in `limits.yaml`, which is owner-approved and CI-controlled, so
turning it off is visible and recorded -- which is the protection that matters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.risk.limits import Breach, as_measurement, load_limits


def cost_ir_drag(annual_cost: Any, allocated_capital: Any, volatility: Any) -> float | None:
    """The IR a pod gives up to its own cost, or None when that cannot be said.

    None rather than 0.0 for every unusable input. Zero is the answer that lets
    the pod through, so it may never be the answer to "we could not tell".
    """
    cost = as_measurement(annual_cost)
    capital = as_measurement(allocated_capital)
    vol = as_measurement(volatility)
    if cost is None or cost < 0.0:
        return None
    if capital is None or capital <= 0.0:
        return None
    if vol is None or vol <= 0.0:
        return None
    return (cost / capital) / vol


def net_of_cost_ir(gross_ir: Any, annual_cost: Any, allocated_capital: Any, volatility: Any) -> float | None:
    """`gross_ir` less the drag its own cost imposes, or None when unmeasurable."""
    gross = as_measurement(gross_ir)
    drag = cost_ir_drag(annual_cost, allocated_capital, volatility)
    if gross is None or drag is None:
        return None
    return gross - drag


def ir_floor(limits: dict[str, Any] | None = None) -> float | None:
    """`min_net_of_cost_ir`, or None when the table's value is not a number."""
    block = (limits or load_limits()).get("cost_attribution") or {}
    return as_measurement(block.get("min_net_of_cost_ir"))


def charging_pods(limits: dict[str, Any] | None = None) -> bool | None:
    """`charge_pods`, or None when it is not a boolean.

    None rather than falsy, because "the flag is missing or malformed" is not
    the owner having decided not to charge. One is an unset switch and the other
    is a decision, and only the second one is an answer.
    """
    block = (limits or load_limits()).get("cost_attribution") or {}
    flag = block.get("charge_pods")
    return flag if isinstance(flag, bool) else None


def check_cost_attribution(
    snapshot: Mapping[str, Any] | None, limits: dict[str, Any] | None = None
) -> list[Breach]:
    """Every pod's IR net of the cost it consumed, against the table's floor.

    snapshot: {pod: {gross_ir, annual_cost, allocated_capital, volatility,
    cost_measured}}. `cost_measured` carries the same meaning as
    `Submission.adv_measured`: absent means the number is a default, and a
    default cost of zero is the one value that clears every floor there is.
    """
    lim = limits or load_limits()
    out: list[Breach] = []

    charging = charging_pods(lim)
    if charging is None:
        block = lim.get("cost_attribution") or {}
        return [
            Breach(
                "COST_ATTRIBUTION_UNUSABLE",
                f"cost_attribution.charge_pods is {block.get('charge_pods')!r}, which is not a "
                "boolean; whether a pod is charged is not a question this can be left open on",
            )
        ]
    if charging is False:
        # A recorded decision not to charge. Nothing to floor, and nothing to
        # report: the decision itself is visible in the owner-approved table.
        return []

    floor = ir_floor(lim)
    if floor is None:
        block = lim.get("cost_attribution") or {}
        out.append(
            Breach(
                "COST_ATTRIBUTION_UNUSABLE",
                f"cost_attribution.min_net_of_cost_ir is {block.get('min_net_of_cost_ir')!r}, "
                "which is not a number; no floor can be applied",
            )
        )
    if not snapshot:
        out.append(
            Breach(
                "COST_UNMEASURED",
                "the cost snapshot names no pods, so no pod was charged; charge_pods is true "
                "and an uncharged pod clears the net-of-cost floor for free",
            )
        )
        return out
    if floor is None:
        return out

    for pod in sorted(snapshot):
        detail = snapshot[pod]
        if not isinstance(detail, Mapping):
            out.append(Breach("COST_UNMEASURED", f"{pod}: the snapshot entry is not a mapping"))
            continue
        out.extend(_pod_breaches(pod, detail, floor))
    return out


def _pod_breaches(pod: str, detail: Mapping[str, Any], floor: float) -> list[Breach]:
    if detail.get("cost_measured") is not True:
        return [
            Breach(
                "COST_UNMEASURED",
                f"{pod}: cost_measured is {detail.get('cost_measured')!r}, so the cost "
                f"{detail.get('annual_cost')!r} is not a measurement and the net-of-cost IR "
                "would be the gross one",
            )
        ]
    net = net_of_cost_ir(
        detail.get("gross_ir"),
        detail.get("annual_cost"),
        detail.get("allocated_capital"),
        detail.get("volatility"),
    )
    if net is None:
        return [
            Breach(
                "COST_UNMEASURED",
                f"{pod}: the net-of-cost IR cannot be computed from gross IR "
                f"{detail.get('gross_ir')!r}, cost {detail.get('annual_cost')!r}, capital "
                f"{detail.get('allocated_capital')!r} and volatility {detail.get('volatility')!r}",
            )
        ]
    if net < floor:
        return [
            Breach(
                "COST_ATTRIBUTION",
                f"{pod}: net-of-cost IR {net:.3f} < {floor:.3f}; the pod does not cover the "
                "data and compute it consumes",
            )
        ]
    return []
