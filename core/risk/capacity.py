"""Capacity as a hard cap, and a capacity nobody measured as a block.

`limits.yaml` has carried a `capacity:` block since the first commit --
`respect_hard_cap: true` and `capacity_utilisation_max: 0.80` -- and until this
module **nothing read it.** CLAUDE.md 7항 says capital above 80% of the estimated
capacity is not allocated, however good the performance is, and that rule had no
enforcer: G6 checks ADV participation and book correlation against the `gates:`
block, which is a different question (can this strategy trade at all) from this
one (how much capital may it hold).

That is the third time this desk has found a limit with no reader: pod limits
passing on absent measurements (ADR-0015), and the whole `fund:` block going
unread (ADR-0016). **A limit written down is not a limit.**

**Why a participation of zero is refused rather than scaled.** Capacity here is
the capital at which the strategy's ADV participation would reach the gate's
cap: `tested_capital x participation_cap / observed_participation`. Divide by a
zero participation and capacity is infinite -- the most forgiving number
available, arrived at by not measuring. `core/backtest/engine.py` returns 0.0
participation when the run had no dollar-volume panel and says so in a note, but
a note is not a gate; so a participation that is not a measurement makes this a
breach, and G6 now fails on it too (ADR-0026).

**Why turning the cap off is itself a breach.** `respect_hard_cap: false` would
read as permission to allocate past the estimate. It cannot be: CLAUDE.md 7항 is
absolute and `limits.yaml` may only be changed with the owner's approval. The
flag exists so that switching it off is *visible*, and this module treats a
missing or false flag as blocking rather than as configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.risk.limits import Breach, as_measurement, load_limits


def estimate_capacity(tested_capital: Any, adv_participation: Any, participation_cap: float) -> float | None:
    """The capital at which participation would reach `participation_cap`.

    `None` when the inputs do not support the question. Returning a number here
    for an unmeasured run is the whole failure this module exists to prevent:
    every unmeasurable case is "infinite capacity" if you let the arithmetic run.
    """
    capital = as_measurement(tested_capital)
    observed = as_measurement(adv_participation)
    if capital is None or capital <= 0.0:
        return None
    if observed is None or observed <= 0.0:
        # Zero participation scales to unlimited capacity. A strategy that
        # really traded nothing has no capacity to allocate against either.
        return None
    return capital * participation_cap / observed


def max_allocatable(capacity: float, limits: dict[str, Any] | None = None) -> float:
    """The most that may be allocated against `capacity`. The allocator's ceiling."""
    utilisation = utilisation_cap(limits)
    if utilisation is None:
        raise ValueError("capacity_utilisation_max is not a usable share; check limits.yaml")
    return capacity * utilisation


def utilisation_cap(limits: dict[str, Any] | None = None) -> float | None:
    """`capacity_utilisation_max`, or None when the table's value cannot be used.

    A value above 1.0 would permit allocating past the estimate, which voids the
    rule the number exists to serve, so it is refused rather than clamped: a
    silent clamp would hide a typo in an owner-approved file.
    """
    block = (limits or load_limits()).get("capacity") or {}
    share = as_measurement(block.get("capacity_utilisation_max"))
    if share is None or not 0.0 < share <= 1.0:
        return None
    return share


def check_capacity(snapshot: Mapping[str, Any], limits: dict[str, Any] | None = None) -> list[Breach]:
    """Every pod's allocated capital against 80% of its estimated capacity.

    snapshot: {pod: {tested_capital, adv_participation, adv_measured,
    allocated_capital}}. An empty snapshot is not a clean book -- a fund with no
    pods allocates nothing, and a snapshot that lost its pods looks identical.
    """
    lim = limits or load_limits()
    block = lim.get("capacity") or {}
    cap = float(lim["gates"]["adv_participation_max"])
    out: list[Breach] = []

    if block.get("respect_hard_cap") is not True:
        out.append(
            Breach(
                "CAPACITY_CAP_DISABLED",
                f"capacity.respect_hard_cap is {block.get('respect_hard_cap')!r}; "
                "the hard cap is not optional (CLAUDE.md 7항) and a limits change needs approval",
            )
        )
    utilisation = utilisation_cap(lim)
    if utilisation is None:
        out.append(
            Breach(
                "CAPACITY_UTILISATION_UNUSABLE",
                f"capacity_utilisation_max is {block.get('capacity_utilisation_max')!r}, "
                "which is not a share in (0, 1]; no ceiling can be computed from it",
            )
        )
    if not snapshot:
        out.append(
            Breach("CAPACITY_UNMEASURED", "the capacity snapshot names no pods, so nothing was checked")
        )
        return out
    if utilisation is None:
        return out

    for pod in sorted(snapshot):
        detail = snapshot[pod]
        if not isinstance(detail, Mapping):
            out.append(Breach("CAPACITY_UNMEASURED", f"{pod}: the snapshot entry is not a mapping"))
            continue
        out.extend(_pod_breaches(pod, detail, cap, utilisation))
    return out


def _pod_breaches(pod: str, detail: Mapping[str, Any], cap: float, utilisation: float) -> list[Breach]:
    if detail.get("adv_measured") is not True:
        return [
            Breach(
                "CAPACITY_UNMEASURED",
                f"{pod}: adv_measured is {detail.get('adv_measured')!r}, so the participation "
                f"{detail.get('adv_participation')!r} is not a measurement and capacity is unknown",
            )
        ]
    capacity = estimate_capacity(detail.get("tested_capital"), detail.get("adv_participation"), cap)
    if capacity is None:
        return [
            Breach(
                "CAPACITY_UNMEASURED",
                f"{pod}: capacity cannot be estimated from capital "
                f"{detail.get('tested_capital')!r} and participation {detail.get('adv_participation')!r}",
            )
        ]
    allocated = as_measurement(detail.get("allocated_capital"))
    if allocated is None or allocated < 0.0:
        return [
            Breach(
                "CAPACITY_ALLOCATION_UNMEASURED",
                f"{pod}: allocated capital is {detail.get('allocated_capital')!r}; "
                "an unmeasured allocation cannot be compared to the cap",
            )
        ]
    ceiling = capacity * utilisation
    if allocated > ceiling:
        return [
            Breach(
                "CAPACITY",
                f"{pod}: allocated {allocated:,.0f} > {utilisation:.0%} of estimated capacity "
                f"{capacity:,.0f} ({ceiling:,.0f})",
            )
        ]
    return []
