"""Capital allocation across pods.

The rule this file implements exists because the obvious one is wrong. Sizing
pods in proportion to their rolling Sharpe amplifies estimation error straight
into the book: over a short sample the standard error of a Sharpe ratio is
large enough that the ranking is mostly noise, so a Sharpe-proportional
allocator spends its time chasing luck. The replacement (PLAN section 1.1, v2)
starts from risk parity, lets realised performance tilt the result but never
decide it, and caps the total at half of the estimated Kelly leverage.

Nothing here can widen risk. The weights this module proposes still pass
through `core.risk.limits.check_pod` and the pre-trade hook, and the capacity
ceiling is clamped to the 80% that CLAUDE.md rule 7 fixes, whatever the
configuration file says.

References
    Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix".
    Maillard, Roncalli & Teiletche (2010) for equal risk contribution.
    Kelly leverage halved is the standard concession to estimation error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core import audit
from core.config import ROOT
from core.risk.limits import load_limits

ALLOCATION_FILE = ROOT / "core" / "portfolio" / "allocation.yaml"

# CLAUDE.md rule 7. The configuration file may tighten this and may not loosen it.
CAPACITY_FRACTION_CEILING = 0.80

TRADING_DAYS = 252


def load_allocation_config(path: Path | None = None) -> dict[str, Any]:
    with (path or ALLOCATION_FILE).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def capacity_fraction(limits: dict[str, Any]) -> float:
    """Capacity utilisation ceiling, read from the limit table, clamped to rule 7.

    The number lives in limits.yaml because exceeding it puts capital at risk.
    The clamp is here because CLAUDE.md rule 7 fixes 80% as an absolute, so the
    table may tighten it and may not loosen it.
    """
    return min(float(limits["capacity"]["capacity_utilisation_max"]), CAPACITY_FRACTION_CEILING)


def horizon_budget(limits: dict[str, Any]) -> float:
    """Largest share of the book one horizon may hold, from the limit table."""
    return float(limits["horizon"]["risk_budget_share_max"])


def kelly_fraction(limits: dict[str, Any]) -> float:
    """Fraction of the estimated Kelly leverage to take, from the limit table."""
    return float(limits["pod"]["kelly_fraction"])


def volatility_band(limits: dict[str, Any]) -> tuple[float, float]:
    """The annualised realised-volatility target band, from the limit table."""
    low, high = limits["pod"]["target_volatility"]
    return float(low), float(high)


def volatility_target(limits: dict[str, Any]) -> float:
    """The single number the gross is sized to: the midpoint of the band.

    The table states a band because the risk engine judges a realised number
    against it. Sizing needs one target, and the midpoint is the only choice
    that leaves equal room on both sides before the book is judged.
    """
    low, high = volatility_band(limits)
    return (low + high) / 2.0


def lock_months(limits: dict[str, Any]) -> int:
    """How long an allocation holds, from the limit table."""
    return int(limits["allocation"]["lock_months"])


# What each action in the limit table's drawdown ladder does to an allocation.
# The ladder already names the action ("halve_capital", "stop_pod"); until now
# nothing carried it out, so a pod in a cut-level drawdown kept its capital and
# only had its orders blocked. Blocking orders is not de-risking: the position
# stays on and comes back at full size the moment the breach clears.
DRAWDOWN_MULTIPLIER = {"report": 1.0, "halve_capital": 0.5, "stop_pod": 0.0}


def drawdown_multiplier(tier: str | None, limits: dict[str, Any]) -> float:
    """How much of its allocation a pod keeps at this drawdown tier.

    An unknown tier or an action the table names but this module does not
    implement raises. Returning 1.0 for something unrecognised would let a
    renamed limit silently stop cutting capital, which is the one failure this
    ladder exists to prevent.
    """
    if tier is None:
        return 1.0
    ladder = limits["pod"]["drawdown"]
    if tier not in ladder:
        raise ValueError(f"unknown drawdown tier {tier!r}; the table has {sorted(ladder)}")
    action = str(ladder[tier]["action"])
    if action not in DRAWDOWN_MULTIPLIER:
        raise ValueError(f"drawdown action {action!r} has no allocation rule")
    return DRAWDOWN_MULTIPLIER[action]


@dataclass
class PodState:
    """What the allocator needs to know about one pod.

    `returns` are daily and already cost-deducted: the data, compute and LLM
    spend attributed to the pod comes out before the allocator sees it, so a
    pod that does not earn its costs cannot present a positive record.
    """

    pod_id: str
    returns: np.ndarray
    capacity_usd: float
    horizon: str = "daily"
    clean_months: int = 0
    months_since_allocation: int = 0
    current_weight: float | None = None
    stopped: bool = False
    gate_failed: bool = False
    # The drawdown tier the risk engine judged, not one the allocator derives.
    # core/risk/limits.py owns that judgment: it needs both the absolute level
    # and the pod's own sealed backtest distribution, and a second opinion here
    # would be a second place to get it wrong.
    drawdown_tier: str | None = None


@dataclass(frozen=True)
class PodAllocation:
    pod_id: str
    weight: float
    risk_parity_weight: float
    ir_tilt: float
    ir_available: bool
    binding: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AllocationResult:
    allocations: list[PodAllocation]
    gross: float
    half_kelly_gross: float
    gross_limit: float
    binding_gross: str = "none"
    expected_volatility: float = 0.0
    notes: list[str] = field(default_factory=list)

    def weight(self, pod_id: str) -> float:
        for allocation in self.allocations:
            if allocation.pod_id == pod_id:
                return allocation.weight
        raise KeyError(pod_id)

    @property
    def by_pod(self) -> dict[str, float]:
        return {a.pod_id: a.weight for a in self.allocations}


def shrunk_correlation(returns: np.ndarray) -> np.ndarray:
    """Sample correlation shrunk toward the identity at the Ledoit-Wolf intensity.

    With a handful of pods and a few hundred observations the sample
    correlation is noisy enough that risk parity built on it concentrates in
    whichever pair happened to look uncorrelated.
    """
    returns = np.asarray(returns, dtype=float)
    n_obs, n_assets = returns.shape
    if n_assets < 2:
        return np.eye(n_assets)

    sd = returns.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    standardised = (returns - returns.mean(axis=0)) / sd
    sample = standardised.T @ standardised / n_obs

    target = np.eye(n_assets)
    dispersion = float(np.sum((sample - target) ** 2))
    if dispersion <= 0:
        return target

    # Variance of the sample correlation entries, the numerator of the optimal
    # shrinkage intensity.
    noise = 0.0
    for t in range(n_obs):
        outer = np.outer(standardised[t], standardised[t])
        noise += float(np.sum((outer - sample) ** 2))
    noise /= n_obs**2

    intensity = min(1.0, max(0.0, min(noise, dispersion) / dispersion))
    shrunk = intensity * target + (1.0 - intensity) * sample
    np.fill_diagonal(shrunk, 1.0)
    return shrunk


def equal_risk_contribution(covariance: np.ndarray, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
    """Weights where every pod contributes the same share of portfolio variance.

    Inverse-volatility weighting is only equal-risk when the pods are
    uncorrelated; once they are not, the correlated pods quietly carry more
    risk than their weight suggests.
    """
    covariance = np.asarray(covariance, dtype=float)
    n_assets = covariance.shape[0]
    if n_assets == 1:
        return np.ones(1)

    variances = np.diag(covariance).copy()
    variances[variances <= 0] = np.finfo(float).eps
    weights = 1.0 / np.sqrt(variances)
    weights /= weights.sum()

    for _ in range(max_iter):
        marginal = covariance @ weights
        contribution = weights * marginal
        portfolio_var = float(weights @ covariance @ weights)
        if portfolio_var <= 0:
            break
        target = portfolio_var / n_assets
        if np.max(np.abs(contribution / portfolio_var - 1.0 / n_assets)) < tol:
            break
        safe = np.where(contribution > 0, contribution, np.finfo(float).eps)
        weights = weights * np.sqrt(target / safe)
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        if total <= 0:
            return np.full(n_assets, 1.0 / n_assets)
        weights /= total

    return weights


def information_ratio(returns: np.ndarray, lookback: int) -> float | None:
    """Annualised IR over the lookback, or None when the history is too short.

    A pod without a year of record has no IR. Substituting zero would read as
    "it lost money" and substituting an average would invent a track record, so
    the caller is told the number does not exist and the ramp does the limiting.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < lookback:
        return None
    window = returns[-lookback:]
    sd = window.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(window.mean() / sd * np.sqrt(TRADING_DAYS))


def ramp_fraction(clean_months: int, ramp: list[dict[str, float]]) -> float:
    """The share of target a pod has earned, by consecutive incident-free months."""
    fraction = 0.0
    for step in sorted(ramp, key=lambda s: s["clean_months"]):
        if clean_months >= step["clean_months"]:
            fraction = float(step["fraction"])
    return fraction


def _average_pairwise_correlation(correlation: np.ndarray) -> float | None:
    """Mean of the off-diagonal entries, or None when there is no pair."""
    n = correlation.shape[0]
    if n < 2:
        return None
    off_diagonal = correlation[~np.eye(n, dtype=bool)]
    return float(off_diagonal.mean())


def _common_window(pods: list[PodState]) -> tuple[np.ndarray, int]:
    """Align pod histories on their shortest common tail."""
    length = min(pod.returns.size for pod in pods)
    matrix = np.column_stack([np.asarray(pod.returns, dtype=float)[-length:] for pod in pods])
    return matrix, length


def annualised(portfolio: np.ndarray) -> tuple[float, float]:
    """Annualised mean and volatility of a daily series."""
    return (
        float(portfolio.mean()) * TRADING_DAYS,
        float(portfolio.std(ddof=1)) * np.sqrt(TRADING_DAYS),
    )


def _half_kelly_gross(portfolio: np.ndarray, kelly_fraction: float) -> float:
    """Kelly leverage for the combined book, halved.

    Kelly is mu / sigma^2 on annualised figures. Both are estimates, and the
    error in mu is the one that bankrupts people, which is why the standard
    concession is to take half.

    The series handed in is the performance window, the same one the IR tilt
    reads. Sizing off the whole available history while tilting off the last
    year would mean the allocator held two different opinions about what
    counts as performance.
    """
    mean, sd = annualised(portfolio)
    if sd <= 0 or mean <= 0:
        return 0.0
    return kelly_fraction * mean / sd**2


def _volatility_target_gross(portfolio: np.ndarray, target_volatility: float) -> float:
    """Gross that puts the unit-gross book at the target volatility."""
    _, sd = annualised(portfolio)
    if sd <= 0:
        return 0.0
    return target_volatility / sd


def allocate(
    pods: list[PodState],
    nav: float,
    config: dict[str, Any] | None = None,
    limits: dict[str, Any] | None = None,
    audit_path: Path | None = None,
) -> AllocationResult:
    """Propose a weight per pod.

    Every step after risk parity can only shrink a weight. That is the property
    worth keeping: a pod cannot end up larger because a constraint fired
    somewhere else, and capital freed by a cap stays uninvested rather than
    being pushed into a pod the risk model never sized for it.
    """
    if nav <= 0:
        raise ValueError("nav must be positive")
    config = config or load_allocation_config()
    limit_table = limits or load_limits()
    gross_limit = float(limit_table["pod"]["gross_leverage_max"])
    capacity_max = capacity_fraction(limit_table)

    notes: list[str] = []
    live = [pod for pod in pods if not pod.stopped]
    if not live:
        return AllocationResult([], 0.0, 0.0, gross_limit, "none", 0.0, ["every pod is stopped"])

    matrix, window = _common_window(live)
    if window < 2:
        raise ValueError("pods need at least two observations to be sized")
    notes.append(f"covariance window: {window} observations")

    correlation = shrunk_correlation(matrix)
    # The fund-level diversification limit. The allocator reports it and does
    # not act on it: correlation is a property of which pods exist, not of how
    # they are sized, so shrinking weights would report compliance without
    # changing the thing measured. Which pod to drop is a cio decision, and the
    # limit table names a number here but no action, unlike the drawdown ladder.
    average_correlation = _average_pairwise_correlation(correlation)
    if average_correlation is not None:
        correlation_max = float(limit_table["fund"]["pod_avg_correlation_max"])
        notes.append(f"average pairwise pod correlation: {average_correlation:.3f}")
        if average_correlation > correlation_max:
            notes.append(
                f"BREACH pod_avg_correlation_max ({correlation_max:.2f}): these pods are "
                "one bet wearing several names; dropping one is a cio decision"
            )

    volatility = matrix.std(axis=0, ddof=1)
    volatility = np.where(volatility > 0, volatility, np.finfo(float).eps)
    covariance = np.outer(volatility, volatility) * correlation
    parity = equal_risk_contribution(covariance)

    lookback = int(config["ir_lookback_days"])
    tilt_cap = float(config["ir_tilt_cap"])
    tilts, availability = [], []
    for pod in live:
        ratio = information_ratio(pod.returns, lookback)
        availability.append(ratio is not None)
        # No history means no tilt, not a bad tilt. The ramp keeps a young pod small.
        tilts.append(1.0 if ratio is None else float(np.clip(ratio, 0.0, tilt_cap) / tilt_cap))
    tilt_array = np.array(tilts)

    tilted = parity * tilt_array
    if tilted.sum() <= 0:
        notes.append("no pod has a positive cost-deducted IR; nothing is allocated")
        allocations = [
            PodAllocation(pod.pod_id, 0.0, float(parity[i]), float(tilt_array[i]), availability[i], "ir_tilt")
            for i, pod in enumerate(live)
        ]
        allocations += [
            PodAllocation(pod.pod_id, 0.0, 0.0, 0.0, False, "stopped") for pod in pods if pod.stopped
        ]
        return _record(
            AllocationResult(allocations, 0.0, 0.0, gross_limit, "ir_tilt", 0.0, notes),
            nav,
            audit_path,
        )

    shares = tilted / tilted.sum()

    # One performance window for both the tilt and the sizing, and the sizing
    # runs on the book actually being proposed. Sizing off the untilted parity
    # weights would let a pod the tilt has already zeroed drag the whole book's
    # gross down with it.
    proposed_book = (matrix @ shares)[-lookback:]
    half_kelly = _half_kelly_gross(proposed_book, kelly_fraction(limit_table))
    vol_target = _volatility_target_gross(proposed_book, volatility_target(limit_table))

    # No separate no-leverage number: gross_leverage_max IS that limit now that
    # the owner's rule is in the table (ADR-0009). A limit with two homes is a
    # limit nobody can change safely.
    candidates = {
        "half_kelly": half_kelly,
        "target_volatility": vol_target,
        "gross_leverage_max": gross_limit,
    }
    binding_gross = min(candidates, key=lambda name: candidates[name])
    gross_budget = candidates[binding_gross]
    if gross_budget <= 0:
        notes.append(f"{binding_gross} allows no capital; nothing is allocated")
    notes.append(f"gross budget set by {binding_gross} at {gross_budget:.3f}")
    _, realised_vol = annualised(proposed_book * gross_budget)
    notes.append(f"expected portfolio volatility at this gross: {realised_vol:.3f}")

    weights = shares * max(gross_budget, 0.0)
    binding = ["risk_parity"] * len(live)
    per_pod_notes: list[list[str]] = [[] for _ in live]

    # The ramp: a young pod may only hold a fraction of what it was sized for.
    ramp = config["ramp"]
    for i, pod in enumerate(live):
        fraction = ramp_fraction(pod.clean_months, ramp)
        if fraction < 1.0:
            weights[i] *= fraction
            binding[i] = "ramp"
            per_pod_notes[i].append(f"ramp at {fraction:.0%} after {pod.clean_months} clean months")

    # The lock: an allocation holds for lock_months. A drawdown trigger or a
    # gate failure is the exception, and so is the capacity ceiling below.
    months = lock_months(limit_table)
    for i, pod in enumerate(live):
        if pod.current_weight is None:
            continue
        unlocked = pod.gate_failed or drawdown_multiplier(pod.drawdown_tier, limit_table) < 1.0
        if pod.months_since_allocation < months and not unlocked:
            weights[i] = float(pod.current_weight)
            binding[i] = "lock"
            remaining = months - pod.months_since_allocation
            per_pod_notes[i].append(f"held at the previous weight, {remaining} month(s) of lock left")

    # The drawdown ladder. It runs after the lock because a pod deep enough in
    # drawdown to be cut is exactly the pod whose lock must not protect it.
    for i, pod in enumerate(live):
        keep = drawdown_multiplier(pod.drawdown_tier, limit_table)
        if keep < 1.0:
            was_locked = binding[i] == "lock"
            weights[i] *= keep
            binding[i] = "drawdown"
            action = limit_table["pod"]["drawdown"][pod.drawdown_tier]["action"]
            per_pod_notes[i].append(f"drawdown tier '{pod.drawdown_tier}' -> {action}")
            if was_locked:
                per_pod_notes[i].append("the drawdown ladder overrides the allocation lock")
        elif pod.drawdown_tier is not None:
            per_pod_notes[i].append(f"drawdown tier '{pod.drawdown_tier}' is report-only")

    # The capacity ceiling. CLAUDE.md rule 7: never above this, for any reason,
    # which includes a locked weight that has outgrown its pod's capacity.
    for i, pod in enumerate(live):
        ceiling = capacity_max * pod.capacity_usd / nav
        if weights[i] > ceiling:
            was_locked = binding[i] == "lock"
            weights[i] = ceiling
            binding[i] = "capacity"
            per_pod_notes[i].append(f"trimmed to {capacity_max:.0%} of estimated capacity")
            if was_locked:
                per_pod_notes[i].append("capacity overrides the allocation lock")

    # The horizon budget: one way of making money may not be most of the book.
    horizon_max = horizon_budget(limit_table)
    total = weights.sum()
    if total > 0:
        for horizon in {pod.horizon for pod in live}:
            members = [i for i, pod in enumerate(live) if pod.horizon == horizon]
            share = weights[members].sum() / total
            if share > horizon_max:
                scale = horizon_max / share
                for i in members:
                    weights[i] *= scale
                    binding[i] = "horizon_budget"
                    per_pod_notes[i].append(f"horizon '{horizon}' trimmed to {horizon_max:.0%}")

    allocations = [
        PodAllocation(
            pod_id=pod.pod_id,
            weight=float(weights[i]),
            risk_parity_weight=float(parity[i]),
            ir_tilt=float(tilt_array[i]),
            ir_available=availability[i],
            binding=binding[i],
            notes=per_pod_notes[i],
        )
        for i, pod in enumerate(live)
    ]
    allocations += [
        PodAllocation(pod.pod_id, 0.0, 0.0, 0.0, False, "stopped", ["pod is stopped"])
        for pod in pods
        if pod.stopped
    ]

    result = AllocationResult(
        allocations=allocations,
        gross=float(weights.sum()),
        half_kelly_gross=half_kelly,
        gross_limit=gross_limit,
        binding_gross=binding_gross,
        expected_volatility=realised_vol,
        notes=notes,
    )
    return _record(result, nav, audit_path)


def _record(result: AllocationResult, nav: float, audit_path: Path | None) -> AllocationResult:
    audit.append(
        "portfolio.allocate",
        {
            "nav": nav,
            "gross": result.gross,
            "half_kelly_gross": result.half_kelly_gross,
            "binding_gross": result.binding_gross,
            "expected_volatility": result.expected_volatility,
            "allocations": [
                {
                    "pod_id": a.pod_id,
                    "weight": a.weight,
                    "binding": a.binding,
                    "ir_available": a.ir_available,
                }
                for a in result.allocations
            ],
        },
        path=audit_path,
    )
    return result
