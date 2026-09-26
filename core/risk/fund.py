"""Fund-level risk: VaR, expected shortfall, pod correlation and the fund halt.

The `fund:` block of `limits.yaml` had no reader at all. `var95_1d_max`,
`es975_1d_max`, `pod_avg_correlation_max` and the fund drawdown tiers -- cut
gross in half at -10%, halt everything at -15% -- existed as four lines of YAML
and nothing else. The fund halt is the last stop between a bad month and a
blown fund, so this is the same finding as ADR-0015 one level up, and it gets the
same treatment: measure it, and read absence as a breach.

Decisions worth stating.

**Historical, not parametric.** VaR and ES come from the empirical distribution
of the fund's own daily returns. A normal distribution understates exactly the
days a limit exists for. The cost is that a historical VaR can never exceed the
worst day in its window, so the window length and the number of observations in
the tail are recorded beside the number.

**The fund's return series is built from declared allocations.** Summing pod
returns equally would invent an allocation, and an invented allocation produces a
VaR for a fund that does not exist. `allocations` is required.

**One pod is not an unmeasured correlation.** With a single pod there is no
pairwise correlation, which is different from having one and not knowing it. That
exemption applies only when `pod_count` is 1, a measured fact -- everything else
absent is a breach.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.risk.limits import Breach, as_measurement, load_limits

#: A year of daily returns. ES at 97.5% puts about six observations in the tail
#: at this length; shorter than this and the number rests on one or two days.
DEFAULT_WINDOW = 250
#: Below this the distribution has no tail to speak of and both numbers are
#: unmeasured rather than optimistic.
MIN_RETURNS = 60


@dataclass(frozen=True)
class FundExposure:
    snapshot: dict[str, Any]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def unmeasured(self) -> tuple[str, ...]:
        return tuple(sorted(key for key, value in self.snapshot.items() if value is None))


def historical_var(returns: np.ndarray, level: float = 0.95) -> float:
    """Loss at the `level` quantile, as a positive fraction of capital.

    A 2% VaR95 means: on the worst 5% of days the fund lost at least 2%.
    """
    if not 0.5 < level < 1.0:
        raise ValueError(f"level {level} is not a tail probability")
    series = np.asarray(returns, dtype=float)
    if series.size < 2 or not np.all(np.isfinite(series)):
        raise ValueError("a VaR needs a finite return series")
    return float(-np.quantile(series, 1.0 - level))


def expected_shortfall(returns: np.ndarray, level: float = 0.975) -> tuple[float, int]:
    """Mean loss in the tail beyond `level`, positive, with the tail's size.

    The count comes back because an ES averaging two observations and one
    averaging sixty are different claims wearing the same number.
    """
    if not 0.5 < level < 1.0:
        raise ValueError(f"level {level} is not a tail probability")
    series = np.asarray(returns, dtype=float)
    if series.size < 2 or not np.all(np.isfinite(series)):
        raise ValueError("an expected shortfall needs a finite return series")
    threshold = float(np.quantile(series, 1.0 - level))
    tail = series[series <= threshold]
    if tail.size == 0:  # every return sits above the quantile, so the quantile is the tail
        tail = np.array([threshold])
    return float(-np.mean(tail)), int(tail.size)


def average_pod_correlation(series: Mapping[str, np.ndarray]) -> float:
    """Mean pairwise correlation across pods.

    The mean, not the maximum: the limit is about the book as a whole being one
    bet. A single crowded pair is what `pod_avg_correlation_max` tolerates and
    what the center book's crowding trim is for.
    """
    names = sorted(series)
    if len(names) < 2:
        raise ValueError("a pairwise correlation needs at least two pods")
    matrix = np.array([np.asarray(series[name], dtype=float) for name in names])
    if not np.all(np.isfinite(matrix)):
        raise ValueError("a pod return series holds a non-finite value")
    deviations = np.std(matrix, axis=1)
    flat = [name for name, spread in zip(names, deviations, strict=True) if spread == 0.0]
    if flat:
        raise ValueError(f"pod(s) {', '.join(flat)} have no variance, so no correlation exists")
    correlations = np.corrcoef(matrix)
    upper = correlations[np.triu_indices(len(names), k=1)]
    return float(np.mean(upper))


def fund_returns(pod_series: Mapping[str, np.ndarray], allocations: Mapping[str, float]) -> np.ndarray:
    """The fund's return series: each pod's returns times its share of capital.

    Allocations name every pod exactly once and sum to at most 1; what is left
    over is cash, which earns nothing here. A missing or extra name raises rather
    than being filled in, because a fund return built on a guessed allocation is
    a fund that does not exist.
    """
    if not pod_series:
        raise ValueError("no pod returns given")
    missing = sorted(set(pod_series) - set(allocations))
    extra = sorted(set(allocations) - set(pod_series))
    if missing or extra:
        raise ValueError(
            f"allocations do not match the pods (missing: {missing or 'none'}, unknown: {extra or 'none'})"
        )
    lengths = {name: np.asarray(series, dtype=float).size for name, series in pod_series.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"pod return series have different lengths: {lengths}")
    total = sum(float(weight) for weight in allocations.values())
    if any(float(weight) < 0.0 for weight in allocations.values()):
        raise ValueError("a negative allocation is a short of a pod, which this does not model")
    if total > 1.0 + 1e-9:
        raise ValueError(f"allocations sum to {total:.3f}, more than the fund's capital")

    stacked = np.array([np.asarray(pod_series[name], dtype=float) for name in sorted(pod_series)])
    weights = np.array([float(allocations[name]) for name in sorted(pod_series)])
    return np.asarray(weights @ stacked, dtype=float)


def fund_snapshot(
    pod_series: Mapping[str, np.ndarray],
    allocations: Mapping[str, float],
    window: int = DEFAULT_WINDOW,
    min_returns: int = MIN_RETURNS,
) -> FundExposure:
    """Measure what `check_fund` reads, from the pods' return series."""
    series = fund_returns(pod_series, allocations)
    if not np.all(np.isfinite(series)):
        raise ValueError("the fund return series holds a non-finite value")

    notes: list[str] = []
    snapshot: dict[str, Any] = {
        "pod_count": len(pod_series),
        "returns_used": 0,
        "var95_1d": None,
        "es975_1d": None,
        "es975_tail_observations": None,
        "pod_avg_correlation": None,
        "drawdown": None,
    }

    recent = series[-window:]
    snapshot["returns_used"] = int(recent.size)
    if recent.size < min_returns:
        notes.append(
            f"VaR and expected shortfall unmeasured: {recent.size} return(s), short of the "
            f"{min_returns} needed for a tail to exist"
        )
    else:
        snapshot["var95_1d"] = historical_var(recent, 0.95)
        shortfall, tail = expected_shortfall(recent, 0.975)
        snapshot["es975_1d"] = shortfall
        snapshot["es975_tail_observations"] = tail
        if tail < 5:
            notes.append(
                f"the 97.5% tail holds {tail} observation(s); the expected shortfall rests on "
                "those days and no others"
            )
        notes.append(
            f"historical VaR over {recent.size} days cannot exceed the worst of them "
            f"({-float(np.min(recent)):.2%})"
        )

    if len(pod_series) < 2:
        notes.append(
            "pod correlation does not exist with a single pod; this is not an unmeasured value "
            "and check_fund exempts it while pod_count is 1"
        )
    else:
        try:
            trimmed = {name: np.asarray(values, dtype=float)[-window:] for name, values in pod_series.items()}
            snapshot["pod_avg_correlation"] = average_pod_correlation(trimmed)
        except ValueError as error:
            notes.append(f"pod correlation unmeasured: {error}")

    equity = np.cumprod(1.0 + series)
    peak = np.maximum.accumulate(equity)
    snapshot["drawdown"] = float(equity[-1] / peak[-1] - 1.0)
    return FundExposure(snapshot=snapshot, notes=tuple(notes))


def check_fund(snapshot: dict[str, Any], limits: dict[str, Any] | None = None) -> list[Breach]:
    """Check the fund snapshot against the fund limits. Absence blocks (ADR-0015).

    snapshot keys: var95_1d, es975_1d, pod_avg_correlation, pod_count, drawdown.
    Losses are positive fractions, drawdown is negative.
    """
    lim = (limits or load_limits())["fund"]
    out: list[Breach] = []

    var95 = as_measurement(snapshot.get("var95_1d"))
    if var95 is None:
        out.append(Breach("FUND_VAR_UNMEASURED", f"1-day VaR95 is {snapshot.get('var95_1d')!r}"))
    elif var95 > lim["var95_1d_max"]:
        out.append(Breach("FUND_VAR95", f"{var95:.2%} > {lim['var95_1d_max']:.0%}"))

    shortfall = as_measurement(snapshot.get("es975_1d"))
    if shortfall is None:
        out.append(Breach("FUND_ES_UNMEASURED", f"1-day ES97.5 is {snapshot.get('es975_1d')!r}"))
    elif shortfall > lim["es975_1d_max"]:
        out.append(Breach("FUND_ES975", f"{shortfall:.2%} > {lim['es975_1d_max']:.0%}"))

    out.extend(correlation_breaches(snapshot, lim))
    out.extend(fund_drawdown_breaches(snapshot, lim))
    return out


def correlation_breaches(snapshot: dict[str, Any], fund_limits: dict[str, Any]) -> list[Breach]:
    """Average pod correlation, exempt only for a fund that has one pod.

    The exemption is narrow on purpose. "There is no second pod" is a fact the
    snapshot states; "nobody computed the correlation" is silence, and silence is
    what this whole rule exists to stop.
    """
    count = as_measurement(snapshot.get("pod_count"))
    if count is None or count < 1:
        return [Breach("POD_COUNT_UNMEASURED", f"pod count is {snapshot.get('pod_count')!r}")]
    correlation = as_measurement(snapshot.get("pod_avg_correlation"))
    if correlation is None:
        if count == 1:
            return []
        return [
            Breach(
                "POD_CORRELATION_UNMEASURED",
                f"average pod correlation is {snapshot.get('pod_avg_correlation')!r} across "
                f"{int(count)} pods",
            )
        ]
    if correlation > fund_limits["pod_avg_correlation_max"]:
        return [Breach("POD_CORRELATION", f"{correlation:.2f} > {fund_limits['pod_avg_correlation_max']}")]
    return []


def fund_drawdown_breaches(snapshot: dict[str, Any], fund_limits: dict[str, Any]) -> list[Breach]:
    """The fund drawdown tiers, most severe first.

    Unlike the pod tiers there is no distribution condition: a pod inside its own
    historical drawdown is having a normal bad run, but a fund down 15% is down
    15% whatever its backtest said, and this is the last stop before the capital
    is gone.
    """
    drawdown = as_measurement(snapshot.get("drawdown"))
    if drawdown is None:
        return [
            Breach(
                "FUND_DD_UNMEASURED",
                f"fund drawdown is {snapshot.get('drawdown')!r}; the halt cannot be checked",
            )
        ]
    for name in ("halt", "reduce"):
        tier = fund_limits["drawdown"][name]
        if drawdown <= tier["abs"]:
            return [Breach(f"FUND_DD_{name.upper()}", f"{drawdown:+.2%} -> {tier['action']}")]
    return []


def required_actions(breaches: list[Breach], limits: dict[str, Any] | None = None) -> tuple[str, ...]:
    """The actions the fund limits call for, in the order the table names them."""
    lim = (limits or load_limits())["fund"]["drawdown"]
    actions = []
    for name in ("halt", "reduce"):
        if any(breach.code == f"FUND_DD_{name.upper()}" for breach in breaches):
            actions.append(str(lim[name]["action"]))
    return tuple(actions)
