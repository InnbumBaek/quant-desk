"""Look-ahead scan for G0.

Detecting leakage by reading code does not scale and does not survive a
refactor. This recomputes the signal on truncated data and compares: if the
value at time t changes when data after t is withheld, the signal saw the
future, whatever the code claims.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

SignalFn = Callable[[np.ndarray], np.ndarray]


class SignalWidthChanged(ValueError):
    """The signal returned a different number of columns on a truncated frame.

    This is not leakage, and left unchecked it does not read as itself: numpy
    either refuses to broadcast (an error whose message says nothing about the
    universe) or, when the width collapses to one, broadcasts happily and the
    scan reports a leak that is not there. Both send a reader after a bug that
    does not exist. The universe really does change with listings and
    delistings, so the fix is to fix the universe upstream, not the scan --
    which is what this exception says.
    """


@dataclass
class LeakReport:
    probes: int
    leaks: list[int] = field(default_factory=list)
    max_deviation: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.leaks


def lookahead_scan(
    signal_fn: SignalFn,
    data: np.ndarray,
    probe_points: Sequence[int] | None = None,
    n_probes: int = 24,
    tolerance: float = 1e-9,
    seed: int = 0,
) -> LeakReport:
    """Recompute the signal point-in-time and compare against the full-sample run.

    `signal_fn` takes a data frame and returns one signal per row. A probe at
    index t hands it `data[: t + 1]` and checks the last value against the
    full-sample value at t. A signal may be one number per row or a vector per
    row (a weight per symbol); a vector probe compares every element and keeps
    the largest deviation, so leakage in a single name cannot average away.
    """
    data = np.asarray(data, dtype=float)
    n_rows = data.shape[0]
    full = np.asarray(signal_fn(data), dtype=float)
    if full.shape[0] != n_rows:
        raise ValueError("signal_fn must return one signal per row")

    if probe_points is None:
        rng = np.random.default_rng(seed)
        low = max(2, n_rows // 10)
        probe_points = sorted(set(rng.integers(low, n_rows, size=n_probes).tolist()))

    report = LeakReport(probes=len(probe_points))
    for t in probe_points:
        point_in_time = np.asarray(signal_fn(data[: t + 1]), dtype=float)
        observed, expected = np.shape(point_in_time[-1]), np.shape(full[t])
        if observed != expected:
            raise SignalWidthChanged(
                f"at probe {t} the signal is {observed} wide on the truncated frame but "
                f"{expected} on the full sample. The scan compares values, not universes: "
                "hold the universe fixed and pad absent symbols explicitly."
            )
        deviation = float(np.max(np.abs(point_in_time[-1] - full[t])))
        report.max_deviation = max(report.max_deviation, deviation)
        if deviation > tolerance:
            report.leaks.append(int(t))

    return report
