"""The shared feature catalogue. One copy of each idea, readable by every pod.

CLAUDE.md rule 6 says a pod may not keep a private feature. That rule needs a
place to live, and this is it: a feature exists once it is registered here, and
registration is where three things get checked, because a comment in a review
cannot check them.

- **Point-in-time.** Registration recomputes the feature on truncated data and
  compares (`core/backtest/leakage.py`). A feature that changes its value at time
  `t` when later data is withheld saw the future, whatever its author intended.
  Catching it here means it can never reach a submission; catching it at G0 means
  every alpha built on it has to be thrown away.
- **Not a duplicate.** A new feature is rejected when its rank correlation with an
  existing one exceeds `MAX_ABS_CORRELATION`. Two names for one signal inflate
  the apparent breadth of the book and quietly double an exposure.
- **Finite everywhere.** A feature must state its warm-up value rather than
  return NaN. A NaN propagates into weights, and a NaN weight is a position
  nobody chose.

Rank (Spearman) correlation, not Pearson: a monotone transform of an existing
feature -- a log, a winsorisation, a rank itself -- is the same idea wearing a
different scale, and Pearson is happy to call it new.

`MAX_ABS_CORRELATION` deliberately does **not** live in `core/risk/limits.yaml`.
That file is the risk limit table, changing it needs owner approval and an ADR
(ADR-0003), and a catalogue de-duplication threshold is neither a risk limit nor
something the pre-trade path reads. Keeping it here means a wider catalogue can
never be obtained by editing the risk limits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from scipy import stats as sps

from core import audit
from core.backtest.leakage import lookahead_scan

#: Rank correlation above which a new feature is the same idea as an existing one.
MAX_ABS_CORRELATION = 0.90

#: A feature takes the data panel it is allowed to see and returns one value per
#: row per symbol. Row `t` may only use rows `<= t`.
FeatureFn = Callable[[np.ndarray], np.ndarray]


class Rejection(StrEnum):
    LEAKAGE = "leakage"
    DUPLICATE = "duplicate"
    NON_FINITE = "non_finite"
    SHAPE = "shape"
    NAME_TAKEN = "name_taken"


@dataclass(frozen=True)
class Feature:
    """A feature and the provenance that makes it auditable.

    `owner` records who contributed it, for cost attribution and for asking
    questions later. It is not an access control: every pod reads every feature.
    """

    name: str
    description: str
    source: str
    owner: str
    fn: FeatureFn


@dataclass(frozen=True)
class Registration:
    accepted: bool
    name: str
    reason: str = ""
    rejection: Rejection | None = None
    correlations: dict[str, float] = field(default_factory=dict)
    leak_probes: int = 0


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation over the pooled panel, 0.0 when either side is constant."""
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("features must be computed on the same panel")
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    rho = sps.spearmanr(a, b).statistic
    return 0.0 if np.isnan(rho) else float(rho)


class FeatureCatalog:
    """Registered features and their values on one reference panel.

    The reference panel is what makes correlation comparable: two features are
    only duplicates relative to the same data. The panel is supplied once, at
    construction, so nobody can register a feature against data chosen to make it
    look independent.
    """

    def __init__(
        self,
        panel: np.ndarray,
        max_abs_correlation: float = MAX_ABS_CORRELATION,
        audit_path: Path | None = None,
        leak_probes: int = 16,
    ) -> None:
        self._panel = np.asarray(panel, dtype=float)
        if self._panel.ndim != 2:
            raise ValueError("the reference panel must be 2-D (dates x symbols)")
        self._max_abs_correlation = float(max_abs_correlation)
        self._audit_path = audit_path
        self._leak_probes = leak_probes
        self._features: dict[str, Feature] = {}
        self._values: dict[str, np.ndarray] = {}

    @property
    def panel(self) -> np.ndarray:
        """The reference panel every feature in this catalogue was measured on."""
        return self._panel

    def names(self) -> tuple[str, ...]:
        return tuple(self._features)

    def values(self, name: str) -> np.ndarray:
        """Any pod may read any registered feature. There is no owner filter."""
        if name not in self._values:
            raise KeyError(f"no feature named {name!r}; registered: {self.names()}")
        return self._values[name]

    def feature(self, name: str) -> Feature:
        return self._features[name]

    def register(self, feature: Feature) -> Registration:
        """Check, then admit. A rejection is recorded as carefully as an admission."""
        result = self._evaluate(feature)
        audit.append(
            "features.register",
            {
                "name": feature.name,
                "owner": feature.owner,
                "source": feature.source,
                "accepted": result.accepted,
                "rejection": result.rejection.value if result.rejection else None,
                "reason": result.reason,
                "correlations": result.correlations,
            },
            path=self._audit_path,
        )
        return result

    def _evaluate(self, feature: Feature) -> Registration:
        if feature.name in self._features:
            return Registration(
                False,
                feature.name,
                f"a feature named {feature.name!r} is already registered",
                Rejection.NAME_TAKEN,
            )

        values = np.asarray(feature.fn(self._panel), dtype=float)
        if values.shape != self._panel.shape:
            return Registration(
                False,
                feature.name,
                f"returned {values.shape}, expected one value per row per symbol {self._panel.shape}",
                Rejection.SHAPE,
            )
        if not np.all(np.isfinite(values)):
            return Registration(
                False,
                feature.name,
                "returned a non-finite value; state an explicit warm-up value instead of NaN",
                Rejection.NON_FINITE,
            )

        report = lookahead_scan(feature.fn, self._panel, n_probes=self._leak_probes)
        if not report.ok:
            return Registration(
                False,
                feature.name,
                f"look-ahead at {len(report.leaks)} of {report.probes} probes "
                f"(max deviation {report.max_deviation:.3g})",
                Rejection.LEAKAGE,
                leak_probes=report.probes,
            )

        correlations = {name: rank_correlation(values, existing) for name, existing in self._values.items()}
        worst = max(correlations.items(), key=lambda kv: abs(kv[1]), default=None)
        if worst is not None and abs(worst[1]) > self._max_abs_correlation:
            return Registration(
                False,
                feature.name,
                f"rank correlation {worst[1]:+.2f} with {worst[0]!r} exceeds "
                f"{self._max_abs_correlation:.2f}; extend that feature instead of adding a second name",
                Rejection.DUPLICATE,
                correlations=correlations,
                leak_probes=report.probes,
            )

        self._features[feature.name] = feature
        self._values[feature.name] = values
        return Registration(
            True,
            feature.name,
            "",
            None,
            correlations,
            report.probes,
        )

    def correlations(self) -> dict[tuple[str, str], float]:
        """Pairwise rank correlations of everything registered, for the crowding review."""
        names = self.names()
        out: dict[tuple[str, str], float] = {}
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                out[(left, right)] = rank_correlation(self._values[left], self._values[right])
        return out


def catalogue_summary(catalog: FeatureCatalog) -> list[Mapping[str, object]]:
    """One row per feature, for a report that must cite the catalogue rather than recall it."""
    return [
        {
            "name": name,
            "owner": catalog.feature(name).owner,
            "source": catalog.feature(name).source,
            "description": catalog.feature(name).description,
        }
        for name in catalog.names()
    ]
