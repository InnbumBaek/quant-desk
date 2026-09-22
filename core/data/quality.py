"""Data health checks — gate G0 and the 06:30 pre-market gate.

P0 ships the contract and the look-ahead check; the remaining checks are
declared here so the pipeline can already fail closed on a missing result
rather than silently skipping a check that does not exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass

REQUIRED_CHECKS = (
    "as_of_alignment",
    "lookahead_scan",
    "timestamp_monotonicity",
    "survivorship_adjustment",
    "corporate_action_consistency",
    "missing_data_ratio",
)


@dataclass(frozen=True)
class HealthReport:
    results: dict[str, bool]

    @property
    def missing(self) -> list[str]:
        return [name for name in REQUIRED_CHECKS if name not in self.results]

    @property
    def failed(self) -> list[str]:
        return sorted(name for name, ok in self.results.items() if not ok)

    @property
    def ok(self) -> bool:
        """Unknown counts as failed: a check that did not run is not a pass."""
        return not self.missing and not self.failed


def lookahead_scan(signal_dates: list[str], feature_dates: list[str]) -> bool:
    """True when no feature timestamp is later than the signal it feeds."""
    return all(f <= s for s, f in zip(signal_dates, feature_dates, strict=True))
