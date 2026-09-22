"""Canary contract. See README.md in this directory.

Each fake alpha must be rejected, and rejected for the defect it was built to
have. A canary that fails for the wrong reason proves nothing, so the expected
gate is asserted by name.
"""

import pytest

from core.backtest import gates
from tests.canaries.alphas import CANARIES

EXPECTED_GATE = {
    "canary_random": {"G2_in_sample", "G4_statistics"},
    "canary_lookahead": {"G0_data"},
    "canary_overfit": {"G4_statistics"},
    "canary_factor": {"G4_statistics"},
}

EXPECTED_REASON = {
    "canary_lookahead": "look-ahead",
    "canary_overfit": "PBO",
    "canary_factor": "residual alpha t",
}


@pytest.mark.parametrize("name", sorted(CANARIES))
def test_canary_is_rejected(name, tmp_path):
    submission, leak_report = CANARIES[name]()
    verdicts = gates.evaluate(submission, leak_report, audit_path=tmp_path / "audit.log")

    assert not gates.approved(verdicts), f"{name} passed every gate"
    failed = set(gates.failed_gates(verdicts))
    assert failed & EXPECTED_GATE[name], f"{name} rejected by {failed}, expected {EXPECTED_GATE[name]}"

    if name in EXPECTED_REASON:
        reasons = " ".join(v.reason for v in verdicts if not v.passed)
        assert EXPECTED_REASON[name] in reasons, f"{name} rejected, but not for its defect: {reasons}"


def test_every_canary_is_covered():
    assert set(CANARIES) == set(EXPECTED_GATE)
