from core import audit


def test_chain_verifies(tmp_path):
    log = tmp_path / "audit.log"
    audit.append("gate.verdict", {"alpha": "a1", "gate": "G4", "pass": False}, path=log)
    audit.append("limits.change", {"field": "gross_leverage_max", "to": 1.5}, path=log)
    ok, broken = audit.verify(log)
    assert ok and broken is None
    assert len(list(audit.read(log))) == 2


def test_tampering_is_detected(tmp_path):
    log = tmp_path / "audit.log"
    audit.append("order.submit", {"symbol": "SPY", "qty": 10}, path=log)
    audit.append("order.fill", {"symbol": "SPY", "qty": 10}, path=log)
    lines = log.read_text(encoding="utf-8").splitlines()
    log.write_text(lines[0].replace('"qty": 10', '"qty": 99') + "\n" + lines[1] + "\n", encoding="utf-8")
    ok, broken = audit.verify(log)
    assert not ok and broken
