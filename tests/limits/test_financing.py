from core.risk.financing import check_financing

OK = {
    "margin_utilization": 0.35,
    "pb_shares": {"ibkr": 1.0},
    "financing_cost_bps": 90,
    "cash_buffer": 0.15,
}


def test_healthy_financing_passes():
    assert check_financing(dict(OK)) == []


def test_margin_and_cash_are_fail_closed_when_unknown():
    codes = {b.code for b in check_financing({"pb_shares": {"ibkr": 1.0}})}
    assert {"MARGIN_UTIL", "CASH_BUFFER"} <= codes


def test_each_financing_limit_trips():
    assert "MARGIN_UTIL" in {b.code for b in check_financing({**OK, "margin_utilization": 0.9})}
    assert "FINANCING_COST" in {b.code for b in check_financing({**OK, "financing_cost_bps": 400})}
    assert "CASH_BUFFER" in {b.code for b in check_financing({**OK, "cash_buffer": 0.01})}
