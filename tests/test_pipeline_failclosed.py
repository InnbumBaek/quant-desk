from core.data.quality import REQUIRED_CHECKS, HealthReport, lookahead_scan
from core.execution.orders import Order, State
from core.pipeline import run_day

CLEAN_BOOK = {
    "gross": 1.2,
    "net": 0.0,
    "weights": {"SPY": 0.04},
    "sector_weights": {"broad": 0.18},
    "style_betas": {"mkt": 0.05},
    "liquidation_days": 1.0,
}
HEALTHY = HealthReport({name: True for name in REQUIRED_CHECKS})


def test_clean_day_allows_orders(tmp_path):
    result = run_day("2026-09-21", HEALTHY, dict(CLEAN_BOOK), audit_path=tmp_path / "a.log")
    assert result.orders_allowed and not result.liquidate_only


def test_failed_health_check_fails_closed(tmp_path):
    sick = HealthReport({**{n: True for n in REQUIRED_CHECKS}, "lookahead_scan": False})
    result = run_day("2026-09-21", sick, dict(CLEAN_BOOK), audit_path=tmp_path / "a.log")
    assert result.liquidate_only
    assert any("lookahead_scan" in r for r in result.reasons)


def test_missing_check_counts_as_failure(tmp_path):
    partial = HealthReport({name: True for name in REQUIRED_CHECKS[:-1]})
    result = run_day("2026-09-21", partial, dict(CLEAN_BOOK), audit_path=tmp_path / "a.log")
    assert result.liquidate_only


def test_unresolved_order_blocks_new_orders(tmp_path):
    stuck = Order("2026-09-18", "statarb-001", "SPY", 0, 100)
    stuck.transition(State.SUBMITTED)
    result = run_day(
        "2026-09-21", HEALTHY, dict(CLEAN_BOOK), open_orders=[stuck], audit_path=tmp_path / "a.log"
    )
    assert result.liquidate_only


def test_limit_breach_blocks_new_orders(tmp_path):
    result = run_day("2026-09-21", HEALTHY, {**CLEAN_BOOK, "gross": 4.0}, audit_path=tmp_path / "a.log")
    assert result.liquidate_only and any(r.startswith("GROSS") for r in result.reasons)


def test_lookahead_helper():
    assert lookahead_scan(["2026-01-02", "2026-01-03"], ["2026-01-02", "2026-01-02"])
    assert not lookahead_scan(["2026-01-02"], ["2026-01-05"])


def test_center_book_nets_pods_before_limits(tmp_path):
    # Two pods hold opposite SPY positions: the netted book is flat, so the
    # single-name limit cannot be breached by the sum.
    result = run_day(
        "2026-09-22",
        HEALTHY,
        dict(CLEAN_BOOK),
        audit_path=tmp_path / "a.log",
        pod_targets={"statarb": {"SPY": 0.045}, "trend": {"SPY": -0.045}},
    )
    assert result.orders_allowed
    assert result.netting is not None and result.netting.turnover_saved == 1.0


def test_crowded_name_is_trimmed_not_blocked(tmp_path):
    result = run_day(
        "2026-09-22",
        HEALTHY,
        dict(CLEAN_BOOK),
        audit_path=tmp_path / "a.log",
        pod_targets={"a": {"NVDA": 0.04}, "b": {"NVDA": 0.04}},
    )
    assert result.orders_allowed
    assert result.netting.net_targets == {"NVDA": 0.05}


def test_financing_breach_blocks_orders(tmp_path):
    result = run_day(
        "2026-09-22",
        HEALTHY,
        dict(CLEAN_BOOK),
        audit_path=tmp_path / "a.log",
        financing_snapshot={"margin_utilization": 0.95, "cash_buffer": 0.15},
    )
    assert result.liquidate_only
    assert any(r.startswith("MARGIN_UTIL") for r in result.reasons)
