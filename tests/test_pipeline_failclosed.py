from core.data.quality import REQUIRED_CHECKS, HealthReport, lookahead_scan
from core.execution.orders import Order, State
from core.pipeline import run_day

# Gross is below 1.0 because the table no longer permits leverage, and every
# measurement is present because an unmeasured one blocks (ADR-0009, ADR-0015).
CLEAN_BOOK = {
    "gross": 0.95,
    "net": 0.0,
    "weights": {"SPY": 0.04},
    "sector_weights": {"broad": 0.18},
    "style_betas": {"mkt": 0.05},
    "liquidation_days": 1.0,
    "realised_volatility": 0.12,
    "drawdown": -0.005,
    "backtest_dd_pct": 30.0,
}
# A day only sends orders when every layer has been measured: the pod book, the
# financing state and the fund tiers. An absent snapshot is a breach, not a
# skipped check (ADR-0015 for the pod, ADR-0016 for the fund and financing).
CLEAN_FUND = {
    "pod_count": 1,
    "var95_1d": 0.012,
    "es975_1d": 0.02,
    "pod_avg_correlation": None,
    "drawdown": -0.01,
}
CLEAN_FINANCING = {
    "margin_utilization": 0.30,
    "pb_shares": {"prime": 1.0},
    "financing_cost_bps": 80,
    "cash_buffer": 0.15,
}
HEALTHY = HealthReport({name: True for name in REQUIRED_CHECKS})


def clean_day(tmp_path, **overrides):
    """A day where nothing is wrong, so a test can change one thing at a time."""
    kwargs = {
        "health": HEALTHY,
        "target_snapshot": dict(CLEAN_BOOK),
        "financing_snapshot": dict(CLEAN_FINANCING),
        "fund_snapshot": dict(CLEAN_FUND),
        "audit_path": tmp_path / "a.log",
    }
    kwargs.update(overrides)
    return run_day("2026-09-22", **kwargs)


def test_clean_day_allows_orders(tmp_path):
    result = clean_day(tmp_path)
    assert result.orders_allowed and not result.liquidate_only
    assert result.gross_multiplier == 1.0 and result.actions == ()


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
    result = clean_day(
        tmp_path,
        pod_targets={"statarb": {"SPY": 0.045}, "trend": {"SPY": -0.045}},
    )
    assert result.orders_allowed
    assert result.netting is not None and result.netting.turnover_saved == 1.0


def test_crowded_name_is_trimmed_not_blocked(tmp_path):
    result = clean_day(tmp_path, pod_targets={"a": {"NVDA": 0.04}, "b": {"NVDA": 0.04}})
    assert result.orders_allowed
    assert result.netting.net_targets == {"NVDA": 0.05}


def test_financing_breach_blocks_orders(tmp_path):
    result = clean_day(
        tmp_path,
        financing_snapshot={**CLEAN_FINANCING, "margin_utilization": 0.95},
    )
    assert result.liquidate_only
    assert any(r.startswith("MARGIN_UTIL") for r in result.reasons)


def test_a_book_with_no_measured_volatility_fails_closed(tmp_path):
    """The volatility target is only a target if the day stops when it is unknown."""
    blind = {k: v for k, v in CLEAN_BOOK.items() if k != "realised_volatility"}
    result = run_day("2026-09-22", HEALTHY, blind, audit_path=tmp_path / "a.log")
    assert result.liquidate_only
    assert any(r.startswith("VOL_UNMEASURED") for r in result.reasons)


# --- the fund layer (ADR-0016) ----------------------------------------------


def test_a_day_with_no_fund_snapshot_sends_nothing(tmp_path):
    """An optional fund halt is not a fund halt."""
    result = clean_day(tmp_path, fund_snapshot=None)
    assert result.liquidate_only
    assert any(r.startswith("FUND_UNMEASURED") for r in result.reasons)


def test_a_day_with_no_financing_snapshot_sends_nothing(tmp_path):
    result = clean_day(tmp_path, financing_snapshot=None)
    assert result.liquidate_only
    assert any(r.startswith("FINANCING_UNMEASURED") for r in result.reasons)


def test_the_fund_halt_blocks_new_orders(tmp_path):
    result = clean_day(tmp_path, fund_snapshot={**CLEAN_FUND, "drawdown": -0.16})
    assert result.liquidate_only
    assert result.actions == ("halt_all",)
    assert any("FUND_DD_HALT" in r for r in result.reasons)


def test_the_reduce_tier_halves_the_targets_rather_than_only_reporting_them(tmp_path):
    """ "Cut gross in half" has to reach the book, or it is a line in a log."""
    result = clean_day(
        tmp_path,
        fund_snapshot={**CLEAN_FUND, "drawdown": -0.11},
        pod_targets={"statarb": {"SPY": 0.04}},
    )
    assert result.actions == ("cut_gross_half",)
    assert result.gross_multiplier == 0.5
    assert result.netting.net_targets == {"SPY": 0.02}


def test_the_multiplier_only_ever_reduces(tmp_path):
    """An overlay may shrink a book and never grow one (CLAUDE.md 5)."""
    for drawdown in (-0.01, -0.11, -0.16):
        result = clean_day(tmp_path, fund_snapshot={**CLEAN_FUND, "drawdown": drawdown})
        assert result.gross_multiplier <= 1.0


def test_a_fund_var_breach_blocks_orders(tmp_path):
    result = clean_day(tmp_path, fund_snapshot={**CLEAN_FUND, "var95_1d": 0.05})
    assert result.liquidate_only
    assert any(r.startswith("FUND_VAR95") for r in result.reasons)


def test_an_unmeasured_fund_var_blocks_orders(tmp_path):
    blind = {k: v for k, v in CLEAN_FUND.items() if k != "var95_1d"}
    result = clean_day(tmp_path, fund_snapshot=blind)
    assert result.liquidate_only
    assert any(r.startswith("FUND_VAR_UNMEASURED") for r in result.reasons)
