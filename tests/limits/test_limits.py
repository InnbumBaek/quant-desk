import pytest

from core.risk.limits import check_pod, load_limits

BASE = {
    "gross": 1.2,
    "net": 0.05,
    "weights": {"AAPL": 0.03},
    "sector_weights": {"tech": 0.15},
    "style_betas": {"mom": 0.1},
    "liquidation_days": 1.5,
}


def test_clean_book_has_no_breach():
    assert check_pod(dict(BASE)) == []


@pytest.mark.parametrize(
    "patch, code",
    [
        ({"gross": 2.5}, "GROSS"),
        ({"net": 0.35}, "NET"),
        ({"weights": {"AAPL": 0.09}}, "SINGLE_NAME"),
        ({"sector_weights": {"tech": 0.31}}, "SECTOR"),
        ({"style_betas": {"mom": 0.6}}, "STYLE_BETA"),
        ({"liquidation_days": 5.0}, "LIQUIDITY"),
    ],
)
def test_each_limit_trips(patch, code):
    snapshot = {**BASE, **patch}
    assert code in {b.code for b in check_pod(snapshot)}


def test_drawdown_needs_both_conditions():
    # -6% but inside the strategy's own historical distribution: no cut.
    calm = {**BASE, "drawdown": -0.06, "backtest_dd_pct": 70}
    assert not [b for b in check_pod(calm) if b.code.startswith("DD_")]
    # -6% and beyond the 95th percentile: capital is halved.
    hot = {**BASE, "drawdown": -0.06, "backtest_dd_pct": 97}
    assert {b.code for b in check_pod(hot)} == {"DD_CUT"}
    # -8% beyond the 99th percentile: the pod stops.
    stop = {**BASE, "drawdown": -0.08, "backtest_dd_pct": 99.5}
    assert {b.code for b in check_pod(stop)} == {"DD_STOP"}


def test_gate_thresholds_are_declared():
    gates = load_limits()["gates"]
    assert gates["residual_alpha_tstat_min"] == 3.0
    assert gates["pbo_max"] == 0.05
    assert gates["paper_trading_days_min"] == 60
