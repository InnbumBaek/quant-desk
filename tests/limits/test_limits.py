import pytest

from core.risk.limits import check_pod, load_limits

# A book the table lets through. Gross sits below 1.0 because the table no longer
# permits leverage, and realised volatility is present because a book whose
# volatility is unmeasured is blocked rather than waved through (ADR-0009).
BASE = {
    "gross": 0.95,
    "net": 0.05,
    "weights": {"AAPL": 0.03},
    "sector_weights": {"tech": 0.15},
    "style_betas": {"mom": 0.1},
    "liquidation_days": 1.5,
    "realised_volatility": 0.12,
}


def test_clean_book_has_no_breach():
    assert check_pod(dict(BASE)) == []


@pytest.mark.parametrize(
    "patch, code",
    [
        ({"gross": 2.5}, "GROSS"),
        # 1.0 is the ceiling, not a soft target: 1.05 is already leverage.
        ({"gross": 1.05}, "GROSS"),
        ({"net": 0.35}, "NET"),
        ({"weights": {"AAPL": 0.09}}, "SINGLE_NAME"),
        ({"sector_weights": {"tech": 0.31}}, "SECTOR"),
        ({"style_betas": {"mom": 0.6}}, "STYLE_BETA"),
        ({"liquidation_days": 5.0}, "LIQUIDITY"),
        ({"realised_volatility": 0.22}, "VOL_ABOVE_TARGET"),
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
    assert gates["paper_trading_days_min"] == 63, "minimum three months, 252/4"


# --- the volatility target (ADR-0009) ---------------------------------------


def test_the_table_declares_the_new_pod_limits():
    pod = load_limits()["pod"]
    assert pod["gross_leverage_max"] == 1.0, "no leverage by default"
    assert pod["target_volatility"] == [0.10, 0.15]
    assert pod["kelly_fraction"] == 0.5
    assert load_limits()["allocation"]["lock_months"] == 3


def test_an_unmeasured_volatility_blocks():
    """A target nobody can measure is a target nobody is keeping."""
    blind = {k: v for k, v in BASE.items() if k != "realised_volatility"}
    assert {b.code for b in check_pod(blind)} == {"VOL_UNMEASURED"}


def test_a_volatility_of_zero_is_a_measurement_not_an_absence():
    """0.0 must read as measured, or the absence rule silently stops working."""
    assert check_pod({**BASE, "realised_volatility": 0.0}) == []


def test_undershooting_the_band_is_not_a_breach():
    """The band is an operating target; undershooting costs return, not capital.

    Treating an undershoot as a breach would invite raising gross to clear it,
    which is the one thing the gross limit exists to stop.
    """
    assert check_pod({**BASE, "realised_volatility": 0.04}) == []


@pytest.mark.parametrize("value", [-0.10, "12%", None, True])
def test_a_nonsense_volatility_is_not_read_as_a_measurement(value):
    snapshot = {**BASE, "realised_volatility": value}
    assert "VOL_UNMEASURED" in {b.code for b in check_pod(snapshot)}
