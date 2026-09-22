import pytest

from core.risk.limits import check_pod, load_limits

# A book the table lets through. Gross sits below 1.0 because the table no longer
# permits leverage, and every field is present because an unmeasured field is
# blocked rather than waved through (ADR-0009 for volatility, ADR-0015 for the
# rest). A snapshot is only clean when it is complete.
BASE = {
    "gross": 0.95,
    "net": 0.05,
    "weights": {"AAPL": 0.03},
    "sector_weights": {"tech": 0.15},
    "style_betas": {"mom": 0.1},
    "liquidation_days": 1.5,
    "realised_volatility": 0.12,
    "drawdown": -0.01,
    "backtest_dd_pct": 40.0,
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


# --- absence blocks (ADR-0015) ----------------------------------------------


@pytest.mark.parametrize(
    "key, code",
    [
        ("gross", "GROSS_UNMEASURED"),
        ("net", "NET_UNMEASURED"),
        ("sector_weights", "SECTOR_UNMEASURED"),
        ("style_betas", "STYLE_BETA_UNMEASURED"),
        ("liquidation_days", "LIQUIDITY_UNMEASURED"),
        ("realised_volatility", "VOL_UNMEASURED"),
        ("drawdown", "DD_UNMEASURED"),
        ("backtest_dd_pct", "DD_UNMEASURED"),
    ],
)
def test_a_missing_measurement_blocks(key, code):
    """Every one of these used to pass by absence, which is a limit that checks nothing."""
    snapshot = {name: value for name, value in BASE.items() if name != key}
    assert code in {b.code for b in check_pod(snapshot)}


@pytest.mark.parametrize(
    "key, code",
    [
        ("gross", "GROSS_UNMEASURED"),
        ("net", "NET_UNMEASURED"),
        ("liquidation_days", "LIQUIDITY_UNMEASURED"),
        ("drawdown", "DD_UNMEASURED"),
    ],
)
def test_an_explicit_none_blocks_too(key, code):
    """`core/risk/exposure.py` writes None for what it could not measure."""
    assert code in {b.code for b in check_pod({**BASE, key: None})}


def test_a_string_is_not_a_measurement():
    assert "GROSS_UNMEASURED" in {b.code for b in check_pod({**BASE, "gross": "0.95"})}


def test_true_is_not_a_gross_of_one():
    """`True` is 1.0 to Python, which would read as a fully invested book."""
    assert "GROSS_UNMEASURED" in {b.code for b in check_pod({**BASE, "gross": True})}


def test_a_nan_is_not_a_measurement():
    assert "LIQUIDITY_UNMEASURED" in {b.code for b in check_pod({**BASE, "liquidation_days": float("nan")})}


def test_zero_is_a_measurement_everywhere_it_is_one():
    """Absence and zero are different: a flat, instantly liquidatable book is fine."""
    flat = {
        **BASE,
        "gross": 0.0,
        "net": 0.0,
        "weights": {},
        "sector_weights": {},
        "liquidation_days": 0.0,
        "drawdown": 0.0,
        "backtest_dd_pct": 0.0,
    }
    assert check_pod(flat) == []


def test_an_empty_sector_mapping_blocks_when_the_book_holds_something():
    snapshot = {**BASE, "sector_weights": {}}
    assert "SECTOR_UNMEASURED" in {b.code for b in check_pod(snapshot)}


def test_an_empty_beta_mapping_blocks_even_for_a_flat_book():
    """A measured flat book has a beta of zero on every factor, not no betas."""
    snapshot = {**BASE, "weights": {}, "gross": 0.0, "style_betas": {}}
    assert "STYLE_BETA_UNMEASURED" in {b.code for b in check_pod(snapshot)}


def test_a_beta_of_zero_is_a_measurement():
    assert check_pod({**BASE, "style_betas": {"mkt": 0.0, "smb": 0.0}}) == []


def test_an_unmeasurable_value_inside_a_mapping_blocks():
    snapshot = {**BASE, "style_betas": {"mkt": 0.1, "smb": None}}
    assert "STYLE_BETA_UNMEASURED" in {b.code for b in check_pod(snapshot)}


def test_a_negative_liquidation_horizon_is_not_a_horizon():
    assert "LIQUIDITY_UNMEASURED" in {b.code for b in check_pod({**BASE, "liquidation_days": -1.0})}
