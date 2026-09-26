"""Measuring a pod snapshot, and saying so when a field cannot be measured."""

from __future__ import annotations

import numpy as np
import pytest

from core.backtest.engine import PricePanel
from core.data.factors import FactorPanel
from core.risk.exposure import liquidation_days, pod_returns, pod_snapshot, style_betas
from core.risk.limits import check_pod

WINDOW = 20


def business_days(count: int) -> np.ndarray:
    days = np.arange(np.datetime64("2024-01-01"), np.datetime64("2026-01-01"), dtype="datetime64[D]")
    return days[np.is_busday(days)][:count]


def panel(rows: int = 60, symbols: tuple[str, ...] = ("SPY", "TLT"), volume: bool = True) -> PricePanel:
    rng = np.random.default_rng(5)
    steps = rng.normal(0.0003, 0.01, (rows - 1, len(symbols)))
    close = 100.0 * np.cumprod(np.vstack([np.ones((1, len(symbols))), 1.0 + steps]), axis=0)
    return PricePanel(
        dates=business_days(rows),
        symbols=symbols,
        close=close,
        dollar_volume=np.full_like(close, 5_000_000.0) if volume else None,
    )


def book(rows: int, symbols: int = 2, weight: float = 0.02) -> np.ndarray:
    return np.full((rows, symbols), weight)


def factors_for(price_panel: PricePanel, names: tuple[str, ...] = ("Mkt-RF", "SMB")) -> FactorPanel:
    rng = np.random.default_rng(6)
    return FactorPanel(
        dates=price_panel.dates,
        names=names,
        values=rng.normal(0.0002, 0.008, (price_panel.dates.shape[0], len(names))),
        source="test",
        digest="0" * 64,
    )


# --- the fields that are always measurable ----------------------------------


def test_exposure_comes_from_the_last_row_of_the_history():
    price_panel = panel()
    weights = book(price_panel.dates.shape[0])
    weights[-1] = [0.03, -0.01]

    snapshot = pod_snapshot(weights, price_panel, window=WINDOW).snapshot
    assert snapshot["gross"] == pytest.approx(0.04)
    assert snapshot["net"] == pytest.approx(0.02)
    assert snapshot["weights"] == {"SPY": 0.03, "TLT": -0.01}


def test_the_sector_weights_are_summed_per_bucket():
    price_panel = panel()
    weights = book(price_panel.dates.shape[0])
    weights[-1] = [0.03, 0.02]
    snapshot = pod_snapshot(weights, price_panel, window=WINDOW).snapshot
    assert snapshot["sector_weights"] == {"us_equity_broad": 0.03, "rates_long": 0.02}


def test_an_unclassified_symbol_leaves_the_sector_weights_unmeasured():
    """Blocked rather than guessed: the bucket decides a concentration limit."""
    price_panel = panel(symbols=("SPY", "NVDA"))
    exposure = pod_snapshot(book(price_panel.dates.shape[0]), price_panel, window=WINDOW)
    assert exposure.snapshot["sector_weights"] is None
    assert any("no bucket for NVDA" in note for note in exposure.notes)


def test_the_liquidation_horizon_is_the_worst_name():
    price_panel = panel()
    weights = book(price_panel.dates.shape[0])
    weights[-1] = [0.10, 0.01]

    days = liquidation_days(weights[-1], price_panel, capital=1_000_000.0, participation=0.20, window=WINDOW)
    # 0.10 * 1e6 = 100k to unwind at 20% of a 5m ADV = 1m a day.
    assert days == pytest.approx(0.1)


def test_a_panel_without_volume_leaves_the_horizon_unmeasured():
    price_panel = panel(volume=False)
    exposure = pod_snapshot(book(price_panel.dates.shape[0]), price_panel, window=WINDOW)
    assert exposure.snapshot["liquidation_days"] is None
    assert any("no dollar volume" in note for note in exposure.notes)


def test_a_participation_share_outside_zero_to_one_is_refused():
    price_panel = panel()
    with pytest.raises(ValueError, match="not a share of volume"):
        liquidation_days(book(3)[-1], price_panel, 1_000_000.0, participation=1.5, window=WINDOW)


# --- volatility comes from one implementation -------------------------------


def test_the_snapshot_volatility_is_the_overlays_volatility():
    """Two estimates of one quantity let a book clear the band on one of them."""
    price_panel = panel(rows=80)
    weights = book(price_panel.dates.shape[0])
    exposure = pod_snapshot(weights, price_panel, window=WINDOW, periods_per_year=252)

    series = pod_returns(weights, price_panel)
    by_hand = float(np.std(series[-WINDOW:], ddof=1)) * np.sqrt(252)
    assert exposure.snapshot["realised_volatility"] == pytest.approx(by_hand, rel=1e-9)


def test_a_history_shorter_than_the_window_leaves_volatility_unmeasured():
    price_panel = panel(rows=10)
    exposure = pod_snapshot(book(10), price_panel, window=WINDOW)
    assert exposure.snapshot["realised_volatility"] is None
    assert any("short of the 20-bar window" in note for note in exposure.notes)


# --- style betas ------------------------------------------------------------


def test_a_book_that_is_one_factor_has_a_beta_of_one_on_it():
    matrix = np.random.default_rng(1).normal(0.0, 0.01, (40, 2))
    returns = matrix[:, 0]
    betas = style_betas(returns, matrix, ("Mkt-RF", "SMB"))
    assert betas["Mkt-RF"] == pytest.approx(1.0, abs=1e-6)
    assert betas["SMB"] == pytest.approx(0.0, abs=1e-6)


def test_the_betas_are_measured_when_a_factor_file_covers_the_panel():
    price_panel = panel(rows=80)
    exposure = pod_snapshot(
        book(price_panel.dates.shape[0]), price_panel, factors=factors_for(price_panel), window=WINDOW
    )
    assert set(exposure.snapshot["style_betas"]) == {"Mkt-RF", "SMB"}


def test_without_a_factor_file_the_betas_are_unmeasured():
    price_panel = panel(rows=80)
    exposure = pod_snapshot(book(price_panel.dates.shape[0]), price_panel, window=WINDOW)
    assert exposure.snapshot["style_betas"] is None
    assert any("no factor file" in note for note in exposure.notes)


def test_a_factor_file_that_does_not_reach_the_panel_leaves_the_betas_unmeasured():
    price_panel = panel(rows=80)
    short = factors_for(price_panel)
    clipped = FactorPanel(
        dates=short.dates[:40],
        names=short.names,
        values=short.values[:40],
        source=short.source,
        digest=short.digest,
    )
    exposure = pod_snapshot(book(80), price_panel, factors=clipped, window=WINDOW)
    assert exposure.snapshot["style_betas"] is None
    assert any("style betas unmeasured" in note for note in exposure.notes)


def test_a_mismatched_factor_matrix_is_refused():
    with pytest.raises(ValueError, match="against 3 factor rows"):
        style_betas(np.zeros(5), np.zeros((3, 2)), ("a", "b"))


# --- drawdown ---------------------------------------------------------------


def test_the_drawdown_needs_a_reference_distribution_for_its_percentile():
    price_panel = panel(rows=80)
    exposure = pod_snapshot(book(price_panel.dates.shape[0]), price_panel, window=WINDOW)
    assert exposure.snapshot["drawdown"] is not None
    assert exposure.snapshot["backtest_dd_pct"] is None
    assert any("nothing to be compared against" in note for note in exposure.notes)


def test_the_percentile_is_where_the_live_drawdown_sits_in_the_backtest():
    price_panel = panel(rows=80)
    weights = book(price_panel.dates.shape[0])
    reference = np.array([0.0, 0.01, 0.02, 0.03, 0.50])
    exposure = pod_snapshot(weights, price_panel, window=WINDOW, backtest_drawdowns=reference)

    depth = -exposure.snapshot["drawdown"]
    expected = 100.0 * float(np.mean(reference <= depth))
    assert exposure.snapshot["backtest_dd_pct"] == pytest.approx(expected)


# --- what the snapshot is for ----------------------------------------------


def test_a_measured_snapshot_is_one_check_pod_can_read():
    """The point of this module: the limit engine gets a complete snapshot."""
    price_panel = panel(rows=80)
    weights = book(price_panel.dates.shape[0], weight=0.02)
    exposure = pod_snapshot(
        weights,
        price_panel,
        factors=factors_for(price_panel),
        window=WINDOW,
        backtest_drawdowns=np.array([0.01, 0.02, 0.03]),
    )
    assert exposure.unmeasured == ()
    codes = {breach.code for breach in check_pod(exposure.snapshot)}
    assert not {code for code in codes if code.endswith("UNMEASURED")}


def test_an_incomplete_snapshot_blocks_rather_than_passes():
    price_panel = panel(rows=10, volume=False)
    exposure = pod_snapshot(book(10), price_panel, window=WINDOW)
    codes = {breach.code for breach in check_pod(exposure.snapshot)}
    assert {"LIQUIDITY_UNMEASURED", "VOL_UNMEASURED", "STYLE_BETA_UNMEASURED", "DD_UNMEASURED"} <= codes


# --- refusals ---------------------------------------------------------------


def test_a_single_row_of_weights_is_not_a_history():
    with pytest.raises(ValueError, match="at least two rows"):
        pod_snapshot(np.zeros((1, 2)), panel(rows=3))


def test_a_one_dimensional_weight_row_is_refused():
    with pytest.raises(ValueError, match="2-D"):
        pod_snapshot(np.zeros(2), panel(rows=3))


def test_non_finite_weights_are_refused():
    weights = book(60)
    weights[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pod_snapshot(weights, panel(rows=60))


def test_weights_that_do_not_match_the_panel_are_refused():
    with pytest.raises(ValueError, match="must match the close panel"):
        pod_returns(book(60, symbols=3), panel(rows=60))
