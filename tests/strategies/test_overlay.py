"""The volatility overlay is tested for the two things that would make it unsafe.

It must never scale a book up -- that is borrowing, and the pod gross limit is
1.0 -- and it must not treat an unmeasured volatility as an acceptable one. Both
are failures that a backtest would report as improved performance.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import strategies
from core.backtest.leakage import lookahead_scan
from core.strategies.overlay import realised_volatility, vol_target

TARGET = 0.125  # midpoint of the owner's 10-15% band


def close_panel(n_rows: int = 500, n_cols: int = 4, vol: float = 0.011, seed: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0002, vol, size=(n_rows, n_cols))
    return 100.0 * np.exp(np.cumsum(steps, axis=0))


def book(prices: np.ndarray) -> np.ndarray:
    return strategies.get("ts_momentum").build(lookback=40)(prices)


# --- the overlay only ever reduces ------------------------------------------


def test_the_overlay_never_scales_a_book_up():
    """Scaling up to reach a target is borrowing, and CLAUDE.md rule 5 forbids it."""
    prices = close_panel(vol=0.002)  # a very quiet market: target is far above realised
    weights = book(prices)
    scaled, measured = vol_target(weights, prices, TARGET)

    quiet = np.isfinite(measured) & (measured < TARGET)
    assert quiet.any(), "the fixture must contain bars below target or it proves nothing"
    assert np.all(np.abs(scaled[quiet]) <= np.abs(weights[quiet]) + 1e-12)


def test_a_loud_book_is_cut_to_the_target():
    prices = close_panel(vol=0.05)
    weights = book(prices)
    scaled, measured = vol_target(weights, prices, TARGET)

    loud = np.isfinite(measured) & (measured > TARGET)
    assert loud.any()
    ratio = np.sum(np.abs(scaled[loud]), axis=1) / np.sum(np.abs(weights[loud]), axis=1)
    assert np.all(ratio < 1.0)
    assert np.allclose(ratio, TARGET / measured[loud], atol=1e-12)


def test_an_unmeasured_volatility_holds_nothing():
    """Consistent with the limits engine blocking an unmeasured book (ADR-0009)."""
    prices = close_panel()
    scaled, measured = vol_target(book(prices), prices, TARGET, lookback=60)
    blind = ~np.isfinite(measured)
    assert blind.any()
    assert np.all(scaled[blind] == 0.0)


def test_a_measured_zero_is_not_an_absence():
    """A flat book has zero volatility; that is a measurement and must not zero it."""
    prices = close_panel()
    flat = np.zeros_like(prices)
    scaled, measured = vol_target(flat, prices, TARGET, lookback=60)
    assert np.all(measured[60:] == 0.0)
    assert np.all(scaled[60:] == 0.0)  # a flat book stays flat, but by its own weights


def test_the_overlay_never_turns_a_position_into_its_opposite():
    prices = close_panel(vol=0.04)
    weights = book(prices)
    scaled, _ = vol_target(weights, prices, TARGET)
    assert np.all(np.sign(scaled) * np.sign(weights) >= 0)


# --- point in time ----------------------------------------------------------


def test_the_overlaid_signal_does_not_look_ahead():
    def signal(close: np.ndarray) -> np.ndarray:
        weights = strategies.get("ts_momentum").build(lookback=40)(close)
        return vol_target(weights, close, TARGET, lookback=60)[0]

    report = lookahead_scan(signal, close_panel(), n_probes=24)
    assert report.ok, f"leaked at {len(report.leaks)} of {report.probes} probes"


def test_the_estimate_uses_the_position_that_was_actually_held():
    """Row t must read weights up to t-1 only: weights[t] earns bar t -> t+1."""
    prices = close_panel()
    weights = book(prices)
    baseline = realised_volatility(weights, prices, lookback=60)

    tampered = weights.copy()
    tampered[300:] *= 3.0
    after = realised_volatility(tampered, prices, lookback=60)

    # Row 300 is estimated from holdings up to 299, so it cannot have moved.
    assert np.isclose(baseline[300], after[300])
    assert not np.isclose(baseline[301], after[301])


# --- refusals ---------------------------------------------------------------


def test_a_non_positive_target_is_refused():
    prices = close_panel()
    with pytest.raises(ValueError, match="not a target"):
        vol_target(book(prices), prices, 0.0)


def test_a_one_bar_window_is_refused():
    prices = close_panel()
    with pytest.raises(ValueError, match="too short"):
        realised_volatility(book(prices), prices, lookback=1)


def test_mismatched_shapes_are_refused():
    prices = close_panel()
    with pytest.raises(ValueError, match="weights are"):
        realised_volatility(np.zeros((10, 2)), prices)


# --- it closes the loop with the limits table -------------------------------


def test_the_series_it_returns_is_what_the_limits_engine_checks():
    """One estimate, read by both, or a book can pass the limit while breaching it."""
    from core.risk.limits import check_pod, load_limits

    prices = close_panel(vol=0.05)
    _, measured = vol_target(book(prices), prices, TARGET)
    limits = load_limits()
    snapshot = {
        "gross": 0.8,
        "net": 0.0,
        "realised_volatility": float(measured[-1]),
    }
    breaches = {b.code for b in check_pod(snapshot, limits)}
    high = limits["pod"]["target_volatility"][1]
    assert ("VOL_ABOVE_TARGET" in breaches) == (measured[-1] > high)
