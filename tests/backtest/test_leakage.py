"""G0 recomputes the signal point-in-time. Reading the code is not a control."""

import numpy as np
import pytest

from core.backtest import leakage
from core.backtest.leakage import lookahead_scan


def _past_only(data: np.ndarray) -> np.ndarray:
    returns = data[:, 0]
    return np.concatenate([[0.0], np.cumsum(returns)[:-1]])


def _peeks_one_day_ahead(data: np.ndarray) -> np.ndarray:
    returns = data[:, 0]
    return np.concatenate([returns[1:], [0.0]])


@pytest.fixture
def data():
    return np.random.default_rng(0).normal(0.0, 0.01, size=(400, 1))


def test_a_past_only_signal_passes(data):
    report = lookahead_scan(_past_only, data)
    assert report.ok
    assert report.max_deviation == 0.0


def test_a_signal_that_peeks_is_caught(data):
    report = lookahead_scan(_peeks_one_day_ahead, data)
    assert not report.ok
    assert len(report.leaks) == report.probes


def test_a_signal_of_the_wrong_length_is_an_error(data):
    with pytest.raises(ValueError):
        lookahead_scan(lambda frame: np.zeros(3), data)


def _shrinking_width(close: np.ndarray) -> np.ndarray:
    """A signal that drops a column once it has seen enough rows, like a changing universe."""
    width = close.shape[1] if close.shape[0] > 100 else close.shape[1] - 1
    return np.zeros((close.shape[0], width))


def _collapsing_width(close: np.ndarray) -> np.ndarray:
    """Width collapses to 1 on a short frame, which numpy would happily broadcast."""
    width = close.shape[1] if close.shape[0] > 100 else 1
    return np.arange(close.shape[0] * width, dtype=float).reshape(close.shape[0], width)


def test_a_narrowing_signal_is_named_rather_than_broadcast():
    panel = np.cumprod(1 + np.full((120, 5), 0.001), axis=0) * 100.0
    with pytest.raises(leakage.SignalWidthChanged, match="wide on the truncated frame"):
        leakage.lookahead_scan(_shrinking_width, panel, probe_points=[50])


def test_a_collapsing_signal_is_not_reported_as_a_leak():
    """Before the shape check this reported 8 leaks out of 24 probes, all of them false."""
    panel = np.cumprod(1 + np.full((120, 5), 0.001), axis=0) * 100.0
    with pytest.raises(leakage.SignalWidthChanged, match="universes"):
        leakage.lookahead_scan(_collapsing_width, panel, probe_points=[50])


def test_a_stable_width_still_scans_normally():
    panel = np.cumprod(1 + np.full((120, 5), 0.001), axis=0) * 100.0
    report = leakage.lookahead_scan(lambda close: np.zeros_like(close), panel, probe_points=[50, 80])
    assert report.ok
    assert report.probes == 2
