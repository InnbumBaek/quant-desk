"""G0 recomputes the signal point-in-time. Reading the code is not a control."""

import numpy as np
import pytest

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
