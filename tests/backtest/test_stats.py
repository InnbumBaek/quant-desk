"""The statistics have to be wrong-proof before the gates can rely on them."""

import numpy as np
import pytest

from core.backtest import stats


def test_sharpe_of_a_flat_series_is_zero():
    assert stats.sharpe_ratio(np.zeros(100)) == 0.0


def test_deflation_punishes_the_number_of_trials():
    """Same track record, more trials tried: the deflated Sharpe must fall."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, size=1000)
    few = rng.normal(0.0, 0.02, size=5)
    many = rng.normal(0.0, 0.02, size=5000)
    assert stats.deflated_sharpe_excess(returns, many) < stats.deflated_sharpe_excess(returns, few)


def test_best_of_many_noise_trials_does_not_survive_deflation():
    rng = np.random.default_rng(1)
    trials = rng.normal(0.0, 0.01, size=(1000, 500))
    sharpes = np.array([stats.sharpe_ratio(trials[:, j], 1) for j in range(trials.shape[1])])
    winner = trials[:, int(np.argmax(sharpes))]
    assert stats.deflated_sharpe_excess(winner, sharpes) <= 0.0


def test_pbo_of_pure_noise_is_near_a_coin_flip():
    """With no real signal, picking the best backtest is worth nothing."""
    rng = np.random.default_rng(2)
    noise = rng.normal(0.0, 0.01, size=(800, 40))
    assert 0.3 < stats.probability_of_backtest_overfitting(noise) < 0.7


def test_pbo_is_low_when_one_configuration_is_genuinely_best():
    rng = np.random.default_rng(3)
    trials = rng.normal(0.0, 0.01, size=(1500, 20))
    trials[:, 7] += 0.0015  # a real edge, not a lucky draw
    assert stats.probability_of_backtest_overfitting(trials) < 0.05


def test_pbo_rejects_an_odd_number_of_splits():
    with pytest.raises(ValueError):
        stats.probability_of_backtest_overfitting(np.zeros((100, 4)), n_splits=5)


def test_residual_alpha_of_a_factor_clone_is_indistinguishable_from_zero():
    rng = np.random.default_rng(4)
    factor = rng.normal(0.0004, 0.01, size=2000)
    clone = factor + (rng.normal(0.0, 0.002, size=2000) - 0.0)
    assert abs(stats.residual_alpha_tstat(clone, factor[:, None])) < 3.0


def test_residual_alpha_survives_when_the_edge_is_not_the_factor():
    rng = np.random.default_rng(5)
    factor = rng.normal(0.0004, 0.01, size=2000)
    independent = rng.normal(0.0008, 0.005, size=2000)
    assert stats.residual_alpha_tstat(independent, factor[:, None]) > 3.0


def test_bootstrap_pvalue_is_large_for_a_zero_mean_series():
    rng = np.random.default_rng(6)
    assert stats.block_bootstrap_pvalue(rng.normal(0.0, 0.01, size=500)) > 0.05


def test_drawdown_quantiles_are_ordered():
    rng = np.random.default_rng(7)
    quantiles = stats.drawdown_quantiles(rng.normal(0.0002, 0.01, size=1000))
    assert quantiles["p80"] <= quantiles["p95"] <= quantiles["p99"]
    assert quantiles["p99"] <= stats.max_drawdown(rng.normal(0.0002, 0.01, size=1000)) + 1.0
