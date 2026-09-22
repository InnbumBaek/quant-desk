"""Statistics that decide G4.

Every function here returns a number the gate compares against
`core/risk/limits.yaml`. None of them takes a threshold as an argument: the
thresholds live in one place and the caller reads them from there, so a gate
can never be loosened by passing a different argument at the call site.

References
    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
    Bailey, Borwein, Lopez de Prado & Zhu (2017), "The probability of
    backtest overfitting", J. Computational Finance 20(4).
    Newey & West (1987) for the HAC standard error used on the residual alpha.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy import stats as sps

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe of an excess-return series."""
    returns = np.asarray(returns, dtype=float)
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(returns.mean() / sd * math.sqrt(periods_per_year))


def _expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """SR0 — the Sharpe a researcher expects from the best of N random trials.

    This is what makes the deflation work: with enough trials, a high Sharpe is
    the expected outcome of luck, not evidence of skill.
    """
    n_trials = trial_sharpes.size
    if n_trials < 2:
        return 0.0
    variance = float(np.var(trial_sharpes, ddof=1))
    if variance <= 0:
        return 0.0
    gamma = EULER_MASCHERONI
    quantile_a = sps.norm.ppf(1.0 - 1.0 / n_trials)
    quantile_b = sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(variance) * ((1.0 - gamma) * quantile_a + gamma * quantile_b)


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """Probability that the strategy's true Sharpe exceeds SR0.

    `returns` is the selected strategy's per-period series; `trial_sharpes` is
    the per-period Sharpe of every configuration that was tried, the selected
    one included. Dropping the discarded trials inflates the result, which is
    why G1 pre-registration counts N before any backtest runs.
    """
    returns = np.asarray(returns, dtype=float)
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    n_obs = returns.size
    if n_obs < 3:
        return 0.0

    observed = sharpe_ratio(returns, periods_per_year=1)
    threshold = _expected_max_sharpe(trial_sharpes)

    skew = float(sps.skew(returns, bias=False))
    kurtosis = float(sps.kurtosis(returns, fisher=False, bias=False))
    denominator = 1.0 - skew * observed + (kurtosis - 1.0) / 4.0 * observed**2
    if denominator <= 0:
        return 0.0

    z = (observed - threshold) * math.sqrt(n_obs - 1) / math.sqrt(denominator)
    return float(sps.norm.cdf(z))


def probability_of_backtest_overfitting(performance: np.ndarray, n_splits: int = 8) -> float:
    """PBO by combinatorially symmetric cross-validation.

    `performance` is observations x configurations: column j is the return
    series of trial j. The answer is the share of splits where the in-sample
    winner lands in the bottom half out of sample, i.e. the probability that
    picking the best backtest buys nothing.

    Block moments are precomputed so the cost is O(splits x configurations)
    rather than rescoring every series for each of the C(S, S/2) splits; a
    thousand-configuration grid is the ordinary case, not the exceptional one.
    """
    performance = np.asarray(performance, dtype=float)
    if performance.ndim != 2 or performance.shape[1] < 2:
        raise ValueError("performance must be observations x configurations, with >= 2 configurations")
    if n_splits % 2 or n_splits < 2:
        raise ValueError("n_splits must be even and >= 2")

    n_obs, n_trials = performance.shape
    blocks = np.array_split(np.arange(n_obs), n_splits)
    counts = np.array([b.size for b in blocks], dtype=float)
    sums = np.array([performance[b].sum(axis=0) for b in blocks])
    sums_sq = np.array([(performance[b] ** 2).sum(axis=0) for b in blocks])

    def sharpes(block_ids: tuple[int, ...]) -> np.ndarray:
        n = counts[list(block_ids)].sum()
        total = sums[list(block_ids)].sum(axis=0)
        total_sq = sums_sq[list(block_ids)].sum(axis=0)
        mean = total / n
        var = (total_sq - n * mean**2) / (n - 1)
        sd = np.sqrt(np.maximum(var, 0.0))
        return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)

    all_blocks = range(n_splits)
    below = 0
    total_splits = 0
    for train_blocks in combinations(all_blocks, n_splits // 2):
        test_blocks = tuple(b for b in all_blocks if b not in train_blocks)
        in_sample = sharpes(train_blocks)
        out_sample = sharpes(test_blocks)
        winner = int(np.argmax(in_sample))
        # Relative rank of the winner out of sample, in (0, 1).
        rank = float(sps.rankdata(out_sample)[winner]) / (n_trials + 1)
        rank = min(max(rank, 1e-9), 1 - 1e-9)
        below += math.log(rank / (1.0 - rank)) <= 0.0
        total_splits += 1

    return below / total_splits


def block_bootstrap_pvalue(
    returns: np.ndarray,
    block_size: int = 10,
    n_resamples: int = 2000,
    seed: int = 0,
) -> float:
    """One-sided p-value for a positive mean, under a circular block bootstrap.

    Blocks keep the autocorrelation an i.i.d. bootstrap would destroy, and the
    resamples are centred so they sample the null of zero mean.
    """
    returns = np.asarray(returns, dtype=float)
    n_obs = returns.size
    if n_obs < block_size * 2:
        raise ValueError("series too short for the requested block size")

    observed = float(returns.mean())
    centred = returns - observed
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n_obs / block_size)
    offsets = np.arange(block_size)

    starts = rng.integers(0, n_obs, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + offsets[None, None, :]) % n_obs
    means = centred[idx.reshape(n_resamples, -1)[:, :n_obs]].mean(axis=1)

    return float((np.sum(means >= observed) + 1) / (n_resamples + 1))


def _newey_west_lag(n_obs: int) -> int:
    return max(1, int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


def residual_alpha_tstat(returns: np.ndarray, factor_returns: np.ndarray) -> float:
    """t-statistic of the intercept from regressing returns on factors.

    The common failure is not overfitting but selling factor exposure as alpha,
    so what G4 reads is the intercept left after FF5 + momentum, with a
    Newey-West standard error because strategy residuals are autocorrelated.
    """
    y = np.asarray(returns, dtype=float)
    x = np.asarray(factor_returns, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.size != x.shape[0]:
        raise ValueError("returns and factor_returns must share their first dimension")

    design = np.column_stack([np.ones(y.size), x])
    n_obs, n_params = design.shape
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta

    xtx_inv = np.linalg.pinv(design.T @ design)
    lag = _newey_west_lag(n_obs)
    meat = (design * residuals[:, None]).T @ (design * residuals[:, None])
    for lag_k in range(1, lag + 1):
        weight = 1.0 - lag_k / (lag + 1.0)
        left = (design[lag_k:] * residuals[lag_k:, None]).T @ (design[:-lag_k] * residuals[:-lag_k, None])
        meat += weight * (left + left.T)
    cov = xtx_inv @ meat @ xtx_inv * n_obs / (n_obs - n_params)

    se = math.sqrt(max(cov[0, 0], 0.0))
    if se == 0:
        return 0.0
    return float(beta[0] / se)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of the compounded series, as a positive number."""
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def drawdown_quantiles(returns: np.ndarray, quantiles: tuple[int, ...] = (80, 95, 99)) -> dict[str, float]:
    """Drawdown distribution recorded at G3-G5 and compared against live DD.

    The live trigger needs a distribution, not a single worst case: a 5% loss
    means one thing when the backtest's 95th percentile is 4% and another when
    it is 12%.
    """
    returns = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    series = (peak - equity) / peak
    return {f"p{q}": float(np.percentile(series, q)) for q in quantiles}


def deflated_sharpe_excess(returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """Observed per-period Sharpe minus SR0, the Sharpe N trials buy for free.

    G4 enforces the deflated Sharpe *probability* (see `deflated_sharpe_ratio`);
    this excess is the same quantity before the normal CDF and is reported
    alongside it as a readable effect size (excess > 0 iff probability > 0.5).
    """
    returns = np.asarray(returns, dtype=float)
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    return sharpe_ratio(returns, periods_per_year=1) - _expected_max_sharpe(trial_sharpes)
