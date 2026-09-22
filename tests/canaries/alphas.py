"""The four fake alphas. Each one must be rejected by a named gate.

Everything here is synthetic and seeded: the suite has to give the same verdict
on every CI run, or a canary failure would itself be noise. What is being
tested is not the data, it is whether the gates reject a strategy whose defect
is known in advance.
"""

from __future__ import annotations

import numpy as np

from core.backtest.gates import Submission
from core.backtest.leakage import LeakReport, lookahead_scan

TRADING_DAYS = 1260  # five years
IS_FRACTION = 0.6
N_FOLDS = 5
DAILY_VOL = 0.01


def _factor_returns(rng: np.random.Generator, n_obs: int, momentum_sharpe: float = 1.5) -> np.ndarray:
    """FF5 + momentum stand-ins. Momentum is column 5 and is the one that pays."""
    factors = rng.normal(0.0, DAILY_VOL, size=(n_obs, 6))
    factors[:, 5] += momentum_sharpe * DAILY_VOL / np.sqrt(252.0)
    return factors


def _demeaned(x: np.ndarray) -> np.ndarray:
    """Synthetic noise with its sample mean removed.

    A canary has to fail for the reason it was built to fail for. Leaving a
    sample mean in the noise gives it an accidental alpha and turns the verdict
    into a property of the seed.
    """
    return x - x.mean()


def _split(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    cut = int(len(returns) * IS_FRACTION)
    in_sample, out_of_sample = returns[:cut], returns[cut:]
    return in_sample, out_of_sample, list(np.array_split(out_of_sample, N_FOLDS))


def _clean_report() -> LeakReport:
    """G0 passed. Used by the canaries whose defect is not leakage."""
    return LeakReport(probes=24)


def canary_random(seed: int = 1) -> tuple[Submission, LeakReport]:
    """A pure coin flip. Nothing predicts anything, so G2 must refuse it."""
    rng = np.random.default_rng(seed)
    returns = _demeaned(rng.normal(0.0, DAILY_VOL, size=TRADING_DAYS))
    in_sample, out_of_sample, folds = _split(returns)
    trials = rng.normal(0.0, DAILY_VOL, size=(TRADING_DAYS, 20))
    trials[:, 0] = returns
    return (
        Submission(
            alpha_id="canary_random",
            in_sample=in_sample,
            out_of_sample=out_of_sample,
            fold_returns=folds,
            trial_returns=trials,
            factor_returns=_factor_returns(rng, TRADING_DAYS),
        ),
        _clean_report(),
    )


def leaky_signal(data: np.ndarray) -> np.ndarray:
    """Yesterday's return, with 1% of tomorrow's return mixed in.

    The honest term carries no edge and the leak is small enough that the
    equity curve looks ordinary, which is exactly why reading the code is not
    a control. What it buys is a daily signal-return correlation near 0.1 —
    an annualised Sharpe around 1.5, comfortably through every later gate.
    """
    returns = data[:, 0]
    yesterday = np.concatenate([[0.0], returns[:-1]])
    tomorrow = np.concatenate([returns[1:], [0.0]])
    return 0.06 * yesterday + 0.01 * tomorrow


def honest_signal(data: np.ndarray) -> np.ndarray:
    """The same signal without the leak. The scan must stay silent on this one."""
    returns = data[:, 0]
    return 0.06 * np.concatenate([[0.0], returns[:-1]])


def canary_lookahead(seed: int = 2) -> tuple[Submission, LeakReport]:
    """Tomorrow's return, mixed into today's signal.

    This canary is built to pass G2 through G6: it really does make money, and
    only the point-in-time recomputation in G0 can tell that the money is not
    there. If G0 ever goes quiet, nothing downstream will catch this.
    """
    rng = np.random.default_rng(seed)
    data = _demeaned(rng.normal(0.0, DAILY_VOL, size=TRADING_DAYS))[:, None]
    report = lookahead_scan(leaky_signal, data, seed=seed)

    signal = leaky_signal(data)
    tomorrow = np.concatenate([data[1:, 0], [0.0]])
    returns = np.sign(signal) * tomorrow
    in_sample, out_of_sample, folds = _split(returns)
    trials = rng.normal(0.0, DAILY_VOL, size=(TRADING_DAYS, 20))
    trials[:, 0] = returns
    return (
        Submission(
            alpha_id="canary_lookahead",
            in_sample=in_sample,
            out_of_sample=out_of_sample,
            fold_returns=folds,
            trial_returns=trials,
            factor_returns=_factor_returns(rng, TRADING_DAYS),
        ),
        report,
    )


def canary_overfit(seed: int = 3, n_trials: int = 1000) -> tuple[Submission, LeakReport]:
    """The best of a thousand noise configurations. G4 must reject it on PBO.

    This is the canary that matters most: the winner has a high in-sample
    Sharpe and passes G2 comfortably, exactly like a real overfitted alpha.
    """
    rng = np.random.default_rng(seed)
    trials = rng.normal(0.0, DAILY_VOL, size=(TRADING_DAYS, n_trials))
    cut = int(TRADING_DAYS * IS_FRACTION)
    in_sample_sharpes = trials[:cut].mean(axis=0) / trials[:cut].std(axis=0, ddof=1)
    winner = int(np.argmax(in_sample_sharpes))

    returns = trials[:, winner]
    in_sample, out_of_sample, folds = _split(returns)
    return (
        Submission(
            alpha_id="canary_overfit",
            in_sample=in_sample,
            out_of_sample=out_of_sample,
            fold_returns=folds,
            trial_returns=trials,
            factor_returns=_factor_returns(rng, TRADING_DAYS),
        ),
        _clean_report(),
    )


def canary_factor(seed: int = 4) -> tuple[Submission, LeakReport]:
    """A repackaged momentum factor. G4 must reject it on the residual alpha t.

    It earns money and passes G2 and G3, which is the trap: the returns are
    real, they are simply not the fund's to claim. Every other criterion in G4
    is satisfied, so the residual t is the only thing standing between this and
    a capital allocation.
    """
    rng = np.random.default_rng(seed)
    factors = _factor_returns(rng, TRADING_DAYS)
    momentum = factors[:, 5]
    idiosyncratic = _demeaned(rng.normal(0.0, DAILY_VOL * 0.15, size=TRADING_DAYS))
    returns = momentum + idiosyncratic

    in_sample, out_of_sample, folds = _split(returns)
    # The grid varies how much idiosyncratic noise each configuration carries,
    # so its Sharpe ordering is the same in and out of sample. A grid whose
    # members all share one Sharpe would make the in-sample winner a coin flip
    # and PBO would reject this alpha for the wrong reason.
    noise_scale = np.linspace(2.0, 0.1, 20)
    grid_noise = _demeaned(rng.normal(0.0, DAILY_VOL, size=TRADING_DAYS))
    trials = momentum[:, None] + grid_noise[:, None] * noise_scale[None, :]
    return (
        Submission(
            alpha_id="canary_factor",
            in_sample=in_sample,
            out_of_sample=out_of_sample,
            fold_returns=folds,
            trial_returns=trials,
            factor_returns=factors,
        ),
        _clean_report(),
    )


CANARIES = {
    "canary_random": canary_random,
    "canary_lookahead": canary_lookahead,
    "canary_overfit": canary_overfit,
    "canary_factor": canary_factor,
}
