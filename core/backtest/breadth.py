"""How many independent bets a universe actually offers.

Grinold & Kahn's fundamental law says the information ratio is roughly the
information coefficient times the square root of breadth, which is the argument
for widening a universe: five instruments to twenty-seven would be a factor of
2.3 at the same skill. The argument is only as good as the word *breadth*, and
breadth in the law is the number of **independent** bets, not the number of
tickers. Nine sector SPDRs are slices of one index; adding all nine to SPY does
not add nine bets, and a report that multiplied by the square root of the ticker
count would promise an improvement the correlation structure cannot deliver.

So the number reported here is Meucci's effective number of bets: the entropy of
the eigenvalue spectrum of the correlation matrix, exponentiated.

    p_i = lambda_i / sum(lambda)        ENB = exp(-sum p_i * ln p_i)

It is the number of equally-weighted independent series whose spectrum would be
as spread out as this one. For a set of perfectly correlated names it tends to 1;
for genuinely independent names it equals the count. Meucci (2009), 'Managing
Diversification', Risk 22(5).

**What this is not.** It is a property of the covariance structure, so it bounds
the law's breadth from above rather than predicting an information ratio: a
universe of twenty independent names still earns nothing at zero skill, and a
correlation matrix measured on one sample is itself an estimate. It also says
nothing about capacity, cost or whether a signal exists on the new names. So this
belongs in a report that informs which universe to declare, never in a gate.

**Why it is measured rather than assumed.** The same lesson as ADR-0033: the
direction of an effect is a thing to measure, not to reason about. Sector ETFs
could plausibly carry a great deal of independent variation (they are different
industries) or almost none (they are all the same market). Only the spectrum
says which.
"""

from __future__ import annotations

import numpy as np

#: Below this many rows a correlation matrix on more than a couple of columns is
#: mostly estimation error, and its spectrum says more about the sample than the
#: market. Reported as unmeasured rather than as a number.
MIN_ROWS_PER_SYMBOL = 10


def correlation_spectrum(returns: np.ndarray) -> np.ndarray | None:
    """Eigenvalues of the correlation matrix, descending, or None when unusable."""
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or matrix.shape[0] < 3:
        return None
    if not np.all(np.isfinite(matrix)):
        return None
    deviation = matrix.std(axis=0, ddof=1)
    if np.any(deviation <= 0):
        # A column that never moves has no correlation with anything. Dropping it
        # silently would raise the effective count by pretending it is
        # independent, which is the flattering direction.
        return None
    correlation = np.corrcoef(matrix, rowvar=False)
    if not np.all(np.isfinite(correlation)):
        return None
    values = np.linalg.eigvalsh(correlation)
    return np.sort(np.clip(values, 0.0, None))[::-1]


def effective_bets(returns: np.ndarray) -> float | None:
    """Meucci's effective number of bets, or None when it cannot be measured.

    None rather than the column count: an unmeasured breadth read as the ticker
    count is the most flattering reading available, and the whole point of the
    number is that the two differ.
    """
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim == 2 and matrix.shape[0] < MIN_ROWS_PER_SYMBOL * matrix.shape[1]:
        return None
    spectrum = correlation_spectrum(matrix)
    if spectrum is None:
        return None
    total = float(spectrum.sum())
    if total <= 0:
        return None
    weights = spectrum / total
    positive = weights[weights > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return float(np.exp(entropy))


def law_factor(wide: float | None, narrow: float | None) -> float | None:
    """The fundamental law's multiplier on IR from going wide, at equal skill.

    `sqrt(breadth_wide / breadth_narrow)` on **effective** bets. None when either
    side is unmeasured, because a factor computed against an assumed breadth is a
    promise about a number nobody measured.
    """
    if wide is None or narrow is None or narrow <= 0 or wide <= 0:
        return None
    return float(np.sqrt(wide / narrow))
