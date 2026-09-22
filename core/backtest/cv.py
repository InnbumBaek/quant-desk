"""Cross-validation that does not leak.

A plain K-fold on a time series trains on the future and on observations whose
labels overlap the test window, which is how a backtest reports a Sharpe it
cannot repeat. Splits here purge the overlap and embargo the bars that follow
a test block (Lopez de Prado, Advances in Financial Machine Learning, ch. 7).
"""

from __future__ import annotations

import numpy as np

Split = tuple[np.ndarray, np.ndarray]


def purged_kfold(
    n_samples: int,
    n_splits: int = 5,
    label_horizon: int = 1,
    embargo_pct: float = 0.01,
) -> list[Split]:
    """Contiguous test blocks, with overlapping and adjacent train bars removed.

    `label_horizon` is how many bars the label spans; `embargo_pct` is the share
    of the sample dropped after each test block to break serial correlation that
    purging alone leaves behind.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_samples < n_splits * (label_horizon + 1):
        raise ValueError("sample too short for the requested splits and horizon")

    indices = np.arange(n_samples)
    embargo = int(round(n_samples * embargo_pct))
    splits: list[Split] = []

    for block in np.array_split(indices, n_splits):
        test_start, test_end = int(block[0]), int(block[-1])
        # Purge: a train bar whose label window reaches into the test block.
        purge_start = test_start - label_horizon
        # Embargo: bars right after the test block stay out of training.
        embargo_end = test_end + label_horizon + embargo
        train = indices[(indices < purge_start) | (indices > embargo_end)]
        splits.append((train, block))

    return splits


def walk_forward(n_samples: int, n_splits: int = 5, min_train: int | None = None) -> list[Split]:
    """Expanding-window splits: train only on what preceded the test block."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    blocks = np.array_split(np.arange(n_samples), n_splits + 1)
    floor = min_train if min_train is not None else len(blocks[0])
    splits: list[Split] = []
    for i in range(1, len(blocks)):
        train = np.concatenate(blocks[:i])
        if train.size < floor:
            continue
        splits.append((train, blocks[i]))
    return splits


def fold_sign_stability(fold_returns: list[np.ndarray]) -> float:
    """Share of out-of-sample folds with a positive mean.

    G3 reads this because a strategy that earns everything in one fold is a
    single lucky episode wearing a track record.
    """
    if not fold_returns:
        return 0.0
    positive = sum(1 for fold in fold_returns if float(np.mean(fold)) > 0)
    return positive / len(fold_returns)
