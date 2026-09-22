"""Purging and the embargo are the difference between a backtest and a rehearsal."""

import numpy as np
import pytest

from core.backtest.cv import fold_sign_stability, purged_kfold, walk_forward


def test_train_and_test_never_overlap():
    for train, test in purged_kfold(1000, n_splits=5):
        assert not set(train.tolist()) & set(test.tolist())


def test_bars_adjacent_to_the_test_block_are_removed():
    label_horizon, embargo_pct, n_samples = 5, 0.02, 1000
    embargo = int(round(n_samples * embargo_pct))
    for train, test in purged_kfold(n_samples, 5, label_horizon, embargo_pct):
        start, end = int(test[0]), int(test[-1])
        assert not any(start - label_horizon <= i <= end + label_horizon + embargo for i in train)


def test_every_bar_is_tested_exactly_once():
    tested = np.concatenate([test for _, test in purged_kfold(1000, n_splits=5)])
    assert sorted(tested.tolist()) == list(range(1000))


def test_walk_forward_never_trains_on_the_future():
    for train, test in walk_forward(1000, n_splits=5):
        assert train.max() < test.min()


def test_too_few_splits_is_an_error():
    with pytest.raises(ValueError):
        purged_kfold(1000, n_splits=1)


def test_fold_sign_stability_counts_profitable_folds():
    folds = [np.array([0.01, 0.02]), np.array([-0.01]), np.array([0.005]), np.array([0.003])]
    assert fold_sign_stability(folds) == 0.75
    assert fold_sign_stability([]) == 0.0
