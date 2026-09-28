"""Effective breadth: the number the fundamental law actually asks for.

The law multiplies skill by the square root of breadth, and a report that used
the ticker count would promise an improvement the correlation structure cannot
deliver. These tests pin the two ends -- perfectly correlated names count as one,
independent names count as themselves -- and the refusals, because an unmeasured
breadth read as the ticker count is the most flattering reading available.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.backtest.breadth import MIN_ROWS_PER_SYMBOL, correlation_spectrum, effective_bets, law_factor


def independent(n_rows: int = 3000, n_cols: int = 10, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_rows, n_cols))


def clones(n_rows: int = 3000, n_cols: int = 10, noise: float = 0.02, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    market = rng.normal(size=(n_rows, 1))
    return market + noise * rng.normal(size=(n_rows, n_cols))


def test_independent_columns_count_as_themselves():
    assert effective_bets(independent()) == pytest.approx(10.0, abs=0.2)


def test_near_perfect_clones_count_as_one_bet():
    """Nine sector slices of one index are not nine bets."""
    assert effective_bets(clones()) == pytest.approx(1.0, abs=0.1)


def test_more_tickers_of_the_same_thing_does_not_raise_the_count():
    """The claim that makes the measure worth having: the ticker count rises and
    the effective count does not."""
    five, twenty = effective_bets(clones(n_cols=5)), effective_bets(clones(n_cols=20))
    assert five is not None and twenty is not None
    assert twenty < 1.5 and five < 1.5


def test_a_mix_lands_between_its_two_halves():
    rng = np.random.default_rng(7)
    market = rng.normal(size=(3000, 1))
    mixed = np.hstack([market + 0.2 * rng.normal(size=(3000, 5)), rng.normal(size=(3000, 5))])
    bets = effective_bets(mixed)
    assert bets is not None
    assert 4.0 < bets < 8.0


def test_a_sample_too_short_for_the_matrix_is_unmeasured_not_the_column_count():
    """A correlation matrix estimated from fewer rows than columns says more about
    the sample than the market, and the flattering reading is the ticker count."""
    assert effective_bets(independent(n_rows=MIN_ROWS_PER_SYMBOL * 10 - 1, n_cols=10)) is None


def test_a_column_that_never_moves_is_a_refusal_rather_than_a_free_bet():
    rows = independent(n_cols=3)
    rows[:, 2] = 4.0
    assert effective_bets(rows) is None
    assert correlation_spectrum(rows) is None


def test_one_column_is_not_a_breadth_measurement():
    assert effective_bets(independent(n_cols=1)) is None


def test_a_non_finite_value_is_not_dropped_quietly():
    rows = independent(n_cols=3)
    rows[10, 1] = np.nan
    assert effective_bets(rows) is None


def test_the_spectrum_sums_to_the_column_count():
    """A correlation matrix has unit diagonal, so its trace is the column count.
    If this drifts, the entropy is being taken over something else."""
    spectrum = correlation_spectrum(independent(n_cols=6))
    assert spectrum is not None
    assert float(spectrum.sum()) == pytest.approx(6.0)
    assert np.all(np.diff(spectrum) <= 1e-9), "eigenvalues must come back descending"


def test_the_law_factor_is_the_square_root_of_the_ratio_of_effective_bets():
    assert law_factor(18.0, 2.0) == pytest.approx(3.0)
    assert law_factor(4.0, 4.0) == pytest.approx(1.0)


@pytest.mark.parametrize(("wide", "narrow"), [(None, 4.0), (4.0, None), (4.0, 0.0), (0.0, 4.0)])
def test_an_unmeasured_side_gives_no_factor(wide, narrow):
    """A factor computed against an assumed breadth is a promise about a number
    nobody measured."""
    assert law_factor(wide, narrow) is None
