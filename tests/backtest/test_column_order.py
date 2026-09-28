"""A verdict must not depend on the order the panel's columns happen to be in.

ADR-0041 made a submission run on the universe its declaration named, which also
changed the panel's column order from alphabetical to declared. Re-running the
six existing alphas on the same snapshot afterwards reproduced every return
series to the last bit -- and moved the ensemble's G6 capacity number by 3.2e-3,
because `adv_participation` decided what counted as a trade with `> 0` on a
float sum. The population a percentile is taken over was therefore chosen by
rounding (ADR-0044).

These tests pin the property, not the incident: permuting the columns of a panel
is a relabelling, so every number a gate reads must come back unchanged. Only the
ensemble ever failed it, because only blending rescales a row that did not trade.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import strategies
from core.backtest.engine import DUST_DOLLARS, PricePanel, adv_participation, simulate

ROWS, COLS = 1200, 5
PERMUTATION = (3, 1, 0, 4, 2)


def panel(seed: int = 0, order: tuple[int, ...] | None = None) -> PricePanel:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(ROWS, COLS)), axis=0))
    volume = 1e9 * np.exp(rng.normal(0, 0.3, size=(ROWS, COLS)))
    symbols = np.array(["AAA", "BBB", "CCC", "DDD", "EEE"])
    if order is not None:
        columns = list(order)
        close, volume, symbols = close[:, columns], volume[:, columns], symbols[columns]
    return PricePanel(
        dates=tuple(np.arange("2010-01-01", ROWS, dtype="datetime64[D]").astype(str)),
        symbols=tuple(symbols),
        close=close,
        dollar_volume=volume,
    )


def participation(order: tuple[int, ...] | None, name: str) -> float:
    board = panel(order=order)
    weights = strategies.get(name).build()(board.close)
    return adv_participation(board, simulate(board, weights, cost_bps=5.0).traded_notional)


@pytest.mark.parametrize("name", strategies.names(available_only=True))
def test_capacity_does_not_depend_on_the_column_order(name: str):
    assert participation(None, name) == pytest.approx(participation(PERMUTATION, name), rel=1e-12)


@pytest.mark.parametrize("name", strategies.names(available_only=True))
def test_weights_do_not_depend_on_the_column_order(name: str):
    strategy = strategies.get(name)
    inverse = np.argsort(PERMUTATION)
    plain, shuffled = panel(), panel(order=PERMUTATION)
    for params in strategy.search_grid():
        straight = strategy.fn(plain.close, params)
        relabelled = strategy.fn(shuffled.close, params)[:, inverse]
        assert straight == pytest.approx(relabelled, abs=1e-12)


def test_a_trade_under_a_cent_is_not_a_trade():
    board = panel()
    notional = np.zeros((ROWS - 1, COLS))
    notional[0, 0] = 1_000_000.0
    dusted = notional.copy()
    dusted[1:, :] = DUST_DOLLARS / 2.0

    assert adv_participation(board, dusted) == pytest.approx(adv_participation(board, notional))


def test_a_trade_of_exactly_a_cent_is_a_trade():
    """The boundary is inclusive, so the smallest notional a broker prints counts."""
    board = panel()
    notional = np.zeros((ROWS - 1, COLS))
    notional[0, 0] = DUST_DOLLARS

    assert adv_participation(board, notional) > 0.0


def test_dust_only_ever_made_the_capacity_number_look_better():
    """The direction matters: the threshold must not be a way to pass G6.

    Dropping the smallest trades raises a high percentile, so the measured
    participation goes up and the gate gets harder -- the opposite of the change
    CLAUDE.md forbids, which is moving a bar to let something through.
    """
    board = panel()
    weights = strategies.get("multi_signal").build()(board.close)
    notional = simulate(board, weights, cost_bps=5.0).traded_notional
    shares = notional / np.asarray(board.dollar_volume, dtype=float)[:-1]

    counting_dust = float(np.percentile(shares[notional > 0], 95.0))
    dropping_dust = adv_participation(board, notional)

    assert (notional > 0).sum() > (notional >= DUST_DOLLARS).sum()
    assert dropping_dust >= counting_dust


def test_the_ensemble_is_the_one_that_produces_dust():
    """Why only `blend-001` moved: a single signal holds a position exactly.

    `ts_momentum` and `trend_breakout` repeat the same float from bar to bar, so
    an untraded cell subtracts to exactly zero. Blending rescales every row, so
    the same intent comes out one bit apart and reads as a trade.
    """
    board = panel()
    dust = {}
    for name in ("multi_signal", "trend_breakout", "ts_momentum"):
        weights = strategies.get(name).build()(board.close)
        notional = simulate(board, weights, cost_bps=5.0).traded_notional
        dust[name] = int(np.sum((notional > 0) & (notional < DUST_DOLLARS)))

    assert dust["multi_signal"] > 0
    assert dust["trend_breakout"] == 0
    assert dust["ts_momentum"] == 0
