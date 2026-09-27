"""Amihud illiquidity, and the capacity it implies.

Most of these tests are refusals. The failure this module is built against is a
capacity estimate that grows the less we know: drop the name nobody could
measure, charge it nothing, and the book looks cheaper the more illiquid it is.
So every path where an input is missing must return None with a reason, and the
book-level cost must refuse outright rather than sum the measurable part.

Nothing here asserts a number against a threshold, because there is no
threshold: `core/risk/capacity.py` enforces the participation cap and this module
reports a second opinion (ADR-0031).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.data.sources import PricePanel
from core.execution.impact import (
    DEFAULT_WINDOW,
    MIN_OBSERVATIONS,
    ROUND_TRIP_CROSSINGS,
    CapacityGap,
    amihud_lambda,
    book_cost_fraction,
    capacity_gap,
    cost_capacity,
    panel_illiquidity,
)


def panel(
    rows: int = 80,
    symbols: tuple[str, ...] = ("AAA", "BBB"),
    volume: float | None = 1e7,
    step: float = 0.01,
) -> PricePanel:
    """A panel with a constant-magnitude daily move, so lambda is exactly solvable."""
    dates = np.datetime64("2026-01-01", "D") + np.arange(rows)
    # Alternating +step/-step keeps |return| close to `step` and the level flat.
    swing = np.where(np.arange(rows) % 2 == 0, 1.0 + step, 1.0)
    close = np.outer(swing, np.ones(len(symbols))) * 100.0
    dv = None if volume is None else np.full((rows, len(symbols)), volume)
    return PricePanel(dates=dates, symbols=symbols, close=close, dollar_volume=dv)


# --- the measure itself -------------------------------------------------------


def test_lambda_is_the_mean_of_absolute_return_over_dollar_volume():
    """Not a fitted constant: the arithmetic is recomputed here from the panel."""
    p = panel()
    measure = amihud_lambda(p, "AAA")
    assert measure.measured
    close = p.close[-(DEFAULT_WINDOW + 1) :, 0]
    returns = np.diff(close) / close[:-1]
    expected = float(np.mean(np.abs(returns) / 1e7))
    assert measure.lam == pytest.approx(expected)


def test_lambda_reads_as_a_price_move_per_dollar():
    """`lam * dollars` is a fraction, which is what a capacity question needs."""
    measure = amihud_lambda(panel(), "AAA")
    assert measure.lam is not None
    move = measure.lam * 1e6
    assert 0.0 < move < 1.0, "a million dollars should not move a name by 100%"


def test_a_thinner_name_is_more_illiquid_for_the_same_returns():
    thick = amihud_lambda(panel(volume=1e8), "AAA")
    thin = amihud_lambda(panel(volume=1e6), "AAA")
    assert thick.lam is not None and thin.lam is not None
    assert thin.lam > thick.lam


def test_the_window_is_the_last_sessions_and_not_the_whole_history():
    p = panel(rows=400)
    measure = amihud_lambda(p, "AAA", window=30)
    assert measure.observations == 30


def test_the_median_dollar_volume_comes_back_so_the_lambda_is_interpretable():
    measure = amihud_lambda(panel(volume=5e6), "AAA")
    assert measure.median_dollar_volume == pytest.approx(5e6)


# --- refusals -----------------------------------------------------------------


def test_a_panel_without_dollar_volume_measures_nothing():
    measure = amihud_lambda(panel(volume=None), "AAA")
    assert not measure.measured
    assert "no dollar volume" in measure.unmeasured


def test_a_symbol_the_panel_does_not_carry_is_named_not_guessed():
    measure = amihud_lambda(panel(), "ZZZ")
    assert not measure.measured
    assert "ZZZ" in measure.unmeasured


def test_too_few_sessions_is_a_refusal_and_reports_how_few():
    """A mean of five heavy-tailed ratios is noise with a number attached."""
    measure = amihud_lambda(panel(rows=10), "AAA")
    assert not measure.measured
    assert measure.observations < MIN_OBSERVATIONS
    assert str(measure.observations) in measure.unmeasured


def test_a_window_of_one_session_is_rejected_outright():
    with pytest.raises(ValueError, match="no return"):
        amihud_lambda(panel(), "AAA", window=1)


def test_a_flat_price_has_no_measurable_impact():
    """Zero return over the window is not zero impact; it is no observation of it."""
    rows = 80
    dates = np.datetime64("2026-01-01", "D") + np.arange(rows)
    p = PricePanel(
        dates=dates,
        symbols=("AAA",),
        close=np.full((rows, 1), 100.0),
        dollar_volume=np.full((rows, 1), 1e7),
    )
    measure = amihud_lambda(p, "AAA")
    assert not measure.measured
    assert "not a price move per dollar" in measure.unmeasured


def test_an_unmeasured_symbol_charges_nothing_and_says_so():
    measure = amihud_lambda(panel(volume=None), "AAA")
    assert measure.cost_fraction(1e6) is None


def test_every_column_appears_in_the_panel_measure_measured_or_not():
    """One entry per column, always: a symbol missing from this map would be a
    name with no cost rather than a name with an unknown cost."""
    measures = panel_illiquidity(panel(volume=None))
    assert set(measures) == {"AAA", "BBB"}
    assert not any(m.measured for m in measures.values())


# --- the book ----------------------------------------------------------------


def test_a_round_trip_is_charged_twice():
    measure = amihud_lambda(panel(), "AAA")
    one = measure.cost_fraction(1e6, crossings=1)
    both = measure.cost_fraction(1e6, crossings=ROUND_TRIP_CROSSINGS)
    assert one is not None and both is not None
    assert both == pytest.approx(2 * one)


def test_cost_is_linear_in_size_and_that_is_stated_not_hidden():
    """Amihud's lambda is linear by construction, so double the size doubles the
    charge. Empirical impact is concave, which makes this an upper bound."""
    measure = amihud_lambda(panel(), "AAA")
    small = measure.cost_fraction(1e6)
    large = measure.cost_fraction(2e6)
    assert small is not None and large is not None
    assert large == pytest.approx(2 * small)


def test_the_book_cost_is_a_fraction_of_capital():
    weights = np.array([0.5, 0.5])
    fraction, unmeasured = book_cost_fraction(weights, panel(), capital=1e6)
    assert unmeasured == ()
    assert fraction is not None and 0.0 < fraction < 1.0


def test_one_unmeasurable_holding_refuses_the_whole_book():
    """The illiquid name is exactly the expensive one. Dropping it would make the
    book look cheaper the less we know about it."""
    p = panel(rows=80, symbols=("AAA", "BBB"))
    thin = np.asarray(p.dollar_volume, dtype=float).copy()
    flat = np.asarray(p.close, dtype=float).copy()
    flat[:, 1] = 100.0  # BBB never moves, so it gets no lambda
    p = PricePanel(dates=p.dates, symbols=p.symbols, close=flat, dollar_volume=thin)
    fraction, unmeasured = book_cost_fraction(np.array([0.5, 0.5]), p, capital=1e6)
    assert fraction is None
    assert unmeasured == ("BBB",)


def test_a_name_that_is_not_held_does_not_block_the_book():
    """Only holdings need a cost. A zero weight pays nothing by arithmetic."""
    p = panel(rows=80, symbols=("AAA", "BBB"))
    flat = np.asarray(p.close, dtype=float).copy()
    flat[:, 1] = 100.0
    p = PricePanel(dates=p.dates, symbols=p.symbols, close=flat, dollar_volume=p.dollar_volume)
    fraction, unmeasured = book_cost_fraction(np.array([1.0, 0.0]), p, capital=1e6)
    assert unmeasured == ()
    assert fraction is not None


def test_a_weight_vector_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="one weight per panel symbol"):
        book_cost_fraction(np.array([1.0]), panel(), capital=1e6)


def test_a_book_with_no_capital_is_rejected():
    with pytest.raises(ValueError, match="not a book size"):
        book_cost_fraction(np.array([0.5, 0.5]), panel(), capital=0.0)


# --- cost-based capacity ------------------------------------------------------


def test_capacity_is_where_cost_reaches_the_share_of_the_edge():
    weights = np.array([0.5, 0.5])
    capital, why = cost_capacity(weights, panel(), gross_edge=0.02, cost_share_max=0.25)
    assert why == "" and capital is not None
    # At that capital, the book's cost should be a quarter of the edge.
    fraction, _ = book_cost_fraction(weights, panel(), capital=capital)
    assert fraction is not None
    assert fraction == pytest.approx(0.02 * 0.25, rel=1e-6)


def test_a_bigger_edge_buys_more_capacity():
    weights = np.array([0.5, 0.5])
    small, _ = cost_capacity(weights, panel(), gross_edge=0.01, cost_share_max=0.25)
    large, _ = cost_capacity(weights, panel(), gross_edge=0.04, cost_share_max=0.25)
    assert small is not None and large is not None
    assert large > small


def test_no_edge_means_no_capacity_rather_than_infinite_capacity():
    capital, why = cost_capacity(np.array([0.5, 0.5]), panel(), gross_edge=0.0, cost_share_max=0.25)
    assert capital is None
    assert "no room for any cost" in why


def test_the_cost_share_has_no_default_because_no_limit_declares_one():
    """Inventing one here would be a capital limit nobody approved."""
    with pytest.raises(TypeError):
        cost_capacity(np.array([0.5, 0.5]), panel(), gross_edge=0.02)  # type: ignore[call-arg]


def test_a_share_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match="not a share of the edge"):
        cost_capacity(np.array([0.5, 0.5]), panel(), gross_edge=0.02, cost_share_max=1.5)


def test_an_unmeasurable_book_has_no_cost_capacity_and_names_the_symbol():
    capital, why = cost_capacity(
        np.array([0.5, 0.5]), panel(volume=None), gross_edge=0.02, cost_share_max=0.25
    )
    assert capital is None
    assert "AAA" in why


# --- the gap, which is a finding and never a breach ---------------------------


def test_the_gap_reports_both_estimates_and_which_one_binds():
    gap = capacity_gap(
        participation_capacity=5e9,
        weights=np.array([0.5, 0.5]),
        panel=panel(),
        gross_edge=0.02,
        cost_share_max=0.25,
    )
    assert gap.cost_capacity is not None
    assert gap.binding == min(gap.participation_capacity, gap.cost_capacity)
    assert gap.findings


def test_a_smaller_cost_capacity_says_the_enforced_cap_is_the_forgiving_one():
    gap = capacity_gap(
        participation_capacity=1e12,
        weights=np.array([0.5, 0.5]),
        panel=panel(volume=1e5),
        gross_edge=0.001,
        cost_share_max=0.1,
    )
    assert gap.cost_capacity is not None and gap.cost_capacity < 1e12
    assert any("more forgiving estimate" in line for line in gap.findings)


def test_the_linearity_caveat_travels_with_every_finding():
    """A reader who sees the number must see what it assumes."""
    gap = capacity_gap(
        participation_capacity=5e9,
        weights=np.array([0.5, 0.5]),
        panel=panel(),
        gross_edge=0.02,
        cost_share_max=0.25,
    )
    assert any("upper bound" in line for line in gap.findings)


def test_a_missing_edge_leaves_the_cost_side_unmeasured_not_substituted():
    """Repeating the participation number as a cost answer would turn a missing
    measurement into a confirmation."""
    gap = capacity_gap(
        participation_capacity=5e9,
        weights=np.array([0.5, 0.5]),
        panel=panel(),
        gross_edge=None,
        cost_share_max=0.25,
    )
    assert gap.cost_capacity is None
    assert "gross edge" in gap.unmeasured
    assert gap.findings == ()
    assert gap.binding == 5e9, "the participation estimate is still the one enforced"


def test_a_missing_cost_share_is_also_unmeasured():
    gap = capacity_gap(
        participation_capacity=5e9,
        weights=np.array([0.5, 0.5]),
        panel=panel(),
        gross_edge=0.02,
        cost_share_max=None,
    )
    assert gap.cost_capacity is None
    assert "cost share" in gap.unmeasured


def test_both_estimates_missing_leaves_nothing_binding():
    gap = capacity_gap(
        participation_capacity=None,
        weights=np.array([0.5, 0.5]),
        panel=panel(volume=None),
        gross_edge=0.02,
        cost_share_max=0.25,
    )
    assert gap.binding is None


def test_the_gap_never_produces_a_breach():
    """`core/risk/capacity.py` enforces. This is a second opinion, not a second
    limit, and a Breach here would be a limit nobody approved (ADR-0031)."""
    gap = capacity_gap(
        participation_capacity=1e9,
        weights=np.array([0.5, 0.5]),
        panel=panel(volume=1e5),
        gross_edge=0.001,
        cost_share_max=0.1,
    )
    assert not hasattr(gap, "breaches")
    assert all(isinstance(line, str) for line in gap.findings)
    assert isinstance(gap, CapacityGap)


def test_the_two_conservatisms_are_documented_where_the_number_is_produced():
    """A factor of two that nobody wrote down is a factor of two somebody will
    later mistake for a fill forecast."""
    from core.execution.impact import Illiquidity

    doc = Illiquidity.cost_fraction.__doc__ or ""
    assert "terminal" in doc and "concave" in doc
    assert "not a fill forecast" in doc


def test_a_flat_book_returns_none_and_not_a_free_round_trip():
    """Arithmetically a flat book is free, and that is true and useless: printed
    beside a census that could cost nothing, a 0.0 reads as "trading this book is
    free". Found by a test whose fixture netted to flat (ADR-0031)."""
    fraction, unmeasured = book_cost_fraction(np.array([0.0, 0.0]), panel(), capital=1e6)
    assert fraction is None
    assert unmeasured == (), "an empty tuple is how a caller tells this from unmeasurable"


def test_the_two_reasons_for_no_cost_are_distinguishable():
    p = panel(rows=80, symbols=("AAA", "BBB"))
    flat = np.asarray(p.close, dtype=float).copy()
    flat[:, 1] = 100.0
    unmeasurable = PricePanel(dates=p.dates, symbols=p.symbols, close=flat, dollar_volume=p.dollar_volume)

    _none_held, held_symbols = book_cost_fraction(np.array([0.0, 0.0]), p, capital=1e6)
    _none_costed, costed_symbols = book_cost_fraction(np.array([0.5, 0.5]), unmeasurable, capital=1e6)
    assert held_symbols == () and costed_symbols == ("BBB",)


def test_a_flat_book_has_no_cost_capacity_and_says_which_reason():
    capital, why = cost_capacity(np.array([0.0, 0.0]), panel(), gross_edge=0.02, cost_share_max=0.25)
    assert capital is None
    assert "holds nothing" in why
