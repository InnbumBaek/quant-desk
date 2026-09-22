"""Declaring a symbol's market, and measuring a calendar instead of asserting one."""

from __future__ import annotations

import numpy as np
import pytest

from core.data.markets import (
    MARKETS,
    UNIVERSE,
    UnknownSymbol,
    currencies,
    group_by_market,
    market_for,
    sessions_per_year,
)


def business_days(start: str, end: str, holidays: tuple[str, ...] = ()) -> np.ndarray:
    """Weekdays between two dates, minus the given holidays."""
    dates = np.arange(np.datetime64(start), np.datetime64(end), dtype="datetime64[D]")
    weekdays = dates[np.is_busday(dates)]
    if holidays:
        drop = np.array(holidays, dtype="datetime64[D]")
        weekdays = weekdays[~np.isin(weekdays, drop)]
    return weekdays


# --- declaring a market ------------------------------------------------------


def test_a_declared_symbol_resolves_to_its_market():
    assert market_for("SPY").code == "US"
    assert market_for("005930").code == "KR"


def test_an_undeclared_symbol_raises_rather_than_being_guessed():
    with pytest.raises(UnknownSymbol, match="not in the universe"):
        market_for("NVDA")


def test_a_six_digit_ticker_is_not_assumed_korean():
    """The shape of a ticker is not evidence. A guess here picks a currency."""
    with pytest.raises(UnknownSymbol):
        market_for("123456")


def test_a_symbol_declared_in_a_market_that_does_not_exist_raises():
    with pytest.raises(ValueError, match="not in MARKETS"):
        market_for("XXX", {"XXX": "JP"})


def test_every_universe_entry_names_a_known_market():
    unknown = {symbol: code for symbol, code in UNIVERSE.items() if code not in MARKETS}
    assert unknown == {}


def test_the_two_markets_do_not_share_a_currency():
    """The mixed-currency refusal in align_panels depends on this being true."""
    assert MARKETS["US"].currency != MARKETS["KR"].currency
    assert currencies(["US", "KR"]) == {"US": "USD", "KR": "KRW"}


def test_an_unknown_market_code_raises():
    with pytest.raises(ValueError, match="unknown market"):
        currencies(["JP"])


# --- grouping ----------------------------------------------------------------


def test_grouping_splits_the_two_markets():
    groups = group_by_market(["005930", "SPY", "QQQ", "000660"])
    assert groups == {"KR": ("000660", "005930"), "US": ("QQQ", "SPY")}


def test_grouping_is_order_independent():
    """The grouping feeds a snapshot id, so the same set must give the same output."""
    assert group_by_market(["SPY", "TLT", "005930"]) == group_by_market(["005930", "TLT", "SPY"])


def test_grouping_refuses_the_whole_set_for_one_undeclared_symbol():
    with pytest.raises(UnknownSymbol):
        group_by_market(["SPY", "NVDA"])


# --- measuring the calendar --------------------------------------------------


def test_a_us_year_measures_about_252_sessions():
    """2024 had 252 US sessions by the exchange calendar; the estimator has to land there."""
    dates = business_days(
        "2024-01-01",
        "2025-01-01",
        holidays=(
            "2024-01-01",
            "2024-01-15",
            "2024-02-19",
            "2024-03-29",
            "2024-05-27",
            "2024-06-19",
            "2024-07-04",
            "2024-09-02",
            "2024-11-28",
            "2024-12-25",
        ),
    )
    assert dates.shape[0] == 252
    assert sessions_per_year(dates) == pytest.approx(252.0, abs=1.5)


def test_a_korean_year_measures_fewer_sessions_than_a_us_one():
    """Korea keeps more market holidays, which is the whole reason the panels split."""
    us = business_days("2024-01-01", "2025-01-01", holidays=("2024-01-01", "2024-01-15"))
    kr = business_days(
        "2024-01-01",
        "2025-01-01",
        holidays=("2024-01-01", "2024-02-09", "2024-02-12", "2024-03-01", "2024-05-01"),
    )
    assert sessions_per_year(kr) < sessions_per_year(us)


def test_the_estimator_counts_intervals_not_dates():
    """Two dates a year apart are one session interval, not two sessions a year."""
    dates = np.array(["2024-01-02", "2025-01-02"], dtype="datetime64[D]")
    assert sessions_per_year(dates) == pytest.approx(1.0, abs=0.01)


def test_a_sample_under_a_year_refuses_to_annualise():
    dates = business_days("2024-01-01", "2024-04-01")
    with pytest.raises(ValueError, match="under the 350 required"):
        sessions_per_year(dates)


def test_an_explicit_shorter_span_is_allowed():
    dates = business_days("2024-01-01", "2024-04-01")
    assert sessions_per_year(dates, min_span_days=60) > 200


def test_unsorted_dates_raise():
    dates = np.array(["2024-01-03", "2024-01-02", "2025-06-01"], dtype="datetime64[D]")
    with pytest.raises(ValueError, match="not sorted"):
        sessions_per_year(dates)


def test_repeated_dates_raise():
    dates = np.array(["2024-01-02", "2024-01-02", "2025-06-01"], dtype="datetime64[D]")
    with pytest.raises(ValueError, match="dates repeat"):
        sessions_per_year(dates)


def test_one_date_is_not_a_rate():
    with pytest.raises(ValueError, match="at least two dates"):
        sessions_per_year(np.array(["2024-01-02"], dtype="datetime64[D]"))


def test_a_two_dimensional_array_is_refused():
    with pytest.raises(ValueError, match="one-dimensional"):
        sessions_per_year(np.zeros((3, 2), dtype="datetime64[D]"))
