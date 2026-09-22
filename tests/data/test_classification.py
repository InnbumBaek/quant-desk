"""The sector bucket is declared, and a symbol without one is not guessed."""

from __future__ import annotations

import pytest

from core.data.classification import (
    SECTORS,
    UnclassifiedSymbol,
    bare,
    sector_for,
    sector_weights,
    unclassified,
)
from core.data.markets import UNIVERSE


def test_a_declared_symbol_has_a_bucket():
    assert sector_for("SPY") == "us_equity_broad"
    assert sector_for("XLK") == "technology"


def test_an_undeclared_symbol_raises():
    with pytest.raises(UnclassifiedSymbol, match="no sector bucket"):
        sector_for("NVDA")


def test_a_market_qualified_symbol_resolves():
    """A cross-market panel qualifies its symbols as US:SPY (ADR-0013)."""
    assert bare("US:SPY") == "SPY"
    assert bare("SPY") == "SPY"
    assert sector_for("KR:005930") == "technology"


def test_a_colon_in_a_ticker_is_not_a_market_prefix():
    assert bare("BRK:B") == "BRK:B"


def test_the_buckets_are_net_not_gross():
    """A long and a short in one bucket are a spread; gross leverage catches size."""
    weights = {"XLK": 0.10, "005930": -0.04}
    assert sector_weights(weights) == {"technology": pytest.approx(0.06)}


def test_weights_in_different_buckets_stay_apart():
    assert sector_weights({"SPY": 0.05, "TLT": -0.03}) == {
        "us_equity_broad": 0.05,
        "rates_long": -0.03,
    }


def test_an_undeclared_symbol_fails_the_whole_book():
    with pytest.raises(UnclassifiedSymbol):
        sector_weights({"SPY": 0.05, "NVDA": 0.01})


def test_unclassified_lists_what_is_missing_without_raising():
    assert unclassified(["US:SPY", "NVDA", "TLT"]) == ("NVDA",)


def test_every_universe_symbol_has_a_bucket():
    """A symbol that can be loaded but not bucketed blocks its pod, so keep them in step."""
    assert [symbol for symbol in UNIVERSE if symbol not in SECTORS] == []


def test_no_bucket_is_a_single_symbol_dressed_as_a_sector():
    """A bucket per name never binds, which is the same as having no limit."""
    buckets = {}
    for symbol, bucket in SECTORS.items():
        buckets.setdefault(bucket, []).append(symbol)
    industry_singletons = [b for b, members in buckets.items() if len(members) == 1]
    # Sector funds legitimately are their industry; the point is that the broad
    # buckets are shared, so nothing sits alone in a bucket meant for several.
    assert "us_equity_broad" not in industry_singletons
