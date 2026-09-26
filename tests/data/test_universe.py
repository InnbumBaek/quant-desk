"""Point-in-time membership, halts, delistings and the illiquid tail.

The through-line: a universe of every listed name only works if the code can say
who was listed *then*, and refuses to pretend when it cannot.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from core.backtest.engine import PricePanel
from core.data.classification import UnclassifiedSymbol, sector_weights
from core.data.markets import group_by_market
from core.data.universe import (
    DEFAULT_LIQUIDITY_WINDOW,
    Halt,
    Listing,
    ListingStatus,
    SurvivorshipBias,
    Universe,
    UnknownListing,
    as_date,
    floor_from_limits,
    liquidity_screen,
    load_universe,
    median_dollar_volume,
    seed_universe,
    stale_price_runs,
    survivorship_adjustment,
    survivorship_gap,
    tradable_set,
    write_universe,
)

AS_OF = date(2026, 9, 1)


def listing(symbol: str, **overrides) -> Listing:
    fields = {
        "symbol": symbol,
        "market": "US",
        "sector": "us_equity_broad",
        "listed_on": "2020-01-02",
    }
    fields.update(overrides)
    return Listing(**fields)


def pit_universe(*listings: Listing, as_of: date = AS_OF) -> Universe:
    return Universe(listings=listings, as_of=as_of, source="test-pit", point_in_time=True)


def panel(
    symbols: tuple[str, ...] = ("AAA", "BBB"),
    rows: int = 40,
    volume: float | dict[str, float] | None = 50_000_000.0,
    flat: tuple[str, ...] = (),
) -> PricePanel:
    days = np.arange(np.datetime64("2026-07-01"), np.datetime64("2027-01-01"), dtype="datetime64[D]")
    dates = days[np.is_busday(days)][:rows]
    rng = np.random.default_rng(11)
    steps = rng.normal(0.0, 0.01, (rows - 1, len(symbols)))
    close = 100.0 * np.cumprod(np.vstack([np.ones((1, len(symbols))), 1.0 + steps]), axis=0)
    for symbol in flat:
        close[-10:, symbols.index(symbol)] = close[-11, symbols.index(symbol)]

    dollar_volume = None
    if volume is not None:
        dollar_volume = np.full_like(close, 0.0)
        for index, symbol in enumerate(symbols):
            per_symbol = volume if isinstance(volume, float) else volume[symbol]
            dollar_volume[:, index] = per_symbol
    return PricePanel(dates=dates, symbols=symbols, close=close, dollar_volume=dollar_volume)


# --- a listing's life -------------------------------------------------------


def test_a_date_arrives_as_a_string_a_date_or_a_numpy_day():
    assert as_date("2024-05-06") == date(2024, 5, 6)
    assert as_date(date(2024, 5, 6)) == date(2024, 5, 6)
    assert as_date(np.datetime64("2024-05-06")) == date(2024, 5, 6)


def test_anything_else_is_not_a_date():
    with pytest.raises(TypeError, match="not a date"):
        as_date(20240506)


def test_before_its_listing_date_a_symbol_does_not_exist():
    assert listing("AAA").status_on("2019-12-31") is ListingStatus.PRE_LISTING


def test_on_its_listing_date_it_trades():
    assert listing("AAA").status_on("2020-01-02") is ListingStatus.LISTED


def test_the_delisting_date_is_the_first_day_it_no_longer_trades():
    name = listing("AAA", delisted_on="2024-06-03")
    assert name.status_on("2024-06-02") is ListingStatus.LISTED
    assert name.status_on("2024-06-03") is ListingStatus.DELISTED


def test_a_halt_covers_its_start_and_not_its_end():
    """`end` is the session trading resumed, so a one-day halt is start..start+1."""
    name = listing("AAA", halts=(Halt(start="2023-03-01", end="2023-03-06"),))
    assert name.status_on("2023-02-28") is ListingStatus.LISTED
    assert name.status_on("2023-03-01") is ListingStatus.HALTED
    assert name.status_on("2023-03-05") is ListingStatus.HALTED
    assert name.status_on("2023-03-06") is ListingStatus.LISTED


def test_a_halt_with_no_end_is_still_running():
    name = listing("AAA", halts=(Halt(start="2026-05-01"),))
    assert name.status_on("2030-01-01") is ListingStatus.HALTED


def test_a_listing_with_no_date_refuses_to_guess_its_past():
    with pytest.raises(ValueError, match="no listing date"):
        listing("AAA", listed_on=None).status_on("2019-01-01")


def test_a_delisting_before_the_listing_is_refused():
    with pytest.raises(ValueError, match="before it listed"):
        listing("AAA", delisted_on="2019-01-01")


def test_a_halt_outside_the_listed_life_is_refused():
    with pytest.raises(ValueError, match="before it listed"):
        listing("AAA", halts=(Halt(start="2019-06-01"),))
    with pytest.raises(ValueError, match="after it delisted"):
        listing("AAA", delisted_on="2021-01-04", halts=(Halt(start="2022-01-03"),))


def test_a_halt_that_ends_before_it_starts_is_refused():
    with pytest.raises(ValueError, match="cannot end"):
        Halt(start="2023-03-05", end="2023-03-01")


def test_an_undeclared_market_is_refused():
    with pytest.raises(ValueError, match="known markets"):
        listing("AAA", market="JP")


def test_a_listing_without_a_sector_loads_and_says_it_is_unclassified():
    """One new ticker with no bucket must not stop the whole desk from loading."""
    assert listing("AAA", sector="").classified is False
    assert listing("AAA", sector=None).sector is None
    assert listing("AAA").classified is True


# --- the universe -----------------------------------------------------------


def test_a_symbol_cannot_appear_twice():
    with pytest.raises(ValueError, match="twice: AAA"):
        pit_universe(listing("AAA"), listing("AAA", listed_on="2021-01-04"))


def test_a_point_in_time_universe_needs_a_listing_date_for_every_name():
    with pytest.raises(ValueError, match="needs a listing date"):
        pit_universe(listing("AAA"), listing("BBB", listed_on=None))


def test_a_halted_name_is_a_member_but_not_tradable():
    """It is still a position: it cannot be traded and it cannot be marked away."""
    universe = pit_universe(listing("AAA"), listing("BBB", halts=(Halt(start="2026-08-01"),)))
    assert universe.members("2026-08-15") == ("AAA", "BBB")
    assert universe.tradable("2026-08-15") == ("AAA",)


def test_a_delisted_name_is_neither():
    universe = pit_universe(listing("AAA"), listing("BBB", delisted_on="2025-04-01"))
    assert universe.members("2026-01-05") == ("AAA",)
    assert universe.tradable("2026-01-05") == ("AAA",)


def test_a_name_that_had_not_listed_yet_is_absent_from_the_past():
    universe = pit_universe(listing("AAA"), listing("BBB", listed_on="2024-03-01"))
    assert universe.members("2023-01-03") == ("AAA",)
    assert universe.members("2024-03-01") == ("AAA", "BBB")


def test_a_source_without_delisting_history_refuses_the_past():
    universe = Universe((listing("AAA", listed_on=None),), as_of=AS_OF, source="live-list")
    with pytest.raises(SurvivorshipBias, match="carries no delisting history"):
        universe.members("2019-01-02")


def test_the_bias_can_be_taken_on_deliberately():
    universe = Universe((listing("AAA", listed_on=None),), as_of=AS_OF, source="live-list")
    assert universe.members("2019-01-02", allow_survivorship_bias=True) == ("AAA",)


def test_a_non_point_in_time_source_still_answers_for_today():
    """As-of and later need no history: the file says who is listed now."""
    universe = Universe((listing("AAA", listed_on=None),), as_of=AS_OF, source="live-list")
    assert universe.members(AS_OF) == ("AAA",)
    assert universe.members("2027-01-04") == ("AAA",)


def test_the_market_map_is_what_group_by_market_takes():
    universe = pit_universe(listing("AAA"), listing("005930", market="KR", sector="technology"))
    assert group_by_market(universe.symbols, universe.market_map()) == {
        "KR": ("005930",),
        "US": ("AAA",),
    }


def test_the_sector_map_is_what_sector_weights_takes():
    universe = pit_universe(listing("AAA"), listing("BBB", sector="energy"))
    weights = {"AAA": 0.03, "BBB": -0.01}
    assert sector_weights(weights, universe.sector_map()) == {
        "us_equity_broad": pytest.approx(0.03),
        "energy": pytest.approx(-0.01),
    }


def test_the_states_add_up_to_the_whole_universe():
    universe = pit_universe(
        listing("AAA"),
        listing("BBB", delisted_on="2025-04-01"),
        listing("CCC", listed_on="2027-01-04"),
        listing("DDD", halts=(Halt(start="2026-08-01"),)),
    )
    counts = universe.counts("2026-08-15")
    assert counts == {"pre_listing": 1, "listed": 1, "halted": 1, "delisted": 1}
    assert sum(counts.values()) == len(universe)


def test_the_delistings_inside_a_window_are_listed():
    universe = pit_universe(
        listing("AAA", delisted_on="2023-05-02"),
        listing("BBB", delisted_on="2025-05-02"),
        listing("CCC"),
    )
    assert universe.delisted_between("2023-01-01", "2024-01-01") == ("AAA",)


def test_an_unclassified_name_is_left_out_of_the_sector_map_rather_than_bucketed():
    """`sector_weights` then raises on a book holding it, and the limit engine blocks."""
    universe = pit_universe(listing("AAA"), listing("BBB", sector=None))
    assert universe.unclassified == ("BBB",)
    assert universe.sector_map() == {"AAA": "us_equity_broad"}
    with pytest.raises(UnclassifiedSymbol):
        sector_weights({"AAA": 0.02, "BBB": 0.01}, universe.sector_map())


def test_an_unclassified_name_cannot_be_ordered_and_the_count_is_noted():
    universe = pit_universe(listing("AAA"), listing("BBB", sector=None))
    result = tradable_set(universe, panel(), min_dollar_volume=100_000.0, on="2026-08-15")
    assert result.kept == ("AAA",)
    assert "concentration limit cannot be checked" in result.excluded["BBB"]
    assert any("no sector bucket" in note for note in result.notes)


def test_a_symbol_the_universe_does_not_carry_raises():
    with pytest.raises(UnknownListing, match="not in the universe"):
        pit_universe(listing("AAA")).listing("ZZZ")


# --- the seed table ---------------------------------------------------------


def test_the_seed_universe_carries_the_declared_symbols():
    universe = seed_universe(AS_OF)
    assert "SPY" in universe.symbols
    assert "005930" in universe.symbols
    assert universe.listing("005930").market == "KR"
    assert universe.listing("XLK").sector == "technology"


def test_the_seed_universe_is_not_point_in_time_and_says_so():
    universe = seed_universe(AS_OF)
    assert universe.point_in_time is False
    with pytest.raises(SurvivorshipBias):
        universe.members("2015-01-02")


# --- the listings file ------------------------------------------------------


def test_a_universe_survives_a_round_trip(tmp_path):
    universe = pit_universe(
        listing("AAA", name="Alpha Corp"),
        listing("BBB", delisted_on="2025-04-01", halts=(Halt(start="2024-02-05", end="2024-02-12"),)),
    )
    path = write_universe(tmp_path / "listings.csv", universe)
    loaded = load_universe(path)

    assert loaded.symbols == universe.symbols
    assert loaded.as_of == AS_OF
    assert loaded.point_in_time is True
    assert loaded.listing("AAA").name == "Alpha Corp"
    assert loaded.listing("BBB").delisted_on == date(2025, 4, 1)
    assert loaded.listing("BBB").halts == (Halt(start=date(2024, 2, 5), end=date(2024, 2, 12)),)


def test_an_unclassified_name_survives_a_round_trip(tmp_path):
    """An empty sector column comes back as absent, not as a bucket named ''."""
    universe = pit_universe(listing("AAA", sector=None))
    loaded = load_universe(write_universe(tmp_path / "listings.csv", universe))
    assert loaded.listing("AAA").sector is None
    assert loaded.unclassified == ("AAA",)


def test_an_open_halt_survives_a_round_trip(tmp_path):
    universe = pit_universe(listing("AAA", halts=(Halt(start="2026-08-03"),)))
    loaded = load_universe(write_universe(tmp_path / "listings.csv", universe))
    assert loaded.listing("AAA").halts == (Halt(start=date(2026, 8, 3)),)
    assert loaded.listing("AAA").status_on("2026-08-20") is ListingStatus.HALTED


def test_a_listings_file_without_a_sidecar_cannot_be_used(tmp_path):
    path = tmp_path / "listings.csv"
    path.write_text("symbol,market,sector\nAAA,US,us_equity_broad\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no sidecar"):
        load_universe(path)


def test_a_sidecar_missing_the_point_in_time_flag_is_refused(tmp_path):
    universe = pit_universe(listing("AAA"))
    path = write_universe(tmp_path / "listings.csv", universe)
    meta = json.loads((tmp_path / "listings.source.json").read_text())
    del meta["point_in_time"]
    (tmp_path / "listings.source.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="point_in_time"):
        load_universe(path)


def test_a_point_in_time_flag_that_is_not_a_boolean_is_refused(tmp_path):
    """'yes' would be truthy, and a truthy string would silence the whole guard."""
    universe = pit_universe(listing("AAA"))
    path = write_universe(tmp_path / "listings.csv", universe)
    meta = json.loads((tmp_path / "listings.source.json").read_text())
    meta["point_in_time"] = "yes"
    (tmp_path / "listings.source.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="must be true or false"):
        load_universe(path)


def test_a_file_missing_a_required_column_is_refused(tmp_path):
    path = tmp_path / "listings.csv"
    path.write_text("symbol,sector\nAAA,us_equity_broad\n", encoding="utf-8")
    (tmp_path / "listings.source.json").write_text(
        json.dumps({"as_of": "2026-09-01", "source": "x", "point_in_time": False}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing column"):
        load_universe(path)


# --- the illiquid tail ------------------------------------------------------


def test_the_floor_comes_from_the_liquidation_limit():
    """0.5% of $1m, unwound over 3 days at 20% of ADV, needs $8,333 a day traded."""
    assert floor_from_limits(1_000_000.0, 0.005) == pytest.approx(0.005 * 1_000_000.0 / (0.20 * 3))


def test_a_bigger_minimum_position_needs_a_bigger_name():
    small = floor_from_limits(1_000_000.0, 0.001)
    large = floor_from_limits(1_000_000.0, 0.010)
    assert large == pytest.approx(10.0 * small)


def test_a_minimum_weight_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match="not a share of NAV"):
        floor_from_limits(1_000_000.0, 0.0)


def test_a_liquid_name_passes_and_a_thin_one_is_dropped_with_its_number():
    prices = panel(volume={"AAA": 50_000_000.0, "BBB": 1_000.0})
    screen = liquidity_screen(prices, min_dollar_volume=100_000.0)
    assert screen.kept == ("AAA",)
    assert "below the 100,000 floor" in screen.reason("BBB")


def test_a_panel_without_volume_drops_everything_rather_than_passing_it():
    """Unmeasured liquidity is not liquidity (ADR-0015)."""
    screen = liquidity_screen(panel(volume=None), min_dollar_volume=100_000.0)
    assert screen.kept == ()
    assert all("no dollar volume" in reason for reason in screen.excluded.values())


def test_the_screen_uses_the_median_so_one_block_trade_cannot_qualify_a_name():
    prices = panel(symbols=("AAA",), rows=40, volume=1_000.0)
    volumes = np.asarray(prices.dollar_volume, dtype=float).copy()
    volumes[-1, 0] = 10_000_000_000.0
    spiky = PricePanel(dates=prices.dates, symbols=prices.symbols, close=prices.close, dollar_volume=volumes)
    assert float(np.mean(volumes[-DEFAULT_LIQUIDITY_WINDOW:, 0])) > 100_000.0
    assert median_dollar_volume(spiky)["AAA"] == pytest.approx(1_000.0)
    assert liquidity_screen(spiky, min_dollar_volume=100_000.0).kept == ()


def test_a_symbol_outside_the_panel_is_reported_rather_than_skipped():
    screen = liquidity_screen(panel(), min_dollar_volume=100.0, symbols=["AAA", "ZZZ"])
    assert screen.reason("ZZZ") == "no price history in the panel"


# --- the halt proxy ---------------------------------------------------------


def test_a_price_that_has_not_moved_reads_as_a_halt():
    runs = stale_price_runs(panel(flat=("BBB",)), min_run=5)
    assert "AAA" not in runs
    assert runs["BBB"] >= 10


def test_a_short_flat_patch_is_not_a_halt():
    assert stale_price_runs(panel(flat=("BBB",)), min_run=20) == {}


def test_a_run_of_one_is_not_a_run():
    with pytest.raises(ValueError, match="not a run"):
        stale_price_runs(panel(), min_run=1)


# --- the order-eligible set -------------------------------------------------


def test_every_name_is_either_kept_or_has_a_reason():
    """A symbol that vanishes between the universe and the order list is the bug."""
    universe = pit_universe(
        listing("AAA"),
        listing("BBB", halts=(Halt(start="2026-08-01"),)),
        listing("CCC", delisted_on="2025-01-02"),
        listing("DDD"),
    )
    prices = panel(symbols=("AAA", "BBB", "DDD"), volume={"AAA": 5e7, "BBB": 5e7, "DDD": 1e3})
    result = tradable_set(universe, prices, min_dollar_volume=100_000.0, on="2026-08-15")

    assert set(result.kept) | set(result.excluded) == set(universe.symbols)
    assert set(result.kept) & set(result.excluded) == set()
    assert result.kept == ("AAA",)


def test_a_halted_name_is_reported_as_halted_not_as_illiquid():
    """The first reason wins, and the state of the listing outranks its volume."""
    universe = pit_universe(listing("AAA"), listing("BBB", halts=(Halt(start="2026-08-01"),)))
    prices = panel(volume={"AAA": 5e7, "BBB": 1.0})
    result = tradable_set(universe, prices, min_dollar_volume=100_000.0, on="2026-08-15")
    assert result.excluded["BBB"] == "halted"


def test_a_delisted_name_carries_its_state_as_the_reason():
    universe = pit_universe(listing("AAA"), listing("BBB", delisted_on="2025-01-02"))
    result = tradable_set(universe, panel(), min_dollar_volume=100_000.0, on="2026-08-15")
    assert result.excluded["BBB"] == "delisted"


def test_a_name_with_no_price_history_is_excluded_rather_than_ordered():
    universe = pit_universe(listing("AAA"), listing("ZZZ"))
    result = tradable_set(universe, panel(), min_dollar_volume=100_000.0, on="2026-08-15")
    assert result.excluded["ZZZ"] == "no price history in the panel"


def test_a_stale_price_excludes_a_name_the_calendar_still_calls_listed():
    """The listings file has no halt for it; the tape says otherwise."""
    universe = pit_universe(listing("AAA"), listing("BBB"))
    result = tradable_set(
        universe, panel(flat=("BBB",)), min_dollar_volume=100_000.0, on="2026-08-15", stale_run=5
    )
    assert "reads as a halt" in result.excluded["BBB"]
    assert result.kept == ("AAA",)


def test_the_survivorship_opt_in_is_recorded_on_the_result():
    universe = Universe(
        (listing("AAA", listed_on=None), listing("BBB", listed_on=None)),
        as_of=AS_OF,
        source="live-list",
    )
    result = tradable_set(
        universe,
        panel(),
        min_dollar_volume=100_000.0,
        on="2019-01-02",
        allow_survivorship_bias=True,
    )
    assert any("survivorship-biased" in note for note in result.notes)
    assert result.kept == ("AAA", "BBB")


def test_without_the_opt_in_the_past_dated_order_list_refuses():
    universe = Universe((listing("AAA", listed_on=None),), as_of=AS_OF, source="live-list")
    with pytest.raises(SurvivorshipBias):
        tradable_set(universe, panel(), min_dollar_volume=100_000.0, on="2019-01-02")


# --- survivorship -----------------------------------------------------------


def test_a_source_without_delistings_reports_the_gap_as_unknown_not_as_zero():
    report = survivorship_gap(seed_universe(AS_OF), "2015-01-02", "2026-09-01")
    assert report.delisted is None
    assert report.gap_pct is None
    assert report.ok is False
    assert any("unmeasured rather than zero" in note for note in report.notes)


def test_a_point_in_time_source_measures_the_gap():
    universe = pit_universe(
        listing("AAA"),
        listing("BBB"),
        listing("CCC", delisted_on="2023-06-01"),
    )
    report = survivorship_gap(universe, "2021-01-04", "2026-09-01")
    assert report.delisted == 1
    assert report.survivors == 2
    assert report.gap_pct == pytest.approx(100.0 / 3)
    assert report.ok is True


def test_a_window_that_ends_before_it_starts_is_refused():
    with pytest.raises(ValueError, match="ends"):
        survivorship_gap(pit_universe(listing("AAA")), "2026-01-01", "2025-01-01")


def test_the_health_check_passes_only_on_a_source_that_knows_who_left():
    window = ("2021-01-04", "2026-09-01")
    assert survivorship_adjustment(seed_universe(AS_OF), *window) is False
    assert survivorship_adjustment(pit_universe(listing("AAA")), *window) is True
