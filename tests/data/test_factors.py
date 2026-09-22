"""The factor matrix has to line up with the bars it is regressed against."""

from __future__ import annotations

import numpy as np
import pytest

from core.backtest.engine import PricePanel
from core.backtest.stats import residual_alpha_tstat
from core.data.factors import FRENCH_MARKET, FactorPanel, load_factors, trim_panel_to_factors

DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]


def panel_of(closes: list[list[float]], days: list[str] | None = None) -> PricePanel:
    days = days or DAYS[: len(closes)]
    return PricePanel(
        dates=np.array(days, dtype="datetime64[D]"),
        symbols=tuple(f"S{i}" for i in range(len(closes[0]))),
        close=np.array(closes, dtype=float),
        dollar_volume=None,
    )


def factors_of(days: list[str], columns: dict[str, list[float]]) -> FactorPanel:
    return FactorPanel(
        dates=np.array(days, dtype="datetime64[D]"),
        names=tuple(columns),
        values=np.array([columns[name] for name in columns], dtype=float).T,
        source="test",
        digest="0" * 64,
    )


def market_panel() -> PricePanel:
    closes = [[100.0, 50.0], [101.0, 49.0], [103.0, 50.0], [102.0, 52.0], [104.0, 51.0], [103.0, 53.0]]
    return panel_of(closes)


# --- the alignment ----------------------------------------------------------


def test_a_bars_factor_row_is_the_day_the_bar_closed():
    """`bar_returns[t]` runs from dates[t] to dates[t+1], so its factor is dates[t+1]."""
    panel = market_panel()
    market = panel.bar_returns.mean(axis=1)
    # The factor equals the panel's own market return, dated by the closing day.
    factors = factors_of(DAYS, {"MKT": [np.nan, *market.tolist()]})

    aligned = factors.align_to_bars(panel.dates)
    assert aligned.shape == (panel.dates.shape[0] - 1, 1)
    assert np.allclose(aligned[:, 0], market)


def long_market(bars: int = 300):
    """A panel long enough for a regression, plus the day-dated market factor."""
    rng = np.random.default_rng(11)
    days = np.arange(np.datetime64("2022-01-03"), np.datetime64("2026-01-01"), dtype="datetime64[D]")
    days = days[np.is_busday(days)][: bars + 1]
    steps = rng.normal(0.0002, 0.01, (bars, 2))
    closes = 100.0 * np.cumprod(np.vstack([np.ones((1, 2)), 1.0 + steps]), axis=0)
    panel = PricePanel(
        dates=days,
        symbols=("A", "B"),
        close=closes,
        dollar_volume=None,
    )
    market = panel.bar_returns.mean(axis=1)
    return panel, market, rng


def slope_and_intercept_t(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(y.size), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[1]), residual_alpha_tstat(y, x)


def test_the_off_by_one_alignment_regresses_against_the_wrong_day():
    """Both shifts produce a plausible number, which is why this needs a test.

    A strategy that *is* the market should show a beta of one against the market
    factor. Shifted a day, the same data shows no relationship at all -- and a
    t-statistic either way, with nothing to say which day was compared.
    """
    panel, market, rng = long_market()
    strategy = market + rng.normal(0.0, 0.002, market.size)

    right = factors_of([str(d) for d in panel.dates], {"MKT": [np.nan, *market.tolist()]})
    beta, tstat = slope_and_intercept_t(strategy, right.align_to_bars(panel.dates))
    assert 0.9 < beta < 1.1, "the market factor explains a strategy that is the market"
    assert abs(tstat) < 3.0, "and leaves no residual alpha to speak of"

    # The same returns dated a day early. 0.0 fills the tail only so that the
    # wrong alignment produces a matrix at all instead of hitting the hole check.
    wrong = factors_of([str(d) for d in panel.dates], {"MKT": [*market.tolist(), 0.0]})
    shifted_beta, _ = slope_and_intercept_t(strategy, wrong.align_to_bars(panel.dates))
    assert abs(shifted_beta) < 0.3, "yesterday's market explains today's return no better than nothing"


# --- the holes --------------------------------------------------------------


def test_a_bar_date_the_factor_file_does_not_cover_raises():
    panel = market_panel()
    short = factors_of(DAYS[:4], {"MKT": [0.001] * 4})
    with pytest.raises(ValueError, match="have no factor row"):
        short.align_to_bars(panel.dates)


def test_the_error_says_how_many_dates_are_missing_and_which(tmp_path):
    panel = market_panel()
    short = factors_of(DAYS[:4], {"MKT": [0.001] * 4})
    with pytest.raises(ValueError, match="2 of 5 bar dates"):
        short.align_to_bars(panel.dates)


def test_a_vendor_hole_inside_the_range_is_not_a_zero():
    panel = market_panel()
    holed = factors_of(DAYS, {"MKT": [0.001, 0.001, np.nan, 0.001, 0.001, 0.001]})
    with pytest.raises(ValueError, match="is not a zero"):
        holed.align_to_bars(panel.dates)


def test_covers_answers_without_raising():
    panel = market_panel()
    assert factors_of(DAYS, {"MKT": [0.001] * 6}).covers(panel.dates)
    assert not factors_of(DAYS[:4], {"MKT": [0.001] * 4}).covers(panel.dates)


def test_a_panel_of_one_date_has_no_bar():
    factors = factors_of(DAYS, {"MKT": [0.001] * 6})
    with pytest.raises(ValueError, match="at least two dates"):
        factors.align_to_bars(np.array(["2024-01-02"], dtype="datetime64[D]"))


# --- the shape of the panel -------------------------------------------------


def test_values_must_match_the_dates_and_names():
    with pytest.raises(ValueError, match="do not match"):
        FactorPanel(
            dates=np.array(DAYS[:3], dtype="datetime64[D]"),
            names=("MKT", "SMB"),
            values=np.zeros((3, 1)),
            source="test",
            digest="x",
        )


def test_dropping_the_risk_free_rate_leaves_the_factors():
    factors = factors_of(DAYS, {"MKT": [0.001] * 6, "RF": [0.0002] * 6})
    kept = factors.drop(("RF",))
    assert kept.names == ("MKT",)
    assert kept.values.shape == (6, 1)
    assert kept.digest == factors.digest, "the file is the same file; dropping a column is a view of it"


def test_dropping_everything_raises():
    factors = factors_of(DAYS, {"MKT": [0.001] * 6})
    with pytest.raises(ValueError, match="would leave no factors"):
        factors.drop(("MKT",))


# --- reading the file -------------------------------------------------------


def write_csv(path, body: str):
    path.write_text(body, encoding="utf-8")
    return path


def test_the_normalised_csv_round_trips(tmp_path):
    path = write_csv(
        tmp_path / "ff.csv",
        "Date,Mkt-RF,SMB,RF\n"
        "2024-01-02,0.00110000,-0.00020000,0.00002000\n"
        "2024-01-03,-0.00050000,0.00010000,0.00002000\n",
    )
    factors = load_factors(path)
    assert factors.names == ("Mkt-RF", "SMB", "RF")
    assert factors.values[0].tolist() == [0.0011, -0.0002, 0.00002]
    assert str(factors.dates[1]) == "2024-01-03"
    assert len(factors.digest) == 64


def test_an_empty_field_becomes_a_hole_not_a_zero(tmp_path):
    path = write_csv(tmp_path / "ff.csv", "Date,MOM\n2024-01-02,0.00110000\n2024-01-03,\n")
    factors = load_factors(path)
    assert np.isnan(factors.values[1, 0])


def test_a_file_without_a_date_column_is_refused(tmp_path):
    path = write_csv(tmp_path / "ff.csv", "day,MOM\n2024-01-02,0.001\n2024-01-03,0.002\n")
    with pytest.raises(ValueError, match="does not start with a Date column"):
        load_factors(path)


def test_dates_out_of_order_are_refused(tmp_path):
    path = write_csv(tmp_path / "ff.csv", "Date,MOM\n2024-01-03,0.001\n2024-01-02,0.002\n")
    with pytest.raises(ValueError, match="ascending date order"):
        load_factors(path)


def test_a_short_row_is_refused(tmp_path):
    path = write_csv(tmp_path / "ff.csv", "Date,MOM,SMB\n2024-01-02,0.001\n2024-01-03,0.002,0.003\n")
    with pytest.raises(ValueError, match="values for 2"):
        load_factors(path)


def test_a_file_with_no_history_is_refused(tmp_path):
    path = write_csv(tmp_path / "ff.csv", "Date,MOM\n2024-01-02,0.001\n")
    with pytest.raises(ValueError, match="needs a history"):
        load_factors(path)


# --- trimming to what the vendor has published ------------------------------


def test_full_coverage_returns_the_same_panel():
    panel = market_panel()
    factors = factors_of(DAYS, {"MKT": [0.001] * 6})
    trimmed, dropped = trim_panel_to_factors(panel, factors)
    assert dropped == 0
    assert trimmed is panel


def test_a_factor_file_that_lags_the_panel_trims_the_tail():
    """French publishes with a lag, so a panel fetched today runs past the file."""
    panel = market_panel()
    factors = factors_of(DAYS[:4], {"MKT": [0.001] * 4})

    trimmed, dropped = trim_panel_to_factors(panel, factors)
    assert dropped == 2
    assert [str(d) for d in trimmed.dates] == DAYS[:4]
    assert trimmed.close.shape == (4, 2)
    # And the trimmed panel is one the factor file can actually price.
    assert factors.align_to_bars(trimmed.dates).shape == (3, 1)


def test_a_factor_file_that_starts_late_trims_the_head():
    panel = market_panel()
    factors = factors_of(DAYS[2:], {"MKT": [0.001] * 4})
    trimmed, dropped = trim_panel_to_factors(panel, factors)
    assert dropped == 2
    assert [str(d) for d in trimmed.dates] == DAYS[2:]


def test_dollar_volume_is_trimmed_with_the_closes():
    panel = panel_of([[100.0, 50.0]] * 6)
    panel = PricePanel(
        dates=panel.dates, symbols=panel.symbols, close=panel.close, dollar_volume=panel.close * 10.0
    )
    factors = factors_of(DAYS[:4], {"MKT": [0.001] * 4})
    trimmed, _ = trim_panel_to_factors(panel, factors)
    assert trimmed.dollar_volume.shape == trimmed.close.shape


def test_a_hole_in_the_middle_is_not_trimmed_around():
    panel = market_panel()
    holed = factors_of(DAYS, {"MKT": [0.001, 0.001, np.nan, 0.001, 0.001, 0.001]})
    with pytest.raises(ValueError, match="a hole in the file, not a lag"):
        trim_panel_to_factors(panel, holed)


def test_a_file_that_barely_overlaps_is_refused():
    panel = market_panel()
    factors = factors_of(DAYS[:2], {"MKT": [0.001, 0.001]})
    with pytest.raises(ValueError, match="too few to run"):
        trim_panel_to_factors(panel, factors)


def test_the_french_market_is_named_so_other_markets_keep_the_proxy():
    assert FRENCH_MARKET == "US"
