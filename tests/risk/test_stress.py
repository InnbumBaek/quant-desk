"""Historical scenarios, and the many ways a stress test can quietly pass.

Most of these tests are about absence. A stress suite is read as "we looked", so
every window that could not be evaluated has to be visible in the result, and a
scenario that contributes nothing must never look like a scenario that hurt
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.backtest.engine import PricePanel
from core.data.factors import FactorPanel
from core.risk.stress import (
    NOT_A_FACTOR,
    SCENARIOS,
    TAIL_QUANTILE,
    Scenario,
    cumulative_moves,
    factor_loss,
    ladder_reach,
    residual_band,
    risk_factors,
    run_scenario,
    stress_snapshot,
    stressed_adv_ratio,
    window_rows,
    worst_window,
)

LADDER = {
    "pod": {
        "drawdown": {
            "warn": {"abs": -0.025, "action": "report"},
            "cut": {"abs": -0.050, "action": "halve_capital"},
            "stop": {"abs": -0.075, "action": "stop_pod"},
        }
    }
}


def factor_panel(
    values: dict[str, list[float]],
    start: str = "2008-09-08",
    source: str = "test",
) -> FactorPanel:
    """A factor file with the columns and rows a test needs, on consecutive days."""
    names = tuple(values)
    rows = len(next(iter(values.values())))
    return FactorPanel(
        dates=np.arange(
            np.datetime64(start, "D"),
            np.datetime64(start, "D") + np.timedelta64(rows, "D"),
            dtype="datetime64[D]",
        ),
        names=names,
        values=np.column_stack([np.asarray(values[name], dtype=float) for name in names]),
        source=source,
        digest="0" * 64,
    )


def price_panel(rows: int = 40, volume: bool = True) -> PricePanel:
    rng = np.random.default_rng(11)
    symbols = ("AAA", "BBB")
    steps = rng.normal(0.0, 0.01, (rows - 1, len(symbols)))
    close = 100.0 * np.cumprod(np.vstack([np.ones((1, len(symbols))), 1.0 + steps]), axis=0)
    return PricePanel(
        dates=np.arange(
            np.datetime64("2024-01-01", "D"),
            np.datetime64("2024-01-01", "D") + np.timedelta64(rows, "D"),
            dtype="datetime64[D]",
        ),
        symbols=symbols,
        close=close,
        dollar_volume=np.full_like(close, 4_000_000.0) if volume else None,
    )


# --- the suite itself --------------------------------------------------------


def test_the_suite_carries_dates_and_no_magnitudes():
    """CLAUDE.md 2항: a typed crisis return is a number nobody can audit."""
    for scenario in SCENARIOS:
        assert isinstance(scenario.start, str) and isinstance(scenario.end, str)
        assert np.datetime64(scenario.start) <= np.datetime64(scenario.end)
        assert scenario.note


def test_every_scenario_key_is_unique():
    assert len({s.key for s in SCENARIOS}) == len(SCENARIOS)


def test_the_market_neutral_episode_is_in_the_suite():
    """August 2007 is the one crisis that hit cross-sectional books specifically."""
    assert "quant-quake-2007" in {s.key for s in SCENARIOS}


def test_the_risk_free_rate_is_not_a_risk_factor():
    """Shocking RF would add a financing return to a loss."""
    assert "RF" in NOT_A_FACTOR
    panel = factor_panel({"Mkt-RF": [0.0, 0.0], "RF": [0.0001, 0.0001]})
    assert risk_factors(panel) == ("Mkt-RF",)


# --- windows ----------------------------------------------------------------


def test_a_window_selects_both_end_days():
    panel = factor_panel({"Mkt-RF": [0.0] * 5})
    rows = window_rows(panel, "2008-09-08", "2008-09-12")
    assert rows.tolist() == [0, 1, 2, 3, 4]


def test_an_inverted_window_is_a_code_defect_not_a_gap():
    panel = factor_panel({"Mkt-RF": [0.0] * 5})
    with pytest.raises(ValueError, match="ends before it starts"):
        window_rows(panel, "2008-09-12", "2008-09-08")


def test_moves_are_compounded_not_summed():
    """A scenario is a position held for weeks; adding daily returns overstates a fall."""
    panel = factor_panel({"Mkt-RF": [-0.10, -0.10]})
    moves, days, why = cumulative_moves(panel, "2008-09-08", "2008-09-09")
    assert why == "" and days == 2
    assert moves["Mkt-RF"] == pytest.approx(-0.19)


def test_a_window_outside_the_file_is_unmeasured_not_flat():
    """The failure this module exists to prevent: a scenario that always passes."""
    panel = factor_panel({"Mkt-RF": [0.0] * 5})
    moves, days, why = cumulative_moves(panel, "1987-10-14", "1987-10-26")
    assert moves == {} and days == 0
    assert "no row in 1987-10-14..1987-10-26" in why
    assert "2008-09-08" in why and "2008-09-12" in why


def test_a_missing_value_inside_the_window_is_refused():
    panel = factor_panel({"Mkt-RF": [-0.01, float("nan"), -0.01]})
    moves, _days, why = cumulative_moves(panel, "2008-09-08", "2008-09-10")
    assert moves == {}
    assert "a missing factor return is not a zero" in why


def test_a_file_with_only_the_risk_free_rate_has_nothing_to_shock():
    panel = factor_panel({"RF": [0.0001] * 3})
    moves, _days, why = cumulative_moves(panel, "2008-09-08", "2008-09-10")
    assert moves == {} and "no risk factor" in why


# --- turning moves into a loss ----------------------------------------------


def test_the_loss_is_the_betas_against_the_moves():
    loss, why = factor_loss({"Mkt-RF": 0.5, "SMB": -0.2}, {"Mkt-RF": -0.20, "SMB": 0.10})
    assert why == ""
    assert loss == pytest.approx(-0.12)


def test_a_factor_with_no_beta_refuses_the_whole_scenario():
    """An unmeasured exposure is not a zero one, and dropping it is how a book passes."""
    loss, why = factor_loss({"Mkt-RF": 0.5}, {"Mkt-RF": -0.20, "SMB": -0.30})
    assert loss is None
    assert "no usable beta for SMB" in why
    assert "an absent beta is not a zero one" in why


def test_a_beta_that_is_not_a_number_is_not_a_beta():
    loss, why = factor_loss({"Mkt-RF": "0.5"}, {"Mkt-RF": -0.20})
    assert loss is None and "Mkt-RF" in why


def test_a_nan_beta_is_not_a_beta():
    loss, why = factor_loss({"Mkt-RF": float("nan")}, {"Mkt-RF": -0.20})
    assert loss is None and "Mkt-RF" in why


def test_no_moves_means_no_loss_rather_than_zero():
    loss, why = factor_loss({"Mkt-RF": 0.5}, {})
    assert loss is None and "no factor moves" in why


# --- the residual band ------------------------------------------------------


def test_the_residual_band_grows_with_the_root_of_time():
    assert residual_band(0.01, 4) == pytest.approx(0.02)


def test_an_unmeasured_residual_has_no_band():
    assert residual_band(None, 10) is None
    assert residual_band(float("nan"), 10) is None
    assert residual_band(True, 10) is None


def test_a_zero_length_window_has_no_band():
    assert residual_band(0.01, 0) is None


def test_the_band_widens_the_loss_rather_than_offsetting_it():
    """Reporting the factor loss alone reports the optimistic half."""
    panel = factor_panel({"Mkt-RF": [-0.10, -0.10]})
    result = run_scenario(
        Scenario("x", "2008-09-08", "2008-09-09", "note"),
        panel,
        {"Mkt-RF": 1.0},
        residual_daily_vol=0.02,
    )
    assert result.factor_loss == pytest.approx(-0.19)
    assert result.loss_with_residual < result.factor_loss


def test_a_gaining_scenario_is_also_widened_downward():
    panel = factor_panel({"Mkt-RF": [0.10]})
    result = run_scenario(
        Scenario("x", "2008-09-08", "2008-09-08", "note"),
        panel,
        {"Mkt-RF": 1.0},
        residual_daily_vol=0.02,
    )
    assert result.loss_with_residual == pytest.approx(0.10 - 0.02)


def test_a_scenario_without_a_residual_has_no_widened_loss():
    panel = factor_panel({"Mkt-RF": [-0.10]})
    result = run_scenario(Scenario("x", "2008-09-08", "2008-09-08", "n"), panel, {"Mkt-RF": 1.0})
    assert result.factor_loss == pytest.approx(-0.10) and result.loss_with_residual is None


def test_an_unmeasured_window_is_not_measured_even_with_a_loss_field():
    panel = factor_panel({"Mkt-RF": [-0.10]})
    result = run_scenario(Scenario("x", "1987-10-14", "1987-10-26", "n"), panel, {"Mkt-RF": 1.0})
    assert not result.measured and result.factor_loss is None and result.unmeasured


def test_an_inverted_scenario_raises_rather_than_reporting_a_gap():
    panel = factor_panel({"Mkt-RF": [-0.10] * 3})
    with pytest.raises(ValueError, match="ends before it starts"):
        run_scenario(Scenario("x", "2008-09-10", "2008-09-08", "n"), panel, {"Mkt-RF": 1.0})


# --- observed tails instead of typed shocks ---------------------------------


def test_the_worst_window_is_measured_and_dated():
    """ "What if the market fell X%" with X taken from the file, and auditable."""
    panel = factor_panel({"Mkt-RF": [0.01, -0.05, -0.06, 0.02]})
    move, first, last = worst_window(panel, "Mkt-RF", 2)
    assert move == pytest.approx((1 - 0.05) * (1 - 0.06) - 1.0)
    assert (first, last) == ("2008-09-09", "2008-09-10")


def test_a_one_day_window_finds_the_single_worst_day():
    panel = factor_panel({"Mkt-RF": [0.01, -0.20, -0.06]})
    move, first, last = worst_window(panel, "Mkt-RF", 1)
    assert move == pytest.approx(-0.20) and first == last == "2008-09-09"


def test_a_window_containing_a_hole_is_skipped_not_treated_as_flat():
    panel = factor_panel({"Mkt-RF": [-0.30, float("nan"), -0.01, -0.02]})
    move, first, _last = worst_window(panel, "Mkt-RF", 2)
    assert move == pytest.approx((1 - 0.01) * (1 - 0.02) - 1.0)
    assert first == "2008-09-10"


def test_a_factor_that_is_every_window_a_hole_is_refused():
    panel = factor_panel({"Mkt-RF": [float("nan")] * 3})
    with pytest.raises(ValueError, match="contains a missing value"):
        worst_window(panel, "Mkt-RF", 2)


def test_a_window_longer_than_the_file_is_refused():
    panel = factor_panel({"Mkt-RF": [-0.01] * 3})
    with pytest.raises(ValueError, match="too few for a 10-day window"):
        worst_window(panel, "Mkt-RF", 10)


def test_a_factor_the_file_does_not_carry_is_refused():
    panel = factor_panel({"Mkt-RF": [-0.01] * 3})
    with pytest.raises(ValueError, match="not a column"):
        worst_window(panel, "HML", 1)


def test_a_zero_day_window_is_refused():
    panel = factor_panel({"Mkt-RF": [-0.01] * 3})
    with pytest.raises(ValueError, match="at least one day"):
        worst_window(panel, "Mkt-RF", 0)


# --- how the book's own universe trades on bad days -------------------------


def test_the_stressed_volume_ratio_is_measured_from_the_panel():
    ratio = stressed_adv_ratio(price_panel())
    assert ratio == pytest.approx(1.0)  # flat volume in the fixture


def test_thinner_volume_on_the_worst_days_shows_up_as_a_ratio_below_one():
    panel = price_panel()
    returns = panel.bar_returns
    market = np.mean(returns, axis=1)
    worst = int(np.argmin(market))
    volume = panel.dollar_volume.copy()
    volume[worst + 1, :] = 1_000_000.0  # the closing row of the worst bar
    thinned = PricePanel(dates=panel.dates, symbols=panel.symbols, close=panel.close, dollar_volume=volume)
    assert stressed_adv_ratio(thinned, quantile=0.05) < 1.0


def test_a_panel_without_volume_cannot_answer_and_says_so():
    """`None`, so a caller cannot read "no adjustment" as "no effect"."""
    assert stressed_adv_ratio(price_panel(volume=False)) is None


def test_a_quantile_that_is_not_a_share_of_days_is_refused():
    with pytest.raises(ValueError, match="not a share of days"):
        stressed_adv_ratio(price_panel(), quantile=1.5)


def test_the_tail_quantile_is_a_share():
    assert 0.0 < TAIL_QUANTILE < 1.0


# --- the snapshot -----------------------------------------------------------


def test_the_snapshot_names_every_window_it_could_not_evaluate():
    """Nine unmeasured windows and two results is not a clean stress test."""
    panel = factor_panel({"Mkt-RF": [-0.05] * 3})
    snapshot = stress_snapshot(panel, {"Mkt-RF": 1.0}, shock_horizons=(1,))
    assert len(snapshot["scenarios"]) == len(SCENARIOS)
    assert len(snapshot["unmeasured"]) == len(SCENARIOS) - 1  # only gfc-2008 is covered
    assert snapshot["worst_key"] == "gfc-2008"


def test_an_empty_suite_is_unmeasured_rather_than_all_clear():
    panel = factor_panel({"Mkt-RF": [-0.05] * 3})
    snapshot = stress_snapshot(panel, {"Mkt-RF": 1.0}, scenarios=())
    assert snapshot["worst_loss"] is None
    assert snapshot["unmeasured"] == ["no scenario was supplied, so nothing was stressed"]


def test_the_worst_scenario_is_the_most_negative_one():
    panel = factor_panel({"Mkt-RF": [-0.05] * 30})
    suite = (
        Scenario("mild", "2008-09-08", "2008-09-09", "n"),
        Scenario("severe", "2008-09-08", "2008-09-20", "n"),
    )
    snapshot = stress_snapshot(panel, {"Mkt-RF": 1.0}, scenarios=suite, shock_horizons=(1,))
    assert snapshot["worst_key"] == "severe"
    assert snapshot["worst_loss"] < 0.0


def test_the_snapshot_pins_which_factor_file_it_used():
    """A stress number with no digest cannot be reproduced next quarter."""
    panel = factor_panel({"Mkt-RF": [-0.05] * 3}, source="ken-french-data-library")
    snapshot = stress_snapshot(panel, {"Mkt-RF": 1.0}, shock_horizons=(1,))
    assert snapshot["factor_source"] == "ken-french-data-library"
    assert snapshot["factor_digest"] == "0" * 64
    assert snapshot["factor_span"] == ["2008-09-08", "2008-09-10"]


def test_a_shock_horizon_longer_than_the_file_is_recorded_not_silently_dropped():
    panel = factor_panel({"Mkt-RF": [-0.05] * 3})
    snapshot = stress_snapshot(panel, {"Mkt-RF": 1.0}, shock_horizons=(1, 20))
    assert "move" in snapshot["shocks"]["Mkt-RF/1d"]
    assert "unmeasured" in snapshot["shocks"]["Mkt-RF/20d"]


def test_the_snapshot_carries_no_volume_ratio_without_a_panel():
    panel = factor_panel({"Mkt-RF": [-0.05] * 3})
    snapshot = stress_snapshot(panel, {"Mkt-RF": 1.0}, shock_horizons=(1,))
    assert snapshot["stressed_adv_ratio"] is None


# --- the ladder, as findings and not as actions -----------------------------


def test_a_scenario_is_placed_on_the_existing_drawdown_ladder():
    panel = factor_panel({"Mkt-RF": [-0.06]})
    snapshot = stress_snapshot(
        panel,
        {"Mkt-RF": 1.0},
        scenarios=(Scenario("x", "2008-09-08", "2008-09-08", "n"),),
        shock_horizons=(1,),
    )
    findings = ladder_reach(snapshot, LADDER)
    assert len(findings) == 1
    assert "pod.drawdown.cut" in findings[0] and "halve_capital" in findings[0]


def test_the_most_severe_rung_touched_is_the_one_reported():
    panel = factor_panel({"Mkt-RF": [-0.20]})
    snapshot = stress_snapshot(
        panel,
        {"Mkt-RF": 1.0},
        scenarios=(Scenario("x", "2008-09-08", "2008-09-08", "n"),),
        shock_horizons=(1,),
    )
    assert "pod.drawdown.stop" in ladder_reach(snapshot, LADDER)[0]


def test_a_scenario_below_every_rung_produces_no_finding():
    panel = factor_panel({"Mkt-RF": [-0.001]})
    snapshot = stress_snapshot(
        panel,
        {"Mkt-RF": 1.0},
        scenarios=(Scenario("x", "2008-09-08", "2008-09-08", "n"),),
        shock_horizons=(1,),
    )
    assert ladder_reach(snapshot, LADDER) == []


def test_the_widened_loss_is_what_the_rung_is_compared_against():
    """A scenario that clears a rung on the factor loss alone still reports it."""
    panel = factor_panel({"Mkt-RF": [-0.02]})
    suite = (Scenario("x", "2008-09-08", "2008-09-08", "n"),)
    clean = stress_snapshot(panel, {"Mkt-RF": 1.0}, scenarios=suite, shock_horizons=(1,))
    assert ladder_reach(clean, LADDER) == []
    widened = stress_snapshot(
        panel, {"Mkt-RF": 1.0}, residual_daily_vol=0.01, scenarios=suite, shock_horizons=(1,)
    )
    assert "pod.drawdown.warn" in ladder_reach(widened, LADDER)[0]


def test_an_unmeasured_scenario_produces_a_finding_rather_than_silence():
    """A stress report is read as "we looked"; a silent gap says we did."""
    panel = factor_panel({"Mkt-RF": [-0.06]})
    snapshot = stress_snapshot(
        panel,
        {"Mkt-RF": 1.0},
        scenarios=(Scenario("gone", "1987-10-14", "1987-10-26", "n"),),
        shock_horizons=(1,),
    )
    findings = ladder_reach(snapshot, LADDER)
    assert len(findings) == 1 and findings[0].startswith("gone: not measured")


def test_a_ladder_with_no_readable_level_says_it_cannot_place_anything():
    panel = factor_panel({"Mkt-RF": [-0.20]})
    snapshot = stress_snapshot(
        panel,
        {"Mkt-RF": 1.0},
        scenarios=(Scenario("x", "2008-09-08", "2008-09-08", "n"),),
        shock_horizons=(1,),
    )
    findings = ladder_reach(snapshot, {"pod": {"drawdown": {"warn": {"abs": None}}}})
    assert findings == ["pod.drawdown carries no readable level, so no scenario can be placed on the ladder"]


def test_the_findings_are_strings_and_not_breaches():
    """CLAUDE.md 1항 and the ladder's meaning: 2008 does not stop a pod today."""
    panel = factor_panel({"Mkt-RF": [-0.20]})
    snapshot = stress_snapshot(
        panel,
        {"Mkt-RF": 1.0},
        scenarios=(Scenario("x", "2008-09-08", "2008-09-08", "n"),),
        shock_horizons=(1,),
    )
    assert all(isinstance(line, str) for line in ladder_reach(snapshot, LADDER))


def test_the_real_ladder_is_read_when_no_limits_are_passed():
    """The default reads `limits.yaml`, so a caller cannot supply its own thresholds."""
    panel = factor_panel({"Mkt-RF": [-0.20]})
    snapshot = stress_snapshot(
        panel,
        {"Mkt-RF": 1.0},
        scenarios=(Scenario("x", "2008-09-08", "2008-09-08", "n"),),
        shock_horizons=(1,),
    )
    assert "pod.drawdown.stop" in ladder_reach(snapshot)[0]


def test_a_rung_reached_only_by_the_residual_says_so():
    """The band grows with root-time, so over a long window it reaches a rung alone."""
    panel = factor_panel({"Mkt-RF": [-0.0005] * 60})
    snapshot = stress_snapshot(
        panel,
        {"Mkt-RF": 1.0},
        residual_daily_vol=0.01,
        scenarios=(Scenario("long", "2008-09-08", "2008-11-06", "n"),),
        shock_horizons=(1,),
    )
    finding = ladder_reach(snapshot, LADDER)[0]
    assert "the residual band, not the" in finding
    assert "factor move, is what reaches it" in finding


def test_a_rung_the_factor_move_reaches_on_its_own_names_no_driver():
    panel = factor_panel({"Mkt-RF": [-0.20]})
    snapshot = stress_snapshot(
        panel,
        {"Mkt-RF": 1.0},
        residual_daily_vol=0.01,
        scenarios=(Scenario("x", "2008-09-08", "2008-09-08", "n"),),
        shock_horizons=(1,),
    )
    assert "residual band, not the" not in ladder_reach(snapshot, LADDER)[0]
