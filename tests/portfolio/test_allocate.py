"""The allocator's job is to refuse to chase luck. These tests pin that."""

import copy

import numpy as np
import pytest

from core.portfolio.allocate import (
    CAPACITY_FRACTION_CEILING,
    PodState,
    allocate,
    capacity_fraction,
    drawdown_multiplier,
    equal_risk_contribution,
    horizon_budget,
    information_ratio,
    load_allocation_config,
    ramp_fraction,
    shrunk_correlation,
    volatility_band,
    volatility_target,
)
from core.risk.limits import load_limits

NAV = 100e6
SEASONED = 12  # clean months: past the ramp, so the ramp does not mask other rules


@pytest.fixture
def config():
    return copy.deepcopy(load_allocation_config())


@pytest.fixture
def limits():
    return copy.deepcopy(load_limits())


@pytest.fixture
def loose(limits):
    """The real limit table with the per-horizon budget switched off.

    Most rules here are tested on two or three pods, where one horizon
    legitimately holds most of the book. Switching that one cap off keeps each
    test about the rule it names instead of about the horizon budget.
    """
    limits["horizon"]["risk_budget_share_max"] = 1.0
    return limits


def _pod(pod_id, mean=0.0006, vol=0.008, days=756, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    kwargs.setdefault("clean_months", SEASONED)
    kwargs.setdefault("capacity_usd", 1e12)  # effectively uncapped unless a test says otherwise
    return PodState(pod_id, mean + rng.normal(0, vol, days), horizon="daily", **kwargs)


# --- the rule that was deliberately thrown away -----------------------------


def test_recent_performance_does_not_drive_the_allocation(config, loose, tmp_path):
    """A bad month must not empty a pod that has a year of record.

    This is the v1 rule the design rejected: sizing by rolling Sharpe turns a
    30-day drawdown into a capital cut, which is how an allocator manufactures
    whipsaw out of noise. Here the two pods are the same series except for the
    last 30 days, and over that month one is up and the other is clearly down.
    """
    rng = np.random.default_rng(7)
    shared = 0.0006 + rng.normal(0, 0.008, 756)

    lucky, unlucky = shared.copy(), shared.copy()
    lucky[-30:] += 0.0003
    unlucky[-30:] -= 0.0003

    def month_sharpe(series):
        window = series[-30:]
        return window.mean() / window.std(ddof=1) * np.sqrt(252)

    # The contrast a short-window allocator would act on: opposite signs.
    assert month_sharpe(lucky) > 0 > month_sharpe(unlucky)

    pods = [
        PodState("lucky", lucky, capacity_usd=1e12, clean_months=SEASONED),
        PodState("unlucky", unlucky, capacity_usd=1e12, clean_months=SEASONED),
    ]
    weights = allocate(pods, NAV, config, loose, audit_path=tmp_path / "a.jsonl").by_pod

    # The allocator reads the year, so the two stay comparable in size.
    assert weights["unlucky"] > 0.6 * weights["lucky"]


def test_the_allocator_reads_no_window_shorter_than_the_configured_lookback(config):
    """Structural companion to the test above: there is no 30- or 90-day input."""
    assert information_ratio(np.zeros(config["ir_lookback_days"] - 1), config["ir_lookback_days"]) is None
    assert information_ratio(np.zeros(config["ir_lookback_days"]), config["ir_lookback_days"]) == 0.0


# --- risk parity ------------------------------------------------------------


def test_equal_risk_contribution_equalises_risk_not_weight():
    covariance = np.array([[0.04, 0.018, 0.0], [0.018, 0.09, 0.0], [0.0, 0.0, 0.01]])
    weights = equal_risk_contribution(covariance)
    contributions = weights * (covariance @ weights)
    shares = contributions / contributions.sum()
    assert np.allclose(shares, 1 / 3, atol=1e-6)
    assert not np.allclose(weights, 1 / 3, atol=1e-2)  # equal risk is not equal weight


def test_shrinkage_pulls_toward_independence_and_lets_go_as_evidence_arrives():
    """Ledoit-Wolf on a pair whose true correlation is 0.6.

    A short sample overstates the correlation and gets pulled hard; a long one
    is left almost alone. Risk parity built on the raw sample would size as if
    the short-sample estimate were fact.
    """
    rho = 0.6

    def sample_and_shrunk(n_obs):
        rng = np.random.default_rng(11)
        z = rng.normal(0, 1, (n_obs, 2))
        paired = np.column_stack([z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]])
        return np.corrcoef(paired, rowvar=False)[0, 1], shrunk_correlation(paired)[0, 1]

    short_sample, short_shrunk = sample_and_shrunk(15)
    long_sample, long_shrunk = sample_and_shrunk(5000)

    assert short_shrunk < short_sample  # pulled toward independence
    assert long_shrunk < long_sample
    assert (short_sample - short_shrunk) > (long_sample - long_shrunk)  # and pulled harder
    assert abs(long_shrunk - rho) < 0.05  # evidence eventually wins


def test_shrinkage_keeps_a_unit_diagonal():
    rng = np.random.default_rng(3)
    assert np.allclose(np.diag(shrunk_correlation(rng.normal(0, 1, (40, 5)))), 1.0)


def test_a_single_pod_needs_no_correlation():
    assert shrunk_correlation(np.zeros((100, 1))).shape == (1, 1)
    assert equal_risk_contribution(np.array([[0.04]])) == np.ones(1)


# --- the IR tilt ------------------------------------------------------------


def test_a_pod_that_does_not_earn_its_costs_gets_nothing(config, loose, tmp_path):
    """Cost-deducted IR at or below zero is a zero tilt, by design."""
    pods = [
        _pod("earner", mean=0.0008, seed=1),
        _pod("burner", mean=-0.0002, seed=2),
    ]
    result = allocate(pods, NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.by_pod["burner"] == 0.0
    assert result.by_pod["earner"] > 0.0


def test_a_pod_without_a_year_of_record_is_not_tilted_but_is_ramped(config, loose, tmp_path):
    young = _pod("young", days=120, seed=4, clean_months=0)
    old = _pod("old", seed=5)
    result = allocate([young, old], NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    record = next(a for a in result.allocations if a.pod_id == "young")
    assert record.ir_available is False
    assert record.ir_tilt == 1.0  # no history means no tilt, not a punishing one
    assert record.binding == "ramp"


def test_nothing_is_allocated_when_no_pod_earns(config, loose, tmp_path):
    pods = [_pod("a", mean=-0.0005, seed=6), _pod("b", mean=-0.0009, seed=7)]
    result = allocate(pods, NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.gross == 0.0
    assert all(a.weight == 0.0 for a in result.allocations)


# --- the caps ---------------------------------------------------------------


def test_capacity_is_never_exceeded(config, loose, tmp_path):
    small = _pod("small", capacity_usd=10e6, seed=8)
    large = _pod("large", capacity_usd=1e12, seed=9)
    result = allocate([small, large], NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    ceiling = capacity_fraction(loose) * 10e6 / NAV
    assert result.by_pod["small"] <= ceiling + 1e-12


def test_the_limit_table_may_tighten_the_capacity_ceiling_but_not_loosen_it(limits):
    """CLAUDE.md rule 7 fixes 80%. The table may go below it and not above."""
    assert capacity_fraction({"capacity": {"capacity_utilisation_max": 0.95}}) == CAPACITY_FRACTION_CEILING
    assert capacity_fraction({"capacity": {"capacity_utilisation_max": 0.50}}) == 0.50
    assert capacity_fraction(limits) <= CAPACITY_FRACTION_CEILING


def test_one_horizon_may_not_hold_the_whole_book(config, limits, tmp_path):
    pods = [
        PodState("d1", _pod("x", seed=10).returns, capacity_usd=1e12, horizon="daily", clean_months=SEASONED),
        PodState("d2", _pod("y", seed=11).returns, capacity_usd=1e12, horizon="daily", clean_months=SEASONED),
        PodState(
            "m1", _pod("z", seed=12).returns, capacity_usd=1e12, horizon="medium", clean_months=SEASONED
        ),
    ]
    result = allocate(pods, NAV, config, limits, audit_path=tmp_path / "a.jsonl")
    weights = result.by_pod
    daily = weights["d1"] + weights["d2"]
    assert daily / result.gross <= horizon_budget(limits) + 1e-9


def test_gross_never_exceeds_the_risk_limit_table(config, loose, tmp_path):
    pods = [_pod("a", mean=0.004, seed=13), _pod("b", mean=0.004, seed=14)]
    result = allocate(pods, NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.gross <= result.gross_limit + 1e-12


# --- the ramp and the lock --------------------------------------------------


def test_the_ramp_steps_up_with_clean_months(config):
    ramp = config["ramp"]
    assert ramp_fraction(0, ramp) == 0.25
    assert ramp_fraction(2, ramp) == 0.25
    assert ramp_fraction(3, ramp) == 0.50
    assert ramp_fraction(7, ramp) == 1.00


def test_an_allocation_holds_for_the_lock_period(config, loose, tmp_path):
    pod = _pod("held", seed=15, current_weight=0.11, months_since_allocation=1)
    other = _pod("free", seed=16)
    result = allocate([pod, other], NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.by_pod["held"] == pytest.approx(0.11)


def test_a_drawdown_cut_breaks_the_lock(config, loose, tmp_path):
    pod = _pod("held", seed=15, current_weight=0.11, months_since_allocation=1, drawdown_tier="cut")
    other = _pod("free", seed=16)
    result = allocate([pod, other], NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.by_pod["held"] != pytest.approx(0.11)


# --- the limit table is the single source -----------------------------------


def test_tightening_the_table_tightens_the_book(config, limits, tmp_path):
    """The migration's whole point: these numbers have one home now."""
    pods = [_pod("a", mean=0.01, vol=0.002, seed=60), _pod("b", mean=0.01, vol=0.002, seed=61)]
    before = allocate(pods, NAV, config, limits, audit_path=tmp_path / "before.jsonl")
    limits["pod"]["gross_leverage_max"] = 0.40
    after = allocate(pods, NAV, config, limits, audit_path=tmp_path / "after.jsonl")
    assert after.gross < before.gross
    assert after.gross <= 0.40 + 1e-12


def test_the_volatility_target_is_the_midpoint_of_the_table_band(limits):
    assert volatility_band(limits) == (0.10, 0.15)
    assert volatility_target(limits) == pytest.approx(0.125)
    limits["pod"]["target_volatility"] = [0.04, 0.06]
    assert volatility_target(limits) == pytest.approx(0.05)


def test_the_allocation_config_holds_no_risk_limit(config):
    """Anything whose breach costs money belongs in limits.yaml, not here."""
    banned = {
        "gross_leverage_max",
        "max_gross",
        "target_volatility",
        "target_volatility_band",
        "kelly_fraction",
        "lock_months",
        "capacity_fraction_max",
        "horizon_budget_max",
    }
    assert banned.isdisjoint(config)


def test_a_book_too_quiet_to_reach_the_band_says_so(config, limits, tmp_path):
    """Undershoot is reported, never corrected: reaching the band from below
    is leverage, and the gross limit refuses it (ADR-0009)."""
    quiet = [_pod("q1", vol=0.001, seed=70), _pod("q2", vol=0.001, seed=71)]
    result = allocate(quiet, NAV, config, limits, audit_path=tmp_path / "q.jsonl")
    low, _ = volatility_band(limits)
    assert result.expected_volatility < low
    assert any("not corrected" in n for n in result.notes)
    assert result.gross <= limits["pod"]["gross_leverage_max"] + 1e-12


def test_a_book_inside_the_band_does_not_cry_undershoot(config, limits, tmp_path):
    wild = [_pod("w1", vol=0.030, seed=72), _pod("w2", vol=0.030, seed=73)]
    result = allocate(wild, NAV, config, limits, audit_path=tmp_path / "w.jsonl")
    assert not any("not corrected" in n for n in result.notes)


# --- the drawdown ladder ----------------------------------------------------


def test_a_cut_tier_halves_the_allocation(config, loose, tmp_path):
    """limits.yaml says halve_capital at the cut tier. It must actually halve."""
    calm = allocate(
        [_pod("a", seed=40), _pod("b", seed=41)], NAV, config, loose, audit_path=tmp_path / "calm.jsonl"
    )
    cut = allocate(
        [_pod("a", seed=40, drawdown_tier="cut"), _pod("b", seed=41)],
        NAV,
        config,
        loose,
        audit_path=tmp_path / "cut.jsonl",
    )
    assert cut.by_pod["a"] == pytest.approx(calm.by_pod["a"] * 0.5)
    assert cut.by_pod["b"] == pytest.approx(calm.by_pod["b"])  # the cut does not fund anyone else


def test_a_stop_tier_takes_the_whole_allocation(config, loose, tmp_path):
    pods = [_pod("a", seed=40, drawdown_tier="stop"), _pod("b", seed=41)]
    result = allocate(pods, NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.by_pod["a"] == 0.0
    assert result.weight("a") == 0.0
    assert [a.binding for a in result.allocations if a.pod_id == "a"] == ["drawdown"]


def test_a_warn_tier_is_report_only(config, loose, tmp_path):
    """The table's warn action is `report`. Reporting must not cut capital."""
    calm = allocate(
        [_pod("a", seed=40), _pod("b", seed=41)], NAV, config, loose, audit_path=tmp_path / "calm.jsonl"
    )
    warned = allocate(
        [_pod("a", seed=40, drawdown_tier="warn"), _pod("b", seed=41)],
        NAV,
        config,
        loose,
        audit_path=tmp_path / "warn.jsonl",
    )
    assert warned.by_pod["a"] == pytest.approx(calm.by_pod["a"])
    note = next(a for a in warned.allocations if a.pod_id == "a").notes
    assert any("report-only" in n for n in note)


def test_a_warn_tier_does_not_break_the_lock(config, loose, tmp_path):
    pod = _pod("held", seed=15, current_weight=0.11, months_since_allocation=1, drawdown_tier="warn")
    other = _pod("free", seed=16)
    result = allocate([pod, other], NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.by_pod["held"] == pytest.approx(0.11)


def test_a_stop_tier_beats_the_lock(config, loose, tmp_path):
    pod = _pod("held", seed=15, current_weight=0.11, months_since_allocation=1, drawdown_tier="stop")
    other = _pod("free", seed=16)
    result = allocate([pod, other], NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.by_pod["held"] == 0.0


def test_an_unknown_drawdown_tier_is_an_error_not_a_pass(limits):
    """A renamed tier must fail loudly. Silently keeping full capital is the
    one outcome this ladder exists to prevent."""
    with pytest.raises(ValueError, match="unknown drawdown tier"):
        drawdown_multiplier("severe", limits)


def test_an_action_with_no_allocation_rule_is_an_error(limits):
    limits["pod"]["drawdown"]["cut"]["action"] = "sell_everything_twice"
    with pytest.raises(ValueError, match="no allocation rule"):
        drawdown_multiplier("cut", limits)


def test_the_ladder_reads_the_table_rather_than_its_own_numbers(limits):
    """Tightening the table to stop at the cut tier must change the allocator."""
    assert drawdown_multiplier("cut", limits) == 0.5
    limits["pod"]["drawdown"]["cut"]["action"] = "stop_pod"
    assert drawdown_multiplier("cut", limits) == 0.0


# --- the fund-level diversification limit -----------------------------------


def test_correlated_pods_are_reported_not_quietly_resized(config, limits, tmp_path):
    """Correlation is a property of which pods exist, not of how they are sized,
    so the allocator says so and leaves the choice to a person."""
    rng = np.random.default_rng(77)
    common = rng.normal(0, 0.008, 756)
    pods = [
        PodState(
            f"p{i}", 0.0006 + common + rng.normal(0, 0.002, 756), capacity_usd=1e12, clean_months=SEASONED
        )
        for i in range(3)
    ]
    result = allocate(pods, NAV, config, limits, audit_path=tmp_path / "a.jsonl")
    assert any("BREACH pod_avg_correlation_max" in n for n in result.notes)
    assert result.gross > 0  # reported, not acted on


def test_a_gate_failure_breaks_the_lock(config, loose, tmp_path):
    pod = _pod("held", seed=15, current_weight=0.11, months_since_allocation=1, gate_failed=True)
    other = _pod("free", seed=16)
    result = allocate([pod, other], NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.by_pod["held"] != pytest.approx(0.11)


def test_capacity_overrides_the_lock(config, loose, tmp_path):
    """A locked weight that has outgrown its pod's capacity is still trimmed."""
    pod = _pod("held", seed=15, capacity_usd=5e6, current_weight=0.50, months_since_allocation=1)
    other = _pod("free", seed=16)
    result = allocate([pod, other], NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    record = next(a for a in result.allocations if a.pod_id == "held")
    assert record.binding == "capacity"
    assert result.by_pod["held"] <= capacity_fraction(loose) * 5e6 / NAV + 1e-12
    assert any("overrides the allocation lock" in n for n in record.notes)


# --- invariants -------------------------------------------------------------


def test_no_constraint_ever_increases_a_weight(config, loose, tmp_path):
    """Every step after risk parity shrinks. A cap elsewhere never enlarges a pod."""
    pods = [
        _pod("a", capacity_usd=8e6, seed=17),
        _pod("b", seed=18, clean_months=1),
        _pod("c", seed=19),
    ]
    result = allocate(pods, NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    budget = min(result.half_kelly_gross, result.gross_limit)
    for record in result.allocations:
        ceiling = record.risk_parity_weight * record.ir_tilt * budget
        assert record.weight <= ceiling + 1e-9


def test_a_stopped_pod_gets_nothing(config, loose, tmp_path):
    pods = [_pod("live", seed=20), _pod("dead", seed=21, stopped=True)]
    result = allocate(pods, NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.by_pod["dead"] == 0.0
    assert next(a for a in result.allocations if a.pod_id == "dead").binding == "stopped"


def test_every_pod_stopped_allocates_nothing(config, tmp_path):
    pods = [_pod("a", seed=22, stopped=True)]
    result = allocate(pods, NAV, config, audit_path=tmp_path / "a.jsonl")
    assert result.gross == 0.0


def test_the_decision_is_audited(config, tmp_path):
    from core import audit

    path = tmp_path / "audit.jsonl"
    allocate([_pod("a", seed=23), _pod("b", seed=24)], NAV, config, audit_path=path)
    records = list(audit.read(path))
    assert [r["event"] for r in records] == ["portfolio.allocate"]
    assert audit.verify(path)[0]


def test_a_non_positive_nav_is_an_error(config):
    with pytest.raises(ValueError):
        allocate([_pod("a", seed=25)], 0.0, config)


# --- the owner's operating rules --------------------------------------------


def test_the_book_is_never_levered(config, loose, tmp_path):
    """No leverage: gross may not exceed NAV, however good the numbers look.

    The number enforced is the limit table's own gross_leverage_max, which the
    owner set to 1.0 in ADR-0009. There is no second no-leverage knob here.
    """
    ceiling = loose["pod"]["gross_leverage_max"]
    assert ceiling == 1.0, "the owner's no-leverage rule lives in the limit table"
    pods = [_pod("a", mean=0.01, vol=0.002, seed=30), _pod("b", mean=0.01, vol=0.002, seed=31)]
    result = allocate(pods, NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.gross <= ceiling + 1e-12
    assert result.half_kelly_gross > ceiling  # Kelly wanted more and was refused
    assert result.binding_gross == "gross_leverage_max"


def test_gross_is_sized_to_the_target_volatility(config, loose, tmp_path):
    """A quiet book is sized up toward the band; a wild one is sized down."""
    quiet = [_pod("q1", vol=0.003, seed=32), _pod("q2", vol=0.003, seed=33)]
    wild = [_pod("w1", vol=0.030, seed=34), _pod("w2", vol=0.030, seed=35)]
    quiet_result = allocate(quiet, NAV, config, loose, audit_path=tmp_path / "q.jsonl")
    wild_result = allocate(wild, NAV, config, loose, audit_path=tmp_path / "w.jsonl")

    assert wild_result.gross < quiet_result.gross
    assert wild_result.binding_gross == "target_volatility"
    low, high = volatility_band(loose)
    assert wild_result.expected_volatility <= high + 1e-9
    # The quiet book is capped by the no-leverage rule before it reaches the band.
    assert quiet_result.binding_gross == "gross_leverage_max"


def test_the_sizing_window_is_the_performance_window(config, loose, tmp_path):
    """Sizing and tilting read the same year; an older, worse era cannot size the book."""
    rng = np.random.default_rng(36)
    recent = 0.0008 + rng.normal(0, 0.008, 252)
    ancient = -0.0015 + rng.normal(0, 0.008, 504)
    series = np.concatenate([ancient, recent])

    pods = [
        PodState("long", series, capacity_usd=1e12, clean_months=SEASONED),
        PodState("long2", series.copy(), capacity_usd=1e12, clean_months=SEASONED),
    ]
    result = allocate(pods, NAV, config, loose, audit_path=tmp_path / "a.jsonl")
    assert result.gross > 0.0


def test_a_pod_zeroed_by_the_tilt_does_not_shrink_the_rest_of_the_book(config, loose, tmp_path):
    """The gross budget describes the book being proposed, not the pods refused."""
    earner = _pod("earner", mean=0.0008, seed=37)
    alone = allocate([earner], NAV, config, loose, audit_path=tmp_path / "1.jsonl")
    with_burner = allocate(
        [earner, _pod("burner", mean=-0.0004, seed=38)], NAV, config, loose, audit_path=tmp_path / "2.jsonl"
    )
    assert with_burner.by_pod["burner"] == 0.0
    assert with_burner.by_pod["earner"] == pytest.approx(alone.by_pod["earner"], rel=1e-9)
