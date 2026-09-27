"""What the capacity layer refuses, and why zero is the dangerous number.

The rule (CLAUDE.md 7항) is that capital above 80% of the estimated capacity is
not allocated. Until ADR-0026 the two keys that encode it had no reader, so the
first thing these tests fix is that the reader exists and blocks.
"""

from __future__ import annotations

import pytest

from core.risk.capacity import check_capacity, estimate_capacity, max_allocatable, utilisation_cap
from core.risk.limits import load_limits

# 1M tested at 1% participation, and the gate caps participation at 3%: capacity
# is 3M, so the 80% ceiling is 2.4M.
POD = {
    "tested_capital": 1_000_000.0,
    "adv_participation": 0.01,
    "adv_measured": True,
    "allocated_capital": 500_000.0,
}
CLEAN = {"statarb": dict(POD)}


@pytest.fixture
def limits():
    return load_limits()


def codes(breaches) -> set[str]:
    return {breach.code for breach in breaches}


# --- the estimate -----------------------------------------------------------


def test_capacity_is_the_capital_at_which_participation_reaches_the_cap():
    assert estimate_capacity(1_000_000.0, 0.01, 0.03) == pytest.approx(3_000_000.0)


def test_a_zero_participation_does_not_become_unlimited_capacity():
    """The arithmetic says infinity. Infinity reached by not measuring is not a number."""
    assert estimate_capacity(1_000_000.0, 0.0, 0.03) is None


@pytest.mark.parametrize("bad", [None, "0.01", True, float("nan"), -0.01])
def test_a_participation_that_is_not_a_measurement_yields_no_estimate(bad):
    assert estimate_capacity(1_000_000.0, bad, 0.03) is None


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, "1000000"])
def test_capital_that_is_not_a_measurement_yields_no_estimate(bad):
    assert estimate_capacity(bad, 0.01, 0.03) is None


def test_the_ceiling_is_eighty_percent_of_the_estimate(limits):
    assert max_allocatable(3_000_000.0, limits) == pytest.approx(2_400_000.0)


# --- the table's own values -------------------------------------------------


def test_the_shipped_table_carries_a_usable_ceiling():
    assert utilisation_cap() == pytest.approx(0.80)


def test_a_utilisation_above_one_is_refused_rather_than_clamped(limits):
    """Above 1.0 permits allocating past the estimate, which voids the rule."""
    limits["capacity"]["capacity_utilisation_max"] = 1.2
    assert utilisation_cap(limits) is None
    assert "CAPACITY_UTILISATION_UNUSABLE" in codes(check_capacity(CLEAN, limits))


@pytest.mark.parametrize("bad", [None, 0.0, "0.8", -0.5])
def test_a_utilisation_that_is_not_a_share_is_refused(limits, bad):
    limits["capacity"]["capacity_utilisation_max"] = bad
    assert utilisation_cap(limits) is None
    with pytest.raises(ValueError, match="capacity_utilisation_max"):
        max_allocatable(3_000_000.0, limits)


def test_turning_the_hard_cap_off_is_itself_a_breach(limits):
    """The flag makes switching it off visible. It does not make it allowed."""
    limits["capacity"]["respect_hard_cap"] = False
    assert "CAPACITY_CAP_DISABLED" in codes(check_capacity(CLEAN, limits))


def test_a_missing_hard_cap_flag_blocks_too(limits):
    del limits["capacity"]["respect_hard_cap"]
    assert "CAPACITY_CAP_DISABLED" in codes(check_capacity(CLEAN, limits))


def test_a_missing_capacity_block_does_not_read_as_no_constraint(limits):
    del limits["capacity"]
    assert codes(check_capacity(CLEAN, limits)) >= {
        "CAPACITY_CAP_DISABLED",
        "CAPACITY_UTILISATION_UNUSABLE",
    }


# --- the snapshot -----------------------------------------------------------


def test_a_book_under_the_ceiling_has_no_breach(limits):
    assert check_capacity(CLEAN, limits) == []


def test_allocating_past_the_ceiling_breaches(limits):
    snapshot = {"statarb": {**POD, "allocated_capital": 2_500_000.0}}
    breaches = check_capacity(snapshot, limits)
    assert codes(breaches) == {"CAPACITY"}
    assert "2,400,000" in breaches[0].detail


def test_the_ceiling_itself_is_allowed(limits):
    """The limit is "not above", so the boundary is inside it."""
    assert check_capacity({"statarb": {**POD, "allocated_capital": 2_400_000.0}}, limits) == []


def test_good_performance_is_not_an_argument_the_engine_can_hear(limits):
    """There is no field for it. That is the design (CLAUDE.md 7항)."""
    snapshot = {"statarb": {**POD, "allocated_capital": 2_500_000.0, "sharpe": 4.0}}
    assert codes(check_capacity(snapshot, limits)) == {"CAPACITY"}


def test_an_unmeasured_participation_blocks_the_pod(limits):
    snapshot = {"statarb": {**POD, "adv_measured": False}}
    assert codes(check_capacity(snapshot, limits)) == {"CAPACITY_UNMEASURED"}


def test_a_participation_of_zero_declared_measured_still_blocks(limits):
    """Declaring it measured cannot make zero divisible."""
    snapshot = {"statarb": {**POD, "adv_participation": 0.0}}
    assert codes(check_capacity(snapshot, limits)) == {"CAPACITY_UNMEASURED"}


def test_an_unmeasured_allocation_blocks(limits):
    snapshot = {"statarb": {**POD, "allocated_capital": None}}
    assert codes(check_capacity(snapshot, limits)) == {"CAPACITY_ALLOCATION_UNMEASURED"}


def test_an_empty_snapshot_is_not_a_clean_book(limits):
    """A fund with no pods and a snapshot that lost its pods look identical."""
    assert codes(check_capacity({}, limits)) == {"CAPACITY_UNMEASURED"}


def test_an_entry_that_is_not_a_mapping_blocks_that_pod(limits):
    assert codes(check_capacity({"statarb": 2_500_000.0}, limits)) == {"CAPACITY_UNMEASURED"}


def test_every_pod_is_checked_and_named(limits):
    snapshot = {
        "statarb": dict(POD),
        "trend": {**POD, "allocated_capital": 9_000_000.0},
        "event": {**POD, "adv_measured": None},
    }
    breaches = check_capacity(snapshot, limits)
    assert codes(breaches) == {"CAPACITY", "CAPACITY_UNMEASURED"}
    assert {"trend", "event"} == {breach.detail.split(":")[0] for breach in breaches}
