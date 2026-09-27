"""The net-of-cost IR floor, and the ways a pod could clear it without paying.

The failure this module exists for is arithmetic: subtract nothing and the
net-of-cost IR is the gross IR, so a floor of 0.0 is cleared by any pod that
made money. Every refusal below is one route to that free pass.
"""

from __future__ import annotations

import pytest

from core.risk import cost
from core.risk.limits import load_limits

MEASURED = {
    "gross_ir": 1.00,
    "annual_cost": 12_000.0,
    "allocated_capital": 1_000_000.0,
    "volatility": 0.12,
    "cost_measured": True,
}


def limits(**cost_attribution):
    table = load_limits()
    return {**table, "cost_attribution": {**(table.get("cost_attribution") or {}), **cost_attribution}}


# --- the arithmetic -----------------------------------------------------------


def test_the_drag_is_the_cost_share_of_capital_in_volatility_units():
    # 12,000 on 1,000,000 is a 1.2% return drag; at 12% volatility that is 0.1 IR.
    assert cost.cost_ir_drag(12_000.0, 1_000_000.0, 0.12) == pytest.approx(0.1)
    assert cost.net_of_cost_ir(1.0, 12_000.0, 1_000_000.0, 0.12) == pytest.approx(0.9)


def test_a_bigger_cost_lowers_the_net_ir():
    cheap = cost.net_of_cost_ir(1.0, 12_000.0, 1_000_000.0, 0.12)
    dear = cost.net_of_cost_ir(1.0, 120_000.0, 1_000_000.0, 0.12)
    assert dear < cheap


@pytest.mark.parametrize(
    ("annual_cost", "capital", "volatility"),
    [
        (None, 1_000_000.0, 0.12),
        (-1.0, 1_000_000.0, 0.12),
        (12_000.0, 0.0, 0.12),
        (12_000.0, None, 0.12),
        (12_000.0, 1_000_000.0, 0.0),
        (12_000.0, 1_000_000.0, None),
        ("many", 1_000_000.0, 0.12),
    ],
)
def test_an_unusable_input_gives_no_drag_rather_than_none_charged(annual_cost, capital, volatility):
    assert cost.cost_ir_drag(annual_cost, capital, volatility) is None
    assert cost.net_of_cost_ir(1.0, annual_cost, capital, volatility) is None


def test_an_unmeasured_gross_ir_cannot_be_netted():
    assert cost.net_of_cost_ir(None, 12_000.0, 1_000_000.0, 0.12) is None


# --- the table ----------------------------------------------------------------


def test_the_repository_table_charges_pods_with_a_zero_floor():
    assert cost.charging_pods() is True
    assert cost.ir_floor() == 0.0


@pytest.mark.parametrize("flag", [None, "yes", 1, 0])
def test_a_flag_that_is_not_a_boolean_is_not_a_decision(flag):
    assert cost.charging_pods(limits(charge_pods=flag)) is None
    breaches = cost.check_cost_attribution({"statarb": dict(MEASURED)}, limits(charge_pods=flag))
    assert [b.code for b in breaches] == ["COST_ATTRIBUTION_UNUSABLE"]


def test_a_recorded_decision_not_to_charge_is_configuration_not_a_breach():
    """Unlike capacity.respect_hard_cap: no CLAUDE.md rule makes pass-through absolute."""
    assert cost.check_cost_attribution(None, limits(charge_pods=False)) == []
    assert cost.check_cost_attribution({}, limits(charge_pods=False)) == []


@pytest.mark.parametrize("floor", [None, "zero", [0.0]])
def test_a_floor_that_is_not_a_number_is_refused(floor):
    breaches = cost.check_cost_attribution({"statarb": dict(MEASURED)}, limits(min_net_of_cost_ir=floor))
    assert "COST_ATTRIBUTION_UNUSABLE" in [b.code for b in breaches]


# --- the refusals -------------------------------------------------------------


def test_an_empty_snapshot_is_no_pod_charged_not_a_clean_book():
    breaches = cost.check_cost_attribution({})
    assert [b.code for b in breaches] == ["COST_UNMEASURED"]
    assert "for free" in breaches[0].detail


def test_a_missing_snapshot_is_the_same_refusal():
    assert [b.code for b in cost.check_cost_attribution(None)] == ["COST_UNMEASURED"]


@pytest.mark.parametrize("measured", [None, False, "true", 1])
def test_a_cost_that_is_not_a_measurement_blocks(measured):
    snapshot = {"statarb": {**MEASURED, "cost_measured": measured}}
    breaches = cost.check_cost_attribution(snapshot)
    assert [b.code for b in breaches] == ["COST_UNMEASURED"]


def test_a_zero_cost_declared_as_measured_is_allowed_and_a_default_zero_is_not():
    """The flag is the whole difference: 0 measured is a fact, 0 by absence is not."""
    declared = {"statarb": {**MEASURED, "annual_cost": 0.0}}
    assert cost.check_cost_attribution(declared) == []
    absent = {"statarb": {k: v for k, v in MEASURED.items() if k != "cost_measured"}}
    assert [b.code for b in cost.check_cost_attribution(absent)] == ["COST_UNMEASURED"]


def test_an_entry_that_is_not_a_mapping_is_named():
    breaches = cost.check_cost_attribution({"statarb": 3})
    assert [b.code for b in breaches] == ["COST_UNMEASURED"]
    assert "not a mapping" in breaches[0].detail


def test_a_pod_that_does_not_cover_its_own_cost_breaches():
    thin = {"statarb": {**MEASURED, "gross_ir": 0.05, "annual_cost": 120_000.0}}
    breaches = cost.check_cost_attribution(thin)
    assert [b.code for b in breaches] == ["COST_ATTRIBUTION"]
    assert "does not cover" in breaches[0].detail


def test_a_pod_exactly_at_the_floor_passes():
    at = {"statarb": {**MEASURED, "gross_ir": 0.1}}  # drag is exactly 0.1
    assert cost.check_cost_attribution(at) == []


def test_every_pod_is_checked_not_just_the_first():
    snapshot = {
        "aaa": {**MEASURED, "gross_ir": 0.0, "annual_cost": 120_000.0},
        "bbb": {**MEASURED, "cost_measured": False},
        "ccc": dict(MEASURED),
    }
    codes = sorted(b.code for b in cost.check_cost_attribution(snapshot))
    assert codes == ["COST_ATTRIBUTION", "COST_UNMEASURED"]


# --- the keys are actually read ------------------------------------------------


def test_both_keys_are_read_by_this_module():
    """The point of the module: these were the last two keys with no reader."""
    source = (__import__("pathlib").Path("core/risk/cost.py")).read_text(encoding="utf-8")
    assert "charge_pods" in source
    assert "min_net_of_cost_ir" in source
