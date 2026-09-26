"""SIC codes map to concentration buckets, and an unmapped one stays unmapped."""

from __future__ import annotations

import pytest

from core.data.sic import BUCKETS, RANGES, bucket_for_sic, unmapped

#: Real filers, with the SIC EDGAR shows for them. Spot checks, not a survey:
#: the point is that each of the eleven buckets is reachable by a code a large
#: listed company actually carries.
REAL_FILERS = {
    3571: ("Apple", "technology"),
    7372: ("Microsoft", "technology"),
    3674: ("Nvidia", "technology"),
    6022: ("JPMorgan", "financials"),
    2911: ("Exxon", "energy"),
    2834: ("Pfizer", "health_care"),
    2836: ("Moderna", "health_care"),
    8731: ("a clinical-stage biotech", "health_care"),
    8000: ("HCA Healthcare", "health_care"),
    6798: ("Simon Property", "real_estate"),
    4911: ("NextEra", "utilities"),
    2086: ("Coca-Cola", "consumer_staples"),
    5411: ("Kroger", "consumer_staples"),
    3711: ("Tesla", "consumer_discretionary"),
    5812: ("McDonald's", "consumer_discretionary"),
    4813: ("Verizon", "communication_services"),
    7990: ("Live Nation", "communication_services"),
    3531: ("Caterpillar", "industrials"),
    4512: ("Delta", "industrials"),
    2810: ("Linde", "materials"),
    3310: ("Nucor", "materials"),
}


@pytest.mark.parametrize("code, expected", [(c, b) for c, (_, b) in REAL_FILERS.items()])
def test_a_real_filers_code_lands_in_the_expected_bucket(code, expected):
    assert bucket_for_sic(code) == expected


def test_every_bucket_is_reachable():
    """A bucket nothing maps to is a bucket that never binds."""
    reached = {bucket_for_sic(code) for code in REAL_FILERS}
    assert reached == set(BUCKETS)


# --- the carve-outs are why the order matters -------------------------------


def test_a_drug_maker_is_health_care_not_chemicals():
    assert bucket_for_sic(2834) == "health_care"
    assert bucket_for_sic(2810) == "materials"


def test_a_computer_maker_is_technology_not_machinery():
    assert bucket_for_sic(3571) == "technology"
    assert bucket_for_sic(3531) == "industrials"


def test_a_carmaker_is_consumer_not_industrial():
    assert bucket_for_sic(3711) == "consumer_discretionary"
    assert bucket_for_sic(3721) == "industrials", "aircraft stay industrial"


def test_a_reit_is_real_estate_not_a_holding_company():
    assert bucket_for_sic(6798) == "real_estate"
    assert bucket_for_sic(6770) == "financials", "a blank-cheque shell is a financial"


def test_a_drug_store_is_staples_and_the_rest_of_retail_is_not():
    assert bucket_for_sic(5912) == "consumer_staples"
    assert bucket_for_sic(5945) == "consumer_discretionary"


# --- what stays unmapped ----------------------------------------------------


def test_nonclassifiable_stays_unmapped():
    """9995 is the SEC's own 'we do not know', and a shell is exactly that."""
    assert bucket_for_sic(9995) is None


def test_public_administration_stays_unmapped():
    assert bucket_for_sic(9199) is None


def test_a_missing_or_unreadable_code_is_unmapped_rather_than_an_error():
    """A listings row with a blank SIC must load; the name is simply not orderable."""
    assert bucket_for_sic(None) is None
    assert bucket_for_sic("") is None
    assert bucket_for_sic("n/a") is None
    assert bucket_for_sic(0) is None
    assert bucket_for_sic(123456) is None


def test_a_code_arrives_as_a_string_from_a_csv():
    assert bucket_for_sic("3571") == "technology"
    assert bucket_for_sic(" 3571 ") == "technology"


def test_unmapped_reports_what_a_source_could_not_classify():
    assert unmapped([3571, 9995, "", 9199, "n/a"]) == (9199, 9995)


# --- the table itself -------------------------------------------------------


def test_every_range_names_a_declared_bucket():
    assert {bucket for _, _, bucket in RANGES} <= set(BUCKETS)


def test_no_range_runs_backwards():
    assert [(low, high) for low, high, _ in RANGES if low > high] == []


def test_no_range_is_shadowed_by_an_earlier_one():
    """First match wins, so a carve-out placed after its block would never fire."""
    for index, (low, high, bucket) in enumerate(RANGES):
        for later_low, later_high, later_bucket in RANGES[index + 1 :]:
            shadowed = low <= later_low and later_high <= high
            assert not shadowed, (
                f"({later_low}, {later_high}) -> {later_bucket} sits inside the earlier "
                f"({low}, {high}) -> {bucket} and can never be reached; move the narrow "
                "range above the block that contains it"
            )


def test_a_carve_out_really_is_reached():
    """The invariant above is only worth having if the carve-outs win in practice."""
    assert bucket_for_sic(2834) != bucket_for_sic(2820)
    assert bucket_for_sic(3571) != bucket_for_sic(3561)
    assert bucket_for_sic(7372) != bucket_for_sic(7363)
