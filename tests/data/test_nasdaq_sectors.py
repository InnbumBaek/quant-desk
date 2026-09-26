"""The vendor sector labels map to the same buckets SIC does, or to nothing.

The whole value of this table is that it cannot invent a bucket. A vendor label
we have never seen must come back `None` and show up in the coverage report, not
get rounded to the nearest plausible sector -- unrelated risk under one
concentration limit is exactly what the limit exists to stop.
"""

from __future__ import annotations

import pytest

from core.data.nasdaq_sectors import NASDAQ_SECTORS, bucket_for_nasdaq_sector, unmapped_labels
from core.data.sic import BUCKETS

#: The sector strings the Nasdaq screener carries, as it spells them.
VENDOR_LABELS = (
    "Basic Materials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Finance",
    "Health Care",
    "Industrials",
    "Real Estate",
    "Technology",
    "Telecommunications",
    "Utilities",
)


@pytest.mark.parametrize("label", VENDOR_LABELS)
def test_every_vendor_label_reaches_a_bucket(label):
    assert bucket_for_nasdaq_sector(label) in BUCKETS


def test_the_table_never_names_a_bucket_no_limit_knows():
    """A bucket spelled only here would count against nothing."""
    assert set(NASDAQ_SECTORS.values()) <= set(BUCKETS)


def test_every_bucket_is_reachable_from_some_vendor_label():
    """A bucket the vendor cannot express is a bucket that empties when SIC is out."""
    assert {bucket_for_nasdaq_sector(label) for label in VENDOR_LABELS} == set(BUCKETS)


# --- the two labels the vendor uses for "we did not classify this" ----------


def test_miscellaneous_is_not_a_sector():
    """Reading somebody else's shrug as a sector is how unrelated risk pools."""
    assert bucket_for_nasdaq_sector("Miscellaneous") is None


def test_a_blank_label_is_not_a_sector():
    assert bucket_for_nasdaq_sector("") is None
    assert bucket_for_nasdaq_sector("   ") is None
    assert bucket_for_nasdaq_sector(None) is None


def test_a_label_we_have_never_seen_is_unmapped_rather_than_guessed():
    assert bucket_for_nasdaq_sector("Consumer Services") is None
    assert bucket_for_nasdaq_sector("Transportation") is None


# --- the spelling the vendor happens to use today ---------------------------


def test_case_and_padding_do_not_change_the_answer():
    assert bucket_for_nasdaq_sector("  HEALTH care ") == "health_care"


def test_finance_is_our_financials_and_telecommunications_our_comms():
    """The vendor's names are not our names; the bucket names are ours."""
    assert bucket_for_nasdaq_sector("Finance") == "financials"
    assert bucket_for_nasdaq_sector("Telecommunications") == "communication_services"
    assert bucket_for_nasdaq_sector("Basic Materials") == "materials"


# --- the coverage report ----------------------------------------------------


def test_unmapped_labels_reports_what_the_table_missed():
    labels = ["Technology", "Consumer Services", "Miscellaneous", "Technology"]
    assert unmapped_labels(labels) == ("Consumer Services", "Miscellaneous")


def test_unmapped_labels_ignores_the_blanks():
    """An empty label is a name with no sector, not a table gap worth reporting."""
    assert unmapped_labels(["", "   ", None, "Energy"]) == ()


def test_a_vendor_renaming_a_sector_shows_up_as_a_gap_not_as_silence():
    """If the vendor ships 'Healthcare' one week, we want to hear about it."""
    assert unmapped_labels(["Healthcare"]) == ("Healthcare",)
