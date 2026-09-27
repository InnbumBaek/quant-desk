"""The Korean industry label -> bucket table.

Most of these tests are about the table's *shape*, not its contents. Whether
`반도체 제조업` is technology is a judgement recorded in ADR-0029 and reviewed as a
table; what code can check is that the table covers the market it was written
from, produces only buckets the US side also produces, never invents a bucket for
a label nobody has looked at, and matches `core/data/sic.py` on the five
businesses the two tables were made to agree on.

The census these numbers come from is `registry/universe/kr_industry.source.json`
as committed on 2026-09-27: 158 labels, 2,756 names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.data.ksic import LABELS, bucket_counts, bucket_for_label, coverage, unmapped
from core.data.sic import BUCKETS, bucket_for_sic

CENSUS = Path("registry/universe/kr_industry.source.json")


def census() -> dict:
    if not CENSUS.exists():  # pragma: no cover - the artifact is committed
        pytest.skip(f"{CENSUS} has not been fetched in this checkout")
    return json.loads(CENSUS.read_text(encoding="utf-8"))


# --- the table covers the market it was written from -------------------------


def test_every_label_the_last_census_saw_has_a_bucket():
    """The reason the fetch ran twice before this table existed."""
    missing = unmapped(census()["industry_counts"])
    assert missing == (), f"labels with no bucket: {missing}"


def test_the_table_carries_no_line_the_census_never_saw():
    """A line for a label the vendor does not send is a line nobody can check."""
    seen = set(census()["industry_counts"])
    invented = sorted(set(LABELS) - seen)
    assert invented == [], f"lines for labels the census does not have: {invented}"


def test_every_bucket_is_one_the_us_table_also_produces():
    assert set(LABELS.values()) <= set(BUCKETS)


def test_no_single_bucket_swallows_the_market():
    """A classification fine enough never to bind is the same as no limit; one
    coarse enough to put everything in one bucket is the same thing from the other
    side. Both are invisible without a number, so here is the number."""
    counts = census()["industry_counts"]
    weighted: dict[str, int] = {}
    for label, count in counts.items():
        bucket = bucket_for_label(label)
        assert bucket is not None
        weighted[bucket] = weighted.get(bucket, 0) + count
    total = sum(weighted.values())
    largest = max(weighted.values())
    assert largest / total < 0.40, f"one bucket holds {largest / total:.0%} of names: {weighted}"
    # And every bucket the table can produce is actually reached by this market,
    # so no line of the table is dead.
    assert set(weighted) == set(LABELS.values())


# --- an unknown label is never given a bucket --------------------------------


def test_a_label_the_table_has_not_seen_returns_none():
    assert bucket_for_label("아직 없는 업종") is None


def test_a_blank_or_missing_label_returns_none():
    assert bucket_for_label("") is None
    assert bucket_for_label(None) is None
    assert bucket_for_label("   ") is None


def test_a_label_is_matched_whole_and_never_by_substring():
    """`기타 금융업` is a prefix of nothing here, but `기타 금속 가공제품 제조업`
    shares its first two characters, and a substring rule would put a metal
    fabricator in financials the first time the vendor reworded a line."""
    assert bucket_for_label("기타 금융업") == "financials"
    assert bucket_for_label("기타 금속 가공제품 제조업") == "industrials"
    assert bucket_for_label("기타 금융") is None
    assert bucket_for_label("기타 금융업 및 보험") is None


def test_surrounding_whitespace_is_tolerated_but_nothing_else_is():
    assert bucket_for_label("  반도체 제조업  ") == "technology"
    assert bucket_for_label("반도체제조업") is None


def test_a_non_string_is_not_a_label():
    assert bucket_for_label(42) is None  # type: ignore[arg-type]
    assert bucket_for_label(["반도체 제조업"]) is None  # type: ignore[arg-type]


# --- the five lines the two tables were made to agree on ---------------------


@pytest.mark.parametrize(
    ("label", "sic", "bucket"),
    [
        # Battery cells: electrical equipment on both sides. `sic.py` gained
        # (3691, 3692) for this.
        ("일차전지 및 이차전지 제조업", 3691, "industrials"),
        # Communications and broadcast equipment: technology hardware on both
        # sides. `sic.py` gained (3661, 3669).
        ("통신 및 방송 장비 제조업", 3663, "technology"),
        # Waste treatment: commercial services, not a utility. `sic.py` gained
        # (4950, 4959).
        ("폐기물 처리업", 4959, "industrials"),
        # Food and drink distribution. `sic.py` gained (5140, 5159).
        ("음·식료품 및 담배 도매업", 5141, "consumer_staples"),
        ("산업용 농·축산물 및 동·식물 도매업", 5153, "consumer_staples"),
    ],
)
def test_the_same_business_counts_against_the_same_limit_in_both_markets(label, sic, bucket):
    """`sector_max` is fund-level: a split here makes the cap under-measure."""
    assert bucket_for_label(label) == bucket
    assert bucket_for_sic(sic) == bucket


def test_the_one_disagreement_left_standing_is_recorded_not_hidden():
    """SIC 7370 puts a search engine and a systems integrator on one code, so the
    US side cannot follow KSIC in separating 포털 out. ADR-0029 says so; this test
    is here so the gap cannot close silently and go unnoticed."""
    assert bucket_for_label("자료처리, 호스팅, 포털 및 기타 인터넷 정보매개 서비스업") == (
        "communication_services"
    )
    assert bucket_for_sic(7370) == "technology"


# --- the reporting helpers the run prints ------------------------------------


def test_bucket_counts_names_the_unmapped_rather_than_dropping_them():
    counts = bucket_counts({"A": "반도체 제조업", "B": "반도체 제조업", "C": "아직 없는 업종"})
    assert counts == {"technology": 2, "(unmapped)": 1}


def test_bucket_counts_is_ordered_largest_first():
    counts = bucket_counts({"A": "의약품 제조업", "B": "반도체 제조업", "C": "반도체 제조업"})
    assert list(counts) == ["technology", "health_care"]


def test_coverage_is_the_share_that_can_be_placed():
    assert coverage({"A": "반도체 제조업", "B": "아직 없는 업종"}) == pytest.approx(0.5)


def test_coverage_of_nothing_is_zero_not_one():
    """An empty universe has not been classified; it has not been read."""
    assert coverage({}) == 0.0


def test_unmapped_is_sorted_and_deduplicated():
    assert unmapped(["나", "가", "나", "반도체 제조업"]) == ("가", "나")
