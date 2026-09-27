"""Reading KRX KIND's listed-company table, and refusing what is not it.

The reason this parser exists is that `sector_max` cannot be checked for a
Korean name. The reason these tests are mostly refusals is that a sector source
which quietly returns a short or shifted table is worse than none: it puts a
wrong classification behind a concentration limit, where nobody looks at it
again.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.data.kind import (
    ENCODING,
    EXPECTED_HEADER,
    MARKET,
    KindShapeError,
    check_header,
    decode,
    industry_labels,
    read_listings,
    read_table,
    unmapped,
)
from core.data.ksic import bucket_for_label

ROW = "<tr>" + "".join(f"<td>{cell}</td>" for cell in EXPECTED_HEADER) + "</tr>"


def row(
    name: str = "삼성전자",
    venue: str = "코스피",
    ticker: str = "005930",
    industry: str = "통신 및 방송 장비 제조업",
    product: str = "휴대폰",
    listed: str = "1975-06-11",
    month: str = "12월",
    ceo: str = "한종희",
    site: str = "http://www.samsung.com/sec",
    region: str = "경기도",
) -> str:
    cells = (name, venue, ticker, industry, product, listed, month, ceo, site, region)
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def table(*rows: str, header: tuple[str, ...] = EXPECTED_HEADER) -> bytes:
    head = "<tr>" + "".join(f"<th>{cell}</th>" for cell in header) + "</tr>"
    body = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
        '<meta http-equiv="Content-Type" content="text/html; charset=euc-kr" />'
        "<title>상장법인목록</title></head><body>"
        f'<table class="bbs_tb" border="1">{head}{"".join(rows)}</table></body></html>'
    )
    return body.encode(ENCODING)


def many(count: int) -> bytes:
    return table(*(row(name=f"회사{i}", ticker=f"{i:06d}") for i in range(count)))


# --- the shape the vendor actually sends -------------------------------------


def test_a_row_becomes_a_korean_listing():
    listings, _census = read_listings(table(row()))
    assert len(listings) == 1
    listing = listings[0]
    assert listing.symbol == "005930"
    assert listing.market == MARKET
    assert listing.name == "삼성전자"
    assert listing.listed_on == date(1975, 6, 11)


def test_the_listing_date_comes_along_which_the_us_file_has_never_had():
    listings, census = read_listings(table(row(listed="2020-01-02")))
    assert listings[0].listed_on == date(2020, 1, 2)
    assert census.undated == ()


def test_the_vendor_label_is_preserved_per_symbol_and_counted():
    listings, census = read_listings(
        table(
            row(ticker="005930", industry="통신 및 방송 장비 제조업"),
            row(ticker="000660", industry="통신 및 방송 장비 제조업"),
            row(ticker="005380", industry="자동차용 엔진 및 자동차 제조업"),
        )
    )
    assert len(listings) == 3
    assert census.by_symbol["005380"] == "자동차용 엔진 및 자동차 제조업"
    assert census.industries["통신 및 방송 장비 제조업"] == 2
    assert census.distinct_industries == 2


def test_the_census_orders_labels_by_how_common_they_are():
    """The table is written from the top of this list downwards."""
    _listings, census = read_listings(
        table(
            row(ticker="000001", industry="흔한 업종"),
            row(ticker="000002", industry="흔한 업종"),
            row(ticker="000003", industry="드문 업종"),
        )
    )
    assert list(census.industries) == ["흔한 업종", "드문 업종"]


def test_no_listing_is_given_a_bucket_unless_a_table_is_passed_in():
    """This parser has no opinion about what a label means. A caller that has not
    chosen a table gets names it cannot trade, not names in a guessed bucket."""
    listings, _census = read_listings(table(row()))
    assert listings[0].sector is None
    assert not listings[0].classified


def test_the_caller_supplies_the_table_and_the_bucket_lands_on_the_listing():
    listings, _census = read_listings(table(row()), classify=bucket_for_label)
    assert listings[0].sector == "technology"
    assert listings[0].classified


def test_a_table_that_does_not_know_the_label_leaves_the_listing_unorderable():
    listings, _census = read_listings(table(row(industry="아직 없는 업종")), classify=bucket_for_label)
    assert listings[0].sector is None
    assert not listings[0].classified


def test_the_raw_label_survives_even_when_a_bucket_was_assigned():
    """The bucket is a judgement; the label is the evidence for it."""
    _listings, census = read_listings(table(row()), classify=bucket_for_label)
    assert census.by_symbol["005930"] == "통신 및 방송 장비 제조업"


def test_a_date_with_slashes_is_read_too():
    listings, _census = read_listings(table(row(listed="1975/06/11")))
    assert listings[0].listed_on == date(1975, 6, 11)


# --- refusals ----------------------------------------------------------------


def test_a_body_that_is_not_euc_kr_is_refused_rather_than_partly_read():
    """A partial decode drops exactly the rows with Korean names in them."""
    with pytest.raises(KindShapeError, match="does not decode"):
        decode(b"\xff\xfe\x00not euc-kr\xff")


def test_a_body_with_no_table_row_is_refused_and_shows_what_came():
    with pytest.raises(KindShapeError, match="no table row"):
        read_table("<html><body>maintenance</body></html>")


def test_a_shifted_header_is_refused_naming_both():
    """Reading by position would start taking 주요제품 as the industry."""
    shifted = (
        "회사명",
        "시장구분",
        "종목코드",
        "주요제품",
        "업종",
        "상장일",
        "결산월",
        "대표자명",
        "홈페이지",
        "지역",
    )
    with pytest.raises(KindShapeError, match="not"):
        read_listings(table(row(), header=shifted))


def test_the_market_segment_column_is_read_because_the_first_run_found_it():
    """It arrived at position two and the header check is what caught it."""
    assert EXPECTED_HEADER[1] == "시장구분"
    _listings, census = read_listings(
        table(row(ticker="005930", venue="코스피"), row(ticker="035420", venue="코스닥"))
    )
    assert census.venue_by_symbol == {"005930": "코스피", "035420": "코스닥"}
    assert census.venues == {"코스닥": 1, "코스피": 1}


def test_a_header_missing_the_market_segment_is_refused_too():
    """The shape this parser was first written for is now itself a mismatch."""
    old = ("회사명", "종목코드", "업종", "주요제품", "상장일", "결산월", "대표자명", "홈페이지", "지역")
    with pytest.raises(KindShapeError, match="시장구분"):
        check_header(list(old))


def test_an_added_column_is_refused_rather_than_absorbed():
    extended = (*EXPECTED_HEADER, "신규")
    with pytest.raises(KindShapeError):
        check_header(list(extended))


def test_a_row_with_the_wrong_number_of_cells_is_dropped_by_name():
    short = "<tr>" + "".join(f"<td>{c}</td>" for c in ("회사", "코스피", "005930")) + "</tr>"
    _listings, census = read_listings(table(row(ticker="000660"), short))
    assert len(census.dropped) == 1
    assert "3 cell(s)" in next(iter(census.dropped.values()))


def test_a_ticker_of_the_wrong_length_is_dropped_and_its_name_is_recorded():
    """The name is in the reason so the next run can say what the row was."""
    _listings, census = read_listings(table(row(ticker="000660"), row(ticker="5930")))
    assert "5930" in census.dropped
    assert "six-character" in census.dropped["5930"]
    assert "삼성전자" in census.dropped["5930"]


def test_a_code_with_a_letter_is_kept_and_counted_as_its_own_class():
    """63 of these turned up on the first run, all one shape (ADR-0028)."""
    listings, census = read_listings(table(row(ticker="0001A0", name="어떤회사"), row(ticker="005930")))
    assert sorted(x.symbol for x in listings) == ["0001A0", "005930"]
    assert census.nonnumeric == {"0001A0": "어떤회사"}


def test_a_lowercase_or_punctuated_code_is_still_refused():
    _listings, census = read_listings(table(row(ticker="000660"), row(ticker="00a1b0")))
    assert "00a1b0" in census.dropped


def test_a_duplicate_reason_names_the_columns_that_actually_differ():
    """Two runs reported duplicates as "same name, same venue", which said nothing.
    The comparison now covers all ten columns, including the six this module
    discards, because that is where the difference turned out to be."""
    _listings, census = read_listings(
        table(
            row(ticker="000480", name="시알홀딩스", ceo="김갑"),
            row(ticker="000480", name="시알홀딩스", ceo="이을"),
        )
    )
    reason = census.dropped["000480"]
    assert "대표자명" in reason and "김갑" in reason and "이을" in reason
    assert "회사명" not in reason, "a column that matches is not part of the explanation"


def test_a_verbatim_repeat_says_so_because_that_is_itself_the_answer():
    """A row the vendor sends twice unchanged is a different fact from a row that
    differs in a column we discard, and the old reason could not tell them apart."""
    _listings, census = read_listings(table(row(ticker="000480"), row(ticker="000480")))
    assert "verbatim repeat" in census.dropped["000480"]


def test_a_duplicate_across_two_venues_still_names_the_venue():
    _listings, census = read_listings(
        table(row(ticker="000480", venue="유가"), row(ticker="000480", venue="코스닥"))
    )
    assert "시장구분" in census.dropped["000480"]
    assert "유가" in census.dropped["000480"] and "코스닥" in census.dropped["000480"]


def test_a_repeated_ticker_keeps_the_first_and_records_the_second():
    listings, census = read_listings(
        table(row(ticker="005930", name="첫번째"), row(ticker="005930", name="두번째"))
    )
    assert [x.name for x in listings] == ["첫번째"]
    assert "appears twice" in census.dropped["005930"]


def test_a_table_whose_every_row_is_unreadable_is_refused():
    """An empty universe written to disk looks exactly like a universe."""
    with pytest.raises(KindShapeError, match="none of them was a listing"):
        read_listings(table(row(ticker="abc"), row(ticker="12")))


def test_an_unreadable_listing_date_is_recorded_not_invented():
    listings, census = read_listings(table(row(ticker="005930", listed="상장일")))
    assert listings[0].listed_on is None
    assert census.undated == ("005930",)


def test_an_impossible_date_is_recorded_as_undated():
    _listings, census = read_listings(table(row(listed="2020-02-31")))
    assert census.undated == ("005930",)


def test_a_blank_industry_is_not_counted_as_a_label():
    """An empty string is not a classification, and counting it would invent one."""
    _listings, census = read_listings(table(row(ticker="005930", industry="")))
    assert census.industries == {}
    assert census.by_symbol["005930"] == ""


# --- the census is what the bucket table gets built from ---------------------


def test_the_distinct_labels_can_be_read_on_their_own():
    labels = industry_labels(table(row(ticker="000001", industry="가"), row(ticker="000002", industry="나")))
    assert sorted(labels) == ["가", "나"]


def test_unmapped_labels_are_reportable_before_they_bite():
    assert unmapped(["가", "나", "다"], {"나": "technology"}) == ("가", "다")


def test_a_mapping_that_covers_everything_reports_nothing():
    assert unmapped(["가"], {"가": "technology"}) == ()
