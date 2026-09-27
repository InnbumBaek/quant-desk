"""The DART reader: telling a quiet day from a refused key, since both arrive as 200.

Whether DART still sends these field names is settled on the runner with a key
(this container only reached the key-less error page). These fix what the code
does with each kind of answer.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.data.dart import (
    NO_DATA,
    OK,
    RETRYABLE_STATUS,
    STATUS_TEXT,
    DartRefused,
    DartShapeError,
    read_page,
    read_row,
    status_of,
)


def row(**overrides) -> dict[str, str]:
    base = {
        "corp_code": "00126380",
        "corp_name": "삼성전자",
        "stock_code": "005930",
        "corp_cls": "Y",
        "report_nm": "주요사항보고서(자기주식취득결정)",
        "rcept_no": "20260925000123",
        "flr_nm": "삼성전자",
        "rcept_dt": "20260925",
    }
    base.update(overrides)
    return base


def page(*rows, status: str = OK, **extra) -> dict[str, object]:
    body: dict[str, object] = {
        "status": status,
        "message": STATUS_TEXT.get(status, ""),
        "page_no": 1,
        "page_count": 100,
        "total_count": len(rows) or 1,
        "total_page": 1,
        "list": list(rows) or [row()],
    }
    body.update(extra)
    return body


# --- the status code, read first --------------------------------------------


def test_a_quiet_window_is_an_empty_page_not_a_failure():
    """013 means the window really held nothing, which is what a holiday looks like."""
    got = read_page({"status": NO_DATA, "message": "조회된 데이타가 없습니다"})
    assert got.disclosures == () and got.total_count == 0 and not got.has_more


def test_a_rejected_key_is_not_a_quiet_window():
    """The mistake that matters: both arrive as HTTP 200 with no rows."""
    with pytest.raises(DartRefused) as caught:
        read_page({"status": "010", "message": "등록되지 않은 키입니다"})
    assert caught.value.status == "010"
    assert not caught.value.retryable, "waiting will not register a key"


def test_a_rate_limit_and_a_maintenance_window_are_worth_retrying():
    for status in ("020", "800"):
        with pytest.raises(DartRefused) as caught:
            read_page({"status": status})
        assert caught.value.retryable, status


def test_every_retryable_code_is_a_code_dart_documents():
    assert RETRYABLE_STATUS <= set(STATUS_TEXT)


def test_a_refusal_carries_dart_own_words_when_it_sends_none():
    with pytest.raises(DartRefused, match="시스템 점검"):
        read_page({"status": "800"})


def test_a_reply_without_a_status_is_not_a_dart_reply():
    """The key-less endpoint answers with an HTML error page; that must not parse."""
    with pytest.raises(DartShapeError, match="no status field"):
        status_of({"html": "<!DOCTYPE html>"})


def test_a_reply_that_is_not_an_object_is_refused():
    with pytest.raises(DartShapeError):
        status_of(["status", "000"])


def test_success_with_no_list_is_an_error_because_013_exists_for_that():
    with pytest.raises(DartShapeError, match="013"):
        read_page({"status": OK, "list": None})


# --- one row ----------------------------------------------------------------


def test_a_row_becomes_a_disclosure():
    got = read_row(row())
    assert got.receipt_no == "20260925000123"
    assert got.filed_on == date(2026, 9, 25)
    assert got.corp_name == "삼성전자" and got.stock_code == "005930"


def test_the_receipt_number_is_what_the_filing_is_read_at():
    assert "20260925000123" in read_row(row()).url


def test_an_unlisted_filer_is_marked_rather_than_dropped():
    """Unlisted companies file too; they are just not tradable."""
    assert not read_row(row(stock_code="")).listed
    assert read_row(row()).listed


def test_a_report_title_is_collapsed_to_one_line():
    """Titles arrive with stray whitespace and a multi-line title breaks every report."""
    assert read_row(row(report_nm=" 분기보고서 \n (2026.06) ")).report == "분기보고서 (2026.06)"


def test_a_row_with_no_receipt_number_cannot_be_deduped():
    with pytest.raises(DartShapeError, match="rcept_no"):
        read_row(row(rcept_no=""))


def test_a_missing_field_says_what_was_there_instead():
    """Nobody here has held a keyed response; the first run has to teach us."""
    with pytest.raises(DartShapeError) as caught:
        read_row({"rcept_no": "1", "reportName": "x"})
    assert "report_nm" in str(caught.value) and "reportName" in str(caught.value)


@pytest.mark.parametrize("bad", ["2026-09-25", "260925", "", "2026092X"])
def test_a_filing_date_that_is_not_eight_digits_is_refused(bad):
    with pytest.raises(DartShapeError, match="rcept_dt"):
        read_row(row(rcept_dt=bad))


# --- paging -----------------------------------------------------------------


def test_a_page_knows_whether_another_follows():
    first = read_page(page(row(), page_no=1, total_page=3))
    last = read_page(page(row(), page_no=3, total_page=3))
    assert first.has_more and not last.has_more


def test_the_totals_come_back_as_numbers_even_when_sent_as_strings():
    got = read_page(page(row(), page_no="2", total_page="7", total_count="140"))
    assert (got.page_no, got.total_pages, got.total_count) == (2, 7, 140)


def test_a_total_that_is_not_a_number_is_an_error():
    with pytest.raises(DartShapeError, match="total_page"):
        read_page(page(row(), total_page="many"))


def test_a_row_that_is_not_an_object_fails_the_page():
    with pytest.raises(DartShapeError, match="str"):
        read_page(page(row(), "nonsense"))
