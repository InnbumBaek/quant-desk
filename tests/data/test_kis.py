"""What the KIS adapter refuses, which matters more than what it reads.

This credential can place an order. So the first block here is not about
parsing: it is the proof that the order endpoints are unreachable through this
code, by name and by path, and that a future reader who adds one has to delete
a test to do it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.data.kis import (
    LIVE_HOST,
    ORDER_TR,
    PAPER_HOST,
    READ_ONLY_TR,
    Balance,
    KisRefused,
    KisShapeError,
    OrderPathRefused,
    Token,
    check_read_only,
    host,
    outcome_of,
    read_balance,
    read_quote,
    read_token,
)

NOW = datetime(2026, 9, 27, 1, 0, tzinfo=UTC)
QUOTE_PATH = "/uapi/domestic-stock/v1/quotations/inquire-price"
BALANCE_PATH = "/uapi/domestic-stock/v1/trading/inquire-balance"


def quote_row(**overrides) -> dict[str, str]:
    base = {
        "stck_prpr": "72,100",
        "prdy_vrss": "1,100",
        "prdy_ctrt": "1.55",
        "acml_vol": "12,345,678",
        "stck_hgpr": "72,300",
        "stck_lwpr": "70,800",
        "stck_oprc": "71,000",
        "stck_sdpr": "71,000",
    }
    base.update(overrides)
    return base


def holding_row(symbol: str = "005930", **overrides) -> dict[str, str]:
    base = {
        "pdno": symbol,
        "prdt_name": "삼성전자",
        "hldg_qty": "10",
        "ord_psbl_qty": "10",
        "pchs_avg_pric": "70,000",
        "prpr": "72,100",
        "evlu_amt": "721,000",
        "evlu_pfls_amt": "21,000",
    }
    base.update(overrides)
    return base


def balance_body(*rows, cash: str = "1,000,000", total: str = "1,721,000") -> dict:
    return {
        "rt_cd": "0",
        "msg_cd": "MCA00000",
        "msg1": "정상처리 되었습니다.",
        "output1": list(rows),
        "output2": [{"dnca_tot_amt": cash, "tot_evlu_amt": total}],
    }


# --- the order path is refused by construction -------------------------------


@pytest.mark.parametrize("tr_id", sorted(ORDER_TR))
def test_every_order_tr_id_is_refused_by_name(tr_id: str):
    with pytest.raises(OrderPathRefused, match=tr_id):
        check_read_only(BALANCE_PATH, tr_id)


def test_the_order_path_is_refused_even_with_a_read_tr_id():
    """The tr_id is the vendor's label. The path is what actually moves money."""
    with pytest.raises(OrderPathRefused, match="order"):
        check_read_only("/uapi/domestic-stock/v1/trading/order-cash", "FHKST01010100")


@pytest.mark.parametrize(
    "path",
    [
        "/uapi/domestic-stock/v1/trading/order-cash",
        "/uapi/domestic-stock/v1/trading/order-credit",
        "/uapi/domestic-stock/v1/trading/order-rvsecncl",
        "/uapi/overseas-stock/v1/trading/order",
    ],
)
def test_no_endpoint_that_can_submit_is_reachable(path: str):
    with pytest.raises(OrderPathRefused):
        check_read_only(path, "FHKST01010100")


def test_the_buying_power_endpoint_gets_no_exception():
    """It only reads, but its path says `order`. The gate beats the convenience."""
    with pytest.raises(OrderPathRefused, match="order"):
        check_read_only("/uapi/domestic-stock/v1/trading/inquire-psbl-order", "VTTC8434R")
    assert "VTTC8908R" not in READ_ONLY_TR


def test_an_unlisted_tr_id_is_refused_rather_than_passed_through():
    with pytest.raises(OrderPathRefused, match="not on the read-only list"):
        check_read_only(QUOTE_PATH, "FHKST99999999")


def test_a_path_outside_the_allowlist_is_refused():
    with pytest.raises(OrderPathRefused, match="read-only prefixes"):
        check_read_only("/uapi/domestic-stock/v1/ksdinfo/dividend", "FHKST01010100")


def test_the_reads_this_desk_actually_needs_are_allowed():
    check_read_only(QUOTE_PATH, "FHKST01010100")
    check_read_only(BALANCE_PATH, "VTTC8434R")
    check_read_only(BALANCE_PATH, "TTTC8434R")


def test_the_read_and_order_lists_do_not_overlap():
    assert set(READ_ONLY_TR).isdisjoint(ORDER_TR)


def test_paper_is_the_default_host():
    assert host() == PAPER_HOST and host(paper=False) == LIVE_HOST
    assert PAPER_HOST != LIVE_HOST


# --- the token ---------------------------------------------------------------


def test_a_token_is_never_printed_in_full():
    token = Token("abcdefghijklmnopqrstuvwxyz", NOW + timedelta(hours=24))
    assert token.value not in token.masked
    assert "26 chars" in token.masked


def test_a_token_is_spent_before_it_actually_expires():
    """A token that dies mid-request is a failure that reads like a refusal."""
    token = Token("x" * 20, NOW + timedelta(minutes=5))
    assert token.usable_at(NOW) is False
    assert Token("x" * 20, NOW + timedelta(hours=2)).usable_at(NOW) is True


def test_the_token_reply_becomes_an_expiry_we_can_check():
    token = read_token({"access_token": "t" * 30, "expires_in": "86400", "token_type": "Bearer"}, now=NOW)
    assert token.expires_at == NOW + timedelta(seconds=86400)
    assert token.header.startswith("Bearer ")


def test_a_refused_token_request_is_a_refusal_not_an_empty_token():
    with pytest.raises(KisRefused, match="EGW00133"):
        read_token({"error_code": "EGW00133", "error_description": "접근토큰 발급 잠시 후 다시 시도하세요"})


def test_an_expiry_we_cannot_read_is_an_error_rather_than_a_guess():
    with pytest.raises(KisShapeError, match="expires_in"):
        read_token({"access_token": "t" * 30, "expires_in": "soon"})


def test_a_token_that_is_already_spent_is_refused():
    with pytest.raises(KisShapeError, match="already spent"):
        read_token({"access_token": "t" * 30, "expires_in": "0"})


# --- the outcome behind HTTP 200 ---------------------------------------------


def test_a_reply_with_no_outcome_field_names_what_it_did_carry():
    with pytest.raises(KisShapeError, match="msg1"):
        outcome_of({"msg1": "정상"})


def test_a_refusal_carries_the_vendor_code_a_human_can_look_up():
    with pytest.raises(KisRefused) as caught:
        read_quote({"rt_cd": "1", "msg_cd": "OPSQ0001", "msg1": "모의투자 미지원 API"}, "005930")
    assert caught.value.code == "OPSQ0001" and "미지원" in caught.value.message


def test_a_rate_limit_is_retryable_and_a_rejected_key_is_not():
    assert KisRefused("EGW00201", "초당 거래건수를 초과").retryable is True
    assert KisRefused("EGW00123", "유효하지 않은 appkey").retryable is False


# --- the quote ---------------------------------------------------------------


def test_a_quote_reads_through_the_thousands_separators():
    quote = read_quote({"rt_cd": "0", "output": quote_row()}, "005930")
    assert quote.price == pytest.approx(72100.0) and quote.volume == 12345678


def test_an_empty_field_is_a_shape_error_and_never_a_zero_price():
    """A zero price is a halt. A zero from a parser is a fabrication."""
    with pytest.raises(KisShapeError, match="stck_prpr is empty"):
        read_quote({"rt_cd": "0", "output": quote_row(stck_prpr="")}, "005930")


def test_a_missing_field_names_the_fields_that_were_there():
    row = quote_row()
    del row["stck_hgpr"]
    with pytest.raises(KisShapeError, match="stck_prpr"):
        read_quote({"rt_cd": "0", "output": row}, "005930")


# --- the balance, which is what reconciliation compares to -------------------


def test_a_balance_reads_positions_cash_and_account_value():
    balance = read_balance(balance_body(holding_row()))
    assert balance.symbols == ("005930",)
    assert balance.cash == pytest.approx(1_000_000.0)
    assert balance.total_value == pytest.approx(1_721_000.0)
    assert balance.holdings[0].unrealised == pytest.approx(21_000.0)


def test_a_closed_position_is_not_a_holding():
    balance = read_balance(balance_body(holding_row(), holding_row("000660", hldg_qty="0")))
    assert balance.symbols == ("005930",)


def test_holdings_come_back_sorted_so_a_rerun_compares_cleanly():
    balance = read_balance(balance_body(holding_row("035720"), holding_row("000660")))
    assert balance.symbols == ("000660", "035720")


def test_totals_sent_as_an_object_rather_than_a_list_are_read_too():
    body = balance_body(holding_row())
    body["output2"] = {"dnca_tot_amt": "5", "tot_evlu_amt": "6"}
    assert read_balance(body).cash == pytest.approx(5.0)


def test_a_balance_with_no_totals_row_is_refused():
    body = balance_body(holding_row())
    body["output2"] = []
    with pytest.raises(KisShapeError, match="no totals row"):
        read_balance(body)


def test_a_holding_with_no_symbol_is_refused_rather_than_dropped():
    with pytest.raises(KisShapeError, match="pdno"):
        read_balance(balance_body(holding_row(symbol="")))


def test_an_empty_account_is_an_empty_balance_not_an_error():
    balance = read_balance(balance_body())
    assert balance.holdings == () and balance.total_value == pytest.approx(1_721_000.0)


def test_an_empty_balance_has_no_symbols():
    assert Balance((), 0.0, 0.0).symbols == ()
