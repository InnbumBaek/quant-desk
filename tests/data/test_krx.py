"""What the KRX reader accepts, and what it refuses to turn into a bar.

Whether KRX still sends these field names is settled on the runner with a key
(this container only ever got 401). These fix what the code does with each
shape, including the shape we will find out we guessed wrong: a missing field
has to name the fields that were there, or the first keyed run teaches us
nothing.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.data.krx import (
    BLOCK,
    FIELDS,
    VENUES,
    KrxNoSession,
    KrxShapeError,
    read_day,
    read_day_with_rejects,
    read_row,
    series,
)
from core.data.markets import MARKETS


def row(**overrides) -> dict[str, str]:
    base = {
        "BAS_DD": "20260925",
        "ISU_CD": "005930",
        "ISU_NM": "삼성전자",
        "MKT_NM": "KOSPI",
        "TDD_OPNPRC": "71,000",
        "TDD_HGPRC": "72,300",
        "TDD_LWPRC": "70,800",
        "TDD_CLSPRC": "72,100",
        "ACC_TRDVOL": "12,345,678",
        "ACC_TRDVAL": "889,123,456,700",
        "MKTCAP": "430,512,000,000,000",
        "LIST_SHRS": "5,969,782,550",
    }
    base.update(overrides)
    return base


def day(*rows) -> dict[str, list[dict[str, str]]]:
    return {BLOCK: list(rows) or [row()]}


# --- one row ----------------------------------------------------------------


def test_a_row_becomes_a_bar():
    bar = read_row(row())
    assert bar.day == date(2026, 9, 25)
    assert bar.symbol == "005930" and bar.name == "삼성전자"
    assert (bar.open, bar.high, bar.low, bar.close) == (71000.0, 72300.0, 70800.0, 72100.0)
    assert bar.volume == 12345678
    assert bar.shares == 5969782550


def test_every_venue_maps_to_a_market_the_desk_declares():
    assert set(VENUES.values()) <= set(MARKETS)


def test_the_market_is_the_one_calendar_kospi_and_kosdaq_share():
    assert read_row(row(MKT_NM="KOSPI")).market == read_row(row(MKT_NM="KOSDAQ")).market == "KR"


def test_a_missing_field_says_what_was_there_instead():
    """The schema is from a document, not from bytes; the first keyed run has to teach us."""
    short = {"ISU_CD": "005930", "CLOSE": "72100"}
    with pytest.raises(KrxShapeError) as caught:
        read_row(short)
    message = str(caught.value)
    assert "TDD_CLSPRC" in message, "it must name what it wanted"
    assert "CLOSE" in message, "and what it got, or the next run learns nothing"


@pytest.mark.parametrize("empty", ["", " ", "-", "null"])
def test_an_empty_price_is_an_error_not_a_zero(empty):
    """A halted name has no close, and a zero close reaches a limit report as a real price."""
    with pytest.raises(KrxShapeError, match="TDD_CLSPRC"):
        read_row(row(TDD_CLSPRC=empty))


def test_a_price_that_is_not_a_number_is_an_error():
    with pytest.raises(KrxShapeError, match="not a number"):
        read_row(row(TDD_CLSPRC="상한가"))


def test_a_zero_that_was_actually_sent_is_kept():
    """Zero volume is a measurement: a listed name can trade nothing all day."""
    assert read_row(row(ACC_TRDVOL="0")).volume == 0


def test_an_unknown_venue_is_refused_rather_than_defaulted():
    with pytest.raises(KrxShapeError, match="MKT_NM"):
        read_row(row(MKT_NM="NASDAQ"))


def test_a_row_with_no_code_has_nowhere_to_go():
    with pytest.raises(KrxShapeError, match="ISU_CD"):
        read_row(row(ISU_CD=""))


@pytest.mark.parametrize("bad", ["2026-09-25", "260925", "", "2026092X"])
def test_a_date_that_is_not_eight_digits_is_refused(bad):
    with pytest.raises(KrxShapeError, match="BAS_DD"):
        read_row(row(BAS_DD=bad))


# --- one day ----------------------------------------------------------------


def test_a_day_reads_every_row():
    bars = read_day(day(row(), row(ISU_CD="000660", ISU_NM="SK하이닉스")))
    assert {bar.symbol for bar in bars} == {"005930", "000660"}


def test_a_response_without_the_block_says_which_keys_it_had():
    with pytest.raises(KrxShapeError) as caught:
        read_day({"outBlock1": []})
    assert "outBlock1" in str(caught.value) and BLOCK in str(caught.value)


def test_an_error_payload_is_not_an_empty_day():
    """A key that is refused answers with a message, and that must not read as a holiday."""
    with pytest.raises(KrxShapeError):
        read_day({"error": "AUTH_KEY is invalid"})


def test_an_empty_block_is_a_holiday_and_says_so_in_its_own_type():
    """KRX is the only Korean trading calendar here, so an empty weekday is the measurement."""
    with pytest.raises(KrxNoSession):
        read_day({BLOCK: []})


def test_a_holiday_is_still_an_error_for_a_caller_that_does_not_care_which():
    """Subclassing keeps `except KrxShapeError` correct for callers that want a day or nothing."""
    assert issubclass(KrxNoSession, KrxShapeError)


def test_a_halted_name_does_not_cost_the_whole_day():
    bars, rejected = read_day_with_rejects(day(row(), row(ISU_CD="900110", TDD_CLSPRC="")))
    assert [bar.symbol for bar in bars] == ["005930"]
    assert len(rejected) == 1 and "900110" in rejected[0]


def test_the_strict_reader_fails_the_day_and_says_how_many():
    with pytest.raises(KrxShapeError, match="1 of 2 rows"):
        read_day(day(row(), row(ISU_CD="900110", TDD_CLSPRC="")))


def test_a_row_that_is_not_an_object_is_counted_not_crashed_on():
    _bars, rejected = read_day_with_rejects({BLOCK: [row(), "nonsense"]})
    assert len(rejected) == 1 and "str" in rejected[0]


# --- the transpose ----------------------------------------------------------


def test_days_of_every_symbol_become_symbols_of_every_day():
    """KRX answers one day at a time; a panel wants one symbol at a time."""
    monday = read_day(day(row(BAS_DD="20260921"), row(BAS_DD="20260921", ISU_CD="000660")))
    tuesday = read_day(day(row(BAS_DD="20260922"), row(BAS_DD="20260922", ISU_CD="000660")))

    grouped = series([*tuesday, *monday])
    assert set(grouped) == {"005930", "000660"}
    assert [bar.day for bar in grouped["005930"]] == [date(2026, 9, 21), date(2026, 9, 22)]


def test_the_fields_we_read_are_the_fields_we_declare():
    """A field recorded and never read rots; this keeps the table and the bar in step."""
    bar = read_row(row())
    for attribute in FIELDS.values():
        assert hasattr(bar, attribute), attribute
