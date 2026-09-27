"""What the macro reader does with a revised number, a gap, and a refusal.

The expensive mistake here is not a crash. It is a backtest that reads GDP for
2015Q1 as revised in 2026 and looks perfectly well-dated while doing it. So the
vintage flag is fixed by tests, not just documented, and so is the difference
between "no value for this period" and "zero".
"""

from __future__ import annotations

from datetime import date

import pytest

from core.data.macro import (
    ECOS_NO_DATA,
    VINTAGE_CURRENT,
    VINTAGE_INITIAL,
    VINTAGE_UNKNOWN,
    MacroRefused,
    MacroSeries,
    MacroShapeError,
    read_ecos,
    read_fred,
    to_rows,
)


def fred(*observations) -> dict[str, object]:
    return {
        "realtime_start": "2026-09-27",
        "observation_start": "1776-07-04",
        "units": "lin",
        "count": len(observations),
        "observations": list(observations),
    }


def obs(day: str, value: str) -> dict[str, str]:
    return {"realtime_start": "2026-09-27", "realtime_end": "9999-12-31", "date": day, "value": value}


def ecos(*rows) -> dict[str, object]:
    return {"StatisticSearch": {"list_total_count": len(rows), "row": list(rows)}}


def ecos_row(time: str, value: str, **overrides) -> dict[str, str]:
    base = {
        "STAT_CODE": "722Y001",
        "STAT_NAME": "1.3.2. 한국은행 기준금리",
        "ITEM_CODE1": "0101000",
        "ITEM_NAME1": "한국은행 기준금리",
        "UNIT_NAME": "연%",
        "TIME": time,
        "DATA_VALUE": value,
    }
    base.update(overrides)
    return base


# --- the vintage, which is the whole point ----------------------------------


def test_a_series_carries_which_vintage_it_is():
    series = read_fred(fred(obs("2026-08-01", "3.1")), "CPIAUCSL", vintage=VINTAGE_INITIAL)
    assert series.vintage == VINTAGE_INITIAL
    assert series.point_in_time is True


def test_the_current_revision_is_not_point_in_time():
    """Today's value at a 2015 timestamp is look-ahead, and the flag has to say so."""
    series = read_fred(fred(obs("2015-01-01", "3.1")), "GDP", vintage=VINTAGE_CURRENT)
    assert series.point_in_time is False


def test_an_unspecified_vintage_is_unknown_rather_than_assumed_safe():
    assert read_fred(fred(obs("2026-08-01", "3.1")), "CPIAUCSL").vintage == VINTAGE_UNKNOWN


def test_a_vintage_nobody_defined_is_refused():
    with pytest.raises(MacroShapeError, match="vintage"):
        MacroSeries("CPIAUCSL", "fred", (), vintage="point_in_time_probably")


def test_ecos_never_claims_to_be_point_in_time():
    """ECOS revises in place and exposes no vintage parameter, so it cannot claim one."""
    series = read_ecos(ecos(ecos_row("202609", "2.5")), "722Y001")
    assert series.vintage == VINTAGE_UNKNOWN and series.point_in_time is False


# --- a missing value is missing ---------------------------------------------


def test_a_fred_gap_is_dropped_and_never_zeroed():
    series = read_fred(fred(obs("2026-08-01", "3.1"), obs("2026-09-01", ".")), "CPIAUCSL")
    assert [o.day for o in series.observations] == [date(2026, 8, 1)]
    assert all(o.value != 0.0 for o in series.observations)


def test_a_real_zero_survives():
    """A zero policy rate is an event. Only `.` is absence."""
    series = read_fred(fred(obs("2015-01-01", "0.0")), "DFF")
    assert [o.value for o in series.observations] == [0.0]


def test_an_ecos_dash_is_a_gap_too():
    series = read_ecos(ecos(ecos_row("202608", "2.5"), ecos_row("202609", "-")), "722Y001")
    assert [o.day for o in series.observations] == [date(2026, 8, 1)]


def test_a_thousands_separator_is_a_number_not_a_shape_error():
    series = read_ecos(ecos(ecos_row("2026Q2", "1,234.5")), "200Y001")
    assert series.observations[0].value == pytest.approx(1234.5)


def test_a_value_that_is_not_a_number_is_refused_rather_than_guessed():
    with pytest.raises(MacroShapeError, match="not a number"):
        read_fred(fred(obs("2026-08-01", "n/a")), "CPIAUCSL")


# --- ECOS's period stamps ---------------------------------------------------


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2026", date(2026, 1, 1)),
        ("2026Q1", date(2026, 1, 1)),
        ("2026Q3", date(2026, 7, 1)),
        ("202609", date(2026, 9, 1)),
        ("20260925", date(2026, 9, 25)),
    ],
)
def test_every_ecos_period_becomes_the_first_day_of_that_period(stamp: str, expected: date):
    series = read_ecos(ecos(ecos_row(stamp, "2.5")), "722Y001")
    assert series.observations[0].day == expected


def test_a_period_ecos_does_not_document_is_refused():
    with pytest.raises(MacroShapeError, match="not a period"):
        read_ecos(ecos(ecos_row("2026-09", "2.5")), "722Y001")


# --- the outcome hiding behind HTTP 200 -------------------------------------


def test_ecos_no_data_is_an_empty_series_not_an_exception():
    """A quiet window is an answer. A red weekly job for a quiet quarter is not."""
    series = read_ecos({"RESULT": {"CODE": ECOS_NO_DATA, "MESSAGE": "해당 자료가 없습니다."}}, "722Y001")
    assert series.observations == () and series.span is None


def test_every_other_ecos_result_code_is_a_refusal():
    with pytest.raises(MacroRefused) as caught:
        read_ecos({"RESULT": {"CODE": "INFO-100", "MESSAGE": "인증키가 유효하지 않습니다."}}, "722Y001")
    assert caught.value.code == "INFO-100"


def test_a_fred_error_message_is_a_refusal():
    with pytest.raises(MacroRefused) as caught:
        read_fred(
            {"error_code": 400, "error_message": "Bad Request. The value for variable api_key is not"}, "X"
        )
    assert caught.value.code == "400"


def test_a_reply_with_neither_a_series_nor_a_result_names_what_it_did_carry():
    with pytest.raises(MacroShapeError, match="Cookie"):
        read_ecos({"Cookie": {}}, "722Y001")


def test_a_fred_reply_without_observations_names_its_keys():
    with pytest.raises(MacroShapeError, match="seriess"):
        read_fred({"seriess": []}, "CPIAUCSL")


def test_a_row_missing_its_period_names_the_fields_that_were_there():
    with pytest.raises(MacroShapeError, match="STAT_CODE"):
        read_ecos(ecos({"STAT_CODE": "722Y001", "DATA_VALUE": "2.5"}), "722Y001")


def test_a_fred_observation_missing_its_date_is_refused():
    with pytest.raises(MacroShapeError, match="missing date"):
        read_fred({"observations": [{"value": "3.1"}]}, "CPIAUCSL")


# --- what a caller gets -----------------------------------------------------


def test_observations_come_back_in_time_order_whatever_the_source_sent():
    series = read_fred(fred(obs("2026-09-01", "3.2"), obs("2026-08-01", "3.1")), "CPIAUCSL")
    assert [o.day for o in series.observations] == [date(2026, 8, 1), date(2026, 9, 1)]
    assert series.span == (date(2026, 8, 1), date(2026, 9, 1))


def test_units_and_name_come_from_the_source_rather_than_from_us():
    series = read_ecos(ecos(ecos_row("202609", "2.5")), "722Y001")
    assert series.units == "연%" and "기준금리" in series.name


def test_rows_start_with_a_header_so_the_file_says_what_it_holds():
    rows = to_rows(read_fred(fred(obs("2026-08-01", "3.1")), "CPIAUCSL"))
    assert rows[0] == ("date", "value", "published_on") and rows[1][0] == "2026-08-01"


def test_the_publication_date_is_a_column_not_a_footnote():
    """A file carrying only the period is a file whose reader cannot avoid look-ahead."""
    rows = to_rows(read_fred(fred(obs("2026-08-01", "3.1")), "CPIAUCSL"))
    assert rows[1][2] == "2026-09-27"


def test_an_empty_series_still_writes_its_header():
    assert to_rows(MacroSeries("X", "fred", ())) == [("date", "value", "published_on")]


# --- reading the series as of a past day ------------------------------------


def test_a_number_published_later_is_invisible_to_an_earlier_reader():
    """August CPI is published in September. A 1 September backtest must not see it."""
    payload = fred(
        {"date": "2026-08-01", "value": "3.1", "realtime_start": "2026-09-11"},
        {"date": "2026-07-01", "value": "3.0", "realtime_start": "2026-08-12"},
    )
    series = read_fred(payload, "CPIAUCSL", vintage=VINTAGE_INITIAL)
    assert [o.day for o in series.as_of(date(2026, 9, 1))] == [date(2026, 7, 1)]
    assert [o.day for o in series.as_of(date(2026, 9, 30))] == [date(2026, 7, 1), date(2026, 8, 1)]


def test_a_number_published_on_the_day_itself_is_visible():
    series = read_fred(
        fred({"date": "2026-08-01", "value": "3.1", "realtime_start": "2026-09-11"}), "CPIAUCSL"
    )
    assert len(series.as_of(date(2026, 9, 11))) == 1


def test_a_series_with_no_publication_dates_refuses_the_question():
    """Approximating the stamp would put back exactly the look-ahead the caller came to avoid."""
    series = read_ecos(ecos(ecos_row("202608", "2.5")), "722Y001")
    with pytest.raises(MacroShapeError, match="no publication date"):
        series.as_of(date(2026, 9, 1))


def test_fred_sentinels_are_not_mistaken_for_publication_dates():
    series = read_fred(fred({"date": "2026-08-01", "value": "3.1", "realtime_start": "9999-12-31"}), "X")
    assert series.observations[0].published_on is None
