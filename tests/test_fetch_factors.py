"""The factor fetcher is tested for what it refuses and for the units it writes.

No test touches the network. This container has no route to the data library, so
the live format could not be checked from here: the fixtures below are the shape
the library documents, and the first runner fetch is what proves the parser
against the real bytes. That is why every refusal path is tested and why the
errors carry the first bytes of whatever arrived instead.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date, timedelta

import pytest

from scripts import fetch_factors as ff

HEADER = """This file was created by CMPT_ME_BEME_RETS using the CRSP database.
The 1-month TBill return is from Ibbotson and Associates, Inc.

"""


FIVE = "  0.11,  0.02, -0.35,  0.03,  0.13, 0.012"


def daily_body(columns: str, rows: int = 600, values: str = FIVE) -> str:
    """A French-shaped daily CSV: prose, a blank first field on the header, then dates."""
    lines = [HEADER + f",{columns}"]
    day = date(2022, 1, 3)
    for i in range(rows):
        lines.append(f" {(day + timedelta(days=i)):%Y%m%d},{values}")
    return "\n".join(lines) + "\n"


def zipped(body: str, name: str = "F-F_Research_Data_5_Factors_2x3_daily.CSV") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, body)
    return buffer.getvalue()


def five_factor_file(rows: int = 600) -> ff.FactorFile:
    return ff.parse_french_csv(daily_body("Mkt-RF,SMB,HML,RMW,CMA,RF", rows=rows))


def momentum_file(rows: int = 600) -> ff.FactorFile:
    return ff.parse_french_csv(daily_body("Mom", rows=rows, values="  0.44"))


# --- the archive ------------------------------------------------------------


def test_the_single_csv_member_is_read():
    text = ff.unzip_single_csv(zipped(daily_body("Mkt-RF,SMB,HML,RMW,CMA,RF")), ff.FF5_URL)
    assert "Mkt-RF" in text


def test_a_body_that_is_not_a_zip_is_an_error():
    with pytest.raises(ff.FetchError, match="did not return a zip archive"):
        ff.unzip_single_csv(b"<html>503 Service Unavailable</html>", ff.FF5_URL)


def test_an_archive_with_two_csvs_is_an_error():
    """Two members means the layout changed, and guessing which is the data is how a silent swap happens."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.csv", "x")
        archive.writestr("b.csv", "y")
    with pytest.raises(ff.FetchError, match="holds 2 CSV members"):
        ff.unzip_single_csv(buffer.getvalue(), ff.FF5_URL)


# --- parsing ----------------------------------------------------------------


def test_the_columns_come_from_the_header_row():
    parsed = five_factor_file()
    assert parsed.names == ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF")
    assert len(parsed.rows) == 600


def test_percent_becomes_decimal_once():
    parsed = five_factor_file()
    assert parsed.rows["2022-01-03"][0] == pytest.approx(0.0011)
    assert parsed.rows["2022-01-03"][5] == pytest.approx(0.00012)


def test_the_date_becomes_an_iso_day():
    assert "2022-01-03" in five_factor_file().rows


def test_a_missing_marker_becomes_a_hole_and_is_recorded():
    body = daily_body("Mkt-RF,SMB,HML,RMW,CMA,RF")
    body = body.replace(" 20220104,  0.11", " 20220104, -99.99", 1)
    parsed = ff.parse_french_csv(body)
    assert parsed.rows["2022-01-04"][0] is None
    assert parsed.missing["Mkt-RF"] == ["2022-01-04"]


def test_an_annual_block_after_the_daily_one_is_not_folded_in():
    """Some files append a second table. Reading it as daily data would add fake rows."""
    body = daily_body("Mkt-RF,SMB,HML,RMW,CMA,RF")
    body += "\n Annual Factors: January-December\n\n"
    body += ",Mkt-RF,SMB,HML,RMW,CMA,RF\n 2022, 10.0, 1.0, 2.0, 3.0, 4.0, 0.5\n"
    parsed = ff.parse_french_csv(body)
    assert len(parsed.rows) == 600
    assert "2022-01-01" not in parsed.rows


def test_a_file_with_no_header_is_an_error():
    with pytest.raises(ff.FetchError, match="no column header found"):
        ff.parse_french_csv("just some prose\nand more prose\n")


def test_data_before_any_header_is_an_error():
    with pytest.raises(ff.FetchError, match="before any column header"):
        ff.parse_french_csv("prose\n 20220103,  0.11\n")


def test_a_short_row_is_an_error():
    body = daily_body("Mkt-RF,SMB,HML,RMW,CMA,RF")
    body = body.replace(" 20220104,  0.11,  0.02, -0.35,  0.03,  0.13, 0.012", " 20220104,  0.11,  0.02", 1)
    with pytest.raises(ff.FetchError, match="values for 6 columns"):
        ff.parse_french_csv(body)


def test_a_non_numeric_value_is_an_error():
    body = daily_body("Mkt-RF,SMB,HML,RMW,CMA,RF")
    body = body.replace(" 20220104,  0.11", " 20220104,  n/a", 1)
    with pytest.raises(ff.FetchError, match="is not a number"):
        ff.parse_french_csv(body)


def test_a_date_that_is_not_a_calendar_day_is_an_error():
    body = daily_body("Mkt-RF,SMB,HML,RMW,CMA,RF")
    body = body.replace(" 20220104,", " 20220132,", 1)
    with pytest.raises(ff.FetchError, match="not a calendar date"):
        ff.parse_french_csv(body)


def test_a_repeated_date_is_an_error():
    body = daily_body("Mkt-RF,SMB,HML,RMW,CMA,RF", rows=600)
    body += " 20220103,  0.11,  0.02, -0.35,  0.03,  0.13, 0.012\n"
    with pytest.raises(ff.FetchError, match="repeats 2022-01-03"):
        ff.parse_french_csv(body)


def test_a_truncated_file_is_an_error_not_a_short_history():
    with pytest.raises(ff.FetchError, match="need at least"):
        ff.parse_french_csv(daily_body("Mkt-RF,SMB,HML,RMW,CMA,RF", rows=10))


# --- joining the two files --------------------------------------------------


def test_the_join_keeps_only_the_dates_both_files_cover():
    """Momentum is published separately and lags; the intersection is the only honest join."""
    five, mom = five_factor_file(rows=600), momentum_file(rows=598)
    names, rows, dropped = ff.merge(five, mom)
    assert names == ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom")
    assert len(rows) == 598
    assert dropped == 2


def test_the_join_refuses_a_momentum_file_with_several_columns():
    with pytest.raises(ff.FetchError, match="one momentum column"):
        ff.merge(five_factor_file(), five_factor_file())


def test_files_that_share_no_date_are_an_error():
    start = date(1962, 1, 3)
    other = ff.parse_french_csv(
        "\n".join([HEADER + ",Mom"] + [f" {(start + timedelta(days=i)):%Y%m%d},  0.44" for i in range(600)])
        + "\n"
    )
    with pytest.raises(ff.FetchError, match="share no date"):
        ff.merge(five_factor_file(), other)


# --- what gets written -----------------------------------------------------


def test_the_written_file_is_in_decimals_with_holes_left_empty(tmp_path):
    five, mom = five_factor_file(), momentum_file()
    names, rows, dropped = ff.merge(five, mom)
    rows["2022-01-04"] = (None,) + rows["2022-01-04"][1:]

    path = ff.write_factors(names, rows, tmp_path, dropped, {"Mkt-RF": ["2022-01-04"]})
    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "Date,Mkt-RF,SMB,HML,RMW,CMA,RF,Mom"
    expected = "2022-01-03,0.00110000,0.00020000,-0.00350000,0.00030000,0.00130000,0.00012000,0.00440000"
    assert lines[1] == expected
    assert lines[2].startswith("2022-01-04,,"), "a hole is an empty field, never a zero"


def test_the_sidecar_records_the_units_and_the_vendor_holes(tmp_path):
    five, mom = five_factor_file(), momentum_file()
    names, rows, dropped = ff.merge(five, mom)
    ff.write_factors(names, rows, tmp_path, dropped, {"Mkt-RF": ["2022-01-04"]})

    sidecar = json.loads((tmp_path / "ff5_mom_daily.source.json").read_text(encoding="utf-8"))
    assert sidecar["source"] == "ken-french-data-library"
    assert "divides by 100" in sidecar["units"]
    assert sidecar["vendor_missing_markers"] == {"Mkt-RF": ["2022-01-04"]}
    assert sidecar["rows"] == len(rows)
    assert sidecar["columns"] == list(names)


def test_the_written_file_is_what_core_reads_back(tmp_path):
    """The fetcher's output and the reader's input are the same contract."""
    from core.data.factors import load_factors

    five, mom = five_factor_file(), momentum_file()
    names, rows, dropped = ff.merge(five, mom)
    path = ff.write_factors(names, rows, tmp_path, dropped, {})

    panel = load_factors(path)
    assert panel.names == names
    assert panel.values.shape == (len(rows), len(names))
    assert panel.drop(("RF",)).names == ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")


def test_no_pdf_or_monthly_file_is_requested():
    assert "daily" in ff.FF5_URL and "daily" in ff.MOM_URL
    assert ff.FF5_URL.startswith("https://") and ff.MOM_URL.startswith("https://")
