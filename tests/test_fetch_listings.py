"""Parsing the SEC listings sources, and refusing a response that is not one.

The parsers are what stands between a format change at the SEC and a universe
that quietly shrinks, so the refusals get as much attention as the happy path.
Whether the live files still have this shape is settled on the runner, not here
(this container reaches no SEC host); these fix what the code accepts.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date

import pytest

from core.data.universe import load_universe, write_universe
from scripts.fetch_listings import (
    BACKOFF_SECONDS,
    RETRYABLE,
    USER_AGENT,
    FetchError,
    TickerRow,
    _explain,
    _get,
    build,
    completed_quarters,
    parse_company_tickers,
    parse_dera_sub,
    parse_nasdaq_listed,
    parse_other_listed,
    read_sub_member,
    sectors_from_previous,
)

FIELDS = ["cik", "name", "ticker", "exchange"]
ROWS = [
    [320193, "Apple Inc.", "AAPL", "Nasdaq"],
    [789019, "MICROSOFT CORP", "MSFT", "Nasdaq"],
    [19617, "JPMorgan Chase & Co", "JPM", "NYSE"],
    [1090727, "A PINK SHEET CO", "PNKX", "OTC"],
]


def tickers_json(fields=None, rows=None) -> bytes:
    payload = {"fields": FIELDS if fields is None else fields, "data": ROWS if rows is None else rows}
    return json.dumps(payload).encode("utf-8")


def sub_txt(pairs: dict[int, int], columns: tuple[str, ...] = ("adsh", "cik", "name", "sic")) -> str:
    lines = ["\t".join(columns)]
    for cik, sic in pairs.items():
        cells = {"adsh": f"0000-{cik}", "cik": str(cik), "name": "X", "sic": str(sic)}
        lines.append("\t".join(cells[column] for column in columns))
    return "\n".join(lines) + "\n"


def zipped(text: str, member: str = "sub.txt") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(member, text)
    return buffer.getvalue()


# --- how a refusal is reported ----------------------------------------------


class _Response:
    """The little of `urlopen`'s result that `_get` reads."""

    def __init__(self, body: bytes, headers: dict[str, str] | None = None):
        self._body, self.headers = body, headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def http_error(code: int, body: bytes, headers: dict[str, str] | None = None):
    import email.message
    import urllib.error

    message = email.message.Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urllib.error.HTTPError("https://www.sec.gov/x", code, "Forbidden", message, io.BytesIO(body))


def test_a_refusal_carries_what_the_server_said():
    """The first runner attempt got a bare 403 and cost a whole run to diagnose."""
    body = b"Your Request Originates from an Undeclared Automated Tool.\nPlease declare your traffic."
    assert "Undeclared Automated Tool" in _explain(http_error(403, body))


def test_a_gzipped_refusal_is_still_readable():
    import gzip as gziplib

    packed = gziplib.compress(b"rate limited")
    assert _explain(http_error(429, packed, {"Content-Encoding": "gzip"})) == "rate limited"


def test_an_empty_refusal_says_so_rather_than_being_blank():
    assert _explain(http_error(403, b"")) == "empty response body"


def test_the_user_agent_names_us_and_a_way_to_reach_us():
    """The SEC refuses clients that do not declare themselves; no personal email."""
    assert "quant-desk" in USER_AGENT
    assert "https://" in USER_AGENT
    assert "@" not in USER_AGENT


def patched(monkeypatch, responses: list):
    """Serve `responses` in order to `_get`, and record every call."""
    import urllib.request

    calls: list[str] = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        nxt = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(nxt, Exception):
            raise nxt
        return nxt if isinstance(nxt, _Response) else _Response(nxt)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def test_a_rate_limit_is_waited_out_rather_than_failing_the_week(monkeypatch):
    """The SEC answers a threshold breach with 403, not 429, on a shared runner IP."""
    calls = patched(monkeypatch, [http_error(403, b"Request Rate Threshold Exceeded"), b"{}"])
    slept: list[float] = []
    assert _get("https://www.sec.gov/x", sleep=slept.append) == b"{}"
    assert len(calls) == 2
    assert slept == [60.0], "the first backoff, and no second wait once it succeeds"


def test_a_refusal_that_will_not_clear_is_not_retried(monkeypatch):
    """404 does not become a file next minute; waiting on it wastes the run."""
    calls = patched(monkeypatch, [http_error(404, b"not found")])
    with pytest.raises(FetchError, match="HTTP 404"):
        _get("https://www.sec.gov/x", sleep=lambda _: None)
    assert len(calls) == 1


def test_every_attempt_failing_reports_the_last_reason(monkeypatch):
    calls = patched(monkeypatch, [http_error(503, b"down")])
    with pytest.raises(FetchError, match="4 attempts failed.*HTTP 503"):
        _get("https://www.sec.gov/x", sleep=lambda _: None)
    assert len(calls) == 4


def test_a_gzipped_body_is_decompressed(monkeypatch):
    import gzip as gziplib

    patched(monkeypatch, [_Response(gziplib.compress(b"hello"), {"Content-Encoding": "gzip"})])
    assert _get("https://www.sec.gov/x", sleep=lambda _: None) == b"hello"


def test_the_backoff_outlasts_the_secs_ten_minute_block():
    """Measured: 5s/20s/60s were all refused (run 36270027620). Shorter is theatre."""
    assert sum(BACKOFF_SECONDS) >= 600.0


def test_the_rate_limit_status_is_in_the_retryable_set():
    assert 403 in RETRYABLE
    assert 429 in RETRYABLE
    assert 404 not in RETRYABLE


# --- the membership list ----------------------------------------------------


def test_the_ticker_file_parses_into_listings():
    rows = parse_company_tickers(tickers_json())
    assert [row.ticker for row in rows] == ["AAPL", "MSFT", "JPM", "PNKX"]
    assert rows[0].cik == 320193


def test_columns_are_read_by_name_not_by_position():
    """The SEC has reordered this file before; a positional read would not notice."""
    reordered = [["Nasdaq", "AAPL", 320193, "Apple Inc."]]
    rows = parse_company_tickers(tickers_json(fields=["exchange", "ticker", "cik", "name"], rows=reordered))
    assert rows[0] == TickerRow(cik=320193, name="Apple Inc.", ticker="AAPL", exchange="Nasdaq")


def test_an_extra_column_does_not_break_the_read():
    fields = [*FIELDS, "sic"]
    row = [[320193, "Apple", "AAPL", "Nasdaq", 3571]]
    rows = parse_company_tickers(tickers_json(fields=fields, rows=row))
    assert rows[0].ticker == "AAPL"


def test_a_filer_with_no_ticker_is_not_a_listing():
    rows = parse_company_tickers(tickers_json(rows=[[1, "A PRIVATE FILER", "", "NYSE"], *ROWS]))
    assert len(rows) == len(ROWS)


def test_an_html_error_page_is_an_error_not_an_empty_universe():
    with pytest.raises(FetchError, match="not JSON"):
        parse_company_tickers(b"<html>rate limited</html>")


def test_a_renamed_column_is_an_error():
    with pytest.raises(FetchError, match="missing column"):
        parse_company_tickers(tickers_json(fields=["cik", "title", "ticker", "exchange"]))


def test_a_row_of_the_wrong_width_is_an_error():
    with pytest.raises(FetchError, match="cells for"):
        parse_company_tickers(tickers_json(rows=[[320193, "Apple", "AAPL"]]))


def test_a_file_that_parses_to_nothing_is_an_error():
    with pytest.raises(FetchError, match="zero listings"):
        parse_company_tickers(tickers_json(rows=[]))


# --- the SIC codes ----------------------------------------------------------


def test_the_quarters_walk_back_from_the_last_completed_one():
    """The current quarter has no data set, so it is never asked for."""
    assert completed_quarters(date(2026, 9, 26), 4) == ("2026q2", "2026q1", "2025q4", "2025q3")


def test_the_walk_back_crosses_a_year():
    assert completed_quarters(date(2026, 1, 15), 2) == ("2025q4", "2025q3")


def test_asking_for_no_quarters_is_refused():
    with pytest.raises(ValueError, match="not a number of quarters"):
        completed_quarters(date(2026, 9, 26), 0)


def test_sub_txt_gives_a_cik_to_sic_mapping():
    assert parse_dera_sub(sub_txt({320193: 3571, 19617: 6022})) == {320193: 3571, 19617: 6022}


def test_a_blank_sic_is_skipped_rather_than_read_as_zero():
    """Zero is a bucket lookup that fails silently; absence is the honest reading."""
    text = "adsh\tcik\tsic\n0000-1\t320193\t\n0000-2\t19617\t6022\n"
    assert parse_dera_sub(text) == {19617: 6022}


def test_a_sub_file_without_the_columns_is_an_error():
    with pytest.raises(FetchError, match="no 'sic' column"):
        parse_dera_sub("adsh\tcik\tname\n0000-1\t320193\tX\n")


def test_a_sub_file_with_no_pairs_is_an_error():
    with pytest.raises(FetchError, match="zero CIK/SIC pairs"):
        parse_dera_sub("adsh\tcik\tsic\n")


def test_the_sub_member_comes_out_of_the_quarterly_zip():
    assert "320193" in read_sub_member(zipped(sub_txt({320193: 3571})), "2026q2")


def test_an_archive_that_is_not_a_zip_is_an_error():
    with pytest.raises(FetchError, match="not a zip archive"):
        read_sub_member(b"404 not found", "2026q2")


def test_an_archive_without_sub_txt_is_an_error():
    with pytest.raises(FetchError, match="0 sub.txt members"):
        read_sub_member(zipped("x", member="num.txt"), "2026q2")


# --- the Nasdaq Trader membership list --------------------------------------

NASDAQ_TXT = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust|Q|N|N|100|Y|N
ZZZT|NASDAQ TEST STOCK|Q|Y|N|100|N|N
File Creation Time: 09262026 21:00
"""

OTHER_TXT = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
JPM|JPMorgan Chase & Co.|N|JPM|N|100|N|JPM
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
PNKX|Some Pink Sheet Co|U|PNKX|N|100|N|PNKX
ATEST|NYSE TEST|N|ATEST|N|100|Y|ATEST
File Creation Time: 09262026 21:00
"""


def test_the_nasdaq_directory_parses():
    rows = parse_nasdaq_listed(NASDAQ_TXT)
    assert [row.ticker for row in rows] == ["AAPL", "MSFT", "QQQ"]
    assert rows[2].etf is True
    assert rows[0].exchange == "Nasdaq"


def test_a_test_issue_is_never_a_listing():
    """These rows exist so members can exercise their systems; an order in one is real."""
    assert "ZZZT" not in [row.ticker for row in parse_nasdaq_listed(NASDAQ_TXT)]
    assert "ATEST" not in [row.ticker for row in parse_other_listed(OTHER_TXT)]


def test_the_trailing_timestamp_line_is_not_a_symbol():
    """Both files end with `File Creation Time:`; read as a row it becomes a ticker."""
    assert all("File Creation" not in row.ticker for row in parse_nasdaq_listed(NASDAQ_TXT))


def test_the_other_venues_resolve_from_their_letter():
    rows = {row.ticker: row for row in parse_other_listed(OTHER_TXT)}
    assert rows["JPM"].exchange == "NYSE"
    assert rows["SPY"].exchange == "NYSE Arca"
    assert rows["SPY"].etf is True


def test_a_venue_we_do_not_take_is_dropped():
    assert "PNKX" not in [row.ticker for row in parse_other_listed(OTHER_TXT)]


def test_the_act_symbol_is_the_one_taken():
    """The NASDAQ Symbol column is that venue's spelling; no price file has it."""
    text = OTHER_TXT.replace("JPM|JPMorgan Chase & Co.|N|JPM|N|100|N|JPM", "JPM|JPM|N|JPM|N|100|N|JPMX")
    assert "JPM" in [row.ticker for row in parse_other_listed(text)]
    assert "JPMX" not in [row.ticker for row in parse_other_listed(text)]


def test_a_row_of_the_wrong_width_is_an_error_here_too():
    with pytest.raises(FetchError, match="cells for"):
        parse_nasdaq_listed(NASDAQ_TXT.replace("|100|N|N\n", "|100|N\n", 1))


def test_a_renamed_directory_column_is_an_error():
    with pytest.raises(FetchError, match="missing column"):
        parse_nasdaq_listed(NASDAQ_TXT.replace("Test Issue", "TestIssue", 1))


def test_an_empty_directory_is_an_error():
    with pytest.raises(FetchError, match="zero listings"):
        parse_nasdaq_listed(
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        )


# --- the join ---------------------------------------------------------------


def membership() -> tuple[TickerRow, ...]:
    return parse_nasdaq_listed(NASDAQ_TXT) + parse_other_listed(OTHER_TXT)


CIK = {"AAPL": 320193, "MSFT": 789019, "JPM": 19617}
SIC = {320193: 3571, 789019: 7372, 19617: 6022}


def test_sectors_come_from_the_sic_join():
    universe, coverage = build(membership(), date(2026, 9, 26), CIK, SIC)
    assert universe.sector_map()["AAPL"] == "technology"
    assert universe.sector_map()["JPM"] == "financials"
    assert coverage["classified_from_sic"] == 3


def test_an_etf_takes_its_declared_bucket_and_never_a_trust_sic():
    """SPY's filer SIC is a trust code; taking it would file SPY under financials."""
    universe, coverage = build(membership(), date(2026, 9, 26), {"SPY": 1}, {1: 6726})
    assert universe.sector_map()["SPY"] == "us_equity_broad"
    assert universe.sector_map()["QQQ"] == "us_equity_growth"
    assert coverage["classified_etf_declared"] == 2


def test_a_name_with_no_sector_anywhere_loads_and_cannot_be_ordered():
    universe, coverage = build(membership(), date(2026, 9, 26), {}, {})
    assert "AAPL" in universe.unclassified
    assert "AAPL" in universe.symbols
    assert coverage["classified_from_sic"] == 0


def test_the_previous_file_fills_in_when_the_sec_is_out():
    """A sector is a slow-moving fact; losing the week's membership is not recoverable."""
    universe, coverage = build(
        membership(), date(2026, 9, 26), {}, {}, carried={"AAPL": "technology", "JPM": "financials"}
    )
    assert universe.sector_map()["AAPL"] == "technology"
    assert coverage["classified_from_previous_file"] == 2


def test_a_fresh_sic_beats_a_carried_one():
    universe, _ = build(membership(), date(2026, 9, 26), CIK, SIC, carried={"JPM": "energy"})
    assert universe.sector_map()["JPM"] == "financials"


def test_a_repeated_ticker_is_recorded_rather_than_crashing_the_run():
    doubled = (*membership(), TickerRow(ticker="AAPL", name="APPLE AGAIN", exchange="NYSE"))
    universe, coverage = build(doubled, date(2026, 9, 26), CIK, SIC)
    assert coverage["dropped_duplicate_ticker"] == ["AAPL"]
    assert universe.listing("AAPL").name.startswith("Apple")


def test_the_universe_is_written_as_not_point_in_time():
    """Neither source publishes a listing date, and guessing one is the whole trap."""
    universe, _ = build(membership(), date(2026, 9, 26), CIK, SIC)
    assert universe.point_in_time is False
    assert all(listing.listed_on is None for listing in universe.listings)


def test_the_carry_over_reads_the_file_the_last_run_wrote(tmp_path):
    first, _ = build(membership(), date(2026, 9, 19), CIK, SIC)
    write_universe(tmp_path / "us.csv", first)
    assert sectors_from_previous(tmp_path / "us.csv")["AAPL"] == "technology"


def test_a_missing_previous_file_carries_nothing_rather_than_failing(tmp_path):
    assert sectors_from_previous(tmp_path / "us.csv") == {}


def test_the_written_file_round_trips_with_its_coverage(tmp_path):
    universe, coverage = build(membership(), date(2026, 9, 26), CIK, SIC)
    path = write_universe(tmp_path / "us.csv", universe, extra={"coverage": coverage})

    loaded = load_universe(path)
    assert loaded.symbols == universe.symbols
    assert loaded.point_in_time is False
    sidecar = json.loads((tmp_path / "us.source.json").read_text())
    assert sidecar["coverage"]["classified_from_sic"] == 3
