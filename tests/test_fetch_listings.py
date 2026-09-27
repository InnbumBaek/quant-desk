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

from core.config import SEC_CONTACT_ENV
from core.data.universe import load_universe, write_universe
from scripts import yahoo_profiles
from scripts.fetch_listings import (
    BACKOFF_SECONDS,
    EXIT_COVERAGE_BELOW_FLOOR,
    RETRYABLE,
    USER_AGENT,
    FetchError,
    TickerRow,
    _explain,
    _get,
    _sector_source,
    build,
    completed_quarters,
    fetch_sector_inputs,
    fetch_vendor_sectors,
    next_empty_pool,
    parse_company_tickers,
    parse_dera_sub,
    parse_nasdaq_listed,
    parse_other_listed,
    read_sub_member,
    rotate,
    sectors_from_previous,
    symbols_needing_sectors,
    vendor_queue,
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


@pytest.fixture(autouse=True)
def _declared(monkeypatch):
    """sec.gov refuses an undeclared automated tool by name, so every test that
    reaches it declares one. The tests that care about the missing case unset it
    themselves (ADR-0030)."""
    monkeypatch.setenv(SEC_CONTACT_ENV, "desk@example.invalid")


# --- the User-Agent sec.gov asks for -----------------------------------------


def test_sec_gets_the_declared_agent_and_other_hosts_do_not(monkeypatch):
    """`data.sec.gov` answered the 2026-09-26 probe with "Your Request Originates
    from an Undeclared Automated Tool" -- a different refusal from the rate
    threshold `www.sec.gov` gave in the same run."""
    import urllib.request

    seen: list[str] = []

    def fake_urlopen(request, timeout=None):
        seen.append(request.get_header("User-agent"))
        return _Response(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _get("https://www.sec.gov/files/x.json")
    _get("https://data.sec.gov/submissions/x.json")
    _get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt")
    assert "desk@example.invalid" in seen[0]
    assert "desk@example.invalid" in seen[1]
    assert seen[2] == USER_AGENT, "only sec.gov asked for a contact address"
    assert "Mozilla" not in " ".join(seen)


def test_a_lookalike_host_does_not_get_the_address(monkeypatch):
    """`sec.gov.example.com` is not the SEC, and an address is a person's."""
    import urllib.request

    seen: list[str] = []

    def fake_urlopen(request, timeout=None):
        seen.append(request.get_header("User-agent"))
        return _Response(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _get("https://sec.gov.example.com/x")
    assert seen == [USER_AGENT]


def test_without_a_contact_address_sec_is_not_asked_at_all(monkeypatch):
    """Fail-closed on the request, not on the run: sending the string SEC has
    already said it refuses only burns the shared address further."""
    monkeypatch.delenv(SEC_CONTACT_ENV, raising=False)
    calls = patched(monkeypatch, [b"{}"])
    with pytest.raises(RuntimeError, match=SEC_CONTACT_ENV):
        _get("https://www.sec.gov/files/x.json")
    assert calls == [], "no request left the runner"


def test_an_address_without_an_at_sign_is_not_an_address(monkeypatch):
    monkeypatch.setenv(SEC_CONTACT_ENV, "not-an-address")
    with pytest.raises(RuntimeError, match=SEC_CONTACT_ENV):
        _get("https://www.sec.gov/files/x.json")


def test_the_missing_address_is_recorded_as_a_reason_not_raised_upward(monkeypatch):
    """The membership snapshot is what the run cannot lose. A configuration gap
    reads like a host outage here, and both leave the sectors to the fallback."""
    monkeypatch.delenv(SEC_CONTACT_ENV, raising=False)
    cik_by_ticker, sic_by_cik, read, why = fetch_sector_inputs(1, date(2026, 9, 27))
    assert (cik_by_ticker, sic_by_cik, read) == ({}, {}, ())
    assert why and SEC_CONTACT_ENV in why


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


# --- the vendor sectors, which now arrive one symbol at a time --------------


def test_only_operating_companies_without_a_bucket_are_queued():
    """Five thousand ETFs would spend the whole budget learning nothing."""
    needing = symbols_needing_sectors(membership(), carried={"AAPL": "technology"})
    assert needing == ["JPM", "MSFT"]  # AAPL carried, SPY and QQQ are ETFs


def test_a_symbol_already_known_to_have_no_sector_is_not_asked_again():
    needing = symbols_needing_sectors(membership(), carried={}, skip=["MSFT"])
    assert needing == ["AAPL", "JPM"]


def test_the_queue_takes_new_symbols_first():
    assert vendor_queue(["A", "B"], ["X", "Y"], budget=3) == ["A", "B", "X"]


def test_a_leftover_budget_re_asks_the_oldest_empties():
    """Never re-asking would make a permanent blind spot out of a new listing."""
    assert vendor_queue([], ["X", "Y", "Z"], budget=2) == ["X", "Y"]


def test_a_full_queue_of_new_symbols_leaves_no_room_for_re_asking():
    assert vendor_queue(["A", "B", "C"], ["X"], budget=2) == ["A", "B"]


def test_the_ones_just_asked_move_to_the_back():
    assert rotate(["X", "Y", "Z"], asked=["X"]) == ["Y", "Z", "X"]


def test_the_vendor_reports_its_outage_rather_than_failing_the_run(monkeypatch):
    def refuse(symbols, budget=0, pause=None):
        raise yahoo_profiles.ProfileError("crumb refused: HTTP 429 Too Many Requests")

    monkeypatch.setattr(yahoo_profiles, "harvest", refuse)
    run = fetch_vendor_sectors(["AAPL"], budget=10)
    assert run.buckets == {} and run.unmapped == ()
    assert run.detail["remaining"] == 1
    assert run.error is not None and "429" in run.error
    assert run.no_sector == (), "a session that never opened answered about nobody"


def test_an_empty_queue_asks_nobody(monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("a run with nothing to ask must not open a session")

    monkeypatch.setattr(yahoo_profiles, "harvest", explode)
    run = fetch_vendor_sectors([], budget=10)
    assert run.buckets == {} and run.error is None and run.detail["asked"] == 0


def test_what_the_vendor_read_is_reported_symbol_by_symbol(monkeypatch):
    harvested = yahoo_profiles.Harvest(
        buckets={"AAPL": "technology"},
        no_sector=["TRST"],
        failed=["ZZZZ"],
        unmapped=("Conglomerates",),
        asked=3,
    )
    monkeypatch.setattr(yahoo_profiles, "harvest", lambda symbols, budget=0, pause=None: harvested)
    run = fetch_vendor_sectors(["AAPL", "TRST", "ZZZZ", "MSFT"], budget=4)

    assert run.buckets == {"AAPL": "technology"} and run.unmapped == ("Conglomerates",)
    assert run.detail["answered_with_no_sector"] == 1 and run.detail["failed"] == 1
    assert run.detail["remaining"] == 1
    assert run.error is None
    assert run.no_sector == ("TRST",), "only what Yahoo answered about, never the whole queue"


# --- which source a bucket came from ----------------------------------------


def test_a_vendor_label_fills_in_where_the_sic_join_could_not():
    universe, coverage = build(membership(), date(2026, 9, 26), {}, {}, vendor={"AAPL": "technology"})
    assert universe.sector_map()["AAPL"] == "technology"
    assert coverage["classified_from_vendor"] == 1


def test_a_filing_beats_a_vendor_opinion():
    """SIC is the filer's own registration; anybody can check it against EDGAR."""
    universe, coverage = build(membership(), date(2026, 9, 26), CIK, SIC, vendor={"JPM": "energy"})
    assert universe.sector_map()["JPM"] == "financials"
    assert coverage["classified_from_vendor"] == 0


def test_a_named_vendor_beats_a_carried_value_whose_source_is_lost():
    universe, coverage = build(
        membership(), date(2026, 9, 26), {}, {}, carried={"AAPL": "energy"}, vendor={"AAPL": "technology"}
    )
    assert universe.sector_map()["AAPL"] == "technology"
    assert coverage["classified_from_previous_file"] == 0


def test_an_etf_never_takes_a_vendor_bucket_either():
    """The screener is a stock screener; a sector on a fund row is somebody's error."""
    universe, _ = build(membership(), date(2026, 9, 26), {}, {}, vendor={"SPY": "financials"})
    assert universe.sector_map()["SPY"] == "us_equity_broad"
    assert universe.sector_map()["QQQ"] == "us_equity_growth"


def test_the_floor_is_measured_over_operating_companies_not_the_whole_file():
    """Two in five real listings are ETFs; counting them measures the ETF tail.

    An ETF takes a bucket only where we declared one, and we have declared a
    handful, so a whole-file share falls as the ETF tail grows even when every
    operating company was classified. The floor has to see through that.
    """
    undeclared_etf = TickerRow(ticker="XXETF", name="Some New Fund", exchange="NYSE Arca", etf=True)
    universe, coverage = build((*membership(), undeclared_etf), date(2026, 9, 26), CIK, SIC)

    assert coverage["operating_companies"] == 3
    assert coverage["operating_classified"] == 3
    assert coverage["operating_share"] == 1.0
    assert "XXETF" in universe.unclassified
    assert coverage["classified_share"] < coverage["operating_share"]


def test_the_sector_source_names_which_one_answered():
    assert _sector_source(None, None, {}) == "sec_sic"
    assert _sector_source("sec down", None, {"AAPL": "technology"}) == "nasdaq_screener"
    assert _sector_source("sec down", "vendor down", {}) == "carried_over_or_absent"


def test_a_vendor_that_answered_and_classified_nothing_is_not_a_sector_source():
    """Otherwise a silent format change reads as a working source with no coverage."""
    assert _sector_source("sec down", None, {}) == "carried_over_or_absent"


def test_the_shortfall_exit_code_is_not_the_crash_exit_code():
    """The workflow commits the listings on 3 and stops on anything else."""
    assert EXIT_COVERAGE_BELOW_FLOOR not in (0, 1, 2)


# --- the pool the next run skips --------------------------------------------


def vendor_run(no_sector=(), asked=0, buckets=None, error=None):
    from scripts.fetch_listings import VendorRun

    return VendorRun(buckets or {}, (), {"asked": asked}, error, tuple(no_sector))


def test_a_dead_session_adds_nobody_to_the_skip_pool():
    """Run 36286809639 never opened a session and recorded 1,200 symbols as answered."""
    queue = [f"S{i}" for i in range(1200)]
    run = vendor_run(error="vendor sectors unavailable: crumb refused: HTTP 429")
    assert next_empty_pool([], run, queue) == []


def test_only_what_the_vendor_answered_joins_the_pool():
    queue = ["AAPL", "TRST", "ZZZZ"]
    run = vendor_run(no_sector=["TRST"], asked=3, buckets={"AAPL": "technology"})
    # ZZZZ failed: it is unmeasured, so the next run must ask about it again.
    assert next_empty_pool([], run, queue) == ["TRST"]


def test_the_symbols_just_asked_about_go_to_the_back():
    queue = ["OLD1", "NEW1"]
    run = vendor_run(no_sector=["NEW1"], asked=2)
    assert next_empty_pool(["OLD1", "OLD2"], run, queue) == ["OLD2", "OLD1", "NEW1"]


def test_a_budget_that_ran_out_rotates_only_what_it_reached():
    """Rotating the unasked tail would push it behind names already answered for."""
    queue = ["A", "B", "C", "D"]
    run = vendor_run(no_sector=["A"], asked=2)
    assert next_empty_pool(["A", "C"], run, queue) == ["C", "A"]
