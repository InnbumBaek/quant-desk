"""The KRX fetcher: what it writes, what it refuses, and what it never leaks.

Whether KRX answers at all is settled on the runner with a key (this container
only ever got 401). These fix the parts that have to be right the first time it
does, because the first keyed run is the only cheap one.
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import date

import pytest

from core.data.universe import load_universe
from scripts import fetch_krx as fk

SESSION = date(2026, 9, 25)


def row(symbol: str = "005930", **overrides) -> dict[str, str]:
    base = {
        "BAS_DD": SESSION.strftime("%Y%m%d"),
        "ISU_CD": symbol,
        "ISU_NM": "삼성전자",
        "MKT_NM": "KOSPI",
        "TDD_OPNPRC": "71000",
        "TDD_HGPRC": "72300",
        "TDD_LWPRC": "70800",
        "TDD_CLSPRC": "72100",
        "ACC_TRDVOL": "12345678",
        "ACC_TRDVAL": "889123456700",
        "MKTCAP": "430512000000000",
        "LIST_SHRS": "5969782550",
    }
    base.update(overrides)
    return base


def payload(count: int = fk.MIN_ROWS_PER_DAY, day: date = SESSION) -> bytes:
    rows = [
        row(f"{100000 + index:06d}", BAS_DD=day.strftime("%Y%m%d"), ISU_NM=f"종목{index}")
        for index in range(count)
    ]
    return json.dumps({fk.BLOCK: rows}).encode("utf-8")


def http_error(code: int, body: bytes = b""):
    import email.message

    return urllib.error.HTTPError(
        "https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd",
        code,
        "refused",
        email.message.Message(),
        io.BytesIO(body),
    )


def serve(monkeypatch, answer):
    """Route `_get`'s urlopen through `answer(request, n)`, recording every request."""
    import urllib.request

    seen: list = []

    class Reply:
        def __init__(self, body: bytes):
            self._body = body

        def read(self, *_):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        result = answer(request, len(seen))
        if isinstance(result, Exception):
            raise result
        return Reply(result)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


# --- the request ------------------------------------------------------------


def test_the_key_rides_in_a_header_and_never_in_the_url(monkeypatch):
    """A key in a query string lands in access logs, proxies and our own probe records."""
    seen = serve(monkeypatch, lambda request, n: payload())
    fk._get(fk.day_url("KOSPI", SESSION), "secret-key", sleep=lambda _: None)

    assert seen[0].headers[fk.KEY_HEADER.capitalize()] == "secret-key"
    assert "secret-key" not in seen[0].full_url


def test_the_url_carries_the_date_in_krx_form():
    assert "basDd=20260925" in fk.day_url("KOSPI", SESSION)


def test_each_venue_has_its_own_endpoint():
    urls = {venue: fk.day_url(venue, SESSION) for venue in fk.ENDPOINTS}
    assert len(set(urls.values())) == len(urls)


def test_a_venue_we_do_not_know_is_refused_before_a_request_is_made():
    with pytest.raises(fk.FetchError, match="not a KRX venue"):
        fk.day_url("NYSE", SESSION)


def test_we_identify_ourselves_the_way_every_other_fetcher_does(monkeypatch):
    from core.config import USER_AGENT

    seen = serve(monkeypatch, lambda request, n: payload())
    fk._get(fk.day_url("KOSPI", SESSION), "k", sleep=lambda _: None)
    assert seen[0].headers["User-agent"] == USER_AGENT


# --- refusals ---------------------------------------------------------------


def test_a_rate_limit_is_waited_out(monkeypatch):
    seen = serve(monkeypatch, lambda request, n: payload() if n == 2 else http_error(429))
    waits: list[float] = []
    fk._get(fk.day_url("KOSPI", SESSION), "k", sleep=waits.append)
    assert len(seen) == 2 and waits == [fk.BACKOFF_SECONDS[0]]


def test_a_bad_key_fails_at_once_rather_than_waiting_out_a_no(monkeypatch):
    """401 means the key is wrong, and ninety seconds will not make it right."""
    seen = serve(monkeypatch, lambda request, n: http_error(401, b'{"message":"AUTH_KEY invalid"}'))
    waits: list[float] = []
    with pytest.raises(fk.FetchError, match="AUTH_KEY invalid"):
        fk._get(fk.day_url("KOSPI", SESSION), "k", sleep=waits.append)
    assert len(seen) == 1 and waits == []


def test_the_refusal_body_comes_back_with_the_error(monkeypatch):
    """A status code alone cost this desk four runner cycles on SEC."""
    serve(monkeypatch, lambda request, n: http_error(403, b"blocked by policy"))
    with pytest.raises(fk.FetchError, match="blocked by policy"):
        fk._get(fk.day_url("KOSPI", SESSION), "k", sleep=lambda _: None)


def test_an_html_page_is_not_a_trading_day(monkeypatch):
    serve(monkeypatch, lambda request, n: b"<html>error</html>")
    with pytest.raises(fk.FetchError, match="not JSON"):
        fk.fetch_day("KOSPI", SESSION, "k", sleep=lambda _: None)


def test_a_short_day_is_refused_rather_than_written(monkeypatch):
    """Twenty rows from a market of 950 is a partial answer, not a quiet market."""
    serve(monkeypatch, lambda request, n: payload(count=20))
    with pytest.raises(fk.FetchError, match="readable rows"):
        fk.fetch_day("KOSPI", SESSION, "k", sleep=lambda _: None)


# --- the shape record --------------------------------------------------------


def test_the_schema_record_carries_names_and_no_values():
    """The parser was written from a document; this is what checks it against bytes."""
    shape = fk.schema_of(json.loads(payload(count=3)))
    assert shape["rows"] == 3
    assert "TDD_CLSPRC" in shape["row_keys"]
    assert "72100" not in json.dumps(shape), "a shape record must not publish a price"
    assert "삼성전자" not in json.dumps(shape, ensure_ascii=False)


def test_the_schema_record_survives_a_response_we_cannot_parse():
    """It is most useful exactly when the parse failed, so it must not need the parse."""
    shape = fk.schema_of({"errMsg": "invalid key", "errCode": "010"})
    assert shape["top_level_keys"] == ["errCode", "errMsg"]
    assert shape["row_keys"] == []


# --- the calendar ------------------------------------------------------------


def test_weekends_are_never_asked_for():
    days = fk.weekdays(date(2026, 9, 21), date(2026, 9, 27))
    assert days == [date(2026, 9, d) for d in (21, 22, 23, 24, 25)]


def test_a_weekday_that_did_not_trade_is_recorded_not_failed(monkeypatch):
    """There is no Korean holiday table here on purpose: KRX is the calendar."""
    closed = date(2026, 9, 24)

    def answer(request, n):
        if closed.strftime("%Y%m%d") in request.full_url:
            return json.dumps({fk.BLOCK: []}).encode("utf-8")
        day = date(2026, 9, int(request.full_url[-2:]))
        return payload(day=day)

    serve(monkeypatch, answer)
    report = fk.fetch(
        directory=_tmp(),
        days=4,
        as_of=date(2026, 9, 25),
        venues=("KOSPI",),
        key="k",
        sleep=lambda _: None,
        universe_path=_tmp() / "kr.csv",
    )
    assert report["weekdays_closed"] == [closed.isoformat()]


def test_a_window_where_nothing_traded_at_all_is_an_error(monkeypatch):
    """A run of empty weekdays is a query that matches nothing, not a fortnight of holidays."""
    serve(monkeypatch, lambda request, n: json.dumps({fk.BLOCK: []}).encode("utf-8"))
    with pytest.raises(fk.FetchError, match="came back empty"):
        fk.fetch(
            directory=_tmp(),
            days=5,
            as_of=date(2026, 9, 25),
            venues=("KOSPI",),
            key="k",
            sleep=lambda _: None,
            universe_path=_tmp() / "kr.csv",
        )


# --- what lands on disk ------------------------------------------------------

_TMP: list = []


def _tmp():
    import tempfile
    from pathlib import Path

    handle = tempfile.TemporaryDirectory()
    _TMP.append(handle)
    return Path(handle.name)


def test_prices_become_the_csv_shape_the_rest_of_the_desk_reads(monkeypatch):
    serve(monkeypatch, lambda request, n: payload(day=date(2026, 9, int(request.full_url[-2:]))))
    out = _tmp()
    fk.fetch(
        directory=out,
        days=2,
        as_of=date(2026, 9, 25),
        venues=("KOSPI",),
        key="k",
        sleep=lambda _: None,
        universe_path=_tmp() / "kr.csv",
    )
    written = sorted(out.glob("*.csv"))
    assert written, "a run that fetched bars must leave CSVs"
    body = written[0].read_text(encoding="utf-8").splitlines()
    assert body[0] == fk.EXPECTED_HEADER
    assert body[1].startswith("2026-09-")


def test_membership_is_written_with_the_market_the_desk_declares(monkeypatch):
    serve(monkeypatch, lambda request, n: payload(count=120, day=date(2026, 9, int(request.full_url[-2:]))))
    universe_path = _tmp() / "kr.csv"
    fk.fetch(
        directory=_tmp(),
        days=2,
        as_of=date(2026, 9, 25),
        venues=("KOSPI",),
        key="k",
        sleep=lambda _: None,
        universe_path=universe_path,
    )
    universe = load_universe(universe_path)
    assert len(universe.listings) == 120
    assert {listing.market for listing in universe.listings} == {"KR"}


def test_a_name_already_trading_on_the_first_day_gets_no_listing_date(monkeypatch):
    """Writing the backfill's start date as a listing event would invent one."""
    serve(monkeypatch, lambda request, n: payload(count=120, day=date(2026, 9, int(request.full_url[-2:]))))
    universe_path = _tmp() / "kr.csv"
    fk.fetch(
        directory=_tmp(),
        days=3,
        as_of=date(2026, 9, 25),
        venues=("KOSPI",),
        key="k",
        sleep=lambda _: None,
        universe_path=universe_path,
    )
    universe = load_universe(universe_path)
    assert all(listing.listed_on is None for listing in universe.listings)
    assert not universe.point_in_time, "a universe that cannot date its members must say so"
