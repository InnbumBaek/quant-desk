"""The DART fetcher: paging, refusals, and never printing the key.

DART takes the key as a query parameter, so every error message that quotes a
URL is a credential in a public job log unless something stops it. These fix
that, and fix the three things that arrive as HTTP 200 and mean different
things.
"""

from __future__ import annotations

import io
import json
import tempfile
import urllib.error
from datetime import date
from pathlib import Path

import pytest

from core.data.dart import NO_DATA, OK
from scripts import fetch_dart as fd

DAY = date(2026, 9, 25)
KEY = "0123456789abcdef0123456789abcdef01234567"
_TMP: list = []


def tmp() -> Path:
    handle = tempfile.TemporaryDirectory()
    _TMP.append(handle)
    return Path(handle.name)


def row(receipt: str = "20260925000123", **overrides) -> dict[str, str]:
    base = {
        "corp_code": "00126380",
        "corp_name": "삼성전자",
        "stock_code": "005930",
        "corp_cls": "Y",
        "report_nm": "주요사항보고서",
        "rcept_no": receipt,
        "flr_nm": "삼성전자",
        "rcept_dt": DAY.strftime("%Y%m%d"),
    }
    base.update(overrides)
    return base


def body(*rows, status: str = OK, page_no: int = 1, total_page: int = 1) -> bytes:
    return json.dumps(
        {
            "status": status,
            "message": "정상",
            "page_no": page_no,
            "total_page": total_page,
            "total_count": len(rows),
            "list": list(rows),
        }
    ).encode("utf-8")


def http_error(code: int, payload: bytes = b""):
    import email.message

    return urllib.error.HTTPError(fd.ENDPOINT, code, "refused", email.message.Message(), io.BytesIO(payload))


def serve(monkeypatch, answer):
    import urllib.request

    seen: list = []

    class Reply:
        def __init__(self, data: bytes):
            self._data = data

        def read(self, *_):
            return self._data

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


# --- the key ----------------------------------------------------------------


def test_the_key_never_reaches_an_error_message(monkeypatch):
    """DART takes the key in the URL, so an unredacted error is a leak in a public log."""
    serve(monkeypatch, lambda request, n: http_error(403, b"forbidden"))
    with pytest.raises(fd.FetchError) as caught:
        fd._get(fd.page_url(KEY, DAY), KEY, sleep=lambda _: None)
    assert KEY not in str(caught.value)


def test_the_key_never_reaches_a_parse_error_either(monkeypatch):
    serve(monkeypatch, lambda request, n: b"<html>nope</html>")
    with pytest.raises(fd.FetchError) as caught:
        fd.fetch_day(DAY, KEY, sleep=lambda _: None)
    assert KEY not in str(caught.value)


def test_redaction_leaves_everything_else_alone():
    assert fd.redact(f"HTTP 403 for {fd.ENDPOINT}?crtfc_key={KEY}", KEY).endswith("<DART_API_KEY>")
    assert fd.redact("nothing secret here", KEY) == "nothing secret here"


def test_redaction_with_no_key_is_not_a_crash():
    assert fd.redact("plain", "") == "plain"


def test_the_window_is_one_day_wide():
    url = fd.page_url(KEY, DAY)
    assert "bgn_de=20260925" in url and "end_de=20260925" in url


# --- the three things that arrive as 200 ------------------------------------


def test_a_quiet_day_is_written_as_a_quiet_day(monkeypatch):
    """013 is an answer. An empty file records it; a hole would look like a day we skipped."""
    serve(monkeypatch, lambda request, n: json.dumps({"status": NO_DATA}).encode())
    out = tmp()
    report = fd.fetch(out, days=1, as_of=DAY, key=KEY, sleep=lambda _: None)

    assert report["disclosures"] == 0
    assert DAY.isoformat() in report["quiet_days"]
    written = json.loads((out / f"{DAY.isoformat()}.json").read_text(encoding="utf-8"))
    assert written["count"] == 0 and written["disclosures"] == []


def test_a_rejected_key_stops_the_run_rather_than_writing_empty_days(monkeypatch):
    serve(monkeypatch, lambda request, n: json.dumps({"status": "010"}).encode())
    with pytest.raises(fd.FetchError, match="010"):
        fd.fetch(tmp(), days=1, as_of=DAY, key=KEY, sleep=lambda _: None)


def test_a_maintenance_window_is_waited_out_and_retried(monkeypatch):
    calls = {"n": 0}

    def answer(request, n):
        calls["n"] += 1
        return body(row()) if calls["n"] > 1 else json.dumps({"status": "800"}).encode()

    serve(monkeypatch, answer)
    waits: list[float] = []
    got = fd.fetch_day(DAY, KEY, sleep=waits.append)
    assert len(got) == 1 and fd.BACKOFF_SECONDS[-1] in waits


# --- paging -----------------------------------------------------------------


def test_every_page_is_followed(monkeypatch):
    def answer(request, n):
        return body(row(f"2026092500{n:04d}"), page_no=n, total_page=3)

    seen = serve(monkeypatch, answer)
    got = fd.fetch_day(DAY, KEY, sleep=lambda _: None)
    assert len(seen) == 3 and len({d.receipt_no for d in got}) == 3


def test_paging_that_never_ends_is_refused(monkeypatch):
    serve(monkeypatch, lambda request, n: body(row(f"x{n}"), page_no=1, total_page=99))
    with pytest.raises(fd.FetchError, match="more pages past"):
        fd.fetch_day(DAY, KEY, max_pages=3, sleep=lambda _: None)


def test_the_page_number_rides_in_the_query():
    assert "page_no=4" in fd.page_url(KEY, DAY, page=4)


# --- transport --------------------------------------------------------------


def test_a_rate_limited_transport_is_retried(monkeypatch):
    seen = serve(monkeypatch, lambda request, n: body(row()) if n == 2 else http_error(429))
    waits: list[float] = []
    fd._get(fd.page_url(KEY, DAY), KEY, sleep=waits.append)
    assert len(seen) == 2 and waits == [fd.BACKOFF_SECONDS[0]]


def test_a_forbidden_transport_fails_at_once(monkeypatch):
    seen = serve(monkeypatch, lambda request, n: http_error(403))
    waits: list[float] = []
    with pytest.raises(fd.FetchError):
        fd._get(fd.page_url(KEY, DAY), KEY, sleep=waits.append)
    assert len(seen) == 1 and waits == []


# --- what lands on disk -----------------------------------------------------


def test_a_day_is_one_file_so_a_rerun_replaces_one_day(monkeypatch):
    serve(monkeypatch, lambda request, n: body(row(), row("20260925000999", corp_name="SK하이닉스")))
    out = tmp()
    fd.fetch(out, days=1, as_of=DAY, key=KEY, sleep=lambda _: None)

    # `days=1` means "today and the day before", so the window is two weekdays
    # and each gets its own file. That is the point: a re-run of one bad day
    # replaces one file.
    files = sorted(out.glob("*.json"))
    assert [path.name for path in files] == ["2026-09-24.json", "2026-09-25.json"]
    written = json.loads((out / f"{DAY.isoformat()}.json").read_text(encoding="utf-8"))
    assert written["count"] == 2
    assert [d["receipt_no"] for d in written["disclosures"]] == [
        "20260925000123",
        "20260925000999",
    ], "sorted by receipt number, so a re-run of the same day is byte-identical"


def test_an_unlisted_filer_is_counted_separately(monkeypatch):
    serve(monkeypatch, lambda request, n: body(row(), row("20260925000999", stock_code="")))
    out = tmp()
    fd.fetch(out, days=1, as_of=DAY, key=KEY, sleep=lambda _: None)
    written = json.loads((out / f"{DAY.isoformat()}.json").read_text(encoding="utf-8"))
    assert written["count"] == 2 and written["listed_filers"] == 1


def test_weekends_are_never_asked_for():
    assert fd.weekdays(date(2026, 9, 21), date(2026, 9, 27)) == [
        date(2026, 9, day) for day in (21, 22, 23, 24, 25)
    ]
