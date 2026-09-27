"""The KIND fetch: what it writes, what it refuses to write, and what it never touches.

The refusals matter more than the happy path. This file becomes the Korean
sector source, and a fetch that writes a short or filtered table replaces the
universe with that table for a week.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from core.data.kind import KindShapeError
from scripts import fetch_kind as fk
from tests.data.test_kind import many, row, table

NO_SLEEP = lambda _seconds: None  # noqa: E731


@pytest.fixture
def serve(monkeypatch):
    def install(body: bytes | Exception):
        def answer(url=fk.ENDPOINT, timeout=60.0, sleep=None):
            if isinstance(body, Exception):
                raise body
            return body

        monkeypatch.setattr(fk, "_get", answer)

    return install


def full(count: int = fk.MIN_ROWS) -> bytes:
    return many(count)


# --- what a good run leaves behind -------------------------------------------


def test_the_file_is_one_row_per_listing(serve, tmp_path):
    serve(full())
    path, census = fk.fetch(tmp_path, as_of=date(2026, 9, 27))

    assert path.name == fk.OUTPUT_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(fk.COLUMNS)
    assert len(lines) == fk.MIN_ROWS + 1
    assert census.rows == fk.MIN_ROWS


def test_the_vendor_label_lands_in_the_row_it_belongs_to(serve, tmp_path):
    serve(
        table(
            row(ticker="005930", industry="통신 및 방송 장비 제조업"),
            *(row(ticker=f"{i:06d}") for i in range(fk.MIN_ROWS)),
        )
    )
    path, _census = fk.fetch(tmp_path)
    body = path.read_text(encoding="utf-8")
    assert "005930,삼성전자,코스피,통신 및 방송 장비 제조업,1975-06-11,krx-kind" in body


def test_the_sidecar_carries_every_distinct_label_with_its_count(serve, tmp_path):
    """This census is the input to the bucket table. Without it the table is memory."""
    rows = [row(ticker=f"{i:06d}", industry="가" if i % 2 else "나") for i in range(fk.MIN_ROWS)]
    serve(table(*rows))
    path, _census = fk.fetch(tmp_path)

    sidecar = json.loads(path.with_suffix(".source.json").read_text(encoding="utf-8"))
    assert sidecar["distinct_industries"] == 2
    assert sorted(sidecar["industry_counts"]) == ["가", "나"]
    assert sum(sidecar["industry_counts"].values()) == fk.MIN_ROWS


def test_the_sidecar_says_out_loud_that_no_bucket_was_assigned(serve, tmp_path):
    """So nobody reads this file as a classification."""
    serve(full())
    path, _census = fk.fetch(tmp_path)
    sidecar = json.loads(path.with_suffix(".source.json").read_text(encoding="utf-8"))
    assert sidecar["sector_buckets_assigned"] == 0
    assert "typed from memory" in sidecar["why_no_buckets"]


def test_the_sidecar_records_the_columns_it_expected(serve, tmp_path):
    serve(full())
    path, _census = fk.fetch(tmp_path)
    sidecar = json.loads(path.with_suffix(".source.json").read_text(encoding="utf-8"))
    assert sidecar["columns_expected"].index("업종") == 3
    assert sidecar["columns_used"] == list(fk.USED)
    assert sidecar["encoding"] == "euc-kr"


def test_dropped_rows_are_named_in_the_sidecar(serve, tmp_path):
    rows = [row(ticker=f"{i:06d}") for i in range(fk.MIN_ROWS)]
    serve(table(*rows, row(ticker="nope")))
    path, _census = fk.fetch(tmp_path)
    sidecar = json.loads(path.with_suffix(".source.json").read_text(encoding="utf-8"))
    assert "nope" in sidecar["dropped"]
    assert sidecar["rows_written"] == fk.MIN_ROWS


# --- what it refuses ---------------------------------------------------------


def test_a_short_table_is_refused_rather_than_written(serve, tmp_path):
    """A filtered page has the right shape and the wrong content."""
    serve(many(10))
    with pytest.raises(fk.FetchError, match="below the"):
        fk.fetch(tmp_path)
    assert not (tmp_path / fk.OUTPUT_NAME).exists()


def test_a_shifted_header_is_refused_and_writes_nothing(serve, tmp_path):
    shifted = ("회사명", "종목코드", "주요제품", "업종", "상장일", "결산월", "대표자명", "홈페이지", "지역")
    serve(table(*(row(ticker=f"{i:06d}") for i in range(fk.MIN_ROWS)), header=shifted))
    with pytest.raises(KindShapeError):
        fk.fetch(tmp_path)
    assert not (tmp_path / fk.OUTPUT_NAME).exists()


def test_a_maintenance_page_is_refused(serve, tmp_path):
    serve("<html><body>점검 중입니다</body></html>".encode("euc-kr"))
    with pytest.raises(KindShapeError, match="no table row"):
        fk.fetch(tmp_path)


def test_the_run_reports_a_refusal_instead_of_raising(serve, tmp_path, capsys):
    serve(many(10))
    assert fk.main(["--out", str(tmp_path)]) == 1
    assert "KIND:" in capsys.readouterr().err


def test_a_good_run_prints_the_census_top(serve, tmp_path, capsys):
    rows = [row(ticker=f"{i:06d}", industry="흔한 업종" if i else "드문 업종") for i in range(fk.MIN_ROWS)]
    serve(table(*rows))
    assert fk.main(["--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "distinct industry label(s); no bucket assigned yet" in out
    assert "흔한 업종" in out


# --- the boundaries this fetch must not cross --------------------------------


def test_this_fetch_never_writes_the_membership_file(serve, tmp_path):
    """`scripts/fetch_krx.py` owns kr.csv. Two writers, and nobody can change it."""
    serve(full())
    fk.fetch(tmp_path)
    assert not (tmp_path / "kr.csv").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["kr_industry.csv", "kr_industry.source.json"]


def test_the_request_does_not_pretend_to_be_a_browser(monkeypatch):
    """The same rule every fetcher here follows: say who we are."""
    seen: dict[str, str] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return full()

    def fake_urlopen(request, timeout=60.0):
        seen.update(request.headers)
        return FakeResponse()

    monkeypatch.setattr(fk.urllib.request, "urlopen", fake_urlopen)
    fk._get(sleep=NO_SLEEP)
    agent = seen.get("User-agent", "")
    assert "quant-desk" in agent and "Mozilla" not in agent


def test_the_endpoint_carries_no_key(serve):
    """The whole reason this source was chosen over DART's per-name endpoint."""
    assert "key" not in fk.ENDPOINT.lower()
    assert fk.ENDPOINT.startswith("https://")


def test_a_refused_request_carries_the_hosts_own_words(monkeypatch):
    """Guessing at a 403 cost four runner cycles once."""
    import urllib.error

    error = urllib.error.HTTPError(
        fk.ENDPOINT,
        403,
        "Forbidden",
        {},
        None,  # type: ignore[arg-type]
    )
    error.read = lambda: "접근이 거부되었습니다".encode("euc-kr")  # type: ignore[method-assign]

    def fake_urlopen(request, timeout=60.0):
        raise error

    monkeypatch.setattr(fk.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(fk.FetchError, match="접근이 거부되었습니다"):
        fk._get(sleep=NO_SLEEP)


def test_a_retryable_status_is_retried_then_reported(monkeypatch):
    import urllib.error

    calls = {"n": 0}

    def fake_urlopen(request, timeout=60.0):
        calls["n"] += 1
        raise urllib.error.HTTPError(fk.ENDPOINT, 503, "busy", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(fk.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(fk.FetchError, match="after 4 attempts"):
        fk._get(sleep=NO_SLEEP)
    assert calls["n"] == len(fk.BACKOFF_SECONDS) + 1
