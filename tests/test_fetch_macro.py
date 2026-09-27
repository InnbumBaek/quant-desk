"""The macro fetcher: the initial release, two keys in two bad places, refusals.

Three things here are worth a test each. FRED must be asked for the *initial
release* or the files are quietly look-ahead. Both keys ride in the URL -- one
in the query, one in the path -- so a public job log is one unredacted error
away from a disclosed credential. And a refused series must neither stop the
other nine nor look like an empty window.
"""

from __future__ import annotations

import io
import json
import tempfile
import urllib.error
from datetime import date
from pathlib import Path

import pytest

from core.data.macro import VINTAGE_INITIAL, VINTAGE_UNKNOWN
from scripts import fetch_macro as fm

START = date(2026, 1, 1)
END = date(2026, 9, 25)
FRED_KEY = "0123456789abcdef0123456789abcdef"
ECOS_KEY = "ABCDEFGHIJKLMNOPQRSTUVWX"
_TMP: list = []

ONE_FRED = (fm.FredSeries("DGS10", "10년 국채", "percent"),)
ONE_ECOS = (fm.EcosSeries("722Y001", "0101000", "M", "기준금리"),)


def tmp() -> Path:
    handle = tempfile.TemporaryDirectory()
    _TMP.append(handle)
    return Path(handle.name)


def fred_body(*values: tuple[str, str, str]) -> bytes:
    return json.dumps(
        {
            "observations": [
                {"date": day, "value": value, "realtime_start": published, "realtime_end": "9999-12-31"}
                for day, value, published in values
            ]
        }
    ).encode("utf-8")


def ecos_body(*values: tuple[str, str]) -> bytes:
    return json.dumps(
        {
            "StatisticSearch": {
                "list_total_count": len(values),
                "row": [
                    {
                        "STAT_CODE": "722Y001",
                        "STAT_NAME": "기준금리",
                        "ITEM_CODE1": "0101000",
                        "UNIT_NAME": "연%",
                        "TIME": time,
                        "DATA_VALUE": value,
                    }
                    for time, value in values
                ],
            }
        }
    ).encode("utf-8")


def tables_body(rows: int = 2) -> bytes:
    return json.dumps(
        {"StatisticTableList": {"row": [{"STAT_CODE": f"{n}Y001", "STAT_NAME": "표"} for n in range(rows)]}}
    ).encode("utf-8")


def http_error(code: int, payload: bytes = b"") -> urllib.error.HTTPError:
    import email.message

    return urllib.error.HTTPError(
        fm.FRED_ENDPOINT, code, "refused", email.message.Message(), io.BytesIO(payload)
    )


def serve(monkeypatch, answer):
    """Answer by URL, and keep every URL asked for so a test can read it back."""
    import urllib.request

    asked: list[str] = []

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
        asked.append(request.full_url)
        result = answer(request.full_url, len(asked))
        if isinstance(result, Exception):
            raise result
        return Reply(result)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return asked


def route(fred: bytes | Exception, ecos: bytes | Exception, tables: bytes | Exception | None = None):
    tables = tables_body() if tables is None else tables

    def answer(url: str, n: int):
        if "stlouisfed" in url:
            return fred
        if "StatisticTableList" in url:
            return tables
        return ecos

    return answer


def run(out: Path, monkeypatch, answer, **kwargs) -> dict:
    serve(monkeypatch, answer)
    return fm.fetch(
        out,
        start=START,
        end=END,
        fred_key=FRED_KEY,
        ecos_key=ECOS_KEY,
        sleep=lambda _: None,
        fred_catalogue=kwargs.pop("fred_catalogue", ONE_FRED),
        ecos_catalogue=kwargs.pop("ecos_catalogue", ONE_ECOS),
        **kwargs,
    )


# --- the initial release, which is the whole reason for this fetcher ---------


def test_fred_is_asked_for_the_initial_release_and_nothing_else():
    """`output_type=1` would return today's revised numbers on old dates: look-ahead."""
    url = fm.fred_url("DGS10", FRED_KEY, START)
    assert f"output_type={fm.FRED_INITIAL_RELEASE}" in url
    assert "observation_start=2026-01-01" in url


def test_a_fred_series_is_marked_point_in_time_and_an_ecos_one_is_not(monkeypatch):
    out = tmp()
    run(out, monkeypatch, route(fred_body(("2026-08-03", "4.2", "2026-08-04")), ecos_body(("202608", "2.5"))))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    vintages = {entry["source"]: entry["vintage"] for entry in manifest["series"]}
    assert vintages == {"fred": VINTAGE_INITIAL, "ecos": VINTAGE_UNKNOWN}
    assert [entry["point_in_time"] for entry in manifest["series"] if entry["source"] == "ecos"] == [False]


def test_the_publication_day_lands_in_the_file(monkeypatch):
    out = tmp()
    run(out, monkeypatch, route(fred_body(("2026-08-01", "3.1", "2026-09-11")), ecos_body(("202608", "2.5"))))
    lines = (out / "us" / "DGS10.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "date,value,published_on"
    assert lines[1].startswith("2026-08-01,3.1,2026-09-11")


# --- the keys ---------------------------------------------------------------


def test_the_fred_key_never_reaches_an_error_message(monkeypatch):
    serve(monkeypatch, lambda url, n: http_error(403, b"forbidden"))
    with pytest.raises(fm.FetchError) as caught:
        fm._get(fm.fred_url("DGS10", FRED_KEY, START), FRED_KEY, sleep=lambda _: None)
    assert FRED_KEY not in str(caught.value)


def test_the_ecos_key_never_reaches_an_error_message(monkeypatch):
    """ECOS puts the key in the path, where stripping a query string would miss it."""
    serve(monkeypatch, lambda url, n: b"<html>nope</html>")
    with pytest.raises(fm.FetchError) as caught:
        fm.fetch_ecos(ONE_ECOS[0], ECOS_KEY, START, END, sleep=lambda _: None)
    assert ECOS_KEY not in str(caught.value)


def test_the_ecos_key_is_a_path_segment_so_this_is_not_a_hypothetical():
    assert f"/{ECOS_KEY}/" in fm.ecos_url(ONE_ECOS[0], ECOS_KEY, START, END)


def test_a_refusal_reported_by_ecos_itself_is_also_scrubbed(monkeypatch):
    serve(
        monkeypatch,
        lambda url, n: json.dumps({"RESULT": {"CODE": "INFO-100", "MESSAGE": "bad key"}}).encode(),
    )
    with pytest.raises(fm.FetchError) as caught:
        fm.fetch_ecos(ONE_ECOS[0], ECOS_KEY, START, END, sleep=lambda _: None)
    assert ECOS_KEY not in str(caught.value) and "INFO-100" in str(caught.value)


def test_redaction_covers_both_keys_at_once():
    text = f"{FRED_KEY} and {ECOS_KEY}"
    assert fm.redact(text, FRED_KEY, ECOS_KEY) == "<REDACTED_KEY> and <REDACTED_KEY>"


def test_the_shape_report_carries_field_names_and_never_values(monkeypatch):
    """A value could be the key echoed back by the vendor. Names only."""
    payload = json.loads(fred_body(("2026-08-01", "3.1", "2026-09-11")))
    shape = fm.schema_of(payload, "", "observations")
    assert shape["fields"] == ["date", "realtime_end", "realtime_start", "value"]
    assert "3.1" not in json.dumps(shape)


# --- a refused series is not an empty series --------------------------------


def test_one_refused_series_does_not_cost_the_others(monkeypatch):
    out = tmp()
    report = run(out, monkeypatch, route(http_error(403), ecos_body(("202608", "2.5"))))
    assert report["series"] == 1 and len(report["refused"]) == 1
    assert (out / "kr" / "722Y001_0101000_M.csv").exists()
    assert not (out / "us").exists(), "a refused series must not leave a file behind"


def test_a_refusal_makes_the_job_exit_non_zero(monkeypatch):
    serve(monkeypatch, route(http_error(403), ecos_body(("202608", "2.5"))))
    monkeypatch.setattr(fm, "FRED_CATALOGUE", ONE_FRED)
    monkeypatch.setattr(fm, "ECOS_CATALOGUE", ONE_ECOS)
    monkeypatch.setenv("FRED_API_KEY", FRED_KEY)
    monkeypatch.setenv("ECOS_API_KEY", ECOS_KEY)
    monkeypatch.setattr(fm, "PAUSE_SECONDS", 0.0)
    assert fm.main(["--out", str(tmp()), "--start", START.isoformat()]) == 1


def test_a_clean_run_exits_zero(monkeypatch):
    serve(monkeypatch, route(fred_body(("2026-08-01", "3.1", "2026-09-11")), ecos_body(("202608", "2.5"))))
    monkeypatch.setattr(fm, "FRED_CATALOGUE", ONE_FRED)
    monkeypatch.setattr(fm, "ECOS_CATALOGUE", ONE_ECOS)
    monkeypatch.setenv("FRED_API_KEY", FRED_KEY)
    monkeypatch.setenv("ECOS_API_KEY", ECOS_KEY)
    monkeypatch.setattr(fm, "PAUSE_SECONDS", 0.0)
    assert fm.main(["--out", str(tmp()), "--start", START.isoformat()]) == 0


def test_an_empty_window_is_written_as_an_empty_series(monkeypatch):
    """INFO-200 is an answer. A header-only file records that we asked."""
    out = tmp()
    quiet = json.dumps({"RESULT": {"CODE": "INFO-200", "MESSAGE": "없음"}}).encode()
    report = run(out, monkeypatch, route(fred_body(("2026-08-01", "3.1", "2026-09-11")), quiet))
    assert report["refused"] == [] and report["series"] == 2
    assert (out / "kr" / "722Y001_0101000_M.csv").read_text(
        encoding="utf-8"
    ).strip() == "date,value,published_on"


# --- the window ECOS is asked for -------------------------------------------


@pytest.mark.parametrize(("cycle", "expected"), [("D", "20260101/20260925"), ("M", "202601/202609")])
def test_the_window_is_stamped_the_way_the_cycle_requires(cycle: str, expected: str):
    series = fm.EcosSeries("817Y002", "010190000", cycle, "국고채")
    assert expected in fm.ecos_url(series, ECOS_KEY, START, END)


def test_a_cycle_ecos_does_not_document_is_refused():
    with pytest.raises(fm.FetchError, match="not one ECOS documents"):
        fm.ecos_url(fm.EcosSeries("817Y002", "0", "H", "없는 주기"), ECOS_KEY, START, END)


def test_the_item_code_is_the_last_segment():
    assert fm.ecos_url(ONE_ECOS[0], ECOS_KEY, START, END).endswith("/0101000")


# --- transport --------------------------------------------------------------


def test_a_rate_limited_transport_is_retried(monkeypatch):
    body = fred_body(("2026-08-01", "3.1", "2026-09-11"))
    asked = serve(monkeypatch, lambda url, n: body if n == 2 else http_error(429))
    waits: list[float] = []
    fm._get(fm.fred_url("DGS10", FRED_KEY, START), FRED_KEY, sleep=waits.append)
    assert len(asked) == 2 and waits == [fm.BACKOFF_SECONDS[0]]


def test_a_forbidden_transport_fails_at_once(monkeypatch):
    asked = serve(monkeypatch, lambda url, n: http_error(403))
    with pytest.raises(fm.FetchError):
        fm._get(fm.fred_url("DGS10", FRED_KEY, START), FRED_KEY, sleep=lambda _: None)
    assert len(asked) == 1


# --- the catalogue ----------------------------------------------------------


def test_nothing_unredistributable_is_in_the_catalogue():
    """These files are committed, so a licensed series would be a redistribution."""
    catalogued = {series.series_id for series in fm.FRED_CATALOGUE}
    assert catalogued.isdisjoint(fm.NOT_REDISTRIBUTABLE)


def test_the_manifest_says_why_a_wanted_series_is_absent(monkeypatch):
    out = tmp()
    run(out, monkeypatch, route(fred_body(("2026-08-01", "3.1", "2026-09-11")), ecos_body(("202608", "2.5"))))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "VIXCLS" in manifest["not_redistributable"]


def test_the_ecos_table_list_is_kept_so_the_codes_can_be_checked(monkeypatch):
    out = tmp()
    report = run(
        out, monkeypatch, route(fred_body(("2026-08-01", "3.1", "2026-09-11")), ecos_body(("202608", "2.5")))
    )
    assert report["ecos_tables"] == 2
    assert json.loads((out / "ecos_tables.json").read_text(encoding="utf-8"))[0]["STAT_CODE"] == "0Y001"


def test_a_missing_table_list_is_a_refusal_not_a_silent_skip(monkeypatch):
    out = tmp()
    report = run(
        out,
        monkeypatch,
        route(
            fred_body(("2026-08-01", "3.1", "2026-09-11")),
            ecos_body(("202608", "2.5")),
            tables=json.dumps({"RESULT": {"CODE": "INFO-100", "MESSAGE": "bad key"}}).encode(),
        ),
    )
    assert len(report["refused"]) == 1 and "table list" in report["refused"][0]
