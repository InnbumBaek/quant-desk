"""The fetcher's job is to refuse bad responses, so that is what is tested.

No test here touches the network: every source is exercised against a recorded
response body. The container these tests run in has no route to a vendor host
anyway, and a test that needed one would be a test that fails for the wrong
reason.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from scripts import fetch_prices as fp

START, END = date(2024, 1, 1), date(2024, 3, 1)


def stooq_csv(rows: int = 80, header: str = fp.EXPECTED_HEADER) -> bytes:
    lines = [header]
    for i in range(rows):
        day = date(2024, 1, 1).toordinal() + i
        lines.append(f"{date.fromordinal(day)},100,101,99,{100 + i * 0.1:.4f},1000000")
    return ("\n".join(lines) + "\n").encode()


def yahoo_json(rows: int = 80, closes=None) -> bytes:
    stamps = [1704067200 + i * 86400 for i in range(rows)]
    closes = closes if closes is not None else [100.0 + i for i in range(rows)]
    return json.dumps(
        {
            "chart": {
                "error": None,
                "result": [
                    {
                        "timestamp": stamps,
                        "indicators": {
                            "quote": [
                                {
                                    "open": closes,
                                    "high": closes,
                                    "low": closes,
                                    "close": closes,
                                    "volume": [1_000_000] * rows,
                                }
                            ]
                        },
                    }
                ],
            }
        }
    ).encode()


@pytest.fixture
def no_network(monkeypatch):
    """Any unexpected HTTP call fails the test rather than reaching out."""

    def refuse(url, timeout=30.0):
        raise AssertionError(f"unexpected network call to {url}")

    monkeypatch.setattr(fp, "_get", refuse)
    return monkeypatch


def serve(monkeypatch, body: bytes):
    monkeypatch.setattr(fp, "_get", lambda url, timeout=30.0: body)


# --- parsing ----------------------------------------------------------------


def test_stooq_csv_parses_into_bars(no_network):
    serve(no_network, stooq_csv(rows=70))
    bars = fp.fetch_stooq("SPY", START, END)
    assert len(bars) == 70
    assert bars[0].day == "2024-01-01"
    assert bars[0].close == pytest.approx(100.0)


def test_yahoo_json_parses_into_bars(no_network):
    serve(no_network, yahoo_json(rows=70))
    bars = fp.fetch_yahoo("SPY", START, END)
    assert len(bars) == 70
    assert bars[0].close == pytest.approx(100.0)


def test_yahoo_padding_rows_are_dropped_not_filled(no_network):
    closes = [100.0 + i for i in range(70)]
    closes[10] = None
    serve(no_network, yahoo_json(rows=70, closes=closes))
    bars = fp.fetch_yahoo("SPY", START, END)
    assert len(bars) == 69, "a null close is a day that did not trade, not a price to invent"


# --- refusals ---------------------------------------------------------------


def test_a_rate_limit_notice_is_an_error_not_an_empty_panel(no_network):
    serve(no_network, b"Exceeded the daily hits limit\n")
    with pytest.raises(fp.FetchError, match="not the daily CSV header"):
        fp.fetch_stooq("SPY", START, END)


def test_an_html_error_page_is_an_error(no_network):
    serve(no_network, b"<!DOCTYPE html><html><body>404</body></html>")
    with pytest.raises(fp.FetchError, match="not the daily CSV header"):
        fp.fetch_stooq("SPY", START, END)


def test_a_yahoo_error_block_is_an_error(no_network):
    serve(no_network, json.dumps({"chart": {"error": {"code": "Not Found"}}}).encode())
    with pytest.raises(fp.FetchError, match="yahoo error"):
        fp.fetch_yahoo("NOPE", START, END)


def test_yahoo_length_disagreement_is_an_error(no_network):
    serve(
        no_network,
        json.dumps(
            {
                "chart": {
                    "error": None,
                    "result": [
                        {
                            "timestamp": [1, 2, 3],
                            "indicators": {"quote": [{"close": [1.0]}]},
                        }
                    ],
                }
            }
        ).encode(),
    )
    with pytest.raises(fp.FetchError, match="stamps and"):
        fp.fetch_yahoo("SPY", START, END)


def test_a_short_history_is_refused(tmp_path):
    bars = [fp.Bar(f"2024-01-{i + 1:02d}", 1, 1, 1, 1, 1) for i in range(10)]
    with pytest.raises(fp.FetchError, match="need at least"):
        fp.write_bars("SPY", bars, tmp_path, "stooq", "shape")


def test_a_repeated_date_is_refused(tmp_path):
    bars = [fp.Bar("2024-01-01", 1, 1, 1, 1, 1)] * fp.MIN_ROWS
    with pytest.raises(fp.FetchError, match="repeated date"):
        fp.write_bars("SPY", bars, tmp_path, "stooq", "shape")


def test_a_non_positive_close_is_refused(tmp_path):
    bars = [fp.Bar(f"2024-{1 + i // 28:02d}-{i % 28 + 1:02d}", 1, 1, 1, 1, 1) for i in range(fp.MIN_ROWS)]
    bars[5] = fp.Bar(bars[5].day, 1, 1, 1, 0.0, 1)
    with pytest.raises(fp.FetchError, match="non-positive close"):
        fp.write_bars("SPY", bars, tmp_path, "stooq", "shape")


def test_an_unknown_source_is_named(tmp_path):
    with pytest.raises(fp.FetchError, match="unknown source"):
        fp.fetch_all(["SPY"], tmp_path, source="bloomberg", pause=0)


# --- the written file is what core.data.sources reads ------------------------


def test_written_csv_loads_as_a_panel(tmp_path, no_network):
    from core.data.sources import load_csv_panel

    serve(no_network, stooq_csv(rows=70))
    for symbol in ("SPY", "QQQ"):
        bars = fp.fetch_stooq(symbol, START, END)
        fp.write_bars(symbol, bars, tmp_path, "stooq", fp.URL_SHAPES["stooq"])

    panel, manifest = load_csv_panel({"SPY": tmp_path / "spy.csv", "QQQ": tmp_path / "qqq.csv"})
    assert panel.close.shape == (70, 2)
    assert manifest.symbols == ("QQQ", "SPY")
    assert manifest.has_volume


def test_provenance_is_written_beside_the_csv(tmp_path, no_network):
    serve(no_network, stooq_csv(rows=70))
    bars = fp.fetch_stooq("SPY", START, END)
    fp.write_bars("SPY", bars, tmp_path, "stooq", fp.URL_SHAPES["stooq"])

    provenance = json.loads((tmp_path / "spy.source.json").read_text())
    assert provenance["source"] == "stooq"
    assert provenance["rows"] == 70
    assert provenance["first_date"] == "2024-01-01"
    assert "stooq.com" in provenance["url_shape"]


# --- the fallback -----------------------------------------------------------


def test_auto_falls_through_to_the_next_source(tmp_path, monkeypatch):
    """A source that is blocked or rate-limited must not end the run."""
    calls: list[str] = []

    def stooq_fails(symbol, start, end):
        calls.append("stooq")
        raise fp.FetchError("stooq says no")

    def yahoo_works(symbol, start, end):
        calls.append("yahoo")
        return [
            fp.Bar(date.fromordinal(date(2024, 1, 1).toordinal() + i).isoformat(), 1, 1, 1, 100.0 + i, 1e6)
            for i in range(70)
        ]

    monkeypatch.setitem(fp.SOURCES, "stooq", stooq_fails)
    monkeypatch.setitem(fp.SOURCES, "yahoo", yahoo_works)

    report = fp.fetch_all(["SPY"], tmp_path, source="auto", pause=0)
    assert report["ok"]
    assert report["source"] == "yahoo"
    assert calls == ["stooq", "yahoo"]
    assert (tmp_path / "spy.csv").exists()


def test_when_every_source_fails_the_report_says_what_each_one_said(tmp_path, monkeypatch):
    def fails(name):
        def inner(symbol, start, end):
            raise fp.FetchError(f"{name} says no")

        return inner

    monkeypatch.setitem(fp.SOURCES, "stooq", fails("stooq"))
    monkeypatch.setitem(fp.SOURCES, "yahoo", fails("yahoo"))

    report = fp.fetch_all(["SPY"], tmp_path, source="auto", pause=0)
    assert not report["ok"]
    assert [attempt["source"] for attempt in report["attempts"]] == ["stooq", "yahoo"]
    assert "stooq says no" in report["attempts"][0]["failures"]["SPY"]
    assert not list(tmp_path.glob("*.csv"))


def test_a_partial_panel_is_not_accepted(tmp_path, monkeypatch):
    """Four of five symbols is a different universe, so it must not pass as success."""

    def one_symbol_missing(symbol, start, end):
        if symbol == "GLD":
            raise fp.FetchError("no data for GLD")
        return [
            fp.Bar(date.fromordinal(date(2024, 1, 1).toordinal() + i).isoformat(), 1, 1, 1, 100.0 + i, 1e6)
            for i in range(70)
        ]

    monkeypatch.setitem(fp.SOURCES, "stooq", one_symbol_missing)
    monkeypatch.setitem(fp.SOURCES, "yahoo", one_symbol_missing)

    report = fp.fetch_all(["SPY", "GLD"], tmp_path, source="auto", pause=0)
    assert not report["ok"]
    assert report["attempts"][0]["written"] == ["SPY"]
    assert "GLD" in report["attempts"][0]["failures"]
