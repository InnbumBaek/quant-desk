"""The impact census has to actually run on the runner, or the module is unread.

`core/execution/impact.py` has its own unit tests. These are about the wiring:
that the census reaches the snapshot panel, that the report labels the book it
measured, that the cost side of capacity stays unmeasured rather than borrowing a
number, and that a run which could cost nothing fails rather than writing a
clean-looking file of zeroes.
"""

from __future__ import annotations

import json

import numpy as np

from scripts.impact_report import BOOK_NOTE, QUOTE_NOTIONAL, build, census, main, markdown
from tests.test_data_snapshot import sidecar, synthetic_market


def desk(tmp_path):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    synthetic_market(data, "SPY", ("2023-01-02", "2023-07-04"), 400.0)
    synthetic_market(data, "QQQ", ("2023-01-02", "2023-07-04"), 300.0)
    return data


# --- the wiring --------------------------------------------------------------


def test_the_census_runs_end_to_end_on_the_smoke_book(tmp_path):
    report = build(desk(tmp_path), allow_dirty=True)
    assert report["book"] == BOOK_NOTE
    assert report["symbols"] == 2
    assert report["measured"] == 2
    assert report["measurable_share"] == 1.0
    assert report["estimator"].startswith("amihud-2002")


def test_the_book_round_trip_cost_is_quoted_beside_its_notional(tmp_path):
    """A cost fraction without a size is meaningless: impact is a function of size."""
    report = build(desk(tmp_path), allow_dirty=True)
    assert report["quote_notional"] == QUOTE_NOTIONAL
    cost = report["book_round_trip_cost_fraction"]
    assert cost is not None and 0.0 < cost < 1.0
    assert report["book_unmeasured"] == ""


def test_the_conservatism_travels_in_the_artifact_and_not_only_in_the_code(tmp_path):
    """A reader of the JSON must see what the number assumes."""
    report = build(desk(tmp_path), allow_dirty=True)
    note = report["conservatism"]
    assert "upper bound" in note and "not a fill forecast" in note


def test_the_cost_side_of_capacity_stays_unmeasured_because_no_alpha_has_passed(tmp_path):
    """Typing in an edge to make the number appear is what ADR-0031 forbids."""
    report = build(desk(tmp_path), allow_dirty=True)
    assert report["capacity_unmeasured"]
    assert "gross edge" in report["capacity_unmeasured"]
    assert "cost_capacity" not in report


def test_the_report_is_written_and_printed_with_its_run_id(tmp_path, capsys):
    data = desk(tmp_path)
    out = tmp_path / "impact"
    assert main(["--data", str(data), "--out", str(out), "--allow-dirty"]) == 0
    written = list(out.glob("*.impact.json"))
    assert len(written) == 1
    body = json.loads(written[0].read_text(encoding="utf-8"))
    assert written[0].name.startswith(body["run_id"])
    printed = capsys.readouterr().out
    assert "costable" in printed
    assert "upper bound" in printed


# --- the headline is the count that cannot be costed -------------------------


def test_a_panel_without_dollar_volume_fails_the_run(tmp_path):
    """Not a quiet zero: it means the capacity machinery is unusable, and a run
    that reports that as a clean pass hides it."""
    data = tmp_path / "data"
    data.mkdir(parents=True)
    days = np.arange(np.datetime64("2023-01-02"), np.datetime64("2024-07-01"), dtype="datetime64[D]")
    days = days[np.is_busday(days)]
    rng = np.random.default_rng(7)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(days))))
    for symbol in ("SPY", "QQQ"):
        body = "Date,Close\n" + "".join(f"{d},{p:.4f}\n" for d, p in zip(days, prices, strict=True))
        (data / f"{symbol.lower()}.csv").write_text(body, encoding="utf-8")
        sidecar(data, symbol, source="synthetic", rows=len(days))

    out = tmp_path / "impact"
    assert main(["--data", str(data), "--out", str(out), "--allow-dirty"]) == 1
    body = json.loads(next(out.glob("*.impact.json")).read_text(encoding="utf-8"))
    assert body["measured"] == 0
    assert body["measurable_share"] == 0.0
    assert body["book_round_trip_cost_fraction"] is None
    assert body["unmeasurable_reasons"]


def test_identical_refusals_are_grouped_so_a_hundred_reads_as_one_problem(tmp_path):
    """A hundred identical reasons is one problem, not a hundred."""
    from core.data.sources import PricePanel

    rows = 80
    dates = np.datetime64("2026-01-01", "D") + np.arange(rows)
    symbols = tuple(f"S{i}" for i in range(5))
    panel = PricePanel(
        dates=dates,
        symbols=symbols,
        close=np.full((rows, len(symbols)), 100.0),  # flat, so no lambda anywhere
        dollar_volume=np.full((rows, len(symbols)), 1e7),
    )
    summary, lambdas = census(panel)
    assert lambdas == {}
    assert summary["unmeasurable"] == 5
    assert len(summary["unmeasurable_reasons"]) == 1
    assert next(iter(summary["unmeasurable_reasons"].values())) == 5


def test_the_most_illiquid_names_are_named_so_the_worst_case_is_visible(tmp_path):
    report = build(desk(tmp_path), allow_dirty=True)
    worst = report["most_illiquid"]
    assert worst and set(worst[0]) == {"symbol", "lambda"}
    assert worst[0]["lambda"] >= worst[-1]["lambda"]
    assert report["lambda_median"] > 0.0


def test_the_summary_says_how_many_names_could_not_be_costed(tmp_path):
    report = build(desk(tmp_path), allow_dirty=True)
    text = markdown(report)
    assert f"{report['measured']} of {report['symbols']} symbol(s) costable" in text
    assert "cost-based capacity" in text


def test_a_missing_data_directory_is_a_refusal_and_not_an_empty_report(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="fetch_prices"):
        build(tmp_path / "nothing", allow_dirty=True)
