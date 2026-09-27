"""The stress suite has to actually run on the runner, or the module is unread.

`core/risk/stress.py` has its own unit tests. These are about the wiring: that the
factor file is read whole rather than trimmed to the panel, that the report labels
the book it measured, and that a run which evaluated nothing fails instead of
writing a clean-looking file.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.stress_report import BOOK_NOTE, build, main
from tests.test_data_snapshot import synthetic_market, write_factor_file

# The synthetic price panel spans 2023-01 to 2024-06 (see `synthetic_market`), so a
# factor file that reaches back to 1987 covers scenarios the panel never could --
# which is the whole point of reading it whole.
DEEP_START = "1987-01-01"


def desk(tmp_path, factor_start: str = DEEP_START):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    synthetic_market(data, "SPY", ("2023-01-02", "2023-07-04"), 400.0)
    synthetic_market(data, "QQQ", ("2023-01-02", "2023-07-04"), 300.0)
    factors = write_factor_file(tmp_path / "factors", start=factor_start)
    return data, factors


# --- the wiring --------------------------------------------------------------


def test_the_suite_runs_end_to_end_on_the_smoke_book(tmp_path):
    data, factors = desk(tmp_path)
    report = build(data, factors, allow_dirty=True)

    assert report["book"] == BOOK_NOTE
    assert report["worst_key"]
    assert report["worst_loss"] is not None
    assert set(report["betas"]) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"}
    assert report["residual_daily_vol"] > 0.0


def test_the_risk_free_rate_is_not_among_the_shocked_factors(tmp_path):
    """A financing return added to a loss is not a loss."""
    data, factors = desk(tmp_path)
    report = build(data, factors, allow_dirty=True)
    assert "RF" not in report["betas"]
    assert not any(key.startswith("RF/") for key in report["shocks"])


def test_the_factor_file_is_read_whole_and_not_trimmed_to_the_panel(tmp_path):
    """Trimming would delete every scenario but the most recent and look clean."""
    data, factors = desk(tmp_path)
    report = build(data, factors, allow_dirty=True)

    # `np.is_busday` knows weekends, not holidays, so the file starts on New Year's Day.
    assert report["factor_span"][0] == "1987-01-01"
    measured = {row["key"] for row in report["scenarios"] if row["factor_loss"] is not None}
    assert "black-monday-1987" in measured
    assert "gfc-2008" in measured


def test_a_shallow_factor_file_leaves_the_old_scenarios_unmeasured(tmp_path):
    """And it says so, rather than reporting two windows as the suite."""
    data, factors = desk(tmp_path, factor_start="2023-01-01")
    report = build(data, factors, allow_dirty=True)

    assert report["worst_key"] is None
    assert len(report["unmeasured"]) == report["scenarios_in_suite"]
    assert all("no row in" in line for line in report["unmeasured"])


def test_the_betas_are_measured_on_the_overlap_not_on_the_whole_file(tmp_path):
    data, factors = desk(tmp_path)
    report = build(data, factors, allow_dirty=True)

    assert report["beta_window"][0] >= "2023-01-01"
    assert 200 < report["book_bars"] < 400  # 18 months of weekdays, minus the warm-up


def test_the_report_pins_the_factor_file_it_used(tmp_path):
    """A stress number with no digest cannot be reproduced next quarter."""
    data, factors = desk(tmp_path)
    report = build(data, factors, allow_dirty=True)
    assert len(report["factor_digest"]) == 64
    assert report["run_id"]


def test_the_findings_are_read_against_the_real_ladder(tmp_path):
    data, factors = desk(tmp_path)
    report = build(data, factors, allow_dirty=True)
    assert isinstance(report["findings"], list)
    assert all(isinstance(line, str) for line in report["findings"])


# --- refusals ----------------------------------------------------------------


def test_no_factor_file_is_a_refusal_not_a_typed_scenario(tmp_path):
    data, _factors = desk(tmp_path)
    with pytest.raises(SystemExit, match="run scripts/fetch_factors.py first"):
        build(data, tmp_path / "absent.csv", allow_dirty=True)


def test_no_price_data_is_a_refusal(tmp_path):
    empty = tmp_path / "data"
    empty.mkdir()
    _data, factors = desk(tmp_path / "other")
    with pytest.raises(SystemExit, match="run scripts/fetch_prices.py first"):
        build(empty, factors, allow_dirty=True)


def test_a_market_the_factor_model_does_not_describe_is_refused(tmp_path):
    """A US factor model does not price a Korean book."""
    data = tmp_path / "data"
    data.mkdir()
    synthetic_market(data, "005930", ("2023-01-23",), 70_000.0)
    synthetic_market(data, "000660", ("2023-01-23",), 100_000.0)
    factors = write_factor_file(tmp_path / "factors", start=DEEP_START)
    with pytest.raises(SystemExit, match="does not price another market's book"):
        build(data, factors, allow_dirty=True)


# --- what the run leaves behind ---------------------------------------------


def test_the_run_writes_one_report_and_returns_zero(tmp_path, capsys):
    data, factors = desk(tmp_path)
    out = tmp_path / "stress"

    code = main(["--data", str(data), "--out", str(out), "--factors", str(factors), "--allow-dirty"])
    assert code == 0

    written = list(out.glob("*.stress.json"))
    assert len(written) == 1
    body = json.loads(written[0].read_text(encoding="utf-8"))
    assert body["book"] == BOOK_NOTE
    summary = capsys.readouterr().out
    assert "## Stress suite" in summary
    assert "black-monday-1987" in summary


def test_a_run_that_evaluated_nothing_fails_rather_than_looking_clean(tmp_path, capsys):
    """The one exit code that matters: an empty suite is not a quiet result."""
    data, factors = desk(tmp_path, factor_start="2023-01-01")
    out = tmp_path / "stress"

    code = main(["--data", str(data), "--out", str(out), "--factors", str(factors), "--allow-dirty"])
    assert code == 1
    # Written first: the report naming the gap is what a reader needs.
    assert len(list(out.glob("*.stress.json"))) == 1
    assert "No scenario could be evaluated" in capsys.readouterr().out


def test_the_summary_names_every_window_it_skipped(tmp_path, capsys):
    data, factors = desk(tmp_path, factor_start="2009-01-01")
    main(["--data", str(data), "--out", str(tmp_path / "s"), "--factors", str(factors), "--allow-dirty"])
    summary = capsys.readouterr().out
    assert "Not measured" in summary
    assert "black-monday-1987" in summary  # named as skipped, not omitted


def test_the_summary_says_the_estimate_is_the_optimistic_half(tmp_path, capsys):
    data, factors = desk(tmp_path)
    main(["--data", str(data), "--out", str(tmp_path / "s"), "--factors", str(factors), "--allow-dirty"])
    summary = capsys.readouterr().out
    assert "make the loss look smaller" in summary


def test_the_report_holds_no_nan(tmp_path):
    """`allow_nan=False`, so an unmeasurable metric must already be None."""
    data, factors = desk(tmp_path)
    out = tmp_path / "stress"
    main(["--data", str(data), "--out", str(out), "--factors", str(factors), "--allow-dirty"])
    raw = next(iter(out.glob("*.stress.json"))).read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw


def test_the_worst_observed_moves_carry_their_dates(tmp_path):
    data, factors = desk(tmp_path)
    report = build(data, factors, allow_dirty=True)
    dated = [value for value in report["shocks"].values() if "move" in value]
    assert dated
    for value in dated:
        assert np.datetime64(value["start"]) <= np.datetime64(value["end"])
