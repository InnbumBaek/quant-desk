"""The frontier report: what history a wider universe costs, measured on files.

`tests/backtest/test_breadth.py` checks the breadth measure. These are about the
wiring -- that a late-listing symbol shows up as a shorter window rather than as a
crash, that a universe which will not load is recorded with its reason instead of
being dropped, and that the exit code says whether more tickers would buy
anything the fundamental law can use (ADR-0042).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.data.sources import series_spans
from core.data.universe import history_frontier, history_screen
from scripts.universe_frontier import build, main, markdown


def series(directory, symbol: str, first: str, last: str = "2024-06-28", seed: int = 0) -> int:
    """Weekday closes between two dates, on one calendar."""
    days = np.arange(np.datetime64(first), np.datetime64(last), dtype="datetime64[D]")
    days = days[np.is_busday(days)]
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(days))))
    body = "Date,Close,Volume\n" + "".join(
        f"{d},{p:.4f},{1_000_000 + i}\n" for i, (d, p) in enumerate(zip(days, prices, strict=True))
    )
    (directory / f"{symbol.lower()}.csv").write_text(body, encoding="utf-8")
    return len(days)


def desk(tmp_path):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    for index, symbol in enumerate(("SPY", "QQQ", "IWM", "TLT", "GLD")):
        series(data, symbol, "2019-01-02", seed=index)
    # The case the report exists for: one name that listed late.
    series(data, "VOO", "2022-06-01", seed=99)
    return data


def test_a_late_listing_name_appears_as_a_shorter_window_not_as_an_error(tmp_path):
    report = build(desk(tmp_path), allow_dirty=True)
    rows = {row["window_start"]: row for row in report["frontier"]}
    assert len(rows) == 2, "two inception dates, two candidate universes"
    early, late = rows["2019-01-02"], rows["2022-06-01"]
    assert early["count"] == 5 and late["count"] == 6
    assert early["observations"] > late["observations"], "the extra name costs history"


def test_the_names_a_window_excludes_carry_their_inception_date(tmp_path):
    report = build(desk(tmp_path), allow_dirty=True)
    early = next(row for row in report["frontier"] if row["window_start"] == "2019-01-02")
    assert "VOO" in early["excluded"]
    assert "2022-06-01" in early["excluded"]["VOO"]


def test_the_baseline_is_the_universe_the_desk_actually_trades(tmp_path):
    """ "Wider" means nothing without the universe the rejections came from."""
    report = build(desk(tmp_path), allow_dirty=True)
    assert report["baseline"]["symbols"] == ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    assert report["baseline"]["effective_bets"] is not None


def test_independent_names_raise_the_effective_count_and_the_law_factor(tmp_path):
    """Synthetic series are independent by construction, so this is the ceiling
    case: six independent names should measure as more bets than five."""
    report = build(desk(tmp_path), allow_dirty=True)
    late = next(row for row in report["frontier"] if row["count"] == 6)
    assert late["effective_bets"] > report["baseline"]["effective_bets"]
    assert late["law_factor_vs_baseline"] > 1.0


def test_a_universe_that_will_not_load_is_recorded_with_its_reason(tmp_path):
    """A hole *inside* the shared window still refuses the panel, and the row
    stays: a universe that cannot be loaded is a fact about that universe.

    This is the case windowing must not swallow. A late listing is a shorter
    window; a vendor dropping a year of sessions everyone else traded is a fault,
    and `min_coverage` has to keep catching it."""
    data = tmp_path / "data"
    data.mkdir(parents=True)
    for index, symbol in enumerate(("SPY", "QQQ")):
        series(data, symbol, "2019-01-02", seed=index)
    series(data, "IWM", "2019-01-02", seed=5)
    holey = data / "iwm.csv"
    kept = [
        line
        for line in holey.read_text(encoding="utf-8").splitlines()
        if not line.startswith(("2020-", "2021-", "2022-"))
    ]
    holey.write_text("\n".join(kept) + "\n", encoding="utf-8")

    report = build(data, allow_dirty=True)
    widest = next(row for row in report["frontier"] if row["count"] == 3)
    assert not widest["loaded"]
    assert "common to every symbol" in str(widest["reason"])
    assert "not loaded" in markdown(report)


def test_an_unreadable_file_is_named_rather_than_skipped(tmp_path):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    series(data, "SPY", "2019-01-02")
    series(data, "QQQ", "2019-01-02", seed=2)
    (data / "junk.csv").write_text("Nope,Nothing\n1,2\n", encoding="utf-8")
    report = build(data, allow_dirty=True)
    assert "JUNK" in report["unreadable"]
    assert "Not readable:" in markdown(report)


def test_a_readable_file_for_an_undeclared_symbol_is_named_not_crashed_on(tmp_path):
    """A stray ticker in `data/` must not take the report down. Its market and
    currency are genuinely unknown (ADR-0017), which is a reason, not a crash."""
    data = tmp_path / "data"
    data.mkdir(parents=True)
    series(data, "SPY", "2019-01-02")
    series(data, "QQQ", "2019-01-02", seed=2)
    series(data, "ZZZZ", "2019-01-02", seed=3)
    report = build(data, allow_dirty=True)
    assert "not declared" in report["unreadable"]["ZZZZ"]
    assert all("ZZZZ" not in row["symbols"] for row in report["frontier"])


def test_no_readable_series_is_a_refusal_not_an_empty_frontier(tmp_path):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "junk.csv").write_text("Nope\n1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        build(data, allow_dirty=True)


def test_the_exit_code_says_whether_more_tickers_buy_anything(tmp_path, capsys):
    """0 when some universe on the frontier carries more effective bets than the
    baseline; 1 when more tickers would buy nothing the law can use."""
    assert main(["--data", str(desk(tmp_path)), "--out", str(tmp_path / "out"), "--allow-dirty"]) == 0
    assert "Universe frontier" in capsys.readouterr().out
    assert list((tmp_path / "out").glob("*.frontier.json"))


def test_a_baseline_only_fetch_gains_nothing_and_says_so(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    for index, symbol in enumerate(("SPY", "QQQ", "IWM", "TLT", "GLD")):
        series(data, symbol, "2019-01-02", seed=index)
    assert main(["--data", str(data), "--out", str(tmp_path / "out"), "--allow-dirty"]) == 1
    capsys.readouterr()


# --- the screen and the curve, without the report ------------------------------


def test_the_screen_and_the_frontier_agree_on_every_step(tmp_path):
    spans, unreadable = series_spans({"SPY": tmp_path / "nowhere.csv"})
    assert spans == {} and "does not exist" in unreadable["SPY"]

    data = desk(tmp_path)
    from scripts.data_snapshot import discover

    spans, _ = series_spans(discover(data))
    for start, kept in history_frontier(spans):
        assert history_screen(spans, start).kept == kept
