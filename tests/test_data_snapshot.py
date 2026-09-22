"""The snapshot script's job is to leave a record that is true about the fetch.

The manifest already says *which* bytes a result came from, by hash. These tests
cover the part that was missing: *where* those bytes came from. The sidecars that
`fetch_prices` writes live in git-ignored `data/` and do not survive the runner,
so the source name has to be copied into the committed run file or the
reproduction claim in ADR-0007 is only half true.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from core.data.markets import UnknownSymbol
from core.data.sources import load_csv_panel, read_market_manifest
from scripts.data_snapshot import discover, json_safe, main, read_provenance


def sidecar(directory, symbol: str, source: str = "yahoo", rows: int = 755) -> None:
    (directory / f"{symbol.lower()}.source.json").write_text(
        json.dumps(
            {
                "symbol": symbol,
                "source": source,
                "url_shape": f"https://{source}.example/{{symbol}}",
                "fetched_at": "2026-09-22T10:59:25+00:00",
                "rows": rows,
                "first_date": "2023-09-18",
                "last_date": "2026-09-21",
            }
        )
    )


def test_the_source_of_every_symbol_is_read_from_its_sidecar(tmp_path):
    for symbol in ("SPY", "QQQ"):
        sidecar(tmp_path, symbol)

    provenance = read_provenance(tmp_path, ["SPY", "QQQ"])

    assert provenance["sources"] == ["yahoo"]
    assert provenance["no_provenance"] == []
    assert provenance["by_symbol"]["SPY"]["source"] == "yahoo"
    assert provenance["by_symbol"]["SPY"]["rows"] == 755


def test_a_mixed_fetch_names_every_source_it_used(tmp_path):
    sidecar(tmp_path, "SPY", source="yahoo")
    sidecar(tmp_path, "KS200", source="stooq")

    provenance = read_provenance(tmp_path, ["SPY", "KS200"])

    assert provenance["sources"] == ["stooq", "yahoo"]


def test_a_missing_sidecar_is_recorded_as_missing_not_guessed(tmp_path):
    sidecar(tmp_path, "SPY")

    provenance = read_provenance(tmp_path, ["SPY", "QQQ"])

    assert provenance["no_provenance"] == ["QQQ"]
    assert "QQQ" not in provenance["by_symbol"]
    assert provenance["sources"] == ["yahoo"], "the absent symbol must not borrow SPY's source"


def test_an_unreadable_sidecar_is_recorded_as_unreadable(tmp_path):
    (tmp_path / "spy.source.json").write_text("{not json")

    provenance = read_provenance(tmp_path, ["SPY"])

    assert provenance["no_provenance"] == ["SPY"]
    assert "JSONDecodeError" in provenance["by_symbol"]["SPY"]["unreadable"]
    assert provenance["sources"] == []


def test_provenance_files_are_not_mistaken_for_symbols(tmp_path):
    (tmp_path / "spy.csv").write_text("Date,Open,High,Low,Close,Volume\n")
    sidecar(tmp_path, "SPY")

    assert list(discover(tmp_path)) == ["SPY"]


# --- the committed record must be standard JSON -----------------------------


def test_an_infinite_metric_becomes_null_and_is_named(tmp_path):
    """G4's drawdown/return ratio is infinite whenever the return is not positive.

    `json.dump` would write `Infinity`, which only Python reads back. The value
    becomes null and the key is listed, so nothing is lost and the file parses
    everywhere.
    """
    found: list[str] = []
    safe = json_safe(
        {"verdicts": [{"metrics": {"dd_to_return": math.inf, "pbo": 0.31}}]},
        found=found,
    )

    assert safe["verdicts"][0]["metrics"]["dd_to_return"] is None
    assert safe["verdicts"][0]["metrics"]["pbo"] == 0.31
    assert found == ["verdicts[0].metrics.dd_to_return=inf"]
    json.dumps(safe, allow_nan=False)  # would raise if anything non-finite survived


def test_a_nan_is_caught_too(tmp_path):
    found: list[str] = []
    json_safe({"sharpe": math.nan}, found=found)
    assert found == ["sharpe=nan"]


def test_finite_values_are_untouched():
    payload = {"a": 1, "b": [0.5, "x", None], "c": {"d": True}}
    assert json_safe(payload) == payload


# --- the two-market path -----------------------------------------------------


def synthetic_market(directory, symbol: str, holidays: tuple[str, ...], start: float) -> int:
    """A year and a half of weekday closes on one calendar, minus its holidays."""
    days = np.arange(np.datetime64("2023-01-02"), np.datetime64("2024-07-01"), dtype="datetime64[D]")
    days = days[np.is_busday(days)]
    days = days[~np.isin(days, np.array(holidays, dtype="datetime64[D]"))]
    rng = np.random.default_rng(abs(hash(symbol)) % 2**31)
    prices = start * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(days))))
    body = "Date,Close,Volume\n" + "".join(
        f"{d},{p:.4f},{1_000_000 + i}\n" for i, (d, p) in enumerate(zip(days, prices, strict=True))
    )
    (directory / f"{symbol.lower()}.csv").write_text(body, encoding="utf-8")
    sidecar(directory, symbol, source="synthetic", rows=len(days))
    return len(days)


def test_each_market_is_snapshotted_on_its_own_calendar(tmp_path):
    """The script has to keep both calendars, not their intersection (ADR-0013)."""
    data, snapshots = tmp_path / "data", tmp_path / "snapshots"
    data.mkdir()
    us_rows = synthetic_market(data, "SPY", ("2023-01-02", "2023-07-04", "2024-01-15"), 400.0)
    synthetic_market(data, "QQQ", ("2023-01-02", "2023-07-04", "2024-01-15"), 300.0)
    kr_rows = synthetic_market(data, "005930", ("2023-01-23", "2023-01-24", "2024-02-12"), 70_000.0)

    assert main(["--data", str(data), "--snapshots", str(snapshots), "--allow-dirty"]) == 0

    index = list(snapshots.glob("*.markets.json"))
    assert len(index) == 1
    per_market = read_market_manifest(index[0])
    assert sorted(per_market) == ["KR", "US"]
    assert per_market["US"].rows == us_rows
    assert per_market["KR"].rows == kr_rows
    assert per_market["US"].dates_dropped == 0 and per_market["KR"].dates_dropped == 0

    # And the merged panel, which is what the old path built, is shorter than both.
    merged, _ = load_csv_panel(discover(data), min_coverage=0.0)
    assert merged.dates.shape[0] < min(us_rows, kr_rows)

    smokes = [json.loads(p.read_text()) for p in snapshots.glob("*.smoke.json")]
    assert sorted(s["market"] for s in smokes) == ["KR", "US"]
    assert len({s["run_id"] for s in smokes}) == 2, "each market pins its own data, so its own run id"
    for smoke in smokes:
        # 18 months clears the annualising floor, so the rate is measured and the
        # record says which number the backtest actually used (ADR-0013).
        assert 240 < smoke["sessions_per_year"]["measured"] < 262
        assert smoke["sessions_per_year"]["used_by_config"] == 252
        assert smoke["provenance"]["sources"] == ["synthetic"]


def test_a_single_market_fetch_writes_no_index(tmp_path):
    """One market's own manifest already names the whole read."""
    data, snapshots = tmp_path / "data", tmp_path / "snapshots"
    data.mkdir()
    synthetic_market(data, "SPY", ("2023-01-02",), 400.0)
    synthetic_market(data, "QQQ", ("2023-01-02",), 300.0)

    assert main(["--data", str(data), "--snapshots", str(snapshots), "--allow-dirty"]) == 0
    assert list(snapshots.glob("*.markets.json")) == []
    assert len(list(snapshots.glob("*.smoke.json"))) == 1


def test_an_undeclared_symbol_stops_the_snapshot(tmp_path):
    data, snapshots = tmp_path / "data", tmp_path / "snapshots"
    data.mkdir()
    synthetic_market(data, "SPY", ("2023-01-02",), 400.0)
    synthetic_market(data, "NVDA", ("2023-01-02",), 100.0)

    with pytest.raises(UnknownSymbol):
        main(["--data", str(data), "--snapshots", str(snapshots), "--allow-dirty"])


def write_factor_file(directory, start="2023-01-02", end="2024-07-01"):
    """A factor file covering every weekday in the synthetic range."""
    days = np.arange(np.datetime64(start), np.datetime64(end), dtype="datetime64[D]")
    days = days[np.is_busday(days)]
    rng = np.random.default_rng(3)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ff5_mom_daily.csv"
    lines = ["Date,Mkt-RF,SMB,HML,RMW,CMA,RF,Mom"]
    for day in days:
        values = rng.normal(0.0002, 0.008, 7)
        lines.append(f"{day}," + ",".join(f"{v:.8f}" for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_us_smoke_uses_the_factor_file_and_korea_does_not(tmp_path):
    """A US factor model does not price a Korean book, and the record says so."""
    data, snapshots = tmp_path / "data", tmp_path / "snapshots"
    data.mkdir()
    synthetic_market(data, "SPY", ("2023-01-02", "2023-07-04"), 400.0)
    synthetic_market(data, "QQQ", ("2023-01-02", "2023-07-04"), 300.0)
    synthetic_market(data, "005930", ("2023-01-23", "2023-01-24"), 70_000.0)
    factors = write_factor_file(tmp_path / "factors")

    argv = ["--data", str(data), "--snapshots", str(snapshots), "--factors", str(factors), "--allow-dirty"]
    assert main(argv) == 0

    smokes = {
        json.loads(p.read_text())["market"]: json.loads(p.read_text()) for p in snapshots.glob("*.smoke.json")
    }
    assert smokes["US"]["factors"]["used"] is True
    assert smokes["US"]["factor_source"] == "supplied"
    assert smokes["US"]["factors"]["columns"] == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"], (
        "the risk-free rate is not a risk factor"
    )
    assert smokes["KR"]["factors"]["used"] is False
    assert "not KR" in smokes["KR"]["factors"]["why_not"]
    assert smokes["KR"]["factor_source"] == "panel_proxy"


def test_without_a_factor_file_the_engine_says_it_used_the_proxy(tmp_path):
    data, snapshots = tmp_path / "data", tmp_path / "snapshots"
    data.mkdir()
    synthetic_market(data, "SPY", ("2023-01-02",), 400.0)
    synthetic_market(data, "QQQ", ("2023-01-02",), 300.0)

    argv = ["--data", str(data), "--snapshots", str(snapshots), "--factors", str(tmp_path / "nope.csv")]
    assert main([*argv, "--allow-dirty"]) == 0

    smoke = json.loads(next(snapshots.glob("*.smoke.json")).read_text())
    assert smoke["factors"] == {"used": False, "why_not": "no factor file was fetched"}
    assert smoke["factor_source"] == "panel_proxy"


def test_a_lagging_factor_file_trims_the_panel_and_records_the_cost(tmp_path):
    """The file stops before the panel does, which is the normal case."""
    data, snapshots = tmp_path / "data", tmp_path / "snapshots"
    data.mkdir()
    synthetic_market(data, "SPY", ("2023-01-02",), 400.0)
    synthetic_market(data, "QQQ", ("2023-01-02",), 300.0)
    factors = write_factor_file(tmp_path / "factors", end="2024-06-01")

    argv = ["--data", str(data), "--snapshots", str(snapshots), "--factors", str(factors), "--allow-dirty"]
    assert main(argv) == 0

    smoke = json.loads(next(snapshots.glob("*.smoke.json")).read_text())
    assert smoke["factors"]["used"] is True
    assert smoke["factors"]["bars_dropped_to_factor_coverage"] > 0
    assert smoke["factors"]["panel_span_used"][1] < "2024-06-01"
