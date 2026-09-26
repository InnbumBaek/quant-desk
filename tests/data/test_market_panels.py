"""One panel per market, and what it costs to put two of them on one date axis."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from core.backtest.engine import PricePanel
from core.data.markets import UnknownSymbol
from core.data.sources import (
    SnapshotManifest,
    align_panels,
    load_csv_panel,
    load_market_panels,
    market_of,
    read_market_manifest,
    write_market_manifest,
)

HEADER = "Date,Close,Volume\n"

# The two markets keep different holidays. The US was shut on 2024-01-15 (MLK Day)
# while Korea traded, and Korea was shut on 2024-02-12 (the Seollal substitute
# holiday) while the US traded. Those two days are the whole problem in miniature:
# intersecting the two calendars throws away a real session in each market.
US_DATES = ["2024-01-12", "2024-01-16", "2024-01-17", "2024-02-12"]
KR_DATES = ["2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17"]
BOTH = ["2024-01-12", "2024-01-16", "2024-01-17"]


def write_csv(path: Path, dates: list[str], start: float) -> Path:
    rows = "".join(f"{d},{start + i},{1_000 + i}\n" for i, d in enumerate(dates))
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def mixed_files(tmp_path: Path) -> dict[str, Path]:
    return {
        "SPY": write_csv(tmp_path / "spy.csv", US_DATES, 100.0),
        "QQQ": write_csv(tmp_path / "qqq.csv", US_DATES, 200.0),
        "005930": write_csv(tmp_path / "005930.csv", KR_DATES, 70_000.0),
    }


def panel_of(dates: list[str], symbols: tuple[str, ...], volume: bool = True) -> PricePanel:
    close = np.arange(len(dates) * len(symbols), dtype=float).reshape(len(dates), len(symbols)) + 1.0
    return PricePanel(
        dates=np.array(dates, dtype="datetime64[D]"),
        symbols=symbols,
        close=close,
        dollar_volume=close * 1_000.0 if volume else None,
    )


# --- the regression this exists to prevent -----------------------------------


def test_one_mixed_panel_loses_a_real_session_in_each_market(tmp_path):
    """The old path, shown failing: the intersection is neither market's calendar."""
    with pytest.raises(ValueError, match="common to every symbol"):
        load_csv_panel(mixed_files(tmp_path))

    # Lowering the coverage bar is how this becomes silent rather than loud.
    panel, manifest = load_csv_panel(mixed_files(tmp_path), min_coverage=0.0)
    assert [str(d) for d in panel.dates] == BOTH
    assert manifest.dates_dropped == 2


def test_per_market_panels_keep_every_session_each_market_traded(tmp_path):
    snapshot = load_market_panels(mixed_files(tmp_path))

    assert snapshot.markets == ("KR", "US")
    assert [str(d) for d in snapshot.panels["US"].dates] == US_DATES
    assert [str(d) for d in snapshot.panels["KR"].dates] == KR_DATES
    assert snapshot.panels["US"].symbols == ("QQQ", "SPY")
    assert snapshot.panels["KR"].symbols == ("005930",)
    assert snapshot.manifests["US"].dates_dropped == 0
    assert snapshot.manifests["KR"].dates_dropped == 0


# --- the snapshot id ---------------------------------------------------------


def test_the_same_bytes_give_the_same_snapshot_id(tmp_path):
    first = load_market_panels(mixed_files(tmp_path))
    second = load_market_panels(mixed_files(tmp_path))
    assert first.snapshot_id == second.snapshot_id


def test_a_changed_byte_changes_the_snapshot_id(tmp_path):
    first = load_market_panels(mixed_files(tmp_path))
    files = mixed_files(tmp_path)
    write_csv(files["005930"], KR_DATES, 70_001.0)
    assert load_market_panels(files).snapshot_id != first.snapshot_id


def test_the_snapshot_id_is_not_any_one_markets_id(tmp_path):
    """It names a different claim: several panels, no shared date axis."""
    snapshot = load_market_panels(mixed_files(tmp_path))
    per_market = {m.snapshot_id for m in snapshot.manifests.values()}
    assert snapshot.snapshot_id not in per_market


def test_a_single_market_set_is_a_snapshot_of_one(tmp_path):
    files = {"SPY": write_csv(tmp_path / "spy.csv", US_DATES, 100.0)}
    snapshot = load_market_panels(files)
    assert snapshot.markets == ("US",)


# --- refusals ----------------------------------------------------------------


def test_an_undeclared_symbol_stops_the_load(tmp_path):
    files = mixed_files(tmp_path)
    files["NVDA"] = write_csv(tmp_path / "nvda.csv", US_DATES, 50.0)
    with pytest.raises(UnknownSymbol):
        load_market_panels(files)


def test_a_market_with_no_files_names_itself_in_the_error(tmp_path):
    files = mixed_files(tmp_path)
    files["005930"] = tmp_path / "gone.csv"
    with pytest.raises(ValueError, match="market KR:"):
        load_market_panels(files)


def test_no_files_at_all_is_refused(tmp_path):
    with pytest.raises(ValueError, match="panel of nothing"):
        load_market_panels({})


# --- the written manifest ----------------------------------------------------


def test_the_market_manifest_round_trips(tmp_path):
    snapshot = load_market_panels(mixed_files(tmp_path))
    path = write_market_manifest(snapshot, tmp_path / "snapshots")

    read_back = read_market_manifest(path)
    assert sorted(read_back) == ["KR", "US"]
    assert read_back["US"].symbols == ("QQQ", "SPY")
    assert read_back["KR"].rows == len(KR_DATES)
    assert path.name.endswith(".markets.json")


def test_the_written_manifest_carries_no_prices(tmp_path):
    """Prices stay out of a public repository (ADR-0007); manifests do not."""
    snapshot = load_market_panels(mixed_files(tmp_path))
    path = write_market_manifest(snapshot, tmp_path / "snapshots")
    body = json.loads(path.read_text(encoding="utf-8"))

    assert sorted(body) == ["created_at", "markets", "snapshot_id"]
    assert sorted(body["markets"]["US"]) == sorted(f.name for f in fields(SnapshotManifest))


def test_rewriting_the_same_id_with_a_different_claim_is_refused(tmp_path):
    snapshot = load_market_panels(mixed_files(tmp_path))
    directory = tmp_path / "snapshots"
    write_market_manifest(snapshot, directory)

    tampered = replace(snapshot.manifests["US"], rows=999)
    forged = replace(snapshot, manifests={**snapshot.manifests, "US": tampered})
    with pytest.raises(ValueError, match="a different manifest already exists"):
        write_market_manifest(forged, directory)


def test_writing_the_same_claim_twice_is_fine(tmp_path):
    """Re-reading the same files later is the same claim; only the clock moved."""
    directory = tmp_path / "snapshots"
    write_market_manifest(load_market_panels(mixed_files(tmp_path)), directory)
    write_market_manifest(load_market_panels(mixed_files(tmp_path)), directory)


# --- aligning ----------------------------------------------------------------


def test_mixed_currencies_are_refused_by_default():
    panels = {"US": panel_of(US_DATES, ("SPY",)), "KR": panel_of(KR_DATES, ("005930",))}
    with pytest.raises(ValueError, match="KRW and USD"):
        align_panels(panels)


def test_an_allowed_mixed_panel_records_that_fx_is_not_applied():
    panels = {"US": panel_of(US_DATES, ("SPY",)), "KR": panel_of(KR_DATES, ("005930",))}
    aligned = align_panels(panels, allow_mixed_currency=True)

    assert [str(d) for d in aligned.panel.dates] == BOTH
    assert aligned.dates_kept == 3
    assert aligned.dropped_by_market == {"KR": 1, "US": 1}
    assert aligned.currencies == {"US": "USD", "KR": "KRW"}
    assert any("no FX conversion" in note for note in aligned.notes)
    assert any("KR lost 1 of its 4 sessions" in note for note in aligned.notes)


def test_dollar_volume_is_withheld_across_currencies():
    panels = {"US": panel_of(US_DATES, ("SPY",)), "KR": panel_of(KR_DATES, ("005930",))}
    aligned = align_panels(panels, allow_mixed_currency=True)
    assert aligned.panel.dollar_volume is None
    assert any("dollar volume withheld" in note for note in aligned.notes)


def test_symbols_carry_their_market_even_for_one_market():
    aligned = align_panels({"US": panel_of(US_DATES, ("QQQ", "SPY"))})
    assert aligned.panel.symbols == ("US:QQQ", "US:SPY")
    assert market_of("US:QQQ") == "US"


def test_a_single_currency_panel_keeps_its_dollar_volume():
    aligned = align_panels({"US": panel_of(US_DATES, ("SPY",))})
    assert aligned.panel.dollar_volume is not None
    assert aligned.dropped_by_market == {"US": 0}
    assert aligned.notes == ()


def test_a_missing_volume_withholds_it_and_says_which_market():
    aligned = align_panels({"US": panel_of(US_DATES, ("SPY",), volume=False)})
    assert aligned.panel.dollar_volume is None
    assert any("no volume for US" in note for note in aligned.notes)


def test_the_aligned_rows_are_the_rows_of_the_kept_dates():
    us = panel_of(US_DATES, ("SPY",))
    kr = panel_of(KR_DATES, ("005930",))
    aligned = align_panels({"US": us, "KR": kr}, allow_mixed_currency=True)

    # BOTH[1] is 2024-01-16: row 1 of the US panel, row 2 of the Korean one, and
    # the columns are ordered by market code, so Korea comes first.
    assert aligned.panel.symbols == ("KR:005930", "US:SPY")
    assert aligned.panel.close[1].tolist() == [kr.close[2, 0], us.close[1, 0]]


def test_markets_that_share_no_session_are_refused():
    us = panel_of(["2024-01-12", "2024-01-16", "2024-01-17"], ("SPY",))
    kr = panel_of(["2023-01-12", "2023-01-16", "2023-01-17"], ("005930",))
    with pytest.raises(ValueError, match="share no trading day"):
        align_panels({"US": us, "KR": kr}, allow_mixed_currency=True)


def test_aligning_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing to align"):
        align_panels({})


def test_an_unknown_market_code_is_refused():
    with pytest.raises(ValueError, match="unknown market"):
        align_panels({"JP": panel_of(US_DATES, ("7203",))})


def test_an_unqualified_symbol_is_not_a_market_symbol():
    with pytest.raises(ValueError, match="not a market-qualified symbol"):
        market_of("SPY")
