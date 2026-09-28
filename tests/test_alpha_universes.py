"""The universe a hypothesis was judged on, checked rather than described.

`universe:` prose was required from the first declaration and no code read it, so
the instruments were the one part of a pre-registration that the day's fetch
decided. These tests pin the two halves of the fix: the panel is restricted to
the declared list, and the grandfathered list cannot quietly grow (ADR-0041).
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from core.alphas import universes
from core.alphas.implementations import ImplementationError
from core.backtest import prereg
from core.backtest.engine import PricePanel
from scripts import submit_alpha


def panel(symbols=("SPY", "QQQ", "IWM", "TLT", "GLD"), n_rows: int = 300) -> PricePanel:
    rng = np.random.default_rng(3)
    steps = rng.normal(0.0004, 0.01, size=(n_rows, len(symbols)))
    return PricePanel(
        dates=np.datetime64("2019-01-01") + np.arange(n_rows),
        symbols=tuple(symbols),
        close=100.0 * np.exp(np.cumsum(steps, axis=0)),
        dollar_volume=np.tile(np.arange(1, len(symbols) + 1) * 1e8, (n_rows, 1)),
    )


# --- restricting a panel -------------------------------------------------------


def test_the_panel_keeps_the_declared_symbols_in_the_declared_order():
    restricted = panel().select(["TLT", "SPY"])
    assert restricted.symbols == ("TLT", "SPY")
    assert np.array_equal(restricted.close[:, 0], panel().close[:, 3])
    assert np.array_equal(restricted.dates, panel().dates)


def test_the_dollar_volume_follows_the_same_columns():
    """A capacity number computed against another symbol's volume is not wrong by
    a little."""
    restricted = panel().select(["GLD", "QQQ"])
    assert restricted.dollar_volume is not None
    assert restricted.dollar_volume[0, 0] == pytest.approx(5e8)
    assert restricted.dollar_volume[0, 1] == pytest.approx(2e8)


def test_a_missing_declared_symbol_is_a_refusal_not_a_smaller_universe():
    """Four of five declared names is a different universe, not a partial one."""
    with pytest.raises(ValueError, match="does not carry"):
        panel().select(["SPY", "EEM"])


def test_an_empty_universe_is_not_a_universe():
    with pytest.raises(ValueError, match="no symbols"):
        panel().select([])


def test_a_repeated_symbol_collapses_rather_than_duplicating_a_column():
    assert panel().select(["SPY", "SPY", "TLT"]).symbols == ("SPY", "TLT")


def test_a_panel_with_no_volume_stays_without_volume():
    bare = PricePanel(dates=panel().dates, symbols=panel().symbols, close=panel().close)
    assert bare.select(["SPY"]).dollar_volume is None


# --- where the universe comes from ---------------------------------------------


@pytest.mark.parametrize("alpha_id", sorted(prereg.declared_ids()))
def test_every_declared_alpha_has_a_universe(alpha_id):
    symbols, source = universes.for_alpha(alpha_id)
    assert symbols and source in {"declaration", "legacy"}


def test_the_grandfathered_list_is_exactly_the_alphas_declared_before_the_field():
    """Bidirectional, like `NO_READER_YET`: an exemption list that can grow
    silently is how an unenforced rule hides. A new declaration carries the field
    in the declaration, where git holds it unmodified."""
    assert tuple(sorted(universes.legacy_universes())) == universes.LEGACY
    for alpha_id in universes.LEGACY:
        assert prereg.declared_universe(alpha_id) is None, (
            f"{alpha_id} now declares its own universe; drop it from the wiring file"
        )


def test_a_declaration_that_names_its_universe_is_preferred_over_the_wiring_file(tmp_path):
    alphas = tmp_path / "alphas"
    alphas.mkdir()
    (alphas / "new-001.yaml").write_text(
        yaml.safe_dump(
            {"id": "new-001", "hypothesis": {"universe_symbols": ["SPY", "EEM"]}},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    wiring = tmp_path / "_implementations.yaml"
    wiring.write_text(
        yaml.safe_dump({"legacy_universes": {"new-001": ["TLT"]}}),
        encoding="utf-8",
    )
    assert universes.for_alpha("new-001", alphas, wiring) == (("SPY", "EEM"), "declaration")


def test_an_alpha_with_no_universe_anywhere_cannot_be_judged(tmp_path):
    alphas = tmp_path / "alphas"
    alphas.mkdir()
    (alphas / "new-001.yaml").write_text("id: new-001\n", encoding="utf-8")
    wiring = tmp_path / "_implementations.yaml"
    wiring.write_text("legacy_universes: {}\n", encoding="utf-8")
    with pytest.raises(universes.UniverseError, match="whatever the fetch held"):
        universes.for_alpha("new-001", alphas, wiring)


@pytest.mark.parametrize("value", [[], "SPY", ["SPY", "SPY"], ["SPY", " "]])
def test_a_malformed_declared_universe_raises_rather_than_reading_as_absent(tmp_path, value):
    alphas = tmp_path / "alphas"
    alphas.mkdir()
    (alphas / "new-001.yaml").write_text(
        yaml.safe_dump({"id": "new-001", "hypothesis": {"universe_symbols": value}}),
        encoding="utf-8",
    )
    with pytest.raises(prereg.PreregistrationError):
        prereg.declared_universe("new-001", directory=alphas)


def test_a_wiring_file_with_a_malformed_entry_is_named(tmp_path):
    wiring = tmp_path / "_implementations.yaml"
    wiring.write_text(yaml.safe_dump({"legacy_universes": {"a-001": []}}), encoding="utf-8")
    with pytest.raises(ImplementationError, match="names no universe"):
        universes.legacy_universes(wiring)


# --- the submission refuses rather than substituting ---------------------------


def test_a_submission_on_a_snapshot_missing_a_declared_name_is_refused():
    """The loophole this closes: a fetch that no longer carries GLD would have
    judged the same declaration on four instruments."""
    from tests.test_submit_alpha import declaration, pin

    with pytest.raises(submit_alpha.NotSubmittable, match="declared universe is not in this snapshot"):
        submit_alpha.submit(
            alpha_id="tsmom-001",
            strategy_name="ts_momentum",
            panel=panel(("SPY", "QQQ", "IWM", "TLT")),
            pin=pin(),
            declaration=declaration(),
            chosen={"lookback": 60.0, "gross": 0.8},
        )


def test_a_wider_snapshot_is_narrowed_to_the_declaration_rather_than_used():
    """The case that made this necessary: the fetch widens, and the old
    declaration must still be judged on its own five."""
    from tests.test_submit_alpha import declaration, pin

    record = submit_alpha.submit(
        alpha_id="tsmom-001",
        strategy_name="ts_momentum",
        panel=panel(("SPY", "QQQ", "IWM", "TLT", "GLD", "EEM", "XLF"), n_rows=700),
        pin=pin(),
        declaration=declaration(),
        chosen={"lookback": 60.0, "gross": 0.8},
    )
    assert record["universe"]["panel_symbols"] == ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    assert record["universe"]["source"] == "legacy"


def test_the_batch_table_names_each_universe_and_its_window():
    """Two alphas on different instrument sets have different windows, and a table
    that hid that would invite a comparison nobody should make."""
    import glob
    import json

    files = sorted(glob.glob("registry/submissions/*.tsmom-001.json"))
    rows = [json.load(open(files[-1], encoding="utf-8"))]
    rows[0]["universe"] = {"declared": ["SPY", "TLT"], "source": "legacy", "panel_symbols": []}
    table = submit_alpha.batch_markdown(rows)
    assert "Universes in this run:" in table
    assert "SPY, TLT" in table
    assert f"{rows[0]['is_rows']} in-sample" in table
