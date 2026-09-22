"""The snapshot script's job is to leave a record that is true about the fetch.

The manifest already says *which* bytes a result came from, by hash. These tests
cover the part that was missing: *where* those bytes came from. The sidecars that
`fetch_prices` writes live in git-ignored `data/` and do not survive the runner,
so the source name has to be copied into the committed run file or the
reproduction claim in ADR-0007 is only half true.
"""

from __future__ import annotations

import json

from scripts.data_snapshot import discover, read_provenance


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
