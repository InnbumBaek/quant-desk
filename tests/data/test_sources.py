"""Reading a panel from files, and naming the bytes it came from."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.data.sources import (
    load_csv_panel,
    read_manifest,
    write_manifest,
)

HEADER = "Date,Open,High,Low,Close,Volume\n"


def write_csv(path: Path, rows: list[tuple[str, float, float]]) -> Path:
    body = "".join(f"{d},{c},{c},{c},{c},{v}\n" for d, c, v in rows)
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def three_days(tmp_path: Path) -> dict[str, Path]:
    a = write_csv(
        tmp_path / "aaa.csv",
        [("2024-01-02", 100.0, 1_000.0), ("2024-01-03", 101.0, 1_100.0), ("2024-01-04", 102.0, 900.0)],
    )
    b = write_csv(
        tmp_path / "bbb.csv",
        [("2024-01-02", 50.0, 2_000.0), ("2024-01-03", 49.0, 2_100.0), ("2024-01-04", 51.0, 1_900.0)],
    )
    return {"AAA": a, "BBB": b}


def test_panel_is_built_in_symbol_and_date_order(tmp_path):
    panel, manifest = load_csv_panel(three_days(tmp_path))

    assert panel.symbols == ("AAA", "BBB")
    assert panel.close.shape == (3, 2)
    assert panel.close[0].tolist() == [100.0, 50.0]
    assert str(panel.dates[0]) == "2024-01-02"
    assert manifest.rows == 3
    assert manifest.first_date == "2024-01-02"
    assert manifest.last_date == "2024-01-04"
    assert manifest.missing == ()


def test_dollar_volume_is_price_times_shares(tmp_path):
    panel, manifest = load_csv_panel(three_days(tmp_path))
    assert manifest.has_volume
    assert panel.dollar_volume[0].tolist() == [100.0 * 1_000.0, 50.0 * 2_000.0]


def test_a_missing_volume_column_gives_a_panel_without_volume(tmp_path):
    path = tmp_path / "aaa.csv"
    path.write_text("Date,Close\n2024-01-02,100\n2024-01-03,101\n2024-01-04,102\n", encoding="utf-8")
    panel, manifest = load_csv_panel({"AAA": path})
    assert panel.dollar_volume is None
    assert not manifest.has_volume


def test_dates_are_intersected_and_the_drop_is_recorded(tmp_path):
    files = three_days(tmp_path)
    write_csv(
        files["BBB"],
        [
            ("2024-01-02", 50.0, 2_000.0),
            ("2024-01-03", 49.0, 2_100.0),
            ("2024-01-04", 51.0, 1_900.0),
            ("2024-01-05", 52.0, 1_800.0),
        ],
    )
    panel, manifest = load_csv_panel(files, min_coverage=0.5)
    assert panel.close.shape == (3, 2), "the date only one symbol has is dropped, not filled"
    assert manifest.dates_dropped == 1


def test_a_hole_too_big_to_ignore_is_refused(tmp_path):
    files = three_days(tmp_path)
    write_csv(files["BBB"], [("2024-01-02", 50.0, 2_000.0)])
    with pytest.raises(ValueError, match="common to every symbol"):
        load_csv_panel(files)


def test_a_missing_file_is_recorded_not_guessed(tmp_path):
    files = three_days(tmp_path)
    files["CCC"] = tmp_path / "ccc.csv"
    panel, manifest = load_csv_panel(files)
    assert manifest.missing == ("CCC",)
    assert manifest.symbols == ("AAA", "BBB")
    assert "CCC" in manifest.requested
    assert panel.close.shape[1] == 2


def test_no_readable_file_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="none of the"):
        load_csv_panel({"AAA": tmp_path / "nope.csv"})


def test_a_repeated_date_is_an_error(tmp_path):
    path = write_csv(
        tmp_path / "aaa.csv",
        [("2024-01-02", 100.0, 1.0), ("2024-01-02", 101.0, 1.0), ("2024-01-03", 102.0, 1.0)],
    )
    with pytest.raises(ValueError, match="repeats the date"):
        load_csv_panel({"AAA": path})


def test_a_missing_close_column_names_the_column(tmp_path):
    path = tmp_path / "aaa.csv"
    path.write_text("Date,Adj\n2024-01-02,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'Close'"):
        load_csv_panel({"AAA": path})


# --- the snapshot id --------------------------------------------------------


def test_the_same_bytes_give_the_same_snapshot_id(tmp_path):
    first = load_csv_panel(three_days(tmp_path))[1]
    second = load_csv_panel(three_days(tmp_path))[1]
    assert first.snapshot_id == second.snapshot_id


def test_one_changed_price_changes_the_snapshot_id(tmp_path):
    files = three_days(tmp_path)
    before = load_csv_panel(files)[1].snapshot_id
    write_csv(
        files["AAA"],
        [("2024-01-02", 100.0, 1_000.0), ("2024-01-03", 101.5, 1_100.0), ("2024-01-04", 102.0, 900.0)],
    )
    after = load_csv_panel(files)[1].snapshot_id
    assert before != after


def test_a_manifest_round_trips(tmp_path):
    _, manifest = load_csv_panel(three_days(tmp_path))
    path = write_manifest(manifest, tmp_path / "snapshots")
    assert path.name == f"{manifest.snapshot_id}.json"
    assert read_manifest(path) == manifest


def test_rewriting_a_manifest_with_different_content_is_refused(tmp_path):
    files = three_days(tmp_path)
    _, manifest = load_csv_panel(files)
    directory = tmp_path / "snapshots"
    write_manifest(manifest, directory)

    tampered = type(manifest)(**{**manifest.__dict__, "rows": 999})
    with pytest.raises(ValueError, match="a different manifest already exists"):
        write_manifest(tampered, directory)


def test_writing_the_same_manifest_twice_is_fine(tmp_path):
    _, manifest = load_csv_panel(three_days(tmp_path))
    directory = tmp_path / "snapshots"
    assert write_manifest(manifest, directory) == write_manifest(manifest, directory)


def test_a_second_read_of_the_same_files_is_the_same_claim(tmp_path):
    """Only `created_at` differs, and the read time is not part of the claim."""
    files = three_days(tmp_path)
    directory = tmp_path / "snapshots"
    first = load_csv_panel(files)[1]
    write_manifest(first, directory)

    second = load_csv_panel(files)[1]
    assert second.snapshot_id == first.snapshot_id
    write_manifest(second, directory)  # must not raise


def test_the_panel_the_loader_returns_is_the_validated_one(tmp_path):
    """Validation lives in PricePanel, so a bad file fails there rather than silently."""
    path = write_csv(tmp_path / "aaa.csv", [("2024-01-02", 100.0, 1.0), ("2024-01-03", 0.0, 1.0)])
    with pytest.raises(ValueError):
        load_csv_panel({"AAA": path})


def test_close_values_are_floats_not_strings(tmp_path):
    panel, _ = load_csv_panel(three_days(tmp_path))
    assert panel.close.dtype == np.dtype(float)
