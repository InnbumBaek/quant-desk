"""Building a `PricePanel` from files on disk, and pinning the data that was used.

This environment's network policy denies outbound access to market-data hosts, so
there is no live fetch here and inventing one would be a fiction. What a research
run actually needs is narrower anyway: a rectangular panel, and a name for the
exact bytes it came from. A backtest whose data cannot be identified cannot be
reproduced, and an unreproducible Sharpe is a rumour.

So this module does two things:

- reads per-symbol daily CSV files (the shape every vendor exports) and returns a
  validated `PricePanel`, and
- returns a `SnapshotManifest` whose `snapshot_id` is a digest of the file
  contents. That id is the data leg of the reproducibility pin in `core/repro.py`
  (git SHA + data snapshot id + seed).

Two decisions worth stating. **Dates are intersected, not filled.** A symbol
missing a day is a hole, and forward-filling a hole invents a price that never
traded; the intersection is what every symbol really has, and the number of dates
dropped is recorded. **A large hole is refused, not reported.** Below
`min_coverage` the panel is not returned at all, because a panel missing a
meaningful share of its history answers a different question than the one asked.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core.backtest.engine import PricePanel


@dataclass(frozen=True)
class SnapshotManifest:
    """What was read, from which bytes. Written next to the snapshot, never regenerated."""

    snapshot_id: str
    created_at: str
    symbols: tuple[str, ...]
    requested: tuple[str, ...]
    missing: tuple[str, ...]
    rows: int
    first_date: str
    last_date: str
    dates_dropped: int
    has_volume: bool
    file_digests: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=False)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_id(digests: Mapping[str, str]) -> str:
    """A digest of the digests, so the same bytes always name the same snapshot."""
    joined = "\n".join(f"{symbol}:{digest}" for symbol, digest in sorted(digests.items()))
    return hashlib.sha256(joined.encode()).hexdigest()[:32]


def _read_series(
    path: Path,
    date_col: str,
    close_col: str,
    volume_col: str | None,
) -> dict[str, tuple[float, float | None]]:
    rows: dict[str, tuple[float, float | None]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or date_col not in reader.fieldnames:
            raise ValueError(f"{path.name} has no {date_col!r} column; found {reader.fieldnames}")
        if close_col not in reader.fieldnames:
            raise ValueError(f"{path.name} has no {close_col!r} column")
        for line in reader:
            date = (line[date_col] or "").strip()
            if not date:
                continue
            close = float(line[close_col])
            volume: float | None = None
            if volume_col and line.get(volume_col):
                volume = float(line[volume_col])
            if date in rows:
                raise ValueError(f"{path.name} repeats the date {date}")
            rows[date] = (close, volume)
    if not rows:
        raise ValueError(f"{path.name} contains no rows")
    return rows


def load_csv_panel(
    files: Mapping[str, Path],
    date_col: str = "Date",
    close_col: str = "Close",
    volume_col: str | None = "Volume",
    min_coverage: float = 0.98,
) -> tuple[PricePanel, SnapshotManifest]:
    """Read one CSV per symbol and return the panel plus the manifest that names it.

    `files` maps symbol to path. A path that does not exist is recorded as missing
    rather than guessed at: a survivorship hole should be visible in the manifest,
    which is the thing a reader will still have in six months.
    """
    if not files:
        raise ValueError("no files given; a panel of nothing cannot be validated")

    requested = tuple(files)
    present = {symbol: path for symbol, path in files.items() if Path(path).exists()}
    missing = tuple(symbol for symbol in requested if symbol not in present)
    if not present:
        raise ValueError(f"none of the {len(requested)} requested files exist")

    series = {
        symbol: _read_series(Path(path), date_col, close_col, volume_col) for symbol, path in present.items()
    }

    all_dates: set[str] = set()
    common: set[str] | None = None
    for rows in series.values():
        all_dates |= rows.keys()
        common = set(rows) if common is None else (common & rows.keys())
    assert common is not None  # `present` is non-empty, so the loop ran

    coverage = len(common) / len(all_dates)
    if coverage < min_coverage:
        raise ValueError(
            f"only {len(common)} of {len(all_dates)} dates are common to every symbol "
            f"({coverage:.1%} < {min_coverage:.1%}); fix the source rather than the panel"
        )

    dates = sorted(common)
    symbols = tuple(sorted(present))
    close = np.array([[series[s][d][0] for s in symbols] for d in dates], dtype=float)

    has_volume = all(series[s][d][1] is not None for s in symbols for d in dates)
    dollar_volume = None
    if has_volume:
        volume = np.array([[series[s][d][1] for s in symbols] for d in dates], dtype=float)
        # Dollar volume, not share volume: a 3% participation limit is about money.
        dollar_volume = np.maximum(close * volume, 1.0)

    panel = PricePanel(
        dates=np.array(dates, dtype="datetime64[D]"),
        symbols=symbols,
        close=close,
        dollar_volume=dollar_volume,
    )

    digests = {symbol: file_digest(Path(path)) for symbol, path in present.items()}
    manifest = SnapshotManifest(
        snapshot_id=_snapshot_id(digests),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        symbols=symbols,
        requested=requested,
        missing=missing,
        rows=len(dates),
        first_date=dates[0],
        last_date=dates[-1],
        dates_dropped=len(all_dates) - len(common),
        has_volume=has_volume,
        file_digests=digests,
    )
    return panel, manifest


def _claim(manifest: SnapshotManifest) -> dict[str, object]:
    """The part of a manifest that is a claim about bytes, i.e. everything but the clock."""
    body = asdict(manifest)
    body.pop("created_at")
    return body


def write_manifest(manifest: SnapshotManifest, directory: Path) -> Path:
    """Write the manifest under its own snapshot id, and refuse to change one that exists.

    A manifest is a claim about bytes. Overwriting it silently would let the same
    snapshot id mean two different things, which is worse than having no id. The
    comparison ignores `created_at`: re-reading the same files later is the same
    claim, and the read time is not part of it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{manifest.snapshot_id}.json"
    if path.exists() and _claim(read_manifest(path)) != _claim(manifest):
        raise ValueError(f"a different manifest already exists for snapshot {manifest.snapshot_id}")
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def read_manifest(path: Path) -> SnapshotManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SnapshotManifest(
        snapshot_id=data["snapshot_id"],
        created_at=data["created_at"],
        symbols=tuple(data["symbols"]),
        requested=tuple(data["requested"]),
        missing=tuple(data["missing"]),
        rows=int(data["rows"]),
        first_date=data["first_date"],
        last_date=data["last_date"],
        dates_dropped=int(data["dates_dropped"]),
        has_volume=bool(data["has_volume"]),
        file_digests=dict(data["file_digests"]),
    )
