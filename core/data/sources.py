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

That intersection is only sound inside one market. Two markets keep different
holidays, so intersecting across them removes real sessions from both. Panels are
therefore built one market at a time by `load_market_panels`, and combining them
is an explicit call to `align_panels` that says what each market lost. See
`core/data/markets.py`.
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
from core.data.markets import MARKETS, currencies, group_by_market


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

    Every symbol here is assumed to trade on one calendar, because the dates are
    intersected. For symbols from more than one market use `load_market_panels`,
    which groups them first.
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
    return _manifest_from(data)


def _manifest_from(data: Mapping[str, object]) -> SnapshotManifest:
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


# --- more than one market ----------------------------------------------------


@dataclass(frozen=True)
class MarketSnapshot:
    """One panel and one manifest per market, and one id tying them together.

    The per-market manifests are the ones that mean something on their own: each
    is a claim about a rectangular panel on one calendar. The top-level
    `snapshot_id` is a digest of those ids, so it names the whole read without
    pretending the markets share a date axis.

    `panels` is not serialised. Prices stay out of the repository (ADR-0007).
    """

    snapshot_id: str
    created_at: str
    panels: dict[str, PricePanel]
    manifests: dict[str, SnapshotManifest]

    @property
    def markets(self) -> tuple[str, ...]:
        return tuple(sorted(self.manifests))

    def to_json(self) -> str:
        body = {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "markets": {code: asdict(manifest) for code, manifest in self.manifests.items()},
        }
        return json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False)


def load_market_panels(
    files: Mapping[str, Path],
    universe: Mapping[str, str] | None = None,
    date_col: str = "Date",
    close_col: str = "Close",
    volume_col: str | None = "Volume",
    min_coverage: float = 0.98,
) -> MarketSnapshot:
    """Group symbols by market, then build one panel per market.

    This is the loader to reach for by default. A single-market set comes back as
    a snapshot with one entry, which costs nothing, and a mixed set comes back
    with each market on its own trading calendar instead of on the intersection
    of two.

    A symbol the universe does not declare raises (`markets.UnknownSymbol`), and
    so does a market whose files are all absent: the message names the market
    rather than leaving a caller to work out which half of the panel vanished.
    """
    if not files:
        raise ValueError("no files given; a panel of nothing cannot be validated")

    groups = group_by_market(files, universe)
    panels: dict[str, PricePanel] = {}
    manifests: dict[str, SnapshotManifest] = {}
    for code, symbols in groups.items():
        subset = {symbol: files[symbol] for symbol in symbols}
        try:
            panel, manifest = load_csv_panel(
                subset,
                date_col=date_col,
                close_col=close_col,
                volume_col=volume_col,
                min_coverage=min_coverage,
            )
        except ValueError as err:
            raise ValueError(f"market {code}: {err}") from err
        panels[code] = panel
        manifests[code] = manifest

    return MarketSnapshot(
        snapshot_id=_snapshot_id({code: manifest.snapshot_id for code, manifest in manifests.items()}),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        panels=panels,
        manifests=manifests,
    )


def write_market_manifest(snapshot: MarketSnapshot, directory: Path) -> Path:
    """Write the multi-market manifest, refusing to change one that already exists.

    Same rule as `write_manifest`, for the same reason: an id that can mean two
    things is worse than no id. The suffix keeps it from colliding with a
    single-market manifest, whose ids live in the same space but describe a
    different claim.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{snapshot.snapshot_id}.markets.json"
    if path.exists():
        existing = read_market_manifest(path)
        mine = {code: _claim(m) for code, m in snapshot.manifests.items()}
        theirs = {code: _claim(m) for code, m in existing.items()}
        if mine != theirs:
            raise ValueError(f"a different manifest already exists for snapshot {snapshot.snapshot_id}")
    path.write_text(snapshot.to_json(), encoding="utf-8")
    return path


def read_market_manifest(path: Path) -> dict[str, SnapshotManifest]:
    """The per-market manifests. The panels are not in here and cannot be."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {code: _manifest_from(body) for code, body in data["markets"].items()}


@dataclass(frozen=True)
class Alignment:
    """One panel across markets, and what building it cost."""

    panel: PricePanel
    dates_kept: int
    dropped_by_market: dict[str, int]
    currencies: dict[str, str]
    notes: tuple[str, ...]


def align_panels(
    panels: Mapping[str, PricePanel],
    allow_mixed_currency: bool = False,
) -> Alignment:
    """Put several market panels on one date axis, on the dates they all traded.

    The intersection is the only option that does not invent a price, and it is
    lossy by construction: a US-Korea pair loses both countries' holidays, around
    a fortnight of sessions a year. `dropped_by_market` is what each market gave
    up, so the cost is in the record rather than in a footnote.

    Symbols come back qualified as `US:SPY`, always, including for a single
    market. Downstream code that sizes a position has to know which currency it
    is in, and a name that only sometimes carries the market is a name nobody
    parses.

    Mixed currencies raise unless the caller says otherwise. Returns are
    currency-invariant, so a signal built on local closes is fine; a P&L, a gross
    exposure and a participation limit are not, and they are what the risk table
    checks. Passing `allow_mixed_currency=True` is a research decision, and it is
    recorded in `notes` along with the fact that FX has not been applied.
    """
    if not panels:
        raise ValueError("no panels given; there is nothing to align")

    codes = sorted(panels)
    ccy = currencies(codes)
    distinct = sorted(set(ccy.values()))
    notes: list[str] = []
    if len(distinct) > 1 and not allow_mixed_currency:
        raise ValueError(
            f"panels span {' and '.join(distinct)}; local-currency closes do not add up into one "
            "P&L. Pass allow_mixed_currency=True for a signal-only panel and convert before sizing."
        )
    if len(distinct) > 1:
        notes.append(
            f"closes are in local currency ({', '.join(f'{c}={ccy[c]}' for c in codes)}) and no FX "
            "conversion has been applied; returns are comparable, money is not"
        )

    common: set[str] | None = None
    for code in codes:
        seen = {str(date) for date in panels[code].dates}
        common = seen if common is None else (common & seen)
    assert common is not None  # `panels` is non-empty
    if not common:
        spans = ", ".join(f"{code} {panels[code].dates[0]}..{panels[code].dates[-1]}" for code in codes)
        raise ValueError(f"the markets share no trading day: {spans}")

    dates = sorted(common)
    dropped = {code: int(panels[code].dates.shape[0] - len(dates)) for code in codes}
    for code in codes:
        own = panels[code].dates.shape[0]
        if dropped[code]:
            notes.append(f"{code} lost {dropped[code]} of its {own} sessions to the intersection")

    symbols: list[str] = []
    close_blocks: list[np.ndarray] = []
    volume_blocks: list[np.ndarray] = []
    missing_volume = [code for code in codes if panels[code].dollar_volume is None]
    for code in codes:
        panel = panels[code]
        position = {str(date): index for index, date in enumerate(panel.dates)}
        rows = [position[date] for date in dates]
        symbols.extend(f"{code}:{symbol}" for symbol in panel.symbols)
        close_blocks.append(panel.close[rows, :])
        if not missing_volume and len(distinct) == 1:
            volume_blocks.append(panel.dollar_volume[rows, :])

    dollar_volume: np.ndarray | None = None
    if len(distinct) > 1:
        notes.append(
            "dollar volume withheld: volumes in different currencies are not one number, and the "
            "participation limit is about money"
        )
    elif missing_volume:
        notes.append(f"dollar volume withheld: no volume for {', '.join(missing_volume)}")
    else:
        dollar_volume = np.concatenate(volume_blocks, axis=1)

    panel = PricePanel(
        dates=np.array(dates, dtype="datetime64[D]"),
        symbols=tuple(symbols),
        close=np.concatenate(close_blocks, axis=1),
        dollar_volume=dollar_volume,
    )
    return Alignment(
        panel=panel,
        dates_kept=len(dates),
        dropped_by_market=dropped,
        currencies=ccy,
        notes=tuple(notes),
    )


def market_of(qualified: str) -> str:
    """The market code from a symbol `align_panels` qualified, e.g. `US:SPY` -> `US`."""
    code, _, rest = qualified.partition(":")
    if not rest or code not in MARKETS:
        raise ValueError(f"{qualified!r} is not a market-qualified symbol")
    return code
