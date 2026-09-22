"""Turn whatever is in `data/` into a snapshot manifest and a gate smoke run.

Run after `scripts/fetch_prices.py`. What this produces, and what it deliberately
does not:

- `registry/snapshots/<snapshot_id>.json` -- one manifest per market: file
  hashes, symbols, date range, dropped dates. No prices. This is the data leg of
  the reproducibility pin and it is safe to commit in a public repository. When
  the fetch spans more than one market, a `<id>.markets.json` index ties the
  per-market ids together; the markets are loaded separately because their
  trading calendars differ (ADR-0013).
- `registry/snapshots/<snapshot_id>.smoke.json` -- the gate verdicts for one
  throwaway strategy, with the run id that pins (code, data, seed), and the
  vendor each symbol came from. Derived numbers and provenance, not the vendor's
  data. Provenance lives here rather than in the manifest because it is a fact
  about *this fetch*: the same bytes can arrive from either source, and a
  manifest id must keep meaning exactly one thing (ADR-0006).
- Nothing else. The CSVs stay in git-ignored `data/` and are never uploaded, so
  the vendor's bytes do not leave the runner (ADR-0007).

The strategy here is a plain cross-sectional momentum, and it is **not** an alpha
submission: it exists to prove the path from bytes to verdict actually runs. It
will normally be rejected, and that is the expected outcome of a smoke test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from core.backtest import gates
from core.backtest.engine import BacktestConfig, PricePanel, run
from core.data.factors import FRENCH_MARKET, FactorPanel, load_factors, trim_panel_to_factors
from core.data.markets import sessions_per_year
from core.data.sources import load_market_panels, write_manifest, write_market_manifest
from core.repro import ReproPin, pin_current


def momentum(close: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Cross-sectional momentum, dollar-neutral, point-in-time by construction."""
    lookback = int(params["lookback"])
    gross = float(params["gross"])
    weights = np.zeros_like(close)
    for t in range(lookback, close.shape[0]):
        past = close[t] / close[t - lookback] - 1.0
        centred = past - past.mean()
        scale = float(np.sum(np.abs(centred)))
        if scale > 0:
            weights[t] = centred / scale * gross
    return weights


def json_safe(value: object, path: str = "", found: list[str] | None = None) -> object:
    """Replace non-finite floats so the committed record is standard JSON.

    `json.dump` writes `Infinity` and `NaN`, which Python reads back and almost
    nothing else does: a committed artefact that only one parser accepts is not a
    record. A gate metric can legitimately be infinite -- G4's drawdown/return
    ratio is, whenever the return is not positive -- so the value becomes `null`
    and the key is listed in `non_finite_metrics`. Dropping to `null` alone would
    lose the difference between "infinite" and "never measured".
    """
    if found is None:
        found = []
    if isinstance(value, float) and not math.isfinite(value):
        found.append(f"{path}={value}")
        return None
    if isinstance(value, dict):
        return {
            key: json_safe(item, f"{path}.{key}" if path else str(key), found) for key, item in value.items()
        }
    if isinstance(value, list):
        return [json_safe(item, f"{path}[{i}]", found) for i, item in enumerate(value)]
    return value


def read_provenance(directory: Path, symbols: Iterable[str]) -> dict[str, object]:
    """Where each symbol's bytes came from, read from the sidecars `fetch_prices` wrote.

    Those sidecars live in git-ignored `data/`, so they do not survive the run. The
    manifest's digests say *which* bytes a result came from; without the vendor's
    name a reproducer does not know where to fetch them again, which makes the
    reproduction claim in ADR-0007 only half true. So it is recorded here.

    A symbol whose sidecar is missing is recorded as missing. Guessing the source
    would put an unverified vendor name next to a verified hash.
    """
    by_symbol: dict[str, dict[str, object]] = {}
    absent: list[str] = []
    for symbol in symbols:
        sidecar = directory / f"{symbol.lower()}.source.json"
        if not sidecar.is_file():
            absent.append(symbol)
            continue
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            absent.append(symbol)
            by_symbol[symbol] = {"unreadable": f"{type(error).__name__}: {error}"}
            continue
        by_symbol[symbol] = {key: record.get(key) for key in ("source", "url_shape", "fetched_at", "rows")}
    sources = sorted({str(entry["source"]) for entry in by_symbol.values() if entry.get("source")})
    return {"by_symbol": by_symbol, "sources": sources, "no_provenance": sorted(absent)}


def discover(directory: Path) -> dict[str, Path]:
    """Every `*.csv` in the directory is a symbol. `*.source.json` files are provenance."""
    return {path.stem.upper(): path for path in sorted(directory.glob("*.csv"))}


def measured_sessions(panel: PricePanel) -> dict[str, object]:
    """Sessions a year as this panel actually shows them: recorded, not yet used.

    The backtest still annualises at `BacktestConfig`'s default. Feeding the
    measurement in changes gate arithmetic, and that is a change which has to
    re-pass the canaries (CLAUDE.md 8, ADR-0013). Recording it now means the day
    that happens there is a before to compare against.
    """
    used = BacktestConfig().periods_per_year
    try:
        return {"measured": round(sessions_per_year(panel.dates), 2), "used_by_config": used}
    except ValueError as error:
        return {"measured": None, "why_not": str(error), "used_by_config": used}


def factor_matrix(panel: PricePanel, factors: FactorPanel | None, market: str) -> tuple:
    """The factor returns for this market's bars, or nothing and the reason why.

    Two reasons to end up with nothing, and they are different: the market has no
    factor file at all (Korea, until it has a Korean source), or the file does not
    line up with the panel. Neither becomes a zero matrix -- the engine's panel
    proxy takes over and labels the verdict `PANEL_PROXY`, which is what a reader
    needs to see.
    """
    if factors is None:
        return panel, None, {"used": False, "why_not": "no factor file was fetched"}
    if market != FRENCH_MARKET:
        return (
            panel,
            None,
            {
                "used": False,
                "why_not": f"the Ken French factors describe {FRENCH_MARKET}, not {market}",
            },
        )
    try:
        trimmed, dropped = trim_panel_to_factors(panel, factors)
        matrix = factors.align_to_bars(trimmed.dates)
    except ValueError as error:
        return panel, None, {"used": False, "why_not": str(error)}
    return (
        trimmed,
        matrix,
        {
            "used": True,
            "source": factors.source,
            "digest": factors.digest,
            "columns": list(factors.names),
            "file_span": [str(factors.dates[0]), str(factors.dates[-1])],
            "bars_dropped_to_factor_coverage": dropped,
            "panel_span_used": [str(trimmed.dates[0]), str(trimmed.dates[-1])],
        },
    )


def smoke_run(panel: PricePanel, pin: ReproPin, factor_returns=None) -> dict[str, object]:
    grid = [{"lookback": lb, "gross": 1.0} for lb in (5, 10, 20, 40, 60, 120)]
    submission, leak_report, report = run(
        alpha_id=f"smoke-{pin.run_id}",
        panel=panel,
        strategy=momentum,
        grid=grid,
        chosen=2,
        config=BacktestConfig(),
        factor_returns=factor_returns,
    )
    verdicts = gates.evaluate(submission, leak_report)
    return {
        "run_id": pin.run_id,
        "pin": pin.as_dict(),
        "alpha_id": submission.alpha_id,
        "n_trials": report.n_trials,
        "chosen_params": report.chosen_params,
        "factor_source": report.factor_source.value,
        "is_rows": report.is_rows,
        "oos_rows": report.oos_rows,
        "adv_participation": report.adv_participation,
        "performance": report.performance,
        "notes": report.notes,
        "approved": gates.approved(verdicts),
        "failed_gates": gates.failed_gates(verdicts),
        "verdicts": [
            {"gate": v.gate, "passed": v.passed, "reason": v.reason, "metrics": v.metrics} for v in verdicts
        ],
    }


def markdown(
    manifest_path: Path,
    manifest,
    smoke: dict[str, object],
    provenance: dict,
    market: str | None = None,
) -> str:
    perf = smoke["performance"]
    sessions = smoke["sessions_per_year"]
    measured = (
        f"{sessions['measured']}/yr measured"
        if sessions["measured"] is not None
        else f"not measurable ({sessions['why_not']})"
    )
    record = smoke["factors"]
    if record["used"]:
        factor_note = (
            f" ({', '.join(record['columns'])}; {record['bars_dropped_to_factor_coverage']}"
            " bar(s) trimmed to the factor file's coverage)"
        )
    else:
        factor_note = f" ({record['why_not']})"
    # An unmeasurable metric must not take the summary down with it.
    turnover = "unreported" if perf["turnover_annual"] is None else f"{perf['turnover_annual']:.1f}x/yr"
    absent = provenance["no_provenance"]
    unrecorded = f" (no provenance for: {', '.join(absent)})" if absent else ""
    lines = [
        "## Data snapshot" + (f" — {market}" if market else ""),
        "",
        f"- snapshot id: `{manifest.snapshot_id}`",
        f"- symbols: {', '.join(manifest.symbols)}"
        + (f" (missing: {', '.join(manifest.missing)})" if manifest.missing else ""),
        f"- rows: {manifest.rows} ({manifest.first_date} to {manifest.last_date})",
        f"- dates dropped for not being common to every symbol: {manifest.dates_dropped}",
        f"- dollar volume present: {manifest.has_volume}",
        f"- sessions: {measured}, annualised at {sessions['used_by_config']} (ADR-0013)",
        f"- fetched from: {', '.join(provenance['sources']) or 'unrecorded'}" + unrecorded,
        f"- manifest: `{manifest_path}`",
        "",
        "## Gate smoke run",
        "",
        "A throwaway cross-sectional momentum, run only to prove the path from bytes to",
        "verdict works. Rejection is the normal outcome.",
        "",
        f"- run id: `{smoke['run_id']}` (git SHA + snapshot id + seed)",
        f"- trials: {smoke['n_trials']}, factor source: `{smoke['factor_source']}`{factor_note}",
        f"- CAGR {perf['cagr']:.2%}, vol {perf['volatility']:.2%}, Sharpe {perf['sharpe']:.2f}, "
        f"MDD {perf['max_drawdown']:.2%}, turnover {turnover}",
        f"- approved: **{smoke['approved']}**",
        f"- failed gates: {', '.join(smoke['failed_gates']) or 'none'}",
        "",
        "| gate | passed | reason |",
        "| --- | --- | --- |",
    ]
    for verdict in smoke["verdicts"]:
        reason = (verdict["reason"] or "").replace("|", "\\|")
        lines.append(f"| {verdict['gate']} | {verdict['passed']} | {reason} |")
    if smoke["notes"]:
        lines += ["", "Provenance notes:", ""] + [f"- {note}" for note in smoke["notes"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--snapshots", default="registry/snapshots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-coverage", type=float, default=0.98)
    parser.add_argument(
        "--factors",
        default="data/factors/ff5_mom_daily.csv",
        help="the normalised FF5+momentum CSV; absent means the engine's panel proxy is used",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    files = discover(Path(args.data))
    if not files:
        raise SystemExit(f"no CSV files in {args.data}/; run scripts/fetch_prices.py first")

    # One panel per market. Two markets keep different holidays, so a shared date
    # axis would throw away real sessions in both of them (ADR-0013).
    snapshot = load_market_panels(files, min_coverage=args.min_coverage)

    # Every pin is taken before anything is written, so each describes a clean checkout.
    pins = {
        code: pin_current(snapshot.manifests[code].snapshot_id, args.seed, allow_dirty=args.allow_dirty)
        for code in snapshot.markets
    }

    snapshots = Path(args.snapshots)
    sections: list[str] = []
    if len(snapshot.markets) > 1:
        # One market's own manifest already names the whole read; the multi-market
        # index only earns a file when there is more than one id to tie together.
        index = write_market_manifest(snapshot, snapshots)
        sections.append(
            "## Markets\n\n"
            f"- snapshot id: `{snapshot.snapshot_id}` (a digest of the per-market ids)\n"
            f"- markets: {', '.join(snapshot.markets)}\n"
            f"- index: `{index}`\n"
        )

    # The factor file is read once: it is one file, and reading it per market
    # would make two copies of the same digest.
    factor_path = Path(args.factors)
    factors = load_factors(factor_path).drop(("RF",)) if factor_path.is_file() else None

    for code in snapshot.markets:
        manifest = snapshot.manifests[code]
        manifest_path = write_manifest(manifest, snapshots)
        panel, matrix, factor_record = factor_matrix(snapshot.panels[code], factors, code)
        smoke = smoke_run(panel, pins[code], factor_returns=matrix)
        smoke["market"] = code
        smoke["factors"] = factor_record
        smoke["sessions_per_year"] = measured_sessions(panel)
        smoke["provenance"] = read_provenance(Path(args.data), manifest.symbols)
        non_finite: list[str] = []
        written = json_safe(smoke, found=non_finite)
        written["non_finite_metrics"] = sorted(non_finite)
        (snapshots / f"{manifest.snapshot_id}.smoke.json").write_text(
            json.dumps(written, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        sections.append(markdown(manifest_path, manifest, smoke, smoke["provenance"], market=code))

    summary = "\n".join(sections)
    print(summary)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
