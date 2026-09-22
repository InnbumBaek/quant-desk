"""Turn whatever is in `data/` into a snapshot manifest and a gate smoke run.

Run after `scripts/fetch_prices.py`. What this produces, and what it deliberately
does not:

- `registry/snapshots/<snapshot_id>.json` -- the manifest: file hashes, symbols,
  date range, dropped dates. No prices. This is the data leg of the
  reproducibility pin and it is safe to commit in a public repository.
- `registry/snapshots/<snapshot_id>.smoke.json` -- the gate verdicts for one
  throwaway strategy, with the run id that pins (code, data, seed). Derived
  numbers, not the vendor's data.
- Nothing else. The CSVs stay in git-ignored `data/` and are never uploaded, so
  the vendor's bytes do not leave the runner (ADR-0007).

The strategy here is a plain cross-sectional momentum, and it is **not** an alpha
submission: it exists to prove the path from bytes to verdict actually runs. It
will normally be rejected, and that is the expected outcome of a smoke test.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from core.backtest import gates
from core.backtest.engine import BacktestConfig, PricePanel, run
from core.data.sources import load_csv_panel, write_manifest
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


def discover(directory: Path) -> dict[str, Path]:
    """Every `*.csv` in the directory is a symbol. `*.source.json` files are provenance."""
    return {path.stem.upper(): path for path in sorted(directory.glob("*.csv"))}


def smoke_run(panel: PricePanel, pin: ReproPin) -> dict[str, object]:
    grid = [{"lookback": lb, "gross": 1.0} for lb in (5, 10, 20, 40, 60, 120)]
    submission, leak_report, report = run(
        alpha_id=f"smoke-{pin.run_id}",
        panel=panel,
        strategy=momentum,
        grid=grid,
        chosen=2,
        config=BacktestConfig(),
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
        "notes": report.notes,
        "approved": gates.approved(verdicts),
        "failed_gates": gates.failed_gates(verdicts),
        "verdicts": [
            {"gate": v.gate, "passed": v.passed, "reason": v.reason, "metrics": v.metrics} for v in verdicts
        ],
    }


def markdown(manifest_path: Path, manifest, smoke: dict[str, object]) -> str:
    lines = [
        "## Data snapshot",
        "",
        f"- snapshot id: `{manifest.snapshot_id}`",
        f"- symbols: {', '.join(manifest.symbols)}"
        + (f" (missing: {', '.join(manifest.missing)})" if manifest.missing else ""),
        f"- rows: {manifest.rows} ({manifest.first_date} to {manifest.last_date})",
        f"- dates dropped for not being common to every symbol: {manifest.dates_dropped}",
        f"- dollar volume present: {manifest.has_volume}",
        f"- manifest: `{manifest_path}`",
        "",
        "## Gate smoke run",
        "",
        "A throwaway cross-sectional momentum, run only to prove the path from bytes to",
        "verdict works. Rejection is the normal outcome.",
        "",
        f"- run id: `{smoke['run_id']}` (git SHA + snapshot id + seed)",
        f"- trials: {smoke['n_trials']}, factor source: `{smoke['factor_source']}`",
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
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    files = discover(Path(args.data))
    if not files:
        raise SystemExit(f"no CSV files in {args.data}/; run scripts/fetch_prices.py first")

    # The pin is taken before anything is written, so it describes a clean checkout.
    panel, manifest = load_csv_panel(files, min_coverage=args.min_coverage)
    pin = pin_current(manifest.snapshot_id, args.seed, allow_dirty=args.allow_dirty)

    snapshots = Path(args.snapshots)
    manifest_path = write_manifest(manifest, snapshots)
    smoke = smoke_run(panel, pin)
    (snapshots / f"{manifest.snapshot_id}.smoke.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = markdown(manifest_path, manifest, smoke)
    print(summary)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
