"""Run the stress suite against a book and write the report the owner reads.

`core/risk/stress.py` measures; this decides what to measure it on and commits the
result. Without this script the module would be the fourth thing on this desk that
exists and nothing reads (ADR-0015, ADR-0016, ADR-0026), and a stress suite nobody
runs is worth exactly as much as a limit nobody checks.

**The book is the smoke book, and the report says so on its face.** This desk holds
no positions: the only book in the repository is the throwaway cross-sectional
momentum `scripts/data_snapshot.py` runs to prove the path from bytes to verdict.
So the stress report proves the path from bytes to scenario, on that same book, and
labels every number as belonging to it. The alternative -- refusing to run until a
real book exists -- leaves the plumbing untested on the day it first matters.

**The factor file is read whole, not trimmed to the panel.** The gates trim the
factor matrix to the three years of prices they regress on; a scenario is the
opposite question. 2008 is not in a panel fetched today, and it is in the factor
file, which starts in 1963. Trimming here would silently delete every scenario but
the most recent one -- and it would look like a clean run.

**What gets committed.** Ken French's library is free for research and licenses
nothing about redistribution, so the same treatment as prices (ADR-0007): the CSV
stays on the runner, and the report carries the digest, the dates, and the derived
numbers. A stress number without the digest of the file it came from cannot be
reproduced next quarter, which is the whole reason the digest is in the report.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core.backtest.engine import PricePanel
from core.data.factors import FRENCH_MARKET, FactorPanel, load_factors
from core.data.sources import load_market_panels
from core.repro import pin_current
from core.risk.exposure import style_fit
from core.risk.stress import SCENARIOS, ladder_reach, stress_snapshot
from scripts.data_snapshot import discover, factor_matrix, momentum

DEFAULT_OUT = Path("registry/stress")

#: The label every number in this report carries. It is not decoration: a reader
#: who takes a scenario loss here for a claim about real positions has been misled,
#: and the desk holds no positions yet.
BOOK_NOTE = "the throwaway momentum smoke book, not a live position"


def smoke_weights(panel: PricePanel, lookback: int = 20) -> np.ndarray:
    """The smoke book's weight history, from the same strategy function the gates run.

    `scripts.data_snapshot.momentum` rather than a copy of it: a stress report
    about a book nothing else in the repository has measured would be a report
    about nothing. The engine's `run` is not used because it does not hand the
    weight history back, and one fixed lookback needs no parameter search.
    """
    weights = momentum(np.asarray(panel.close, dtype=float), {"lookback": lookback, "gross": 1.0})
    return np.asarray(weights, dtype=float)


def book_fit(
    panel: PricePanel, factors: FactorPanel, market: str
) -> tuple[dict[str, float], float, dict[str, object]]:
    """The book's betas and residual scatter, or the reason there are none.

    The betas are measured on the overlap of prices and factors, the same window
    the gates use. The scenarios then apply those betas to windows decades older,
    which is the linearity this report keeps admitting to.
    """
    trimmed, matrix, record = factor_matrix(panel, factors, market)
    if matrix is None:
        return {}, 0.0, record
    weights = smoke_weights(trimmed)
    returns = np.sum(weights[:-1] * trimmed.bar_returns, axis=1)
    betas, scatter = style_fit(returns, matrix, factors.names)
    record = dict(record)
    record["book_bars"] = int(returns.size)
    return betas, scatter, record


def markdown(report: dict[str, object]) -> str:
    """The step summary. Scenarios ranked worst first, and the gaps named."""
    lines = [
        "## Stress suite",
        "",
        f"- book: **{report['book']}**",
        f"- factor file: `{report['factor_source']}` `{report['factor_digest'][:12]}` "
        f"({report['factor_span'][0]} to {report['factor_span'][1]})",
        f"- betas measured on {report.get('book_bars', 'no')} bar(s) of "
        f"{report.get('beta_window', 'an unrecorded window')}",
        "",
    ]
    rows = [row for row in report["scenarios"] if row["factor_loss"] is not None]
    rows.sort(key=lambda row: row["factor_loss"])
    if rows:
        lines += [
            "| scenario | window | days | factor loss | + residual | why it is in the suite |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            widened = row["loss_with_residual"]
            lines.append(
                f"| {row['key']} | {row['window'][0]}..{row['window'][1]} | {row['days']} "
                f"| {row['factor_loss']:.2%} "
                f"| {'unmeasured' if widened is None else format(widened, '.2%')} "
                f"| {row['note']} |"
            )
    else:
        lines.append("No scenario could be evaluated. That is the finding, not an empty table.")
    if report["unmeasured"]:
        lines += ["", f"Not measured ({len(report['unmeasured'])}):", ""]
        lines += [f"- {line}" for line in report["unmeasured"]]
    if report["findings"]:
        lines += ["", "Against the existing pod drawdown ladder (findings, not actions):", ""]
        lines += [f"- {line}" for line in report["findings"]]
    shocks = [
        f"- {key}: {value['move']:.2%} ({value['start']}..{value['end']})"
        for key, value in sorted(report["shocks"].items())
        if "move" in value
    ]
    if shocks:
        lines += ["", "Worst observed moves, with the days they happened on:", ""] + shocks
    lines += [
        "",
        "A factor scenario is linear in today's betas and explains only the part of the",
        "book the factors explain. Both errors make the loss look smaller, which is why",
        "the residual column is there and why it is one band rather than a worst case.",
    ]
    return "\n".join(lines) + "\n"


def build(
    data: Path,
    factor_path: Path,
    market: str = FRENCH_MARKET,
    seed: int = 0,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Measure the suite and return the report. Raises only on a missing input."""
    if not factor_path.is_file():
        raise SystemExit(
            f"no factor file at {factor_path}; run scripts/fetch_factors.py first. "
            "Without it there is nothing to take a scenario's magnitude from, and "
            "typing one in is what this module exists to avoid."
        )
    files = discover(data)
    if not files:
        raise SystemExit(f"no CSV files in {data}/; run scripts/fetch_prices.py first")

    # Whole, deliberately: trimming to the panel would delete every scenario but
    # the most recent and would look like a clean run.
    factors = load_factors(factor_path).drop(("RF",))
    snapshot = load_market_panels(files)
    if market not in snapshot.manifests:
        raise SystemExit(
            f"the snapshot holds {', '.join(snapshot.markets)} and the factor file describes "
            f"{market}; a US factor model does not price another market's book"
        )

    panel = snapshot.panels[market]
    pin = pin_current(snapshot.manifests[market].snapshot_id, seed, allow_dirty=allow_dirty)
    betas, scatter, record = book_fit(panel, factors, market)

    measured = stress_snapshot(
        factors,
        betas,
        residual_daily_vol=scatter if betas else None,
        panel=panel,
    )
    report: dict[str, object] = {
        "run_id": pin.run_id,
        "pin": pin.as_dict(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "market": market,
        "book": BOOK_NOTE,
        "book_bars": record.get("book_bars"),
        "beta_window": record.get("panel_span_used"),
        "betas": betas,
        "residual_daily_vol": scatter if betas else None,
        "scenarios_in_suite": len(SCENARIOS),
        **measured,
    }
    if not betas:
        report["unmeasured"] = [
            f"no betas, so no scenario could be evaluated: {record.get('why_not', 'unrecorded')}",
            *measured["unmeasured"],
        ]
    report["findings"] = ladder_reach(measured)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--factors", default="data/factors/ff5_mom_daily.csv")
    parser.add_argument("--market", default=FRENCH_MARKET)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    report = build(
        Path(args.data),
        Path(args.factors),
        market=args.market,
        seed=args.seed,
        allow_dirty=args.allow_dirty,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{report['run_id']}.stress.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary = markdown(report)
    print(summary)
    print(f"written to {path}")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)
    # A suite that evaluated nothing is a failure of the run, not a quiet result.
    return 0 if report["worst_key"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
