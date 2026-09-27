"""Measure Amihud illiquidity on the snapshot panel and commit the census.

`core/execution/impact.py` measures; this decides what to measure it on and
writes the result down. Without this script the module would be the fifth thing
on this desk that exists and nothing reads (ADR-0015, ADR-0016, ADR-0026,
ADR-0027's own warning), and an impact model nobody runs is worth exactly as much
as a limit nobody checks.

**What this report can answer today, and what it cannot.** It can answer *how
much of the universe we can even cost*, and what a round trip of a book costs at
a stated notional. Both need only closes and dollar volume. It cannot answer
cost-based **capacity**, because that needs the strategy's gross edge per round
trip and no strategy has passed a gate -- the smoke book fails G2 through G5. So
`cost_share_max` and `gross_edge` are not supplied here, `capacity_gap` reports
the cost side as unmeasured, and this script does not pretend otherwise. Typing
an edge in to make the number appear is the failure ADR-0031 was written against.

**The book is the smoke book, and the report says so on its face.** This desk
holds no positions; the only book in the repository is the throwaway momentum one
`scripts/data_snapshot.py` runs to prove the path from bytes to verdict. Labelling
every number as belonging to it is what stops a reader taking a cost here for a
claim about real positions.

**The unmeasurable count is the headline, not a footnote.** A name with no lambda
is a name whose cost is unknown, and `book_cost_fraction` refuses a book that
holds one. So the share of the panel that cannot be costed is the number that
decides whether this machinery can be used at all, and it goes at the top.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core.data.sources import PricePanel, load_market_panels
from core.execution.impact import (
    DEFAULT_WINDOW,
    ROUND_TRIP_CROSSINGS,
    book_cost_fraction,
    capacity_gap,
    panel_illiquidity,
)
from core.repro import pin_current
from scripts.data_snapshot import discover, momentum

DEFAULT_OUT = Path("registry/impact")

#: The label every number in this report carries.
BOOK_NOTE = "the throwaway momentum smoke book, not a live position"

#: The notional the book's round-trip cost is quoted at. A cost fraction is only
#: meaningful beside a size, because impact is a function of size; a million is a
#: round number to quote at and not a capital the desk holds.
QUOTE_NOTIONAL = 1_000_000.0


def smoke_weights(panel: PricePanel, lookback: int = 20) -> np.ndarray:
    """The same throwaway book `data_snapshot` scores, so the two reports agree."""
    return np.asarray(momentum(panel.close, {"lookback": lookback, "gross": 1.0}), dtype=float)[-1]


def census(panel: PricePanel, window: int = DEFAULT_WINDOW) -> tuple[dict[str, object], dict[str, float]]:
    """Per-symbol lambda, and the counts that say whether it can be used."""
    measures = panel_illiquidity(panel, window=window)
    measured = {s: m for s, m in measures.items() if m.measured}
    lambdas = {s: float(m.lam) for s, m in measured.items()}  # type: ignore[arg-type]
    reasons = {s: m.unmeasured for s, m in measures.items() if not m.measured}
    summary: dict[str, object] = {
        "symbols": len(measures),
        "measured": len(measured),
        "unmeasurable": len(reasons),
        "measurable_share": round(len(measured) / len(measures), 4) if measures else 0.0,
        # Why each refusal happened, grouped, because a hundred identical reasons
        # is one problem and not a hundred.
        "unmeasurable_reasons": _grouped(reasons),
        "window_sessions": window,
        "observations": {s: m.observations for s, m in measures.items()},
        "median_dollar_volume": {
            s: m.median_dollar_volume for s, m in measures.items() if m.median_dollar_volume
        },
    }
    if lambdas:
        ordered = sorted(lambdas.items(), key=lambda kv: -kv[1])
        summary["most_illiquid"] = [{"symbol": s, "lambda": lam} for s, lam in ordered[:10]]
        summary["lambda_median"] = float(np.median(list(lambdas.values())))
    return summary, lambdas


def _grouped(reasons: dict[str, str]) -> dict[str, int]:
    """Reason -> how many symbols gave it, with the symbol name stripped out."""
    counts: dict[str, int] = {}
    for symbol, reason in reasons.items():
        key = reason.replace(repr(symbol), "<symbol>")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _why_no_cost(cost: float | None, unmeasured: tuple[str, ...], holdings: int) -> str:
    """Which of the two reasons there is no book cost, or empty when there is one."""
    if cost is not None:
        return ""
    if unmeasured:
        return f"no lambda for {list(unmeasured)[:5]}"
    if holdings == 0:
        return "the book holds nothing, so there is no round trip to cost"
    return "unrecorded"


def markdown(report: dict[str, object]) -> str:
    """The summary a human reads in the job output."""
    lines = [
        "## Impact (Amihud illiquidity)",
        "",
        f"- run `{report['run_id']}`, market {report['market']}, book: {report['book']}",
        f"- **{report['measured']} of {report['symbols']} symbol(s) costable** "
        f"({float(report['measurable_share']):.0%}), {report['window_sessions']}-session window",
    ]
    cost = report.get("book_round_trip_cost_fraction")
    if cost is None:
        lines.append(f"- book round trip: **not costable** -- {report.get('book_unmeasured')}")
    else:
        lines.append(
            f"- book round trip at {QUOTE_NOTIONAL:,.0f}: **{float(cost) * 1e4:.1f} bps** "
            f"({ROUND_TRIP_CROSSINGS} crossings, terminal displacement, linear -- an upper bound)"
        )
    if report.get("most_illiquid"):
        worst = report["most_illiquid"][0]  # type: ignore[index]
        lines.append(f"- most illiquid: `{worst['symbol']}`, lambda {worst['lambda']:.3e} per dollar")
    reasons = report.get("unmeasurable_reasons") or {}
    for reason, count in list(reasons.items())[:3]:  # type: ignore[union-attr]
        lines.append(f"- {count} symbol(s): {reason}")
    lines.append(f"- cost-based capacity: **{report.get('capacity_unmeasured')}**")
    return "\n".join(lines) + "\n"


def build(
    data: Path,
    market: str | None = None,
    window: int = DEFAULT_WINDOW,
    seed: int = 0,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Measure and return the report. Raises only on a missing input."""
    files = discover(data)
    if not files:
        raise SystemExit(f"no CSV files in {data}/; run scripts/fetch_prices.py first")
    snapshot = load_market_panels(files)
    chosen = market or sorted(snapshot.markets)[0]
    if chosen not in snapshot.manifests:
        raise SystemExit(f"the snapshot holds {', '.join(snapshot.markets)}, not {chosen}")

    panel = snapshot.panels[chosen]
    pin = pin_current(snapshot.manifests[chosen].snapshot_id, seed, allow_dirty=allow_dirty)
    weights = smoke_weights(panel)

    summary, _lambdas = census(panel, window=window)
    cost, unmeasured = book_cost_fraction(weights, panel, QUOTE_NOTIONAL, window=window)
    holdings = int(np.count_nonzero(weights))

    # No gross edge exists: nothing has passed a gate. The gap reports the cost
    # side as unmeasured rather than borrowing the participation number.
    gap = capacity_gap(
        participation_capacity=None,
        weights=weights,
        panel=panel,
        gross_edge=None,
        cost_share_max=None,
        window=window,
    )

    return {
        "run_id": pin.run_id,
        "pin": pin.as_dict(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "market": chosen,
        "book": BOOK_NOTE,
        "quote_notional": QUOTE_NOTIONAL,
        "crossings": ROUND_TRIP_CROSSINGS,
        "estimator": "amihud-2002 mean(|return| / dollar volume)",
        "conservatism": (
            "terminal displacement per crossing and linear in size; both push the cost up "
            "and the capacity down, so this is an upper bound and not a fill forecast"
        ),
        "book_round_trip_cost_fraction": cost,
        # `holdings` is here because `cost` of None means two different things,
        # and a reader cannot tell them apart without it: nothing held, or
        # something held that nobody could cost.
        "holdings": holdings,
        "book_unmeasured": _why_no_cost(cost, unmeasured, holdings),
        "capacity_unmeasured": gap.unmeasured,
        **summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--market", default=None)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    report = build(
        Path(args.data),
        market=args.market,
        window=args.window,
        seed=args.seed,
        allow_dirty=args.allow_dirty,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{report['run_id']}.impact.json"
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
    # A census that could cost nothing is a failed run, not a quiet zero: it means
    # the panel carries no dollar volume, and then the capacity machinery is unusable.
    return 0 if report["measured"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
