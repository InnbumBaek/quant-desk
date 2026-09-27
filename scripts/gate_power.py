"""Report what Sharpe the gates admit on the history this desk actually holds.

`core/backtest/power.py` inverts the criteria; this decides what sample length to
invert them at and writes the answer down. The length is not a parameter anyone
chooses -- it is read from the snapshot panel, so the report says what the gates
demand of *this* desk today rather than of a desk in general.

**Why this is worth an artifact.** Every gate can refuse and none can say whether
refusing is all it will ever do. A desk that cannot tell "no alpha passed" from
"no alpha could pass" keeps researching against a wall and calls it discipline.
This report is the number that tells them apart, and it is also the honest answer
to "why does this desk need more data": it is a requirement, not a preference.

**It is not a gate and it changes nothing.** There is no verdict here, no
`Breach`, no threshold of its own. It reads `limits.yaml` and reports, and the
owner is the only one who moves a threshold (CLAUDE.md 3).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from core.backtest.power import (
    DEFAULT_TRIALS,
    binding,
    expected_max_sharpe,
    observations_for,
    requirements,
)
from core.data.sources import load_market_panels
from core.repro import pin_current
from core.risk.limits import load_limits
from scripts.data_snapshot import discover

DEFAULT_DATA = Path("data")
DEFAULT_OUT = Path("registry/power")

#: True annualised Sharpes to answer the inverse question at. Not predictions:
#: 0.75 is below G2's floor and is there to show that one target no sample length
#: reaches, 1.0 is exactly what G2 asks for, 1.5 is a good equity market-neutral
#: book, 2.0 is unusual and 3.0 is the number a pitch deck claims.
TARGETS = (0.75, 1.0, 1.5, 2.0, 3.0)


def build(
    data: Path = DEFAULT_DATA,
    market: str | None = None,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    allow_dirty: bool = False,
) -> dict[str, object]:
    files = discover(data)
    if not files:
        raise SystemExit(f"no CSV files in {data}/; run scripts/fetch_prices.py first")
    snapshot = load_market_panels(files)
    chosen = market or sorted(snapshot.markets)[0]
    if chosen not in snapshot.manifests:
        raise SystemExit(f"the snapshot holds {', '.join(snapshot.markets)}, not {chosen}")

    panel = snapshot.panels[chosen]
    n_obs = len(panel.dates) - 1  # one return per pair of closes
    pin = pin_current(snapshot.manifests[chosen].snapshot_id, seed, allow_dirty=allow_dirty)
    limits = load_limits()

    reqs = requirements(n_obs, trials, limits)
    worst = binding(reqs)
    return {
        "run_id": pin.run_id,
        "pin": pin.as_dict(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "market": chosen,
        "observations": n_obs,
        "symbols": len(panel.symbols),
        "trials_assumed": trials,
        "expected_max_sharpe_per_period": expected_max_sharpe(n_obs, trials),
        "requirements": [
            {
                "criterion": r.criterion,
                "annualised_sharpe_min": r.annualised_sharpe_min,
                "depends_on_sample": r.depends_on_sample,
                "note": r.note,
            }
            for r in reqs
        ],
        "binding": None if worst is None else worst.criterion,
        "binding_annualised_sharpe_min": None if worst is None else worst.annualised_sharpe_min,
        # The inverse question, which is the one that decides whether waiting for
        # data is the answer or whether no sample length helps.
        "observations_needed": {str(t): observations_for(t, trials, limits) for t in TARGETS},
        "not_inverted": {
            "pbo_max": "a property of the trial matrix's shape, not of one return series",
            "residual_alpha_tstat_min": "depends on which factors the series loads on",
            "dd_to_return_ratio_max": "a path property; two series with one Sharpe differ in it",
        },
        "assumption": (
            "iid normal returns. Fat tails move the deflated-Sharpe criterion by about -0.004 at "
            "this length; positive skew moves it by about +0.025, so that figure is not a "
            "universal lower bound; autocorrelation makes the bootstrap far harder than the "
            "closed form, so that figure is a loose floor (ADR-0033)"
        ),
    }


def _years(observations: int | None) -> str:
    return "-" if observations is None else f"{observations} ({observations / 252:.1f}y)"


def markdown(report: dict[str, object]) -> str:
    lines = [
        "## Gate power (what Sharpe the gates admit)",
        "",
        f"- run `{report['run_id']}`, market {report['market']}",
        f"- sample: **{report['observations']} return(s)** "
        f"({float(report['observations']) / 252:.1f} years) across {report['symbols']} symbol(s), "
        f"{report['trials_assumed']} trial(s) assumed",
        "",
        "| criterion | min annualised Sharpe | moves with sample |",
        "| --- | --- | --- |",
    ]
    for item in report["requirements"]:  # type: ignore[union-attr]
        value = item["annualised_sharpe_min"]
        shown = "not measured" if value is None else f"{float(value):.2f}"
        lines.append(f"| `{item['criterion']}` | {shown} | {'yes' if item['depends_on_sample'] else 'no'} |")
    worst = report.get("binding_annualised_sharpe_min")
    lines += [
        "",
        f"- **binding: `{report.get('binding')}` at "
        f"{'not measured' if worst is None else f'{float(worst):.2f}'} annualised**",
        "",
        "How much history a true Sharpe needs before the battery stops binding:",
        "",
    ]
    for target, needed in report["observations_needed"].items():  # type: ignore[union-attr]
        reach = _years(needed)
        tail = "" if needed is not None else "  (no sample length clears the in-sample floor)"
        lines.append(f"- Sharpe {target}: {reach}{tail}")
    lines += ["", f"Assumption: {report['assumption']}."]
    return "\n".join(lines) + "\n"


def write(report: dict[str, object], directory: Path = DEFAULT_OUT) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report['run_id']}.power.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--market", default=None)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    report = build(Path(args.data), market=args.market, trials=args.trials, allow_dirty=args.allow_dirty)
    print(markdown(report), end="")
    path = write(report, Path(args.out))
    print(f"\nwritten to {path}")
    # Exit 1 when the statistics gate rather than the in-sample floor is binding:
    # that is the state where the sample, not the strategy, is the constraint, and
    # it should be visible in the workflow rather than buried in a file. A `binding`
    # of None means nothing could be inverted at all, which is not a pass either.
    bound = report.get("binding")
    return 0 if isinstance(bound, str) and bound.startswith("G2") else 1


if __name__ == "__main__":
    raise SystemExit(main())
