"""What widening the universe costs in history, and what it buys in breadth.

Six alphas were rejected on five ETFs and the finding was that the sample, not
the search, was binding (ADR-0040). The obvious response is to trade on more
instruments -- the fundamental law pays the square root of breadth -- and the
obvious response has a price nobody had measured: **a panel is the intersection
of its symbols' dates.** A symbol that listed in 2010 does not shorten its own
history, it shortens everyone's, so every instrument added to the request is some
months taken off the window for all of them.

That is a curve, not a single answer, and this report draws it. For each distinct
inception date among the fetched files it reports the universe that date admits,
the window it leaves, the Sharpe the gates then demand, and the **effective**
number of bets that universe carries -- not the ticker count, because nine sector
slices of one index are not nine bets (`core/backtest/breadth.py`).

**Why this is worth an artifact.** "Buy more data" is the most expensive
recommendation a research desk can make and the easiest one to make carelessly.
The two numbers that decide it are the breadth actually gained and the history
actually lost, and neither is guessable: whether sector ETFs carry independent
variation is an empirical question about their correlation matrix, and the
required Sharpe barely moves with the sample once there are fifteen years but
moves a great deal below that. This report puts both on one page so the choice of
universe is a decision with a record rather than a preference (CLAUDE.md 2).

**It is not a gate and it decides nothing.** No verdict, no `Breach`, no
threshold of its own. It reads `limits.yaml` through `core/backtest/power.py` and
reports. Which universe to declare is a pre-registration, and that is a separate,
committed act (ADR-0041).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from core.backtest.breadth import effective_bets, law_factor
from core.backtest.power import DEFAULT_TRIALS, binding, requirements, sharpe_for_deflated_probability
from core.backtest.trials import desk_trials
from core.data.markets import UNIVERSE, UnknownSymbol
from core.data.sources import load_market_panels, series_spans
from core.data.universe import history_frontier, history_screen
from core.repro import pin_current
from core.risk.limits import load_limits
from scripts.data_snapshot import discover

DEFAULT_DATA = Path("data")
DEFAULT_OUT = Path("registry/universe")

#: The universe the desk's committed declarations are judged on today. Every row
#: is compared against it, because "wider" is only meaningful against a baseline
#: and this is the one the rejections came from (ADR-0040).
BASELINE = ("SPY", "QQQ", "IWM", "TLT", "GLD")


def _point(
    files: dict[str, Path],
    symbols: tuple[str, ...],
    limits: dict,
    desk_n: int | None,
    min_coverage: float,
    start: str | None = None,
) -> dict[str, object]:
    """One universe: its window, its effective breadth, and what the gates demand."""
    row: dict[str, object] = {
        "symbols": list(symbols),
        "count": len(symbols),
        "loaded": False,
        "reason": None,
    }
    try:
        # From the window this universe admits, so a late listing reads as a
        # shorter window rather than as a hole the whole panel is refused for.
        snapshot = load_market_panels({s: files[s] for s in symbols}, min_coverage=min_coverage, start=start)
    except (ValueError, UnknownSymbol) as error:
        # Recorded, not dropped. A universe that will not load is a fact about
        # this universe, and the reason is the useful half.
        row["reason"] = str(error)
        return row
    markets = sorted(snapshot.markets)
    if len(markets) != 1:
        row["reason"] = f"spans {markets}; a panel is one calendar (core/data/markets.py)"
        return row

    panel = snapshot.panels[markets[0]]
    n_obs = int(len(panel.dates) - 1)
    bets = effective_bets(panel.bar_returns)
    floor = float(limits["gates"]["deflated_sharpe_probability_min"])
    worst = binding(requirements(n_obs, desk_n or DEFAULT_TRIALS, limits))
    row.update(
        {
            "loaded": True,
            "market": markets[0],
            "snapshot_id": snapshot.manifests[markets[0]].snapshot_id,
            "first": str(panel.dates[0]),
            "last": str(panel.dates[-1]),
            "observations": n_obs,
            "years": round(n_obs / 252.0, 2),
            "effective_bets": bets,
            "binding": None if worst is None else worst.criterion,
            "binding_annualised_sharpe_min": None if worst is None else worst.annualised_sharpe_min,
            "deflated_sharpe_required": sharpe_for_deflated_probability(
                n_obs, desk_n or DEFAULT_TRIALS, floor
            ),
        }
    )
    return row


def build(
    data: Path = DEFAULT_DATA,
    min_coverage: float = 0.98,
    seed: int = 0,
    allow_dirty: bool = False,
) -> dict[str, object]:
    files = discover(data)
    if not files:
        raise SystemExit(f"no CSV files in {data}/; run scripts/fetch_prices.py first")
    spans, unreadable = series_spans(files)
    # A file for a symbol nobody declared is not a candidate. `markets.UNIVERSE`
    # is the declaration and it raises on an unknown ticker on purpose (ADR-0017);
    # this report names it and carries on rather than dying on one stray file.
    for symbol in sorted(set(spans) - set(UNIVERSE)):
        unreadable[symbol] = (
            "not declared in core/data/markets.py UNIVERSE, so its market and currency are unknown"
        )
        spans.pop(symbol)
    if not spans:
        raise SystemExit(f"no readable series in {data}/: {unreadable}")

    limits = load_limits()
    desk = desk_trials()
    desk_n = desk.total if desk.measured else None

    baseline_symbols = tuple(s for s in BASELINE if s in files)
    baseline = _point(files, baseline_symbols, limits, desk_n, min_coverage) if baseline_symbols else None
    baseline_bets = (baseline or {}).get("effective_bets")

    rows: list[dict[str, object]] = []
    for start, symbols in history_frontier(spans):
        row = _point(files, symbols, limits, desk_n, min_coverage, start.isoformat())
        row["window_start"] = start.isoformat()
        row["law_factor_vs_baseline"] = law_factor(row.get("effective_bets"), baseline_bets)
        screen = history_screen(spans, start, unreadable)
        row["excluded"] = dict(sorted(screen.excluded.items()))
        rows.append(row)

    pin = pin_current(
        spans[sorted(spans)[0]].last.replace("-", "") + f"x{len(spans)}",
        seed,
        allow_dirty=allow_dirty,
    )
    return {
        "run_id": pin.run_id,
        "pin": pin.as_dict(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fetched": len(files),
        "readable": len(spans),
        "unreadable": dict(sorted(unreadable.items())),
        "desk_trials": desk.as_dict(),
        "baseline": baseline,
        "spans": {
            symbol: {"first": span.first, "last": span.last, "rows": span.rows}
            for symbol, span in sorted(spans.items())
        },
        "frontier": rows,
        "assumption": (
            "effective bets is the entropy of the correlation spectrum on this window "
            "(Meucci 2009), which bounds the fundamental law's breadth from above rather "
            "than predicting an information ratio: it says nothing about skill, cost or "
            "capacity on the added names"
        ),
    }


def _fmt(value: object, spec: str = ".2f") -> str:
    if value is None:
        return "n/a"
    number = float(value)  # type: ignore[arg-type]
    return "n/a" if number != number else format(number, spec)


def markdown(report: dict[str, object]) -> str:
    baseline = report.get("baseline") or {}
    lines = [
        "## Universe frontier (history traded for breadth)",
        "",
        f"- run `{report['run_id']}`, {report['readable']} readable of {report['fetched']} fetched",
        f"- baseline {', '.join(baseline.get('symbols', [])) or 'none'}: "
        f"{baseline.get('observations', 'n/a')} observations, effective bets "
        f"**{_fmt(baseline.get('effective_bets'))}**",
        "",
        "| window from | names | years | effective bets | law factor | binding | required Sharpe |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["frontier"]:  # type: ignore[union-attr]
        if not row["loaded"]:
            lines.append(
                f"| {row['window_start']} | {row['count']} | - | - | - | not loaded | "
                f"{str(row['reason'])[:60].replace('|', '/')} |"
            )
            continue
        lines.append(
            f"| {row['window_start']} | {row['count']} | {_fmt(row['years'], '.1f')} "
            f"| {_fmt(row['effective_bets'])} | {_fmt(row['law_factor_vs_baseline'])}x "
            f"| `{row['binding']}` | {_fmt(row['binding_annualised_sharpe_min'])} |"
        )
    if report["unreadable"]:
        lines += ["", "Not readable:", ""]
        lines += [f"- `{symbol}`: {reason}" for symbol, reason in report["unreadable"].items()]
    lines += ["", f"Assumption: {report['assumption']}."]
    return "\n".join(lines) + "\n"


def write(report: dict[str, object], directory: Path = DEFAULT_OUT) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report['run_id']}.frontier.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--min-coverage", type=float, default=0.98)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    report = build(Path(args.data), min_coverage=args.min_coverage, allow_dirty=args.allow_dirty)
    print(markdown(report), end="")
    path = write(report, Path(args.out))
    print(f"\nwritten to {path}")
    # Exit 1 when no universe on the frontier is wider in *effective* bets than
    # the one the desk already trades: that is the state where more tickers would
    # buy nothing the law can use, and it should be visible rather than filed.
    baseline_bets = (report.get("baseline") or {}).get("effective_bets")
    gained = [
        row["law_factor_vs_baseline"]
        for row in report["frontier"]  # type: ignore[union-attr]
        if row.get("law_factor_vs_baseline") is not None
    ]
    if baseline_bets is None or not gained:
        return 1
    return 0 if max(gained) > 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
