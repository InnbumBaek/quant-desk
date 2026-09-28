"""Run one pre-registered alpha through the whole gate battery and record the verdict.

`scripts/data_snapshot.py` proves the path from bytes to verdict with a throwaway
strategy. This is the real thing: a strategy from `core/strategies/library.py`,
the grid its declaration fixed before the run, and a record that a reviewer can
check without rerunning anything.

**What this script refuses to do, and why each refusal is fail-closed.**

- **No declaration, no run.** Without one, N is counted after the search and the
  deflated Sharpe is not a claim (ADR-0035). The script exits before touching the
  data rather than producing a number that would have to be thrown away.
- **The code grid may not exceed the declared grid.** `Strategy.grid` lives in
  code and code can be edited; the declaration is committed and checked
  unmodified. So every point the run will search is checked against the declared
  values *before* the run, and a point outside them stops the submission. G1
  catches the count afterwards; this catches the values beforehand, which is the
  half a count cannot see -- five points are five points whether or not they are
  the five that were declared.
- **The panel is restricted to the declared universe.** Both halves of the
  search were pinned and the *instruments* were not: the run took whatever the
  day's fetch held, so widening the fetch would re-run every old declaration on a
  universe nobody pre-registered (ADR-0041). An alpha with no declared universe
  cannot be submitted, and a declared symbol the snapshot lacks is a refusal
  rather than a panel with a hole in it.
- **The reported configuration comes out of the declaration.** Not out of the
  results, and not out of `Strategy.defaults`, which can be edited after seeing
  them. A declaration with no `chosen_declared` is refused.
- **A rejected alpha is a successful run.** The exit code reports whether the
  *process* held, not whether the alpha passed: rejection is the ordinary outcome
  and a red workflow for it would train everyone to ignore the red. Exit 2 is for
  a submission that could not honestly be evaluated.

**Why `--all` submits every declared alpha rather than a chosen one.** The desk's
trial count is the sum of the declared grids (ADR-0039), so an alpha that is
declared and never run makes the count larger than the search anyone actually
looked at. That direction is conservative for the deflation and dishonest about
the research: a hypothesis nobody tested sits in the registry looking tested. So
the default is all of them, in one process on one snapshot, and a declaration with
no implementation stops the batch instead of being skipped (ADR-0040).

The record is `registry/submissions/<run_id>.<alpha_id>.json`: verdicts, metrics,
the three-way pin, the declaration it was judged against, and the catalogue
registration. `registry/alphas/<alpha_id>.yaml` is **not** written back to --
`core/backtest/prereg.py` requires it unmodified, so writing results into it
would make G1 reject the next run for a file that could have been edited after
the fact.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from core.alphas import implementations, universes
from core.backtest import gates, power, prereg
from core.backtest.engine import BacktestConfig, PricePanel, run
from core.backtest.trials import DeskTrials, desk_trials
from core.data.factors import load_factors
from core.data.sources import MarketSnapshot, load_market_panels, write_manifest
from core.features.catalog import (
    MAX_ABS_CORRELATION,
    Feature,
    FeatureCatalog,
    catalogue_summary,
    rank_correlation,
)
from core.repro import ReproPin, pin_current
from core.risk.limits import load_limits
from core.strategies.base import Strategy, get, names
from scripts.data_snapshot import discover, factor_matrix, json_safe, measured_sessions

DEFAULT_OUT = Path("registry/submissions")


class NotSubmittable(RuntimeError):
    """The submission cannot be evaluated honestly. Refuse rather than report."""


def grid_within_declaration(grid: Sequence[Mapping[str, float]], declared: Mapping[str, object]) -> list[str]:
    """The ways `grid` steps outside `declared`, empty when it does not.

    Every parameter of every point must appear in the declared values for that
    parameter. A parameter the declaration does not mention at all is a violation
    too: an undeclared axis is an axis that can be widened silently, which is the
    thing the declaration exists to prevent.
    """
    problems: list[str] = []
    for index, point in enumerate(grid):
        for name, value in point.items():
            allowed = declared.get(name)
            if allowed is None:
                problems.append(f"grid point {index} sets {name}, which the declaration does not mention")
                continue
            if not isinstance(allowed, list | tuple):
                problems.append(f"the declaration gives {name} as {allowed!r}, which is not a list of values")
                continue
            if not any(float(value) == float(candidate) for candidate in allowed):
                problems.append(
                    f"grid point {index} sets {name}={value}, which is not among the declared {list(allowed)}"
                )
    return problems


def chosen_index(grid: Sequence[Mapping[str, float]], chosen: Mapping[str, float]) -> int:
    """Where the declared configuration sits in the grid.

    Exactly one match is required. Zero means the declaration names a point the
    run will not visit; more than one means the grid has duplicates and "the
    reported point" does not identify a run.
    """
    hits = [
        index
        for index, point in enumerate(grid)
        if all(float(point.get(name, np.nan)) == float(value) for name, value in chosen.items())
    ]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise NotSubmittable(
            f"the declared configuration {dict(chosen)} is not a point in the grid "
            f"{[dict(p) for p in grid]}, so the run would report a configuration nobody declared"
        )
    raise NotSubmittable(
        f"the declared configuration {dict(chosen)} matches {len(hits)} grid points; "
        "a duplicated grid point does not identify a run"
    )


def catalogue_the_signal(
    panel: PricePanel,
    strategy: Strategy,
    params: Mapping[str, float],
    audit_path: Path | None = None,
) -> dict[str, object]:
    """Admit the traded signal to the shared catalogue, alongside every other family.

    CLAUDE.md rule 6: a pod may not keep a private feature. The catalogue is where
    that is checked rather than asserted, and the check only means something when
    the other families are in the same catalogue -- a de-duplication threshold with
    one entry rejects nothing. So every available strategy is registered at its
    own defaults, in name order, and the submitted signal takes its turn like the
    others. If it comes back a duplicate of another family, that is a finding
    about the book's breadth, not a reason to reorder the registrations.

    The feature is the weight matrix as traded, not an intermediate score: what
    the catalogue must not contain twice is the position the desk ends up holding.

    **When the submitted signal is itself an ensemble** it is not admitted as a
    peer, because it is a combination of catalogued features and the threshold
    would reject one of its own components as a duplicate of their blend. It is
    still measured against every component, which is the number its hypothesis
    rests on: a blend whose legs are highly correlated has the breadth of one leg,
    whatever the fundamental law would give it (ADR-0040). `accepted` is then
    `None` rather than `True` -- it was not judged, and `True` would read as a
    de-duplication check that never ran.
    """
    catalog = FeatureCatalog(panel.close, audit_path=audit_path)
    registrations: list[dict[str, object]] = []
    for name in names(available_only=True):
        candidate = get(name)
        if candidate.family == "ensemble":
            # An ensemble is a combination of catalogued features, not a feature.
            # Registering it beside its own components would reject one of them as
            # a duplicate of their blend, which says nothing about breadth.
            continue
        bound = dict(params) if name == strategy.name else dict(candidate.defaults)
        try:
            fn = candidate.build(**{k: v for k, v in bound.items() if k in candidate.defaults})
        except Exception as error:  # pragma: no cover - a registry that cannot build itself
            registrations.append({"name": name, "accepted": False, "reason": f"could not build: {error}"})
            continue
        result = catalog.register(
            Feature(
                name=name,
                description=candidate.citation,
                source="core/strategies/library.py",
                owner=candidate.family,
                fn=fn,
            )
        )
        registrations.append(
            {
                "name": name,
                "params": bound,
                "accepted": result.accepted,
                "reason": result.reason,
                "rejection": result.rejection.value if result.rejection else None,
                "correlations": result.correlations,
                "leak_probes": result.leak_probes,
            }
        )
    submitted = next((row for row in registrations if row["name"] == strategy.name), None)
    against_components: dict[str, float] = {}
    if submitted is None:
        # An ensemble: measured against the catalogue, not admitted to it.
        bound = {k: v for k, v in params.items() if k in strategy.defaults}
        traded = np.asarray(strategy.build(**bound)(panel.close), dtype=float)
        against_components = {
            name: rank_correlation(traded, catalog.values(name)) for name in catalog.names()
        }
    worst = max(against_components.values(), key=abs, default=None)
    return {
        "max_abs_correlation": MAX_ABS_CORRELATION,
        "registered": list(catalog.names()),
        "summary": catalogue_summary(catalog),
        "registrations": registrations,
        "submitted_is_ensemble": submitted is None,
        "submitted_feature_accepted": None if submitted is None else bool(submitted["accepted"]),
        "submitted_vs_components": against_components,
        "submitted_worst_component_correlation": worst,
        "pairwise": {f"{a}|{b}": value for (a, b), value in catalog.correlations().items()},
    }


def grid_crowding(panel: PricePanel, strategy: Strategy, grid: Sequence[Mapping[str, float]]) -> dict:
    """Rank correlation between the grid's own configurations.

    Not a gate: a measurement of whether the five trials the deflation counts are
    five ideas or one idea five times. N deflates by count, so a grid of near
    copies is deflated as if it were a wide search, which flatters the result --
    worth seeing in the record even though no verdict reads it.
    """
    weights = {
        json.dumps({k: float(v) for k, v in point.items()}, sort_keys=True): strategy.build(
            **{k: v for k, v in point.items() if k in strategy.defaults}
        )(panel.close)
        for point in grid
    }
    keys = sorted(weights)
    return {
        f"{a}|{b}": rank_correlation(weights[a], weights[b])
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    }


def _required_sharpe(sub: gates.Submission, desk_total: int | None) -> dict[str, object]:
    """What G4's deflated-Sharpe floor demands at this alpha's N and at the desk's.

    Two numbers rather than one, because the gap between them *is* the price of
    searching wide: the same sample, the same floor, a higher bar because more
    was tried. Reported, not gated -- `g4_desk_multiplicity` is the gate
    (ADR-0033 built the inversion, ADR-0039 gave it the desk's N).
    """
    floor = float(load_limits()["gates"]["deflated_sharpe_probability_min"])
    n_obs = int(np.asarray(sub.full, dtype=float).size)
    own = int(np.asarray(sub.trial_returns, dtype=float).shape[1])
    return {
        "n_obs": n_obs,
        "deflated_sharpe_probability_min": floor,
        "at_own_trials": power.sharpe_for_deflated_probability(n_obs, own, floor),
        "own_trials": own,
        "at_desk_trials": (
            power.sharpe_for_deflated_probability(n_obs, desk_total, floor)
            if desk_total is not None
            else None
        ),
        "desk_trials": desk_total,
    }


def submit(
    alpha_id: str,
    strategy_name: str,
    panel: PricePanel,
    pin: ReproPin,
    declaration: prereg.Preregistration,
    chosen: Mapping[str, float],
    factor_returns: np.ndarray | None = None,
    config: BacktestConfig | None = None,
    desk: DeskTrials | None = None,
    universe: tuple[tuple[str, ...], str] | None = None,
) -> dict[str, object]:
    """Evaluate one declared alpha and return the record, verdict included."""
    strategy = get(strategy_name)
    if not strategy.available:
        raise NotSubmittable(f"{strategy_name}: {strategy.unavailable_because}")

    # Before anything is measured: the panel is the universe the declaration
    # named, not the one the fetch happened to hold (ADR-0041).
    try:
        declared_symbols, universe_source = universe or universes.for_alpha(alpha_id)
    except (universes.UniverseError, implementations.ImplementationError) as error:
        raise NotSubmittable(str(error)) from error
    try:
        panel = panel.select(declared_symbols)
    except ValueError as error:
        raise NotSubmittable(f"{alpha_id}'s declared universe is not in this snapshot: {error}") from error

    grid = strategy.search_grid()
    problems = grid_within_declaration(grid, declaration.parameters_declared)
    if problems:
        raise NotSubmittable(
            f"{strategy_name}'s grid steps outside {declaration.path}: " + "; ".join(problems)
        )
    index = chosen_index(grid, chosen)

    submission, leak_report, report = run(
        alpha_id=alpha_id,
        panel=panel,
        strategy=strategy.fn,
        grid=grid,
        chosen=index,
        config=config or BacktestConfig(),
        factor_returns=factor_returns,
    )
    verdicts = gates.evaluate(submission, leak_report)
    g1 = gates.g1_preregistration(submission, declaration)
    g7 = gates.g7_paper_trading(submission)

    # The desk's whole declared search, not just this alpha's grid. Allocating to
    # whichever alpha clears is a selection over all of them (ADR-0039).
    desk = desk if desk is not None else desk_trials()
    desk_total = desk.total if desk.measured else None
    desk_reason = "; ".join(desk.unusable)
    g4_desk = gates.g4_desk_multiplicity(submission, desk_total, unmeasured_because=desk_reason)
    blockers = gates.live_blockers(
        verdicts,
        submission,
        prereg=declaration,
        desk_trials=desk_total,
        desk_unmeasured_because=desk_reason,
    )

    return {
        "run_id": pin.run_id,
        "pin": pin.as_dict(),
        "alpha_id": alpha_id,
        "strategy": strategy_name,
        "family": strategy.family,
        "citation": strategy.citation,
        "universe": {
            "declared": list(declared_symbols),
            "source": universe_source,
            "panel_symbols": list(panel.symbols),
        },
        "declaration": {
            "path": declaration.path,
            "committed_and_unmodified": declaration.committed,
            "declared_trials": declaration.declared_trials,
            "parameters_declared": declaration.parameters_declared,
            "chosen_declared": dict(chosen),
            "chosen_index": index,
            "economic_rationale": declaration.economic_rationale.strip(),
            "universe": declaration.universe.strip(),
            "horizon": declaration.horizon.strip(),
        },
        "n_trials": report.n_trials,
        "chosen_params": report.chosen_params,
        "factor_source": report.factor_source.value,
        "is_rows": report.is_rows,
        "oos_rows": report.oos_rows,
        "adv_participation": report.adv_participation,
        "performance": report.performance,
        "notes": report.notes,
        "sessions_per_year": measured_sessions(panel),
        "approved": gates.approved(verdicts),
        "failed_gates": gates.failed_gates(verdicts),
        "verdicts": [
            {"gate": v.gate, "passed": v.passed, "reason": v.reason, "metrics": v.metrics}
            for v in [*verdicts, g1, g7, g4_desk]
        ],
        "live_blockers": blockers,
        "desk_trials": desk.as_dict(),
        "sharpe_required": _required_sharpe(submission, desk_total),
        "grid_crowding": grid_crowding(panel, strategy, grid),
        "catalogue": catalogue_the_signal(panel, strategy, report.chosen_params),
    }


def _admitted(catalogue: Mapping[str, object], short: bool = False) -> str:
    """Whether the submitted signal cleared de-duplication, or was not judged."""
    accepted = catalogue.get("submitted_feature_accepted")
    if accepted is None:
        return "not judged" if short else "not judged -- an ensemble is not a peer of its components"
    return str(bool(accepted))


def markdown(record: Mapping[str, object]) -> str:
    """The record as a reviewer reads it: the verdict first, then what it rests on."""
    declaration = record["declaration"]
    verdicts = record["verdicts"]
    failed = list(record["failed_gates"])
    headline = "연구 게이트 통과 (G0-G6)" if record["approved"] else f"기각 -- {', '.join(failed)}"
    lines = [
        f"## Submission `{record['alpha_id']}` -- {headline}",
        "",
        f"- strategy: `{record['strategy']}` ({record['family']}), {record['citation']}",
        f"- run id: `{record['run_id']}`",
        f"- declaration: `{declaration['path']}`, committed and unmodified: "
        f"**{declaration['committed_and_unmodified']}**",
        f"- N declared {declaration['declared_trials']}, N run {record['n_trials']}; "
        f"reported configuration {declaration['chosen_declared']} (grid index {declaration['chosen_index']})",
        f"- sample: {record['is_rows']} in-sample rows, {record['oos_rows']} out of sample; "
        f"factors `{record['factor_source']}`",
        f"- universe: {', '.join(record['universe']['declared'])} "
        f"({len(record['universe']['declared'])} instrument(s)), declared in the "
        f"**{record['universe']['source']}**",
        "",
        "### Gates",
        "",
        "| gate | verdict | reason |",
        "| --- | --- | --- |",
    ]
    for verdict in verdicts:
        mark = "pass" if verdict["passed"] else "**FAIL**"
        reason = str(verdict["reason"] or "").replace("|", "/")
        lines.append(f"| {verdict['gate']} | {mark} | {reason} |")
    lines += [
        "",
        "### What still stands between this and live capital",
        "",
        *[f"- {blocker}" for blocker in record["live_blockers"]],
        "",
        "### Catalogue (CLAUDE.md rule 6)",
        "",
        f"- registered features: {', '.join(record['catalogue']['registered']) or 'none'}",
        f"- submitted signal admitted: **{_admitted(record['catalogue'])}**",
    ]
    worst_component = record["catalogue"].get("submitted_worst_component_correlation")
    if worst_component is not None:
        lines.append(
            f"- the blend against its own components: worst |rho| = {abs(float(worst_component)):.2f}; "
            "a blend whose legs move together has the breadth of one leg"
        )
    for row in record["catalogue"]["registrations"]:
        if not row.get("accepted"):
            lines.append(f"- rejected `{row['name']}`: {row.get('reason', '')}")
    desk = record["desk_trials"]
    required = record["sharpe_required"]
    lines += [
        "",
        "### The desk's whole search (ADR-0039)",
        "",
        f"- declared trials: this alpha {required['own_trials']}, the desk "
        f"**{desk['total']}** across {len(desk['per_alpha'])} alpha(s); measured: {desk['measured']}",
    ]
    if desk["unusable"]:
        lines += [f"  - not counted: {reason}" for reason in desk["unusable"]]
    own_bar, desk_bar = required["at_own_trials"], required["at_desk_trials"]
    if own_bar is not None and desk_bar is not None:
        lines.append(
            f"- annualised Sharpe the deflation floor demands over {required['n_obs']} "
            f"observations: **{own_bar:.2f}** at this alpha's N, **{desk_bar:.2f}** at the desk's"
        )

    crowding = record["grid_crowding"]
    if crowding:
        worst = max(crowding.items(), key=lambda kv: abs(kv[1]))
        lines += [
            "",
            f"- grid crowding: worst pair |rho| = {abs(worst[1]):.2f} across "
            f"{len(crowding)} pairs; a grid of near copies is deflated as a wide search",
        ]
    return "\n".join(lines) + "\n"


def _number(value: object, spec: str = ".2f") -> str:
    """A metric as a cell, and `n/a` when it was not measured.

    Not `0.00`: a metric that could not be computed is not a metric with the value
    zero, and zero is the flattering reading of several of them (ADR-0038).
    """
    if value is None:
        return "n/a"
    number = float(value)
    return "n/a" if number != number else format(number, spec)


def batch_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    """The whole desk's submissions side by side, on one snapshot.

    The comparison is the point of running them together. Six families on one
    sample, one pin and one deflation count answer a question no single record
    can: whether anything here is evidence, or whether the desk searched six ways
    and found the same nothing (ADR-0040).

    Every number is read out of the verdict that computed it rather than out of
    `performance`, which carries the whole book's Sharpe and not the in-sample and
    out-of-sample halves the gates judge. Reading the wrong key printed `n/a`
    against six measured values, and a table that says a number is missing when it
    is not is worse than one that omits the column.
    """
    lines = [
        "## Desk submission run",
        "",
        "| alpha | strategy | research gates | IS Sharpe | OOS Sharpe | deflated p | PBO | failed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        metrics = {v["gate"]: v["metrics"] for v in row["verdicts"]}
        g2, g3, g4 = (metrics.get(name, {}) for name in ("G2_in_sample", "G3_oos", "G4_statistics"))
        lines.append(
            f"| `{row['alpha_id']}` | `{row['strategy']}` "
            f"| {'pass' if row['approved'] else '**FAIL**'} "
            f"| {_number(g2.get('is_sharpe'))} | {_number(g3.get('oos_sharpe'))} "
            f"| {_number(g4.get('deflated_sharpe_probability'), '.3f')} "
            f"| {_number(g4.get('pbo'), '.3f')} "
            f"| {', '.join(row['failed_gates']) or '-'} |"
        )
    # The universes, because they are no longer necessarily one. Two alphas on
    # different instrument sets have different windows, and a table that hid that
    # would invite a comparison nobody should make (ADR-0041).
    by_universe: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (row["run_id"], ", ".join(row["universe"]["declared"]))
        by_universe.setdefault(key, []).append(row["alpha_id"])
    lines += ["", "Universes in this run:", ""]
    for (run_id, declared), members in by_universe.items():
        sample = next(row for row in rows if row["alpha_id"] == members[0])
        lines.append(
            f"- `{run_id}`: {declared} -- {sample['is_rows']} in-sample + "
            f"{sample['oos_rows']} out-of-sample rows, carrying {', '.join(members)}"
        )

    catalogued = [
        f"`{row['alpha_id']}` {_admitted(row['catalogue'], short=True)}"
        for row in rows
        if not row["catalogue"].get("submitted_feature_accepted")
    ]
    if catalogued:
        lines += ["", f"- catalogue (CLAUDE.md 6항): {'; '.join(catalogued)}"]
    return "\n".join(lines) + "\n"


def _prepare(wanted: Sequence[str], alphas: Path) -> tuple[dict[str, tuple], list[str]]:
    """Resolve every planned alpha's declaration before any data is read.

    The ordering is the point. Nothing about a declaration needs prices, and a
    submission that cannot be evaluated honestly must stop before it computes a
    number that would have to be thrown away. With `--all` it also means one bad
    declaration costs no snapshot load.
    """
    prepared: dict[str, tuple] = {}
    refusals: list[str] = []
    for alpha_id in wanted:
        declaration = prereg.load(alpha_id, directory=alphas)
        if declaration is None:
            refusals.append(
                f"refusing to submit {alpha_id}: no declaration at "
                f"{alphas / (alpha_id + '.yaml')}. N would be counted after the search, and a "
                "deflated Sharpe built on that is not a claim (ADR-0035)."
            )
            continue
        try:
            chosen = prereg.declared_chosen(alpha_id, directory=alphas)
        except prereg.PreregistrationError as error:
            refusals.append(f"refusing to submit {alpha_id}: {error}")
            continue
        if chosen is None:
            refusals.append(
                f"refusing to submit {alpha_id}: {declaration.path} declares no "
                "`chosen_declared`, so the reported configuration would be chosen after seeing "
                "the grid's results."
            )
            continue
        prepared[alpha_id] = (declaration, chosen)
    return prepared, refusals


def _write(record: Mapping[str, object], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    non_finite: list[str] = []
    written = json_safe(record, found=non_finite)
    written["non_finite_metrics"] = sorted(non_finite)
    path = out / f"{record['run_id']}.{record['alpha_id']}.json"
    path.write_text(json.dumps(written, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _report(summary: str) -> None:
    print(summary)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", default=None, help="one alpha id; omit with --all")
    parser.add_argument("--strategy", default=None, help="override the implementation map")
    parser.add_argument(
        "--all",
        action="store_true",
        help="submit every declared alpha on one snapshot, using the implementation map",
    )
    parser.add_argument("--market", default="US")
    parser.add_argument("--data", default="data")
    parser.add_argument("--alphas", default="registry/alphas")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--factors", default="data/factors/ff5_mom_daily.csv")
    parser.add_argument("--seed", type=int, default=0)
    # Derived from `--out` rather than defaulting to a path of its own: the two
    # always belong to the same registry, and a caller that redirects one and
    # forgets the other writes half its run into the live tree (ADR-0050).
    parser.add_argument("--snapshots", default=None)
    parser.add_argument("--min-coverage", type=float, default=0.98)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    alphas = Path(args.alphas)
    # The map lives beside the declarations it maps, so pointing --alphas at another
    # directory moves both together and neither can be read against the wrong set.
    mapping = alphas / implementations.DEFAULT_PATH.name
    if args.all and (args.alpha or args.strategy):
        print("--all submits every declaration; --alpha and --strategy do not apply")
        return 64
    wanted = prereg.declared_ids(alphas) if args.all else [args.alpha or "tsmom-001"]
    if not wanted:
        print(f"refusing to submit: no alpha is declared in {alphas}")
        return 2

    # The declarations are read first, and the wiring second. A missing declaration
    # is a fact about the research; a missing map entry is a fact about the plumbing,
    # and reporting the plumbing would bury the one that matters.
    prepared, refusals = _prepare(wanted, alphas)
    if refusals:
        for refusal in refusals:
            print(refusal)
        return 2

    try:
        if args.all:
            plan, problems = implementations.for_all(alphas, mapping)
            if problems:
                # A declaration nobody runs inflates the desk's trial count above the
                # search anyone looked at, so the batch stops rather than skipping.
                for problem in problems:
                    print(f"refusing to submit the desk: {problem}")
                return 2
        else:
            alpha_id = wanted[0]
            plan = {alpha_id: args.strategy or implementations.for_alpha(alpha_id, mapping)}
    except implementations.ImplementationError as error:
        print(f"refusing to submit: {error}")
        return 2

    files = discover(Path(args.data))
    if not files:
        print(f"refusing to submit: no CSV files in {args.data}/; run scripts/fetch_prices.py")
        return 2

    # One panel per declared universe, built from that universe's own files.
    #
    # Restricting the columns of a shared panel would pin the instruments and
    # leave the *dates* to the widest symbol set on disk: `load_csv_panel`
    # intersects dates, so adding one short-history symbol to the fetch shortens
    # every other alpha's sample without a word. The window is part of a universe's
    # identity, so each universe is intersected on its own and carries its own
    # snapshot id and pin (ADR-0041). Alphas that declare the same universe share
    # one pin and stay comparable, which is the only comparison that means
    # anything -- five instruments over twenty years and twenty-seven over
    # nineteen are not the same experiment.
    #
    # The resolved universe is carried down to `submit()` rather than looked up
    # again there. Two lookups of one fact are two facts: `submit()`'s own call
    # takes the default registry path, so a batch pointed at another directory
    # grouped by that directory's declarations and then judged against the live
    # one's. In production the two coincide, which is exactly why it stayed
    # invisible (ADR-0048).
    groups: dict[tuple[str, ...], tuple[list[str], str]] = {}
    refused: list[str] = []
    for alpha_id in list(prepared):
        try:
            symbols, source = universes.for_alpha(alpha_id, alphas, mapping)
        except (universes.UniverseError, implementations.ImplementationError) as error:
            print(f"refusing to submit {alpha_id}: {error}")
            refused.append(alpha_id)
            prepared.pop(alpha_id)
            continue
        groups.setdefault(symbols, ([], source))[0].append(alpha_id)

    factor_path = Path(args.factors)
    factors = load_factors(factor_path).drop(("RF",)) if factor_path.is_file() else None

    # One count for the whole batch: the desk's declared search does not change
    # between two alphas submitted on the same code, and re-reading it per alpha
    # would let it drift mid-run.
    desk = desk_trials(alphas)
    out = Path(args.out)
    records: list[dict[str, object]] = []

    # Every universe's panel is read before anything is pinned or written.
    #
    # `pin_current` refuses a dirty tree, which is the check that keeps an
    # unreproducible number out of the gate pipeline (ADR-0012). Called once per
    # universe it measures the run's own output instead: the first universe's
    # records make the tree dirty and the second universe is refused for a change
    # this run made. That is exactly what happened the first time the desk
    # declared two universes -- six verdicts written, six refused (ADR-0048). So
    # the code is checked once, before the first record exists, and each universe
    # seals that same code state with its own snapshot id.
    loaded: list[tuple[tuple[str, ...], str, list[str], MarketSnapshot]] = []
    for symbols, (members, source) in groups.items():
        absent = [symbol for symbol in symbols if symbol not in files]
        if absent:
            print(
                f"refusing to submit {', '.join(members)}: the declared universe needs {absent}, "
                f"which {args.data}/ does not hold; fetch them rather than judging the "
                "hypothesis on what is there"
            )
            refused += members
            continue
        subset = {symbol: files[symbol] for symbol in symbols}
        try:
            snapshot = load_market_panels(subset, min_coverage=args.min_coverage)
        except ValueError as error:
            print(f"refusing to submit {', '.join(members)}: {error}")
            refused += members
            continue
        if args.market not in snapshot.markets:
            print(
                f"refusing to submit {', '.join(members)}: the run market is {args.market} and "
                f"this universe holds {sorted(snapshot.markets)}"
            )
            refused += members
            continue
        loaded.append((symbols, source, members, snapshot))

    # The manifest is written for every universe this run judges on, not only for
    # the one the data step happened to build. A verdict's `snapshot_id` is a
    # reference, and a reference nobody can look up is decoration: the first two
    # universes produced twelve verdicts of which six named a snapshot that
    # existed in no committed file (ADR-0050). `write_manifest` refuses to change
    # an id that already means something, so re-reading the same bytes is a
    # no-op and a disagreement is loud.
    snapshots = Path(args.snapshots) if args.snapshots else out.parent / "snapshots"
    code_pin: ReproPin | None = None
    for symbols, source, members, snapshot in loaded:
        manifest = snapshot.manifests[args.market]
        if code_pin is None:
            code_pin = pin_current(manifest.snapshot_id, args.seed, allow_dirty=args.allow_dirty)
        pin = code_pin.for_snapshot(manifest.snapshot_id)
        try:
            written = write_manifest(manifest, snapshots)
        except ValueError as error:
            print(f"refusing to submit {', '.join(members)}: {error}")
            refused += members
            continue
        print(f"snapshot {manifest.snapshot_id} -> {written}")
        panel, matrix, factor_record = factor_matrix(snapshot.panels[args.market], factors, args.market)

        for alpha_id in members:
            declaration, chosen = prepared[alpha_id]
            try:
                record = submit(
                    alpha_id=alpha_id,
                    strategy_name=plan[alpha_id],
                    panel=panel,
                    pin=pin,
                    declaration=declaration,
                    chosen=chosen,
                    factor_returns=matrix,
                    desk=desk,
                    universe=(symbols, source),
                )
            except NotSubmittable as error:
                print(f"refusing to submit {alpha_id}: {error}")
                refused.append(alpha_id)
                continue
            except Exception as error:  # noqa: BLE001 - one broken alpha must not hide five verdicts
                # Not forgiveness: the exit code below is still 2 and the step still
                # goes red. What this buys is that the alphas after this one are still
                # judged, instead of the batch dying halfway and committing a partial
                # set of records that looks like the whole desk.
                print(f"refusing to submit {alpha_id}: {type(error).__name__}: {error}")
                refused.append(alpha_id)
                continue
            record["market"] = args.market
            record["factors"] = factor_record
            record["snapshot_symbols"] = list(manifest.symbols)
            path = _write(record, out)
            records.append(record)
            _report(markdown(record) + f"\n- record: `{path}`\n")

    if len(prepared) > 1 and records:
        _report(batch_markdown(records))
    if refused:
        print(f"could not evaluate: {', '.join(refused)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
