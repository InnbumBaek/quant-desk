"""Gate verdicts G0-G6. Code decides, not an agent.

Thresholds come from `core/risk/limits.yaml` and are never passed in, so a
caller cannot loosen a gate by choosing different arguments. A submission that
fails any gate is rejected; there is no weighting and no override.

As of 2026-09-22 (ADR-0002, authorised by the owner) every G4 criterion the
alpha-gate skill documents is enforced from `limits.yaml`: the deflated Sharpe
probability, PBO, the block-bootstrap p-value, the factor-residual t, and the
drawdown-to-return ratio.

**Two thresholds are in this file and not in the table**, both in `g5_robustness`:
the 30% Sharpe decay under +/-20% parameters, and the sign test at double cost.
They come from the alpha-gate skill, which is where the plan put them, and the
module docstring used to claim otherwise. They are controlled the same way the
table is -- `scripts/check_limits_change.py` lists this file, so neither can move
without an ADR (ADR-0032).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core import audit
from core.backtest import stats
from core.backtest.cv import fold_sign_stability
from core.backtest.leakage import LeakReport
from core.risk.limits import as_measurement, load_limits


@dataclass(frozen=True)
class Verdict:
    gate: str
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.gate}: {'pass' if self.passed else 'FAIL'} {self.reason}".strip()


@dataclass
class Submission:
    """Everything a gate needs, produced by the backtest engine, not by an agent."""

    alpha_id: str
    in_sample: np.ndarray
    out_of_sample: np.ndarray
    fold_returns: list[np.ndarray]
    trial_returns: np.ndarray
    factor_returns: np.ndarray
    adv_participation: float = 0.0
    #: Whether `adv_participation` is a measurement. False by default and by
    #: absence: the engine reports 0.0 participation when a run had no
    #: dollar-volume panel, and 0.0 is the one value that passes every capacity
    #: check there is (ADR-0026).
    adv_measured: bool = False
    book_correlation: float = 0.0
    cost_doubled: np.ndarray | None = None
    param_perturbed: list[np.ndarray] = field(default_factory=list)
    periods_per_year: int = 252
    #: Sessions of paper-trading record behind this submission, for G7.
    paper_trading_days: int = 0
    #: Whether `paper_trading_days` is a record rather than a default, on the same
    #: reasoning as `adv_measured`: 0 is a number, and a gate that reads 0 as
    #: "no paper trading yet, so not applicable" would pass every submission that
    #: has never traded. Absent means FAIL (ADR-0032).
    paper_trading_measured: bool = False

    @property
    def full(self) -> np.ndarray:
        return np.concatenate([self.in_sample, self.out_of_sample])

    @property
    def trial_sharpes(self) -> np.ndarray:
        trials = np.asarray(self.trial_returns, dtype=float)
        return np.array([stats.sharpe_ratio(trials[:, j], 1) for j in range(trials.shape[1])])


def g0_data(report: LeakReport) -> Verdict:
    metrics = {"probes": float(report.probes), "leaks": float(len(report.leaks))}
    if report.ok:
        return Verdict("G0_data", True, metrics)
    return Verdict(
        "G0_data",
        False,
        metrics,
        f"look-ahead at {len(report.leaks)} of {report.probes} probes "
        f"(max deviation {report.max_deviation:.3g})",
    )


def g2_in_sample(sub: Submission, limits: dict[str, Any] | None = None) -> Verdict:
    gates = (limits or load_limits())["gates"]
    sharpe = stats.sharpe_ratio(sub.in_sample, sub.periods_per_year)
    floor = float(gates["is_sharpe_min"])
    return Verdict(
        "G2_in_sample",
        sharpe >= floor,
        {"is_sharpe": sharpe},
        "" if sharpe >= floor else f"in-sample Sharpe {sharpe:.2f} < {floor}",
    )


def g3_oos(sub: Submission, limits: dict[str, Any] | None = None) -> Verdict:
    gates = (limits or load_limits())["gates"]
    is_sharpe = stats.sharpe_ratio(sub.in_sample, sub.periods_per_year)
    oos_sharpe = stats.sharpe_ratio(sub.out_of_sample, sub.periods_per_year)
    ratio = oos_sharpe / is_sharpe if is_sharpe > 0 else 0.0
    stability = fold_sign_stability(sub.fold_returns)

    ratio_min = float(gates["oos_to_is_sharpe_ratio_min"])
    stability_min = float(gates["fold_sign_stability_min"])
    metrics = {"oos_sharpe": oos_sharpe, "oos_to_is": ratio, "fold_sign_stability": stability}

    failures = []
    if ratio < ratio_min:
        failures.append(f"OOS/IS Sharpe {ratio:.2f} < {ratio_min}")
    if stability < stability_min:
        failures.append(f"fold sign stability {stability:.2f} < {stability_min}")
    return Verdict("G3_oos", not failures, metrics, "; ".join(failures))


def g4_statistics(sub: Submission, limits: dict[str, Any] | None = None) -> Verdict:
    gates = (limits or load_limits())["gates"]
    full = sub.full

    probability = stats.deflated_sharpe_ratio(full, sub.trial_sharpes)
    deflated_excess = stats.deflated_sharpe_excess(full, sub.trial_sharpes)
    trials = np.asarray(sub.trial_returns, dtype=float)
    if trials.ndim != 2 or trials.shape[1] < 2:
        # CSCV needs at least two configurations to ask which one wins. An
        # unmeasurable criterion fails, exactly as an unknown data check does:
        # a submission of one hand-picked parameter set is the case PBO exists
        # to catch, so it may not pass by being too thin to measure.
        return Verdict(
            "G4_statistics",
            False,
            {"deflated_sharpe_probability": probability, "deflated_sharpe_excess": deflated_excess},
            "PBO is not computable from a single configuration; submit the grid that was run (G1)",
        )
    pbo = stats.probability_of_backtest_overfitting(trials)
    tstat = stats.residual_alpha_tstat(full, sub.factor_returns)
    bootstrap_p = stats.block_bootstrap_pvalue(full)

    annual_return = float(np.mean(full)) * sub.periods_per_year
    drawdown = stats.max_drawdown(full)
    # A losing strategy has no meaningful drawdown-to-return ratio; make it fail, not divide by zero.
    dd_to_return = drawdown / annual_return if annual_return > 0 else float("inf")

    metrics = {
        "deflated_sharpe_probability": probability,
        "deflated_sharpe_excess": deflated_excess,
        "pbo": pbo,
        "residual_alpha_tstat": tstat,
        "bootstrap_pvalue": bootstrap_p,
        "dd_to_return": dd_to_return,
    }

    failures = []
    if probability < float(gates["deflated_sharpe_probability_min"]):
        failures.append(
            f"deflated Sharpe probability {probability:.3f} < {gates['deflated_sharpe_probability_min']}"
        )
    if pbo >= float(gates["pbo_max"]):
        failures.append(f"PBO {pbo:.3f} >= {gates['pbo_max']}")
    if bootstrap_p >= float(gates["bootstrap_pvalue_max"]):
        failures.append(f"bootstrap p {bootstrap_p:.3f} >= {gates['bootstrap_pvalue_max']}")
    if tstat < float(gates["residual_alpha_tstat_min"]):
        failures.append(f"residual alpha t {tstat:.2f} < {gates['residual_alpha_tstat_min']}")
    if dd_to_return > float(gates["dd_to_return_ratio_max"]):
        failures.append(f"drawdown/return {dd_to_return:.2f} > {gates['dd_to_return_ratio_max']}")
    return Verdict("G4_statistics", not failures, metrics, "; ".join(failures))


def g5_robustness(sub: Submission) -> Verdict:
    """Parameter plateau and cost doubling. Both criteria are in the skill, not the limits table."""
    base = stats.sharpe_ratio(sub.full, sub.periods_per_year)
    metrics: dict[str, float] = {"base_sharpe": base}
    failures = []

    if sub.param_perturbed:
        worst = min(stats.sharpe_ratio(r, sub.periods_per_year) for r in sub.param_perturbed)
        decay = 1.0 - worst / base if base > 0 else 1.0
        metrics["worst_perturbed_sharpe"] = worst
        metrics["sharpe_decay"] = decay
        if decay >= 0.30:
            failures.append(f"Sharpe decays {decay:.0%} at +/-20% parameters")

    if sub.cost_doubled is not None:
        net = float(np.mean(sub.cost_doubled)) * sub.periods_per_year
        metrics["annual_return_double_cost"] = net
        if net <= 0:
            failures.append("negative net return at double cost")

    return Verdict("G5_robustness", not failures, metrics, "; ".join(failures))


def g6_capacity(sub: Submission, limits: dict[str, Any] | None = None) -> Verdict:
    gates = (limits or load_limits())["gates"]
    metrics = {
        "adv_participation": sub.adv_participation,
        "adv_measured": sub.adv_measured,
        "book_correlation": sub.book_correlation,
    }
    failures = []
    if sub.adv_measured is not True:
        # Not a threshold: a threshold compares two numbers, and there is no
        # number here. A run with no dollar-volume panel reports 0.0, which
        # clears any participation cap that could be written (ADR-0026).
        failures.append(
            "ADV participation is not a measurement (the run had no dollar-volume panel), "
            "so capacity was not tested"
        )
    elif sub.adv_participation > float(gates["adv_participation_max"]):
        failures.append(f"ADV participation {sub.adv_participation:.3f} > {gates['adv_participation_max']}")
    if abs(sub.book_correlation) > float(gates["book_correlation_abs_max"]):
        failures.append(f"book correlation {sub.book_correlation:.2f} > {gates['book_correlation_abs_max']}")
    return Verdict("G6_capacity", not failures, metrics, "; ".join(failures))


def g7_paper_trading(sub: Submission, limits: dict[str, Any] | None = None) -> Verdict:
    """Has this alpha actually traded on paper for long enough?

    **This gate had no code.** `gates.paper_trading_days_min` sat in the limits
    table with nothing reading it, so the last gate before live capital was the
    one gate that could not refuse anything. `tests/limits/` now catches that
    class of key mechanically; this closes the instance (ADR-0032).

    The judgement half of G7 -- whether the paper record *looked* like the
    backtest -- stays with the cio, and code does not pretend to make it. What is
    code is the half that is arithmetic: how many sessions of record there are,
    against the table's floor.

    Absence fails. A submission that has never paper traded arrives with
    `paper_trading_days == 0` and `paper_trading_measured is False`, and reading
    that as "not applicable" would clear every alpha that has never traded --
    the same failure `adv_measured` exists to stop (ADR-0026).
    """
    gates = (limits or load_limits())["gates"]
    floor = int(gates["paper_trading_days_min"])
    # `is not True` rather than truthiness, and `as_measurement` rather than
    # `float`, for the same reason: a malformed field must produce a refusal that
    # the audit log records, not an exception that stops the whole evaluation.
    measured = sub.paper_trading_measured is True
    days = as_measurement(sub.paper_trading_days)
    metrics: dict[str, float] = {
        "paper_trading_measured": float(measured),
        "paper_trading_days_min": float(floor),
    }
    if days is not None:
        metrics["paper_trading_days"] = days
    if not measured:
        return Verdict(
            "G7_paper",
            False,
            metrics,
            "no paper-trading record; a run that never traded reports 0 sessions, "
            f"which is not the same as clearing a floor of {floor}",
        )
    if days is None:
        return Verdict(
            "G7_paper",
            False,
            metrics,
            f"{sub.paper_trading_days!r} is not a session count, so there is nothing to compare",
        )
    if days < floor:
        return Verdict("G7_paper", False, metrics, f"{days:.0f} paper session(s) < {floor}")
    return Verdict("G7_paper", True, metrics)


def live_blockers(
    verdicts: list[Verdict],
    sub: Submission,
    limits: dict[str, Any] | None = None,
) -> list[str]:
    """Every reason live capital is not permitted. Never empty.

    **G8 is always outstanding here.** Live capital is the owner's decision and
    this repository holds no artifact that records it, so code cannot report it as
    given. A function that could return an empty list would be a function that
    approves live trading, which is exactly what must not exist (ADR-0012 makes
    the live broker domain a G8 item on the same reasoning).

    `evaluate` keeps meaning research approval, G0-G6. This is the deployment
    question asked separately, so neither answer can be mistaken for the other.
    """
    blockers = [f"{v.gate} failed: {v.reason}" for v in verdicts if not v.passed]
    g7 = g7_paper_trading(sub, limits)
    if not g7.passed:
        blockers.append(f"{g7.gate} failed: {g7.reason}")
    blockers.append(
        "G8_live is a human approval: the owner authorises live capital and no code path grants it"
    )
    return blockers


def evaluate(
    sub: Submission,
    leak_report: LeakReport,
    limits: dict[str, Any] | None = None,
    audit_path: Path | None = None,
) -> list[Verdict]:
    """Run G0 through G6 and record every verdict: research approval, not deployment.

    G1 is a human research review. G7's arithmetic half is `g7_paper_trading` and
    is deliberately *not* run here -- folding it in would change what `approved`
    means for every existing caller, from "the research stands up" to "this may
    take live capital". `live_blockers` asks the second question, and G8 stays the
    owner's (ADR-0032).
    """
    lim = limits or load_limits()
    verdicts = [
        g0_data(leak_report),
        g2_in_sample(sub, lim),
        g3_oos(sub, lim),
        g4_statistics(sub, lim),
        g5_robustness(sub),
        g6_capacity(sub, lim),
    ]
    audit.append(
        "gates.evaluate",
        {
            "alpha_id": sub.alpha_id,
            "approved": approved(verdicts),
            "verdicts": [
                {"gate": v.gate, "passed": v.passed, "reason": v.reason, "metrics": v.metrics}
                for v in verdicts
            ],
        },
        path=audit_path,
    )
    return verdicts


def approved(verdicts: list[Verdict]) -> bool:
    return all(v.passed for v in verdicts)


def failed_gates(verdicts: list[Verdict]) -> list[str]:
    return [v.gate for v in verdicts if not v.passed]
