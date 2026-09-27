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


@dataclass(frozen=True)
class Preregistration:
    """What was declared before the backtest ran, read from `registry/alphas/`.

    `parameters_declared` is a grid: each key maps to the values that key was
    allowed to take. The product of those lengths is N, the number of
    configurations the search was permitted -- which is the number the deflated
    Sharpe deflates by. Declaring it afterwards is the same as not declaring it.
    """

    alpha_id: str
    economic_rationale: str = ""
    universe: str = ""
    horizon: str = ""
    parameters_declared: dict[str, Any] = field(default_factory=dict)
    committed: bool = False
    path: str = ""

    @property
    def declared_trials(self) -> int:
        """N: the size of the declared grid, or 0 when nothing was declared."""
        total = 1
        for values in self.parameters_declared.values():
            if not isinstance(values, list | tuple) or not values:
                return 0
            total *= len(values)
        return total if self.parameters_declared else 0

    @property
    def complete(self) -> tuple[str, ...]:
        """The fields a pre-registration is not one without."""
        return tuple(
            name
            for name in ("economic_rationale", "universe", "horizon")
            if not str(getattr(self, name)).strip()
        )


def g1_preregistration(sub: Submission, prereg: Preregistration | None = None) -> Verdict:
    """Was the trial count declared before the search, and does the run match it?

    **G4 rests on this gate.** `deflated_sharpe_ratio` deflates the observed
    Sharpe by what the best of N trials buys for free, so N is the input that
    decides whether a result is evidence. N taken from the run itself is N chosen
    after seeing the answer: drop the configurations that failed and the deflated
    Sharpe rises with no change to the strategy. `core/backtest/power.py` measures
    how much -- the required Sharpe moves with N -- which is why this gate is worth
    code rather than trust (ADR-0035).

    What is arithmetic, and therefore here:

    - a declaration exists at all, and names a rationale, a universe and a horizon;
    - the declared grid is non-empty, so N is a number;
    - the run did not search **more** configurations than were declared;
    - the declaration is committed and unmodified, because a file that can still be
      edited is a file that can be edited after the result.

    What is judgement, and therefore the cio's: whether the rationale is a reason
    an inefficiency exists rather than a description of the backtest.
    """
    trials_run = int(np.asarray(sub.trial_returns, dtype=float).shape[1])
    metrics: dict[str, float] = {"trials_run": float(trials_run)}
    if prereg is None:
        return Verdict(
            "G1_preregistration",
            False,
            metrics,
            "no pre-registration for this alpha, so N was counted after the search "
            "and the deflated Sharpe is not a claim",
        )

    metrics["declared_trials"] = float(prereg.declared_trials)
    metrics["committed"] = float(prereg.committed)
    missing = prereg.complete
    if missing:
        return Verdict(
            "G1_preregistration",
            False,
            metrics,
            f"the declaration leaves {list(missing)} empty, which is a form and not a hypothesis",
        )
    if prereg.declared_trials <= 0:
        return Verdict(
            "G1_preregistration",
            False,
            metrics,
            "no parameter grid was declared, so N is whatever the run reports",
        )
    if trials_run > prereg.declared_trials:
        return Verdict(
            "G1_preregistration",
            False,
            metrics,
            f"the run searched {trials_run} configuration(s), more than the "
            f"{prereg.declared_trials} declared; the deflation is short by the difference",
        )
    if not prereg.committed:
        return Verdict(
            "G1_preregistration",
            False,
            metrics,
            f"{prereg.path or 'the declaration'} is uncommitted or modified, so it could "
            "have been written after the result",
        )
    return Verdict("G1_preregistration", True, metrics)


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


def g4_desk_multiplicity(
    sub: Submission,
    desk_trials: int | None,
    limits: dict[str, Any] | None = None,
    unmeasured_because: str = "",
) -> Verdict:
    """The same deflated Sharpe, deflated by every trial the *desk* declared.

    G4 deflates by this alpha's own grid, which is the right question about this
    alpha and the wrong one about the fund. A desk that declares six families and
    allocates to whichever clears has selected over all of their configurations,
    so the Sharpe that survives 5 trials has not survived 30 (ADR-0039).

    `desk_trials` is a count, not a module import: `core/backtest/trials.py` reads
    it from the committed declarations and would import this module back. None
    means it could not be measured, and that is a failure -- skipping a
    declaration lowers N, which raises every deflated Sharpe on the desk.

    Not in `evaluate`. Two reasons, and they are the ones ADR-0032 and ADR-0035
    gave for G7 and G1: folding it in would change what `approved` means for
    every existing caller, and a criterion that moves when an unrelated pod
    declares a grid would reject a canary for something other than its own
    defect. `live_blockers` is where it binds, because the decision it belongs to
    is whether this may take capital when other candidates exist.
    """
    floor = float((limits or load_limits())["gates"]["deflated_sharpe_probability_min"])
    own_trials = int(np.asarray(sub.trial_returns, dtype=float).shape[1])
    metrics: dict[str, float] = {"own_trials": float(own_trials), "floor": floor}

    if desk_trials is None:
        return Verdict(
            "G4_desk_multiplicity",
            False,
            metrics,
            "the desk-wide trial count could not be measured, so this Sharpe is deflated by "
            f"this alpha's {own_trials} and not by the desk's search"
            + (f": {unmeasured_because}" if unmeasured_because else ""),
        )
    desk = int(desk_trials)
    metrics["desk_trials"] = float(desk)
    if desk < own_trials:
        return Verdict(
            "G4_desk_multiplicity",
            False,
            metrics,
            f"the desk declared {desk} trial(s) but this run searched {own_trials}; the desk "
            "count cannot be smaller than one alpha's, so one of the two is wrong",
        )

    probability = stats.deflated_sharpe_ratio(sub.full, sub.trial_sharpes, n_trials=desk)
    excess = stats.deflated_sharpe_excess(sub.full, sub.trial_sharpes, n_trials=desk)
    metrics["deflated_sharpe_probability_desk"] = probability
    metrics["deflated_sharpe_excess_desk"] = excess
    if probability < floor:
        # When the desk has searched no wider than this alpha, this gate is G4
        # again and saying "rather than" would invent a distinction.
        wider = (
            f" once deflated by the desk's {desk} declared trial(s) rather than this alpha's {own_trials}"
            if desk > own_trials
            else f" at the desk's {desk} declared trial(s), which is this alpha's own search"
        )
        return Verdict(
            "G4_desk_multiplicity",
            False,
            metrics,
            f"deflated Sharpe probability {probability:.3f} < {floor}{wider}",
        )
    return Verdict("G4_desk_multiplicity", True, metrics)


def live_blockers(
    verdicts: list[Verdict],
    sub: Submission,
    limits: dict[str, Any] | None = None,
    prereg: Preregistration | None = None,
    desk_trials: int | None = None,
    desk_unmeasured_because: str = "",
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
    seen = {v.gate for v in verdicts}
    # G1 is here and not in `evaluate` for the reason that docstring gives, but a
    # result whose N was counted after the search must never reach live capital:
    # the deflated Sharpe behind it is not a measurement of anything.
    if "G1_preregistration" not in seen:
        g1 = g1_preregistration(sub, prereg)
        if not g1.passed:
            blockers.append(f"{g1.gate} failed: {g1.reason}")
    # The desk-wide deflation, for the same reason: allocating to the one alpha
    # that cleared is a selection over every alpha that was declared.
    desk = g4_desk_multiplicity(sub, desk_trials, limits, desk_unmeasured_because)
    if not desk.passed:
        blockers.append(f"{desk.gate} failed: {desk.reason}")
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

    **G1 is not here on purpose, and it is now code.** `g1_preregistration` checks
    the arithmetic half -- was N declared before the search, and did the run stay
    inside it. It stays out of this list because a gate that *every* submission
    fails would make `tests/canaries/` pass for the wrong reason: a canary has to
    be rejected for its own statistical defect, and a blanket failure hides which
    gate caught it. `live_blockers` is where it binds (ADR-0035).

    G7's arithmetic half is `g7_paper_trading` and
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
