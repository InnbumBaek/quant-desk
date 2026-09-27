"""Historical scenarios and tail shocks, with every magnitude taken from the data.

The `stress-testing` agent has been defined since the first commit and had no
code. Its job is to say what this book does in 2007-08, 2008-09, 2018-02, 2020-03
and 2021-01 -- and a desk that cannot answer that has no idea what it is carrying,
because the gates measure a strategy in its own sample and a limit measures today.

Three rules shape everything here.

**A scenario is a pair of dates, never a number.** CLAUDE.md 2항 says numbers are
quoted from artifacts, not typed. "2008년 시장이 -40% 빠졌다" typed into a scenario
table is exactly the kind of number that is wrong by a decimal place and nobody
catches, so `SCENARIOS` carries only the windows. The factor moves come from
`data/factors/ff5_mom_daily.csv`, which is the same file G4 regresses against, and
a window the file cannot cover is **unmeasured**, not zero.

**A hypothetical shock is an observed tail, with its date attached.** The other
half of a stress suite is "what if the market fell X%", and X is where invented
numbers live. So `worst_window` returns the worst cumulative move this factor has
actually had over a given horizon *and the days it happened on*. The report can be
audited: somebody can open the file at those dates.

**The estimate is deliberately incomplete, and says so.** A factor scenario is
linear in today's betas: it ignores that betas rise in a crisis, and it explains
only the part of the book the factors explain. Both errors point the same way --
they make the loss look smaller. So every result carries the regression's residual
scatter over the same horizon as a separate band, and a caller that reports the
factor loss alone is reporting the optimistic half.

**No limit is checked here, because there is no limit to check.** `limits.yaml`
has no `stress:` block and CLAUDE.md 3항 puts adding one in the owner's hands. A
hypothetical loss is also not a realised drawdown: emitting a `Breach` would let
2008 halve a pod's capital today. So this module measures, and `ladder_reach`
names which rung of the *existing* drawdown ladder a scenario would touch -- as a
finding for the owner to read, not as an action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.backtest.engine import PricePanel
from core.data.factors import FactorPanel
from core.risk.limits import as_measurement, load_limits


@dataclass(frozen=True)
class Scenario:
    """One historical window. Dates and a reason; no magnitudes.

    `note` is why the window is in the suite, which is the part a reader needs in
    order to argue with the choice. The dates are inclusive on both ends.
    """

    key: str
    start: str
    end: str
    note: str


#: The suite. Every window is here because it broke a *different* thing, and the
#: order is chronological rather than by severity -- ranking them is the report's
#: job, and the report's ranking is computed.
#:
#: `quant-quake-2007` is in the suite although the agent's brief does not name it.
#: It is the one episode that hit market-neutral cross-sectional books
#: specifically, through crowding rather than through direction, and a stat-arb
#: desk that stresses 2008 but not August 2007 is stressing the wrong thing.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("black-monday-1987", "1987-10-14", "1987-10-26", "the largest one-day market move on file"),
    Scenario("ltcm-1998", "1998-08-03", "1998-10-08", "a leveraged relative-value unwind"),
    Scenario("dotcom-unwind-2000", "2000-03-10", "2000-05-31", "a style reversal, not a market fall"),
    Scenario("quant-quake-2007", "2007-08-06", "2007-08-10", "crowding: the market-neutral episode"),
    Scenario("gfc-2008", "2008-09-08", "2008-11-20", "funding withdrawal and a long drawdown"),
    Scenario("flash-crash-2010", "2010-05-06", "2010-05-07", "intraday liquidity evaporation"),
    Scenario("rmb-2015", "2015-08-18", "2015-08-25", "a cross-asset shock from outside the US"),
    Scenario("volmageddon-2018", "2018-02-02", "2018-02-08", "a short-volatility unwind"),
    Scenario("covid-2020", "2020-02-19", "2020-03-23", "the fastest fall on file"),
    Scenario("meme-squeeze-2021", "2021-01-22", "2021-02-01", "a momentum and short-interest unwind"),
    Scenario("rate-shock-2022", "2022-01-03", "2022-10-12", "a year-long discount-rate repricing"),
)

#: Columns of the Ken French file that are not risk factors. `RF` is the risk-free
#: rate and carrying it into a shock would add a financing return to a loss.
NOT_A_FACTOR: tuple[str, ...] = ("RF",)

#: The share of days that counts as "the tail" when measuring how the book's own
#: universe behaves under stress. A twentieth of a three-year panel is about
#: thirty-eight days: enough to take a median of, few enough to still be the tail.
TAIL_QUANTILE = 0.05


@dataclass(frozen=True)
class ScenarioResult:
    """What one window did to this book, and what could not be measured for it."""

    key: str
    start: str
    end: str
    note: str
    days: int = 0
    factor_moves: dict[str, float] = field(default_factory=dict)
    factor_loss: float | None = None
    residual_band: float | None = None
    unmeasured: str = ""

    @property
    def measured(self) -> bool:
        return self.factor_loss is not None and not self.unmeasured

    @property
    def loss_with_residual(self) -> float | None:
        """The factor loss widened by one residual band, which is the honest floor.

        Not a worst case. It is the factor estimate plus one standard deviation of
        everything the factor model does not explain, and the book can do worse.
        """
        if self.factor_loss is None or self.residual_band is None:
            return None
        return self.factor_loss - abs(self.residual_band)


def risk_factors(factors: FactorPanel) -> tuple[str, ...]:
    """The panel's columns that are risk factors, in the panel's own order."""
    return tuple(name for name in factors.names if name not in NOT_A_FACTOR)


def window_rows(factors: FactorPanel, start: str, end: str) -> np.ndarray:
    """The row indices of the file inside `[start, end]`, both ends inclusive."""
    days = factors.dates.astype("datetime64[D]")
    first = np.datetime64(start, "D")
    last = np.datetime64(end, "D")
    if last < first:
        raise ValueError(f"window {start}..{end} ends before it starts")
    return np.flatnonzero((days >= first) & (days <= last))


def cumulative_moves(factors: FactorPanel, start: str, end: str) -> tuple[dict[str, float], int, str]:
    """Each risk factor's compounded move over the window, and why not when not.

    Compounded rather than summed: a scenario is a held position over weeks, and
    adding daily returns overstates a fall and understates a rise. The third
    element is empty when the window was measured and names the obstacle when it
    was not -- an absent window returns no moves rather than moves of zero,
    because a scenario that silently contributes nothing is a scenario that
    always passes.
    """
    rows = window_rows(factors, start, end)
    if rows.size == 0:
        return (
            {},
            0,
            f"the factor file covers {factors.dates[0]}..{factors.dates[-1]} "
            f"and has no row in {start}..{end}",
        )
    names = risk_factors(factors)
    if not names:
        return {}, 0, f"the factor file carries no risk factor (columns: {', '.join(factors.names)})"
    columns = [factors.names.index(name) for name in names]
    block = factors.values[np.ix_(rows, columns)]
    if not np.all(np.isfinite(block)):
        holes = int(np.sum(~np.isfinite(block)))
        return (
            {},
            int(rows.size),
            f"{holes} missing factor value(s) inside {start}..{end}; a missing factor return is not a zero",
        )
    compounded = np.prod(1.0 + block, axis=0) - 1.0
    return (
        {name: float(value) for name, value in zip(names, compounded, strict=True)},
        int(rows.size),
        "",
    )


def factor_loss(betas: Mapping[str, Any], moves: Mapping[str, float]) -> tuple[float | None, str]:
    """The book's move under those factor moves, and why not when not.

    **Every factor in the file needs a beta.** A factor the book was never
    regressed on is not a zero exposure, it is an unknown one, and dropping it
    quietly is how a book with a large unmeasured exposure passes a stress test.
    So a missing or unusable beta refuses the whole scenario.
    """
    if not moves:
        return None, "no factor moves were measured for this window"
    missing = sorted(name for name in moves if as_measurement(betas.get(name)) is None)
    if missing:
        return None, (
            f"no usable beta for {', '.join(missing)}; the book was regressed on "
            f"{', '.join(sorted(betas)) or '(nothing)'} and an absent beta is not a zero one"
        )
    total = sum(float(as_measurement(betas[name])) * move for name, move in moves.items())
    return float(total), ""


def residual_band(residual_daily_vol: Any, days: int) -> float | None:
    """The unexplained scatter over `days`, as one standard deviation.

    Square-root-of-time, which assumes the residual is serially uncorrelated. In a
    crisis it is not -- residuals cluster -- so this band is narrow rather than
    generous, and that is the direction the module keeps saying out loud.
    """
    daily = as_measurement(residual_daily_vol)
    if daily is None or daily < 0.0 or days <= 0:
        return None
    return float(daily * np.sqrt(days))


def run_scenario(
    scenario: Scenario,
    factors: FactorPanel,
    betas: Mapping[str, Any],
    residual_daily_vol: Any = None,
) -> ScenarioResult:
    """One window against one book. Never raises for missing data; records it.

    An inverted window still raises: that is a wrong `SCENARIOS` entry, which is a
    code defect, and a code defect recorded as an unmeasurable scenario would sit
    in the report for months looking like a gap in the vendor's file.
    """
    moves, days, why = cumulative_moves(factors, scenario.start, scenario.end)
    if why:
        return ScenarioResult(
            key=scenario.key,
            start=scenario.start,
            end=scenario.end,
            note=scenario.note,
            days=days,
            unmeasured=why,
        )
    loss, refused = factor_loss(betas, moves)
    return ScenarioResult(
        key=scenario.key,
        start=scenario.start,
        end=scenario.end,
        note=scenario.note,
        days=days,
        factor_moves=moves,
        factor_loss=loss,
        residual_band=residual_band(residual_daily_vol, days),
        unmeasured=refused,
    )


def worst_window(factors: FactorPanel, name: str, days: int) -> tuple[float, str, str]:
    """The worst compounded move this factor has had over any `days` window, dated.

    This is the module's answer to "what if the market fell X%": X is measured and
    comes with the days it happened on, so a reader can open the file and check.
    Windows containing a missing value are skipped rather than treated as flat.
    """
    if name not in factors.names:
        raise ValueError(f"{name} is not a column of the factor file ({', '.join(factors.names)})")
    if days <= 0:
        raise ValueError("a window needs at least one day")
    column = factors.values[:, factors.names.index(name)]
    if column.shape[0] < days:
        raise ValueError(f"the factor file has {column.shape[0]} rows, too few for a {days}-day window")
    stamps = factors.dates.astype("datetime64[D]")
    worst: tuple[float, str, str] | None = None
    for start in range(column.shape[0] - days + 1):
        block = column[start : start + days]
        if not np.all(np.isfinite(block)):
            continue
        move = float(np.prod(1.0 + block) - 1.0)
        if worst is None or move < worst[0]:
            worst = (move, str(stamps[start]), str(stamps[start + days - 1]))
    if worst is None:
        raise ValueError(f"every {days}-day window of {name} contains a missing value")
    return worst


def stressed_adv_ratio(panel: PricePanel, quantile: float = TAIL_QUANTILE) -> float | None:
    """Median dollar volume on the panel's worst days over its median on all days.

    The liquidation horizon in `core/risk/exposure.py` uses recent median volume.
    Under stress that number moves, and which way is an empirical question rather
    than a multiplier to type: volume rose in 2008 and 2020 and thinned in the
    2007 unwind. So it is measured on the book's own universe, on the worst days
    the panel actually contains. `None` when the panel cannot answer, which keeps
    a caller from reading "no adjustment" as "no effect".
    """
    if panel.dollar_volume is None:
        return None
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile {quantile} is not a share of days")
    returns = panel.bar_returns
    market = np.mean(returns, axis=1)
    if market.size < 2:
        return None
    count = max(1, int(round(market.size * quantile)))
    worst_bars = np.argsort(market, kind="stable")[:count]
    # A bar `t` is earned between rows `t` and `t + 1`; the volume that traded on
    # the bad day is the volume of the closing row.
    volume = panel.dollar_volume[1:, :]
    tail = float(np.median(volume[worst_bars, :]))
    whole = float(np.median(volume))
    if not np.isfinite(tail) or not np.isfinite(whole) or whole <= 0.0:
        return None
    return tail / whole


def stress_snapshot(
    factors: FactorPanel,
    betas: Mapping[str, Any],
    residual_daily_vol: Any = None,
    panel: PricePanel | None = None,
    scenarios: Sequence[Scenario] = SCENARIOS,
    shock_horizons: Sequence[int] = (1, 5, 20),
) -> dict[str, Any]:
    """Everything a stress report quotes, measured once.

    The `unmeasured` list is the point of the return shape. A suite where nine of
    eleven windows fall outside the factor file is not a clean stress test, and a
    report that shows two results without saying so reads exactly like one.
    """
    if not scenarios:
        return {
            "scenarios": [],
            "worst_key": None,
            "worst_loss": None,
            "unmeasured": ["no scenario was supplied, so nothing was stressed"],
            "shocks": {},
            "stressed_adv_ratio": None,
            "factor_source": factors.source,
            "factor_digest": factors.digest,
            "factor_span": [str(factors.dates[0]), str(factors.dates[-1])],
        }

    results = [run_scenario(s, factors, betas, residual_daily_vol) for s in scenarios]
    measured = [r for r in results if r.measured]
    # `measured` guarantees a float, so this is a plain minimum rather than a
    # comparison that has to decide what an absent loss ranks as.
    worst = min(measured, key=lambda r: float(r.factor_loss or 0.0), default=None)

    shocks: dict[str, dict[str, Any]] = {}
    for name in risk_factors(factors):
        for horizon in shock_horizons:
            try:
                move, first, last = worst_window(factors, name, horizon)
            except ValueError as error:
                shocks[f"{name}/{horizon}d"] = {"unmeasured": str(error)}
                continue
            shocks[f"{name}/{horizon}d"] = {"move": move, "start": first, "end": last}

    return {
        "scenarios": [
            {
                "key": r.key,
                "window": [r.start, r.end],
                "note": r.note,
                "days": r.days,
                "factor_moves": r.factor_moves,
                "factor_loss": r.factor_loss,
                "residual_band": r.residual_band,
                "loss_with_residual": r.loss_with_residual,
                "unmeasured": r.unmeasured,
            }
            for r in results
        ],
        "worst_key": worst.key if worst else None,
        "worst_loss": worst.factor_loss if worst else None,
        "unmeasured": [f"{r.key}: {r.unmeasured}" for r in results if r.unmeasured],
        "shocks": shocks,
        "stressed_adv_ratio": stressed_adv_ratio(panel) if panel is not None else None,
        "factor_source": factors.source,
        "factor_digest": factors.digest,
        "factor_span": [str(factors.dates[0]), str(factors.dates[-1])],
    }


def ladder_reach(snapshot: Mapping[str, Any], limits: dict[str, Any] | None = None) -> list[str]:
    """Which rung of the existing drawdown ladder each scenario would touch.

    **Findings, not breaches.** A hypothetical loss is not a realised drawdown: a
    `Breach` here would halve a pod's capital today because 2008 happened, and the
    ladder in `limits.yaml` is written about what the book has actually lost. The
    owner reads these lines and decides whether `limits.yaml` should grow a
    `stress:` block (CLAUDE.md 3항 -- this module may not add one).

    A scenario that could not be measured produces a line saying so. Silence
    would be the failure: a stress report is read as "we looked", and a window
    nobody could evaluate is a window nobody looked at.
    """
    ladder = (limits or load_limits()).get("pod", {}).get("drawdown", {})
    rungs = sorted(
        (
            (str(name), float(level["abs"]), str(level.get("action", "")))
            for name, level in ladder.items()
            if isinstance(level, Mapping) and as_measurement(level.get("abs")) is not None
        ),
        key=lambda rung: rung[1],
    )
    findings: list[str] = []
    if not rungs:
        findings.append("pod.drawdown carries no readable level, so no scenario can be placed on the ladder")
    for row in snapshot.get("scenarios", []):
        key = row.get("key", "(unnamed)")
        if row.get("unmeasured"):
            findings.append(f"{key}: not measured -- {row['unmeasured']}")
            continue
        # The residual-widened loss is what the rung is compared against: the
        # factor loss alone is the optimistic half of the estimate.
        bare = as_measurement(row.get("factor_loss"))
        loss = as_measurement(row.get("loss_with_residual"))
        if loss is None:
            loss = bare
        if loss is None or not rungs:
            continue
        touched = [rung for rung in rungs if loss <= rung[1]]
        if not touched:
            continue
        name, level, action = touched[0]
        # **Which half reached the rung is the finding's real content.** The band
        # grows with the square root of the window, so over a year-long scenario it
        # reaches a rung for any book with ordinary idiosyncratic volatility. The
        # comparison stays conservative -- capping the band would be inventing a
        # number -- and the line says when the scenario itself is not what got
        # there, the same way the allocator's clamp is silent but its note is not.
        driver = ""
        if bare is not None and bare > level:
            driver = f"; the residual band, not the {bare:+.4f} factor move, is what reaches it"
        findings.append(
            f"{key}: an estimated {loss:.4f} reaches pod.drawdown.{name} ({level:+.4f}, "
            f"action {action or 'unspecified'}){driver}"
        )
    return findings
