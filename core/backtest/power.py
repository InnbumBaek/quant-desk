"""What Sharpe must an alpha show to clear the gates, given the data we have?

Every gate in `core/backtest/gates.py` can refuse. None of them can say whether
refusing is all they will ever do. With 735 bars and five instruments, a gate
battery that demands a deflated-Sharpe probability of 0.95 may be arithmetically
impossible to clear no matter how good the strategy is -- and a desk that cannot
tell the difference between "no alpha passed" and "no alpha could pass" will keep
researching against a wall and call it discipline.

So this module inverts the gates. Given a sample length and a trial count it
returns the **minimum annualised Sharpe** each statistical criterion admits, and
which one binds. Nothing here changes a gate or a threshold; it reads
`limits.yaml` and reports.

**The three criteria that can be inverted in closed form**

- `is_sharpe_min` (G2) is already an annualised Sharpe, so it is its own answer
  and does not depend on the sample at all. That is worth seeing next to the
  others: it is the only criterion that does not get harder on short data.
- `deflated_sharpe_probability_min` (G4) compares the observed Sharpe against
  SR0, the Sharpe the best of N trials buys for free, and scales the gap by
  `sqrt(n_obs - 1)`. Inverting it is a quadratic.
- `bootstrap_pvalue_max` (G4) is, for a series with no autocorrelation, the
  one-sided test that the mean is positive: `s * sqrt(n) >= z(1 - p)`. The block
  bootstrap this desk actually runs is weaker than that on autocorrelated data,
  so the closed form is a **floor** on what the bootstrap will demand, not an
  equivalent of it.

**Two criteria are deliberately not inverted.** PBO is a property of the trial
matrix's shape rather than of one series, and the factor-residual t depends on
which factors the return series loads on. A number for either would be a number
about an assumed strategy, not about the gate, and this desk does not report those
(CLAUDE.md 2).

**The assumption, and what breaking it actually does.** The closed forms take
returns as iid normal: skew 0, kurtosis 3, no autocorrelation. Rather than assume
which way each departure pushes, the effect of each was measured through the real
`core.backtest.stats` functions at 735 observations and 6 trials, and the answers
were not all what the algebra suggested:

- **Fat tails barely matter here.** Student-t with 4 degrees of freedom moves the
  deflated-Sharpe probability by about -0.004. Kurtosis enters the denominator
  multiplied by the squared *per-period* Sharpe, which is around 0.01, so the term
  is a rounding error at daily frequency.
- **Skew matters, and it is not conservative.** Strong negative skew costs about
  -0.050, but strong *positive* skew gains about +0.025: a lottery-shaped return
  series clears this criterion at a lower Sharpe than a normal one. So the
  deflated-Sharpe figure below is **not** a universal lower bound -- a positively
  skewed strategy can pass under it. That is a property of the criterion, not of
  this module, and it is recorded rather than corrected (ADR-0033).
- **Autocorrelation is where the understatement is large.** The block bootstrap
  widens the null on dependent data: at the stated minimum Sharpe an iid series
  gets p = 0.031, an AR(1) series with rho 0.3 gets p = 0.157, five times worse.
  So `sharpe_for_bootstrap_pvalue` is a real floor and a loose one.

`tests/backtest/test_power.py` asserts each of those directions rather than
trusting the prose, and checks the deflated-Sharpe algebra by running the gate on
series built at exactly the Sharpe it names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats as sps

from core.backtest.stats import EULER_MASCHERONI
from core.risk.limits import load_limits

#: Trials counted the way G1 pre-registration counts them: every configuration
#: tried, not the ones that survived. The smoke runs use 6.
DEFAULT_TRIALS = 6


@dataclass(frozen=True)
class Requirement:
    """One criterion's minimum annualised Sharpe, or None when it has none."""

    criterion: str
    annualised_sharpe_min: float | None
    depends_on_sample: bool
    note: str = ""


def expected_max_sharpe(n_obs: int, n_trials: int) -> float:
    """SR0 in per-period units: the Sharpe the best of `n_trials` buys for free.

    `core.backtest.stats._expected_max_sharpe` takes the *observed* dispersion of
    the trials it was given. Here there are no trials yet, so the dispersion is
    the one a set of worthless trials would have: a per-period Sharpe estimated
    over `n_obs` observations of zero-mean iid returns has standard deviation
    `1 / sqrt(n_obs)`. Same formula, theoretical dispersion.
    """
    if n_trials < 2 or n_obs < 2:
        return 0.0
    dispersion = 1.0 / math.sqrt(n_obs)
    gamma = EULER_MASCHERONI
    quantile_a = sps.norm.ppf(1.0 - 1.0 / n_trials)
    quantile_b = sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(dispersion * ((1.0 - gamma) * quantile_a + gamma * quantile_b))


def sharpe_for_deflated_probability(
    n_obs: int,
    n_trials: int,
    probability_min: float,
    periods_per_year: int = 252,
) -> float | None:
    """The annualised Sharpe at which G4's deflated-Sharpe probability reaches the floor.

    Solves `(s - SR0) * sqrt(n - 1) = z(p) * sqrt(1 + s^2 / 2)` for `s`, which is
    `deflated_sharpe_ratio` with skew 0 and kurtosis 3 substituted in. Returns
    None when the sample is too short for the question to have an answer.
    """
    if n_obs < 3 or not 0.0 < probability_min < 1.0:
        return None
    sr0 = expected_max_sharpe(n_obs, n_trials)
    a2 = float(n_obs - 1)
    z = float(sps.norm.ppf(probability_min))
    if z <= 0.0:  # a floor at or below 0.5 is cleared by any positive gap
        return float(sr0 * math.sqrt(periods_per_year))

    # s^2 (a2 - z^2/2) - 2 a2 SR0 s + (a2 SR0^2 - z^2) = 0
    quad = a2 - z * z / 2.0
    lin = -2.0 * a2 * sr0
    const = a2 * sr0 * sr0 - z * z
    if quad <= 0.0:  # pragma: no cover - needs n_obs <= z^2/2 + 1, i.e. a handful of bars
        return None
    discriminant = lin * lin - 4.0 * quad * const
    if discriminant < 0.0:  # pragma: no cover - impossible for quad > 0, const < a2*SR0^2
        return None
    root = (-lin + math.sqrt(discriminant)) / (2.0 * quad)
    return float(root * math.sqrt(periods_per_year))


def sharpe_for_bootstrap_pvalue(
    n_obs: int,
    pvalue_max: float,
    periods_per_year: int = 252,
) -> float | None:
    """A floor on the annualised Sharpe G4's bootstrap will demand.

    The iid one-sided test needs `s * sqrt(n) >= z(1 - p)`. The circular block
    bootstrap the desk runs resamples blocks, which widens the null distribution
    on autocorrelated data, so it demands at least this much and usually more.
    """
    if n_obs < 3 or not 0.0 < pvalue_max < 1.0:
        return None
    z = float(sps.norm.ppf(1.0 - pvalue_max))
    return float(z / math.sqrt(n_obs) * math.sqrt(periods_per_year))


def requirements(
    n_obs: int,
    n_trials: int = DEFAULT_TRIALS,
    limits: dict[str, Any] | None = None,
    periods_per_year: int = 252,
) -> list[Requirement]:
    """Every invertible criterion's minimum annualised Sharpe, worst last."""
    gates = (limits or load_limits())["gates"]
    found = [
        Requirement(
            "G2_in_sample/is_sharpe_min",
            float(gates["is_sharpe_min"]),
            depends_on_sample=False,
            note="already an annualised Sharpe; the only criterion a short sample does not tighten",
        ),
        Requirement(
            "G4_statistics/deflated_sharpe_probability_min",
            sharpe_for_deflated_probability(
                n_obs, n_trials, float(gates["deflated_sharpe_probability_min"]), periods_per_year
            ),
            depends_on_sample=True,
            note=f"against SR0 for {n_trials} trial(s) over {n_obs} observation(s)",
        ),
        Requirement(
            "G4_statistics/bootstrap_pvalue_max",
            sharpe_for_bootstrap_pvalue(n_obs, float(gates["bootstrap_pvalue_max"]), periods_per_year),
            depends_on_sample=True,
            note="iid floor; the block bootstrap demands at least this much",
        ),
    ]
    return sorted(found, key=lambda r: (r.annualised_sharpe_min is None, r.annualised_sharpe_min or 0.0))


def binding(reqs: list[Requirement]) -> Requirement | None:
    """The criterion that decides, which is the largest requirement."""
    measured = [r for r in reqs if r.annualised_sharpe_min is not None]
    return max(measured, key=lambda r: r.annualised_sharpe_min or 0.0) if measured else None


def observations_for(
    target_sharpe: float,
    n_trials: int = DEFAULT_TRIALS,
    limits: dict[str, Any] | None = None,
    periods_per_year: int = 252,
    ceiling: int = 252 * 40,
) -> int | None:
    """How many observations a true annualised Sharpe of `target_sharpe` needs.

    The inverse question, and the one that says whether waiting for data is the
    answer: every sample-dependent requirement falls as the sample grows, so there
    is a length at which the battery stops being the binding constraint. Returns
    None when `target_sharpe` is below a criterion no sample size can fix -- G2's
    floor does not move, so a target under `is_sharpe_min` never clears.
    """
    if target_sharpe <= 0.0:
        return None
    lo, hi = 3, int(ceiling)
    worst_at_hi = binding(requirements(hi, n_trials, limits, periods_per_year))
    if worst_at_hi is None or (worst_at_hi.annualised_sharpe_min or 0.0) > target_sharpe:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        worst = binding(requirements(mid, n_trials, limits, periods_per_year))
        if worst is not None and (worst.annualised_sharpe_min or 0.0) <= target_sharpe:
            hi = mid
        else:
            lo = mid + 1
    return lo


def monte_carlo_deflated_probability(
    per_period_sharpe: float,
    n_obs: int,
    n_trials: int,
    draws: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """The real `deflated_sharpe_ratio` on synthetic data, for checking the algebra.

    The trials here are worthless by construction -- zero-mean noise, with the
    selected series drawn separately -- which is exactly the dispersion
    `expected_max_sharpe` assumes. That makes this a check on the algebra and not
    on the practice: in a real submission the selected configuration is *one of*
    the N, so a good strategy raises the observed trial dispersion, raises SR0,
    and makes the gate harder than these numbers say. One more reason every figure
    this module reports is the optimistic end.

    Kept in the module rather than the test file because it is the verification
    the closed forms are only trustworthy with, and a reader of `requirements`
    should be able to find it.
    """
    from core.backtest.stats import deflated_sharpe_ratio

    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for i in range(draws):
        returns = rng.normal(per_period_sharpe, 1.0, size=n_obs)
        trials = rng.normal(0.0, 1.0, size=(n_obs, n_trials))
        trial_sharpes = np.array(
            [float(np.mean(trials[:, j]) / np.std(trials[:, j], ddof=1)) for j in range(n_trials)]
        )
        out[i] = deflated_sharpe_ratio(returns, trial_sharpes)
    return out
