# ADR-0004 — The backtest runner, and what it refuses to do

- Date: 2026-09-22
- Status: accepted
- Context: P1. `core/backtest/gates.py` (ADR-0002) can judge a `Submission`, but
  nothing produced one from market data, so every submission packet was
  hand-assembled. That left the honest parts of the protocol — the trial count,
  the point-in-time convention, the cost drag — as things a caller chose rather
  than properties of the code.
- Authorised by: the owner's standing delegation ("너가 판단한대로 해, 근데 어떠한
  판단에는 항상 근거가 있어야 해", 2026-09-22). `core/risk/limits.yaml` is not
  touched by this change.

## Decision

`core/backtest/engine.py` runs a strategy over a validated price panel and
returns `(Submission, LeakReport, RunReport)`. Five things are structural, not
optional:

1. **Alignment.** `weights[t]` earns `bar_returns[t]`, the return of bar
   `t -> t+1`. The last weight row is never paid, because it is a decision about
   a bar the panel does not contain. This lives in one function, `simulate`, and
   is pinned by `test_weight_earns_the_following_bar_not_the_preceding_one`.
   Rationale: off-by-one alignment is the most common way a backtest invents a
   Sharpe, and it is invisible in aggregate statistics.
2. **The trial count is what was run.** Every grid point is simulated and becomes
   a column of `trial_returns`. A caller cannot search a 5,000-point grid and
   submit the winner as N = 1, because the submitted series *is* one column of
   the grid it came from. The deflated Sharpe and PBO are only as honest as N.
3. **Gross leverage is checked before anything is simulated**, against
   `pod.gross_leverage_max` in `limits.yaml`. A strategy above the limit is
   refused, never rescaled: a rescaled strategy is one nobody tested.
4. **Absence is recorded as absence.** With no factor returns, G4's residual
   alpha runs against an equal-weight panel proxy and `RunReport.factor_source`
   says `PANEL_PROXY`, with a note that the t-statistic is provisional. With no
   volume panel, ADV participation is `0.0` *and* a note says it is zero by
   absence. A silent zero reads as "no capacity problem", which is a lie the
   engine will not tell.
5. **ADV participation uses the decision bar's volume**, at the 95th percentile
   of traded shares. Using the execution bar's volume would be look-ahead inside
   the capacity estimate; the mean would hide the days that move the market and
   the maximum would be one bad print.

## Two consequences worth naming

**G4 now fails a single-configuration submission.** CSCV needs at least two
configurations to ask which one wins, so PBO is not computable from one. The
previous code raised `ValueError` from `stats.py` and took the gate run down with
it. A criterion that cannot be measured now **fails**, with the reason
"PBO is not computable from a single configuration; submit the grid that was run
(G1)" — the same fail-closed rule the data-health report and the financing check
already use. This tightens the gate; it does not loosen one. A hand-picked single
parameter set is precisely the case PBO exists to catch, so it may not pass by
being too thin to measure.

**G5 drops infeasible neighbours.** Perturbing the parameters of a strategy
already at 1.5x gross produces neighbours above the pod limit. Those are not
simulated, and the count is recorded in `RunReport.notes`. Simulating them would
report a robustness number measured on a book the pod could not run, which is
worse than a smaller neighbourhood honestly described.

## Alternatives rejected

- **Rescale an over-levered strategy to the limit.** Rejected: it submits a
  strategy that was never tested, under the name of one that was.
- **Let the engine require a grid of at least two points.** Rejected: it reports
  a gate criterion as an engine limitation. The verdict belongs in G4, where the
  reason is readable and audited.
- **Default to a hard-coded factor model when none is supplied.** Rejected: a
  fabricated FF5+MOM would make the residual-alpha t-statistic look authoritative.
  A named proxy with a warning is worse-looking and more truthful.
- **Charge costs as a flat drag on returns.** Rejected: a flat drag cannot
  distinguish a 5%-turnover strategy from a 300% one, which is most of what the
  cost model is for.

## Verification

`tests/backtest/test_engine.py` (24 tests). The load-bearing ones:
the alignment test above; `test_lookahead_strategy_is_profitable_and_only_g0_catches_it`,
which proves the engine does not launder leakage into a failing Sharpe — a
strategy trading tomorrow's sign makes money, passes G2 through G6, and is
stopped by G0 alone; `test_every_grid_point_becomes_a_trial_column`;
`test_absent_inputs_are_recorded_as_absent_not_as_zero_risk`; and
`test_panel_rejects_data_no_gate_could_interpret`.

`core/backtest/leakage.py` was generalised in the same change: a signal may now
be a vector per row (a weight per symbol), and a probe compares every element and
keeps the largest deviation, so leakage in a single name cannot average away.

Full suite at the time of this decision: 89 passed, ruff clean, on Python 3.12.
