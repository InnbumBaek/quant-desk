# ADR-0002 — Gate engine, and the deflated-Sharpe threshold fix

Date: 2026-09-22 · Decided by: 백승주 (authorised Claude to decide, with a recorded basis) · Recorded by: Claude

## Decision

`core/backtest/` now computes the G0-G6 verdicts that `limits.yaml` describes:
the point-in-time look-ahead scan (G0), purged K-fold with an embargo (G3), and
the deflated Sharpe, PBO, block bootstrap, factor-residual t and
drawdown-to-return ratio that G4 reads. The four canary alphas are implemented
against it and their `strict-xfail` marks are removed, so the suite now proves
the gates reject what they were built to reject rather than asserting it.

`core/risk/limits.yaml` was changed once, in the `gates:` block:

```
- deflated_sharpe_min: 0.0
+ deflated_sharpe_probability_min: 0.95   # Bailey & Lopez de Prado (2014)
+ bootstrap_pvalue_max: 0.05              # block bootstrap, one-sided
+ dd_to_return_ratio_max: 2.0             # max drawdown / annualised return
```

Every enforced threshold is read from the table;
`tests/backtest/test_gates.py::test_every_enforced_threshold_comes_from_the_limits_table`
fails if any gate grows its own. G4 now enforces all five documented criteria.

## Basis (why the change was necessary, not merely tidy)

The deflated Sharpe ratio of Bailey & Lopez de Prado (2014) is a **probability**
in [0, 1]. Read literally, the old `deflated_sharpe_min: 0.0` passed every
strategy ever submitted, including a coin flip, because a probability is never
negative. So the single most important G4 criterion was, as written, no
criterion at all. `0.95` is the authors' own rule: significance at the standard
95% confidence that true Sharpe exceeds the luck-of-N-trials benchmark SR0.

The block-bootstrap p-value and the drawdown-to-return ratio were documented in
the alpha-gate skill but had **no key** in the table, so they could not be
enforced at all. They were computed and reported before this change; now they
are gated at the values the skill already named (`p < 0.05`, ratio `<= 2.0`).

## Authorisation

CLAUDE.md rule 3 makes `limits.yaml` an owner-approval change, precisely so gate
strictness is never decided by whoever is editing a module that day. On
2026-09-22 the owner reviewed the recommendation (posted in this thread with the
reasoning above) and authorised Claude to proceed, on the standing condition
that every judgement carry a recorded basis — which this ADR is.

The change makes the gates **stricter**, never looser: it turns a dead criterion
into a live one and enforces two that were previously only observed. The canary
suite was re-run and still rejects all four fakes by their designated gates
(53 tests pass). A future change that *loosened* any of these would have to
re-pass the same suite.

## Consequence

- New runtime dependencies: `numpy` and `scipy`.
- `canary_overfit` also fails G3, not only G4. That is correct - an overfitted
  alpha does collapse out of sample - so the canary asserts the G4 rejection
  names PBO rather than asserting G4 is its only failure.
- `canary_factor` is now caught by both PBO and the residual t; the contract
  test pins the residual t, which is the defect it was built around.

## Alternatives considered

Enforcing the excess `observed Sharpe - SR0 > 0` instead of the probability.
Rejected: excess `> 0` corresponds to probability `> 0.5`, i.e. a coin-flip
confidence, far weaker than the 95% the source prescribes. The excess is kept
only as a reported effect size (`deflated_sharpe_excess` in `G4.metrics`).
