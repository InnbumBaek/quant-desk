# Canary alphas

Four fake alphas that must be rejected by the gates. They run on every CI
build; if any of them passes, the build fails and no new alpha may be approved
that day. Changing a gate threshold means re-passing this suite.

| canary | what it is | gate that must reject it |
|---|---|---|
| `canary_random` | pure random signal | G2 or G4 (deflated Sharpe ≤ 0) |
| `canary_lookahead` | next-day return mixed in at 1% | G0 look-ahead scan |
| `canary_overfit` | best of 5,000 parameter combinations | G4 (PBO > 5%) |
| `canary_factor` | a clone of the momentum factor | G4 residual alpha t < 3.0 |

P0 ships the contract in `test_canaries.py`; each canary is implemented in P1
alongside the backtest engine it needs. The tests are marked `xfail(strict)`
until then, so the day the engine lands, a canary that silently passes turns
the build red instead of going unnoticed.
