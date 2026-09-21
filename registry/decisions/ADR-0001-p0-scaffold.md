# ADR-0001 — P0 scaffold, public repository

Date: 2026-09-21 · Decided by: 백승주 · Recorded by: Claude

## Decision

The quant-desk monorepo is created as a **public** GitHub repository, and P0
ships the skeleton: agent definitions, the deterministic control path
(audit chain, limit engine, order state machine, daily pipeline), the canary
contract, and CI.

## Consequence

- The plan's earlier default of a private repository no longer holds. Strategy
  code, gate thresholds and risk limits are world-readable.
- Therefore: no credential, no licensed market data and no position or P&L
  file may ever be committed. Keys live in the environment (`.env.example`
  lists them); `state/` and data files are git-ignored.
- An alpha that depends on secrecy to work is not a good fit for this
  repository. Nothing in P0-P5 depends on secrecy, and the gates are
  deliberately public: their value is that they are hard to pass, not that
  they are hidden.

## Alternatives considered

A private repository (the original plan default) keeps strategy detail closed
but costs nothing else. The owner chose public; the mitigations above are the
price of that choice.
