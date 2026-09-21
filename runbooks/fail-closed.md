# Runbook — fail-closed

Triggered when the 06:30 data-health check fails, a reconciliation break is
open, an unresolved order exists, or a pre-trade limit check returns a breach.

1. The pipeline refuses to produce an order file. No new positions, closing
   trades only. This is automatic; no agent decides it.
2. `data-quality` (or `pnl-recon`) posts what failed, with the run id of the
   artifact that carries the numbers.
3. Classify the cause with the loss-attribution skill: data error, code bug,
   execution, factor shock or alpha decay.
4. Fix, re-run the check, and record the outcome in the audit log.
5. `risk-officer` approves resuming. If the cause was a data error or a code
   bug, the affected alpha re-enters at G3 or G5 — see the alpha-gate skill.

Do not work around the gate. Widening a limit to clear a red check requires
human approval and its own audit record.
