# Runbook — crash during the trading day

The broker is the source of truth for positions. The internal ledger is a
cache, and when they disagree the broker wins and `pnl-recon` records a break.

1. Query the broker for positions, cash and open orders.
2. Load the order state machine from `state/orders/` and diff it against the
   broker's open orders.
3. Cancel anything unresolved. Client order ids are deterministic, so a
   resubmission cannot duplicate an order — but a partially filled order is
   resolved first, never re-sent.
4. Recompute the difference between target and actual positions. Do not replay
   the original order file.
5. `risk-officer` approves resumption; the day continues, or ends liquidate-only
   if the health checks cannot be re-established.
