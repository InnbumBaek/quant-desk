# ADR-0006 — Data snapshots from files, and enforcing the reproducibility pin

- Date: 2026-09-22
- Status: accepted
- Context: P1. `core/backtest/engine.py` (ADR-0004) and
  `core/features/catalog.py` (ADR-0005) both take a validated `PricePanel` and
  neither could produce one, so the remaining gap was the data. Separately,
  `.claude/skills/backtest-protocol/SKILL.md` requires every artefact to be
  sealed with (git SHA, data snapshot id, seed) and says an artefact missing any
  of the three may not be gate input — a rule with nowhere to live.
- Authorised by: the owner's standing delegation ("너가 판단한대로 해, 근데 어떠한
  판단에는 항상 근거가 있어야 해", 2026-09-22).

## The constraint that shaped this

This session's environment denies outbound network access to market-data hosts
(a `stooq.com:443` request is rejected by the egress proxy under organization
policy). So there is no vendor client here, and writing one would be untestable
fiction. The owner can change the network policy when creating the environment;
until then, data arrives as files.

That turned out to be the smaller half of the problem. What a research run needs
is not a fetcher but **a panel plus a name for the exact bytes it came from**: a
backtest whose data cannot be identified cannot be reproduced, and an
unreproducible Sharpe is a rumour.

## Decision

**`core/data/sources.py`** reads one daily CSV per symbol (the shape every vendor
exports) and returns `(PricePanel, SnapshotManifest)`. `snapshot_id` is a digest
of the digests of the files read, so the same bytes always produce the same id
and one changed price produces a different one.

- **Dates are intersected, not filled.** A symbol missing a day is a hole, and
  forward-filling a hole invents a price that never traded. The intersection is
  what every symbol really has, and `dates_dropped` records the cost.
- **A hole too big to ignore is refused.** Below `min_coverage` (default 98%) no
  panel is returned, because a panel missing a meaningful share of its history
  answers a different question than the one asked.
- **A missing file is recorded, not guessed at.** `manifest.missing` keeps the
  requested-but-absent symbols, which is where a survivorship hole becomes
  visible to a reader six months later.
- **A manifest may not be silently rewritten.** Writing a different claim under
  an existing snapshot id raises, because one id meaning two things is worse than
  no id. The comparison ignores `created_at`: re-reading the same files is the
  same claim, and the read time is not part of it.

**`core/repro.py`** holds `ReproPin(git_sha, snapshot_id, seed)` with a derived
`run_id`, and enforces the skill's rule:

- A pin cannot be constructed with a blank leg.
- `pin_current` **refuses a dirty working tree.** A pin taken on uncommitted code
  names a commit that does not describe the code that ran, which is worse than no
  pin: it looks reproducible and is not. Untracked files count as dirty, because
  an untracked strategy file is exactly what a pin would fail to describe.
- `allow_dirty=True` exists for a scratch run, records `dirty=True`, and
  `gate_input_ok` then returns `False` with the reason. So a scratch number can
  be produced but cannot enter the gate pipeline wearing a commit hash.
- Every pin is appended to the hash-chained audit log.

## Alternatives rejected

- **Write a Polygon/vendor client now.** Rejected: the network policy blocks it,
  so it could not be run or tested, and an untested data client is where silent
  survivorship and as-of bugs live.
- **Forward-fill missing dates.** Rejected: it invents prices, and the invented
  ones cluster exactly where a strategy is most likely to have traded.
- **Use the file's modification time or path as the snapshot id.** Rejected: both
  change without the data changing, and neither changes when the data does.
- **Make the pin advisory (a warning on a dirty tree).** Rejected: the protocol
  already said the rule; what was missing was the refusal.
- **Store the snapshot id inside `Submission`.** Deferred: the pin is about the
  whole run, not one packet, and `RunReport` plus the audit record already tie
  them together. Worth revisiting when `registry/alphas/<id>.yaml` is written.

## Verification

`tests/data/test_sources.py` (17 tests) and `tests/test_repro.py` (7 tests).
The load-bearing ones: `test_dates_are_intersected_and_the_drop_is_recorded`,
`test_a_hole_too_big_to_ignore_is_refused`,
`test_one_changed_price_changes_the_snapshot_id`,
`test_rewriting_a_manifest_with_different_content_is_refused`,
`test_a_dirty_tree_is_refused`, and
`test_a_scratch_run_may_pin_dirty_but_is_not_gate_input`.

Full suite at the time of this decision: 127 passed, ruff clean, on Python 3.12.
