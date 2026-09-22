# ADR-0005 — The feature catalogue, and why its threshold is not a risk limit

- Date: 2026-09-22
- Status: accepted
- Context: P1. CLAUDE.md rule 6 ("포드 전용 피처를 만들지 않는다") and the v3
  decision to share code across pods (Renaissance's single codebase) had no code
  behind them. `core/features/` was an empty package, so "the feature catalogue"
  was a convention, and a convention cannot reject anything.
- Authorised by: the owner's standing delegation ("너가 판단한대로 해, 근데 어떠한
  판단에는 항상 근거가 있어야 해", 2026-09-22). `core/risk/limits.yaml` is not
  touched by this change — see below for why that is deliberate.

## Decision

`core/features/catalog.py` holds `FeatureCatalog`, and registration is where
three properties are checked rather than reviewed:

1. **Point-in-time.** `register` runs the G0 look-ahead scan on the feature
   itself. A feature whose value at `t` changes when later data is withheld is
   rejected as `Rejection.LEAKAGE`. Catching it here costs one registration;
   catching it at G0 throws away every alpha already built on it.
2. **Not a duplicate.** A feature whose **rank** correlation with an already
   registered one exceeds 0.90 is rejected as `Rejection.DUPLICATE`, naming the
   feature it duplicates. Spearman, not Pearson: a log, a winsorisation or a
   rank of an existing feature is the same idea wearing a different scale, and
   Pearson is happy to call it new. Two names for one signal inflate the apparent
   breadth of the book and quietly double an exposure.
3. **Finite everywhere.** A NaN warm-up is rejected as `Rejection.NON_FINITE`
   with the instruction to state an explicit warm-up value. A NaN propagates into
   weights, and a NaN weight is a position nobody chose.

Both admissions and refusals are appended to the hash-chained audit log with the
owner, the source and the measured correlations. A refusal nobody can see is a
refusal that did not happen.

`owner` on a `Feature` is provenance for cost attribution and for asking
questions later. It is **not** access control: `values()` takes no owner argument
and there is no private view. That is rule 6 expressed as an API rather than as a
paragraph.

The reference panel is supplied once, at construction. Correlation only means
something relative to the same data, so a contributor cannot register against
data chosen to make their feature look independent.

## Why the 0.90 threshold is not in limits.yaml

The obvious place for a number like this is the limit table. It is the wrong
place:

- `core/risk/limits.yaml` is the pre-trade and gate threshold table. Everything
  in it is read by the path that blocks orders. A catalogue de-duplication rule
  is read by nothing in that path.
- Changing that file requires owner approval and an ADR (ADR-0003) and is
  enforced by CI. Putting a research-hygiene knob there would make routine
  research changes look like risk-limit changes, which trains everyone to treat
  a limit change as routine. That is the exact erosion the guard exists to stop.
- Keeping the threshold in the catalogue means a **wider catalogue can never be
  obtained by editing the risk limits**, and a per-catalogue override
  (`max_abs_correlation=...`) can tighten it for a specific research question
  without an ADR.

It is still a rule, not a preference: `MAX_ABS_CORRELATION` is a module constant
with a comment, and changing it is a code change that shows up in review.

## Alternatives rejected

- **Pearson correlation.** Rejected: it admits monotone restatements, which is
  the most common way a duplicate feature gets in.
- **Reject on the feature's correlation with returns instead of with other
  features.** Rejected: that is an alpha test, and it belongs to the gates. The
  catalogue's question is "is this a new idea", not "is it a good one".
- **Warn on a duplicate instead of rejecting.** Rejected: a warning in a research
  loop is read once and never again.
- **Let a pod hold a feature privately until it passes a gate.** Rejected
  explicitly by rule 6, and it is how a platform ends up with N copies of
  momentum and no idea of its true exposure.
- **Persist the catalogue to `registry/features/*.yaml` now.** Deferred, not
  rejected: the audit log already records every registration, and a file format
  invented before there is a real data pipeline will be wrong. The in-memory
  catalogue plus the audit trail is what P1 needs.

## Verification

`tests/features/test_catalog.py` (14 tests). Each refusal has its own test; the
load-bearing ones are
`test_a_feature_that_reads_the_future_is_refused_at_registration` (a feature
returning tomorrow's return never enters the catalogue),
`test_a_monotone_restatement_is_a_duplicate` (the log of an accepted momentum
feature is refused and told which feature it duplicates),
`test_every_pod_reads_every_feature`, and
`test_registrations_and_refusals_are_both_audited`, which also verifies the hash
chain after a refusal.

Full suite at the time of this decision: 103 passed, ruff clean, on Python 3.12.
