# S1-13 Long-Run Learner Phase Schedule ADR

- Status: accepted before the first formal 9N or long-run submission
- Scope: S1-13 non-formal long-profile smoke and formal M=4 long run only
- Requirements: INV-01, INV-08, PROG-02, TEL-02, TEL-03

## Context

Profile A requires `Q=M=4`, `Q_fresh=M`, `S_max=0`, and `H=50`.  The asset-stage
default rotated each learner's fragment phase by `floor(learner_index*H/M)`.  In
the real five-node Pythia/FineWeb-Edu correction runs, that rotation made every
fragment's complete quorum arrive near local phase 43/44 while the earliest next
learner publication arrived at phase 5/6/7.  The CPU-only syncer therefore had
only 11--13 local steps, rather than H=50 steps, to commit and expose each fresh
fragment before one learner overwrote its fixed slot.

The preserved correction evidence is
`runtime_runs/S1-13/long-smoke-20260715T151024Z-3cd45bf` and
`runtime_runs/S1-13/long-smoke-20260715T152016Z-41d50f1`.  The latter recorded
approximately 2.0--2.2 seconds per real 160M fragment update while learners ran
at the then-frozen 0.15 second step floor; only two global cycles were complete
after roughly six publication opportunities.  This cannot satisfy the formal
2400-of-2441 cycle margin.  Increasing the uniform step floor enough to cover
the burst would exceed the frozen 21,600-second active budget.

## Decision

For `long_run` only, use `aligned_zero_for_q_equals_m_capacity_v1`:

- retain the byte-weighted fragment offsets `[6, 18, 32, 44]` and H=50;
- set every learner phase offset to zero, so all four learners use those same
  per-fragment offsets;
- keep learners clock-paced independently; no learner polls or waits for a
  syncer update, quorum, or peer;
- retain the exact model, data, token, optimizer, outer-update, and bounded
  storage contracts.

This spreads complete fragment quorums across the H=50 interval and gives each
fragment a full H interval before its next fixed-slot publication.  The tracked
resolved long config, gate contract, analyzer, and role evidence all assert the
overlay.  The five-node same-profile smoke must still demonstrate at least 59
global cycles from 60 publication opportunities before a formal long run is
admissible.

The 9N GPT-2/WikiText-2 baseline retains its original per-learner phase rotation.

## Consequences and rollback

The long run trades cross-learner write staggering for an attainable fresh-quorum
schedule; fragment writes remain spread by the four byte-weighted fragment
offsets.  Mutable state remains fixed-slot and bounded.  If the preflight smoke
does not reach 59/60, this decision is not widened or adjusted after observation:
the formal run remains blocked and the next action is a smaller syncer/storage
capacity correction.  Rollback is the prior resolved offset matrix plus removal
of the gate overlay, which also invalidates any package carrying this ADR's
config and gate-contract identities.
