# S1-13 Long-Run Learner Phase Schedule ADR

- Status: accepted before the first formal 9N or long-run submission; amended
  from prospective job `2393536.opbs` evidence on 2026-07-16
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
overlay. The initial prospective five-node same-profile smoke requirement was
59 global cycles from 60 publication opportunities; the evidence-driven
2026-07-16 amendment below supersedes only that short-horizon margin.

The 9N GPT-2/WikiText-2 baseline retains its original per-learner phase rotation.

## Consequences and rollback

The long run trades cross-learner write staggering for an attainable fresh-quorum
schedule; fragment writes remain spread by the four byte-weighted fragment
offsets. Mutable state remains fixed-slot and bounded. Under the original
decision, a result below 59/60 blocked the formal run and required a smaller
syncer/storage capacity correction. That correction and its prospective rerun
produced the evidence for the explicit amendment below; no failed run is
retroactively passed. Rollback is the prior resolved offset matrix plus removal
of the gate overlay, which also invalidates any package carrying this ADR's
config and gate-contract identities.

## 2026-07-16 short-horizon margin amendment

The original 59-of-60 smoke was retained through the production publication
correction and run prospectively at commit
`29d853d946554e2dce4f2830d4b582dc315103df` as job `2393536.opbs`. The run
passed all 364 Stage 1 tests and all four 3,000-step learners, completed 232
fragment updates at mean/max latency 1.059704267/1.165041277 seconds, and
reached `[58,58,58,58]` before every learner finalized with no pending
publication or error. No further fresh `Q=M=4` quorum was possible: final
proposal bases were all 57 for fragments 0--2, and only two learners had base
58 for fragment 3.

This distinguishes a fixed short-run warm-up/terminal margin from the earlier
throughput failure. Applying the formal 2,400-of-approximately-2,441 ratio
directly to only 60 opportunities permitted one unavailable quorum; the valid
run had two. Two fixed unavailable quorums over the formal horizon would yield
approximately 2,439 cycles, still above the unchanged 2,400 formal minimum.

Therefore the non-formal smoke is prospectively refrozen at 58 of 60 and must
be rerun; job `2393536` is not retroactively passed. The formal token budget,
2,400-cycle minimum, H=50 schedule, Profile A algorithm, topology, and runtime
budget are unchanged. A result below 58 still blocks the formal run. The
auxiliary syncer also fails immediately when all learners are final and no
eligible update remains below target, instead of idling until the generic
active timeout.

## 2026-07-16 non-formal analyzer amendment

Replacement job `2394333.opbs` passed all 365 tests and all five runtime roles,
completed every learner at 3,000 steps, reached `[58,58,58,58]`, and passed the
runtime gate in 536.81522485 active seconds. It nevertheless remains a failed
experiment because three post-run support checks were specified incorrectly:

- the MPI command did not explicitly propagate `PBS_QTIME_UTC`, leaving four
  remote role records without scheduler qtime;
- the 24.576-million-token capacity smoke inherited the formal one-billion-token
  loss trend threshold even though its frozen claim is protocol/runtime capacity
  only; all loss streams were finite, positive, contiguous, and reconciled;
- concurrent proposal inventory required exact orphan/retirement-marker equality
  at every sample, although one publisher can replace visibility between the two
  observations. The only mismatch was one object at cycle 30 and later samples
  reconciled.

The non-formal contract now requires strict loss-stream validity but reports the
ratio/slope only diagnostically. The formal long run still requires ratio at most
0.99 and negative robust slope. Inventory permits only a concurrent unclassified
gap bounded by active writers and additionally enforces the history-independent
orphan bound `retained_recent + visibility_records * inventory_interval`.
Scheduler qtime is explicitly passed to every MPI rank. Job `2394333` is not
retroactively passed; one replacement smoke must pass before formal submission.
