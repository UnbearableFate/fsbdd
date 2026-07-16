# S1-13 Early-Close ADR

## Status

Accepted by explicit user direction on 2026-07-16.

## Context

The original S1-13 long-run contract required four Pythia-160M learners to
complete at least one billion aggregate input tokens, reach global cycle 2400,
produce a final/initial rolling-median loss ratio no greater than 0.99, retain
a negative robust loss slope, and finish within 21,600 active seconds.

Job `2394870.opbs` began that workload from clean commit
`4354da018789498f5b3e1d05a7f0657a1fedb858`. The user first allowed early
completion when loss and projected time were in range, then explicitly directed
the agent to close Stage 1 at the current observation point and preserve the
evidence. The second direction supersedes the original completion-only long-run
gate for this one Stage 1 closure.

## Decision

S1-13 and Stage 1 may close using the retained partial formal observation in
`reports/stage1/S1-13-partial-formal-observation.json`, together with the
already-passing formal 9-node baseline and corrected five-node long-profile
smoke.

This is a one-time, post-observation scope amendment. It does not retroactively
make job `2394870.opbs` a passing one-billion-token run. The job is classified
as `terminated_by_user_after_accepted_partial_observation`, and its original
contract failures remain explicit:

- 75,509,760 aggregate input tokens, not one billion;
- minimum authoritative global version 166, not 2400;
- partial-observation loss ratio `0.9939910635151703`, not at most 0.99;
- scheduler exit 271 after the requested deletion, not exit 0;
- no terminal role results or normal final evidence package.

The accepted evidence is narrower:

- all four loss streams are finite, positive, and contiguous through at least
  9,217 common optimizer steps;
- the frozen loss formula gives an initial median `4.35518`, final median
  `4.32901`, ratio `0.9939910635151703`, and negative robust slope
  `-4.257425742575278e-06`, which establishes that training occurred and the
  observed aggregate trend was decreasing;
- projected full-run learner spans are 21,501.15--21,573.06 seconds, all below
  the frozen 21,600-second active-runtime budget;
- the same production profile previously passed a complete five-node smoke at
  24,576,000 tokens and global version vector `[60,60,60,60]`;
- the independent-jobs 9-node baseline passed topology, loss, runtime,
  protocol, package validation, and Checker Phase B.

## Scope and affected requirements

This amendment changes only the supplemental Stage 1 M=4 long-run closure
condition in `STAGE0-4_SPEC.md` section 5.11, `RESEARCH_PLAN.md` Stage 1, and
the S1-13 execution card. It does not change any `INV-*`, `MODEL-*`, `FRAG-*`,
`LEARN-*`, `PROP-*`, `SYNC-*`, `OPT-*`, `GLOBAL-*`, `FS-*`, or `PROG-*`
semantic requirement, nor any Stage 1 acceptance ID.

The original frozen gate contract and runtime configuration are retained
unchanged as evidence of what the job did not complete. No claim that Stage 1
contains a completed one-billion-token run is permitted. Later work may cite
this run only as partial real-model training/trend/runtime evidence.

## Failure recording and expected solution

Failure-ledger entry 14 records the early termination because the experiment
did not satisfy its original contract. The cause is an explicit user-directed
scope change, not a runtime defect. No retry is required for the amended Stage
1 closure. If a future claim requires the original one-billion-token evidence,
the expected solution is a new uniquely rooted run under the unchanged original
gate contract; this partial run must not be relabeled or extended in place.

## Rollback

Revoke this ADR, return `CAP_BASE_TRAINING` to inactive, and complete a fresh
M=4 run that passes the original token, cycle, loss, runtime, scheduler, and
package gates.
