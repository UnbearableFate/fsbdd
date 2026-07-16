# Stage 1 Checkpoint Candidate

## Status

`CANDIDATE_PENDING_INDEPENDENT_CHECKER`

All Stage 1 acceptance IDs and the formal 9-node baseline are already covered
by passing evidence. The user explicitly approved a one-time S1-13 early-close
amendment for the separate M=4 long-profile requirement. This checkpoint does
not claim that the partial run completed one billion tokens. Stage 1 and
`CAP_BASE_TRAINING` remain unclosed until the independent Checker accepts this
candidate.

## Source, code, and decision identity

- Integration branch: `master` (not automatically merged).
- Stage branch: `codex/S1-13-stage1-close-9n`.
- Early-close evidence amendment commit:
  `10854d9095dfe9babab3e666932ee1a604b3fb46`.
- S1-13 runtime implementation commit:
  `4354da018789498f5b3e1d05a7f0657a1fedb858`.
- Research Plan SHA-256:
  `1d25f18390f81906ad6b944e231e4cfd4ef83639efc0395c8681efdfd4e54739`.
- Stage 0-4 Spec SHA-256:
  `7a038a408bc51f3b225431868960231f4f5a404f7259de743799477803d92ad8`.
- Current Miyabi skill:
  `https://github.com/UnbearableFate/miyabi-development@c3ddfa47cadfd9132edf939d359b90b28ab5f8ed`.
- Decision: `reports/stage1/S1-13-early-close-adr.md`.
- Partial observation:
  `reports/stage1/S1-13-partial-formal-observation.json`.
- Reviewed failure ledger: entry 14 of
  `reports/stage1/S1-13-experiment-failures.json`.

The source amendment is intentionally post-observation and one-time. It changes
only the supplemental S1-13 long-run closure condition; no algorithm,
publication, progress, or acceptance semantics changed.

## Closed implementation loops

| Loop | Result commit | Loop Checker |
|---|---|---|
| `S1-00` | `b9658c5882e75c5ad72b5c828eb43f731c455f3c` | PASS |
| `S1-01` | `25d12c9896bc20843af241cdaf68503f2ce16d3d` | PASS |
| `S1-02` | `46ed2cc60928368aca33ff764ec0da4910971d70` | PASS |
| `S1-03` | `3552c9bdd8f27df2db128ce19511bd14b26f60dd` | PASS |
| `S1-04` | `071e5c1cd6728be9fae8b8d69084ca79905e4e4e` | PASS |
| `S1-05` | `b7af8fa535cd9b136baa5109c1f2d81fb15271b8` | PASS |
| `S1-06` | `e06affde5e1b611a046e411f2f1a2207d95ee5f8` | PASS |
| `S1-07` | `b530f9a8fc861395659b72d41ca3b46922939392` | PASS |
| `S1-08` | `fc7b792fdaf9480490138075c9a261a92fc64e52` | PASS |
| `S1-09` | `1f04f41d995adfe1b4981c73f26319d46e36c723` | PASS_WITH_FOLLOWUPS, closed by later evidence |
| `S1-10` | `78afc59c073a1330b5e06ac2eda0f6581bbef017` | PASS_WITH_FOLLOWUPS, closed by later evidence |
| `S1-11` | `415ac59a76035d7dfe61f667041d63912062ef6b` | PASS_WITH_FOLLOWUPS, closed by later evidence |
| `S1-12` | `2e44bbc3867f1afa344b6ecc9b783bfb2a31250d` | PASS |
| `S1-13` | current candidate | pending final independent Checker |

## Acceptance

| Acceptance ID | Status | Primary evidence |
|---|---|---|
| `A-FRAG-01` | PASS | `evidence/indexes/S1-02.md` |
| `A-FRAG-02` | PASS | `evidence/indexes/S1-02.md` |
| `A-FRAG-03` | PASS | `evidence/indexes/S1-02.md` |
| `A-PROP-01` | PASS | `evidence/indexes/S1-05.md` |
| `A-PROP-02` | PASS | `evidence/indexes/S1-05.md`, `S1-09.md` |
| `A-PROP-03` | PASS | `evidence/indexes/S1-05.md`, `S1-09.md`, `S1-11.md` |
| `A-GLOBAL-01` | PASS | `evidence/indexes/S1-03.md`, `S1-11.md` |
| `A-GLOBAL-02` | PASS | `evidence/indexes/S1-04.md`, `S1-11.md` |
| `A-LEARN-01` | PASS | `evidence/indexes/S1-12.md`, `S1-13.md` |
| `A-LEARN-02` | PASS | `evidence/indexes/S1-08.md` |
| `A-LEARN-03` | PASS | `evidence/indexes/S1-08.md`, `S1-12.md` |
| `A-PERF-01` | PASS | `evidence/indexes/S1-10.md`, `S1-13.md` |
| `A-ALG-01` | PASS | `evidence/indexes/S1-10.md`, `S1-12.md` |
| `A-EVAL-01` | PASS | `evidence/indexes/S1-12.md` |

`plans/ACCEPTANCE_TRACEABILITY.csv` has no unpassed Stage 1 acceptance row.

## Formal 9-node baseline

- Run: `s1-13-formal-9n-20260715T220641Z-e850f51`, PBS array
  `2393100[].opbs`.
- Shape: eight independent single-GPU learners plus one CPU-only syncer on nine
  distinct compute hosts.
- Workload: immutable pretrained GPT-2 small plus WikiText-2 raw packed-512.
- Result: all nine scheduler exits 0, no abnormal node, topology/loss/runtime/
  protocol gates pass, exact 179-file package admissible, same-Checker Phase B
  PASS.
- Runtime: 286.526301569 active seconds of 1,200; 41 updates; final version
  vector `[10,10,11,10]`; 9,226,240 aggregate tokens.
- Loss: aggregate final/initial ratio `0.9392284813790339`, negative robust
  slope; frozen evaluation loss `3.9339581240863524 -> 3.1861374294978146`.
- Evidence: `runtime_runs/S1-13/formal-9n` and
  `evidence/indexes/S1-13.md`.

This establishes `BASELINE-9N-GPT2-WT2-v1` and the post-S1-13 regression
topology.

## Five-node Pythia/FineWeb-Edu smoke

- Run: `s1-13-long-smoke-analyzer-corrected-20260716T023518Z-6bc43ed`, job
  `2394619.opbs`.
- Result: scheduler exit 0, no abnormal node, 367 tests, four learners at 3,000
  steps and 6,144,000 tokens each, 24,576,000 aggregate tokens, 240 updates,
  final vector `[60,60,60,60]`.
- Active runtime: 539.716323747 of 1,200 seconds.
- All topology, non-formal loss-stream, runtime, protocol, bounded-state,
  reclamation, durability, byte-accounting, and forbidden-runtime checks pass.

## User-accepted partial long observation

Job `2394870.opbs` ran the immutable Pythia-160M/FineWeb-Edu long profile on
four GPU learner hosts plus a CPU-only syncer host. At the accepted observation
point:

- learner steps: `[9219,9217,9217,9217]`;
- input tokens: `[18880512,18876416,18876416,18876416]`, aggregate
  `75,509,760`;
- four finite, positive, contiguous loss streams and 9,217 common points;
- initial/final rolling medians `4.355180000000001/4.32901`, ratio
  `0.9939910635151703`, robust slope `-4.257425742575278e-06`;
- projected full learner spans `21,501.15--21,573.06s`, all within `21,600s`;
- authoritative version vector after termination `[169,167,166,168]`.

The user requested immediate closure. The job was deleted, finishing with
scheduler exit 271 after `00:29:09`, no abnormal node. Therefore the following
original gates are explicitly **not passed**:

- one-billion aggregate input tokens;
- minimum global cycle 2400;
- final/initial ratio at most 0.99;
- scheduler exit 0;
- normal role finalization and complete final package.

No Stage 1, paper, or later-stage claim may describe this as a completed
one-billion-token run. The amended conclusion is only that the real M=4
production path trained continuously with a decreasing observed aggregate
trend and an in-budget runtime projection.

## Attempts, queue/runtime, and rework

- Failure ledger: 14 entries. Every failure records the observed failure,
  diagnosed cause, expected solution, and retained raw evidence before the next
  experiment.
- S1-13 Checker history before this final review: 20 recorded candidate cycles.
- Formal 9N active runtime: 286.526301569s. Corrected five-node smoke active
  runtime: 539.716323747s. Partial formal scheduler walltime: 00:29:09.
- Formal partial job queue delay was 23s (`02:52:18Z -> 02:52:41Z`).
- Dominant rework causes were evidence-harness defects, obsolete asset identity,
  a readiness cache race, redundant publication readback, an impossible short
  smoke terminal threshold, and auxiliary analyzer semantics. Production code
  and auxiliary code were subsequently separated and fully reviewed.
- The recovery process removed per-candidate manual repository hashes and
  routine full payload inventories while retaining semantic asset validation,
  raw failure evidence, narrow source-log checksums, and final independent
  review.

## Known limitations and preserved claims

- A completed M=4 one-billion-token result is absent by explicit user-approved
  scope change.
- The partial formal job has no normal terminal role result or formal package.
- The 9N baseline and five-node smoke remain the complete scheduler/package
  evidence. The partial formal run is only trend/runtime corroboration.
- Stage 2–5 independent goodput, two-hour slow-FS, recovery, matched-token, and
  1000+ update requirements remain unchanged.
- User-owned `AGENTS.md` is the only dirty worktree path and is excluded from
  all Stage 1 commits.

## Capability and next-stage gate

`CAP_BASE_TRAINING` remains inactive in this candidate. It may become active
only if the independent Checker accepts the complete closure with the limitation
above. Stage 2 must then run a new 9-node gate according to the existing gate
unit rules; it may not reuse the Stage 1 baseline as its own implementation
gate.

## Checker

Pending independent S1-13/Stage 1 final review. The Checker must verify the raw
partial-loss calculation, scheduler deletion, original unpassed gates, source
amendment, 9N/smoke pass evidence, acceptance traceability, and absence of an
overstated one-billion-token claim.
