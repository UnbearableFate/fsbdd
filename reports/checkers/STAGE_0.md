# Checker Report: Stage 0 / `6400f16`

## Verdict

`PASS`

Stage 0 satisfies its closure contract. All seven loops have valid final
`PASS` Checker reports, all eight required Stage 0 acceptances pass,
traceability is consistent, superseded blocked candidates do not contribute
to closure, and no Stage 0 blocker or follow-up remains.

## Independent inputs and repository identity

- Initial host: `miyabi-g1`, no PBS allocation; login/control-plane
  static-only workflow.
- No project Python, tests, benchmarks, ML imports, MPI, training, or
  inference ran in this Checker session.
- Current branch: `codex/S0B-03-storage-capacity`; checked candidate
  `6400f1633f6ea0cfd7bfb8dc2eac7177cac91e75`, parent
  `c00bb363cb6aff93781a15d9828b3d3477a928fa`.
- Integration branch `master` at `d78415812719468e823795d0234193cdac699e40`
  is an ancestor. Every final loop result is an ancestor.
- Candidate changes only `reports/stage_checkpoints/STAGE_0.md`.
- `git diff --check` passes; tracked tree, index, and worktree were clean and
  byte-identical to the candidate.
- Source hashes independently match: `RESEARCH_PLAN.md =
  2c0db3148110da10ea62f445c189d2d53b405527d8da264b9af9817f5ca9d835`;
  `STAGE0-4_SPEC.md =
  fad26a8add51e70ddf50d0130b61d80fe06e1c3945221c29508412a5bcab2feb`.
- Current `miyabi-development` checkout is clean at
  `c3ddfa47cadfd9132edf939d359b90b28ab5f8ed`; historical `ad1fd34` is its
  ancestor.

## Loop closure

| Loop | Final checked commit | Final Checker SHA-256 | Result |
|---|---|---|---|
| `S0C-01` | `ed9c49246976d31b604b8a79f88c61e137448d9b` | `726d9cedb51cf916d6fb508a6f5579c9281c2f2571abe17909002c3828b54bb7` | PASS |
| `S0C-02` | `a470dcdb32a6c4b8a6e2e6f756903946ef36aefc` | `6bc15d09beb92f311c466570ad48893db77ae2673c21314cdd13fd1c3de1678f` | PASS |
| `S0A-01` | `9dbec27ae562b20174e55e53b84d55b9c2795c6f` | `a547031c5417d26e678ca39d1da6890cb7edf733a6da1ec048b0d8669617a08f` | PASS |
| `S0A-02` | `311c81aa2eaa264adc7ce256060627ea321443ba` | `1e8211bf55014d2be5804d31a3fd287068364c364ae0e111e21043bd5bd0fa9b` | PASS |
| `S0B-01` | `7681510482548c1f1c8c58f50b2510f72bcbeb43` | `67c42dc5eac369e9235c99dbe6fa2747e2f2624601438a8688b71b0d859a8dc7` | PASS |
| `S0B-02` | `f82ad39f5b0101d54cdd6e8cad43a9b06a707456` | `9886aac84e253ba48e348739d8bbc6e1245b408a5b37481adc5fcd9b4f3b8bd7` | PASS |
| `S0B-03` | `4d2e1e2579ec703241a360ae698d2bc3392f7879` | `8b6d80e9d897940bc8263729a8caca32f3271d8f79994208f761772cc5f9a83d` | PASS |

Every loop state says `pass`, selects its final report, and has no blocker.
Blocked candidates `b993911`, `fe7e558`, `ecbc32a`, `e3b7613`, `e6ab81a`,
`454043a`, and `4ef5229` are ancestors of corrected final candidates. Their
defects were addressed by successor commits and, where required, new
immutable evidence. Historical reports remain intact but supply no acceptance
verdict.

## Independent raw recomputation

### Simulation and oracle

The selected S0A-02 manifest verifies all 67 entries, SHA-256
`848f83af206a1b9ec2782b4582349028233135f18f710a492fcb42898dab21b7`.
Independent CSV reconstruction finds exactly 7776 rows, unique indices
`0..7775`, unique full-axis tuples, exact `3*3*4*3*4*3*2*3` axes, and four
1944-row shards. The first PBS session retained 25 rows/shard and exited with
planned status 75; the second reused the same run/config/generator identity
and completed all shards. Analytic fresh sanity has nine rows, interval 50,
zero discard, and zero stale acceptance.

The Stage 4 anchor recomputes to accepted-token efficiency
`0.3348337641501025`, stale rate `0.053107946226540664`, matched `S_max=0`
efficiency `0.3352794366699349`, recovery `-0.0004456725198324074` (or
`-0.0445673` percentage points), and `delay/H=0.02`. It is correctly an
experiment anchor, not runtime/training-quality evidence, and does not cancel
mandatory Stage 4.

Static oracle recomputation confirms old-base stale displacement `[1,1]`
rather than forbidden current-relative `[4,4]`; token/staleness raw weights
`[100,50,100]` normalize to `[0.4,0.2,0.4]`; mixed displacement `[3,3]`
applies to current `[12,-2]` as `[9,-5]`; Nesterov transitions match the
fixture; exact fraction ordering distinguishes `2^53+1`; and all seven shared
selection/consumption vectors cover the required identities and rejections.

### Target filesystem

The S0B-01 shared package verifies all 41 manifest entries. Its canonical
JSON/YAML manifests are byte-identical and prove target
`/work/xg24i002/x10041/fsbdd/runtime_runs` on read/write Lustre, writer
`mg0016`, reader `mg0017`, launcher-only MPI, filesystem-only application
data, current skill `c3ddfa47`, and 4096-byte payload/digest agreement. The
distinct-host `/tmp` control fails for the expected visibility/ack reasons.
Target Lustre is not substituted by login-node or node-local evidence.

### Visibility and atomicity

Both S0B-02 full runs contain 1000 empty and 1000 loaded samples. Primary p99
is `7.938555/7.800906ms`; repeat p99 is `7.496595/7.950762ms`; worst is below
`2.5s`. Every run/size has exactly 100000 unique sequences `0..99999`, four
readers ending at `99999`, and zero raw violations. Unsafe overwrite produces
exactly 5208 detections (`1321/1298/1313/1276`). Selected manifests verify,
ruling out an inert scanner and closing `A-BENCH-01/02`.

### Synchronized metadata, bandwidth, and cleanup

Each corrected S0B-03 run has 1716 rank-local metadata records, 216 unique
rank-0 coordinator intervals, exact ready/done rank sets `0..15`, and no
coordinator interval shorter than its matching local maximum. The primary
limiting rate is `42912 / 1.060520994 = 40463.131086/s`; repeat is
`41792 / 1.060617853 = 39403.447605/s`, versus `5120/s` required.

Each run also has 272 raw bandwidth operations, 32 coordinator rounds, and
only opposite-host verified reads. Worst 16-stream write is `0.323807765s`
versus `5s`. Fixed state is `6272->6272`, history `11000->11000`, control
files 35, payload/temp zero, and roots removed. Both 57-entry manifests and
the corrective RED verify. This closes `A-BENCH-03/04`.

### Current full regression and skill provenance

Package `20260714T174115Z-harden-sync-n9d6803-8ac93ab4b410`, job
`2383871.opbs` on `mg0028`, passes 57/57 tests. All 34 files verify;
checksum-manifest SHA-256 is
`f027b37211f0eb324a61778edc6ff83f98d500e1a716d7742a54b1a396cf438a`.
It binds clean runtime `5424dae`, clean current skill `c3ddfa47`, and exact
UTC qtime `2026-07-14T17:41:15Z`.

Early CPU-only S0C/S0A packages correctly record historical `ad1fd34`;
storage packages and the complete current regression record `c3ddfa47`. No
selected runtime package reports a login host as compute hostname.

## Acceptance determination

| Acceptance | Verdict |
|---|---|
| `A-SIM-01` | PASS |
| `A-SIM-02` | PASS |
| `A-SIM-03` | PASS |
| `A-BENCH-01` | PASS |
| `A-BENCH-02` | PASS |
| `A-BENCH-03` | PASS |
| `A-BENCH-04` | PASS |
| `A-ALG-00` | PASS |

`ACCEPTANCE_TRACEABILITY.csv` has exactly eight Stage 0 rows, all `passed`
with Checker `PASS`. The requirement map has the same acceptance rows and no
unresolved required Stage 0 requirement. `BENCH-05` is correctly
`optional_not_executed` and makes no object-store claim.

## Independent counterexamples

1. A 7776-row file could hide duplicate/missing tuples; exact indices, axes,
   tuple uniqueness, and shard cardinality reject this.
2. A silent atomicity scanner could falsely support zero violations; the
   unsafe path produces 5208 concrete detections.
3. Disjoint local metadata work could be falsely summed; corrected evidence
   uses complete 16-rank barriers and rank-0 start-to-all-done intervals.
4. A node-local path could masquerade as shared; the same `/tmp` path fails
   across two hosts while target Lustre succeeds.
5. Blocked evidence could leak into closure; every blocked candidate is
   superseded, and final hashes/indexes select corrected packages only.
6. Storage/simulator evidence could be overstated as learner/syncer
   correctness; the checkpoint explicitly defers production runtime,
   model/data, training/loss, recovery, and stale-quality claims.

No counterexample survives against the Stage 0 closure claim.

## Scope, gates, and deferred risks

- Stage 0 contains oracle, event/count simulation, and deterministic storage
  tooling only; no learner/syncer production implementation exists.
- The nine-node gate is not applicable before `S1-13` establishes
  `CAP_BASE_TRAINING`.
- Optional `BENCH-05` is unexecuted and non-blocking.
- Stage 1 entry conditions are met: `S0A-02`, `S0B-03`, and `S0C-02` pass;
  simulation/storage profiles are frozen; target Lustre is qualified.
- Later stages own typed runtime/config identity, fragment/model integration,
  real training/loss, recovery, stale matched-token quality, long-run
  boundedness, and post-`S1-13` 8+1/50x10 regressions.
- Stage 1 must preserve fixed `M*F` discovery and rerun capacity if actual
  sync time, payload sizes, concurrency, layout, or mount materially changes.

These are later-stage obligations, not Stage 0 follow-ups.

## Signature

- Checker: fresh independent `/root/stage0_final_checker`.
- Date: 2026-07-15 Asia/Tokyo.
- Initial workflow: `miyabi-g1`, login/control-plane static-only.
- Checked candidate: `6400f1633f6ea0cfd7bfb8dc2eac7177cac91e75`.
