# Stage 0 Checkpoint

## Status

`PASS`

All seven Stage 0 loops and all eight Stage 0 acceptance IDs have independent
loop-level `PASS` verdicts. The fresh independent stage-level Checker also
returns `PASS`; no Stage 0 blocker or follow-up remains.

## Source and code

- Source hashes: `RESEARCH_PLAN.md =
  2c0db3148110da10ea62f445c189d2d53b405527d8da264b9af9817f5ca9d835`;
  `STAGE0-4_SPEC.md =
  fad26a8add51e70ddf50d0130b61d80fe06e1c3945221c29508412a5bcab2feb`.
- Integration branch: `master` (not automatically merged).
- Stage assembly branch: `codex/S0B-03-storage-capacity`.
- Checked checkpoint candidate: `6400f1633f6ea0cfd7bfb8dc2eac7177cac91e75`
  (parent `c00bb363cb6aff93781a15d9828b3d3477a928fa`).
- Current Miyabi skill:
  `https://github.com/UnbearableFate/miyabi-development@c3ddfa47cadfd9132edf939d359b90b28ab5f8ed`.
- Current full regression: job `2383871.opbs`, 57/57 Stage 0 tests on a
  compute node from corrected runtime `5424dae`, with the current skill
  checkout and a clean worktree.

The early CPU-only 0-C/0-A closure packages bind and snapshot the then-recorded
skill commit `ad1fd34`. Storage closure packages bind the current `c3ddfa47`
checkout after the S0B-01 provenance correction. The current-skill audit is
closed by the 57-test compute package, which covers every Stage 0 test module
under `c3ddfa47`; no runtime check was run on the login node.

## Closed loops

| Loop | Checked commit | Checker / SHA-256 | Evidence |
|---|---|---|---|
| `S0C-01` | `ed9c49246976d31b604b8a79f88c61e137448d9b` | `PASS`, `726d9cedb51cf916d6fb508a6f5579c9281c2f2571abe17909002c3828b54bb7` | `evidence/indexes/S0C-01.md` |
| `S0C-02` | `a470dcdb32a6c4b8a6e2e6f756903946ef36aefc` | `PASS`, `6bc15d09beb92f311c466570ad48893db77ae2673c21314cdd13fd1c3de1678f` | `evidence/indexes/S0C-02.md` |
| `S0A-01` | `9dbec27ae562b20174e55e53b84d55b9c2795c6f` | `PASS`, `a547031c5417d26e678ca39d1da6890cb7edf733a6da1ec048b0d8669617a08f` | `evidence/indexes/S0A-01.md` |
| `S0A-02` | `311c81aa2eaa264adc7ce256060627ea321443ba` | `PASS`, `1e8211bf55014d2be5804d31a3fd287068364c364ae0e111e21043bd5bd0fa9b` | `evidence/indexes/S0A-02.md` |
| `S0B-01` | `7681510482548c1f1c8c58f50b2510f72bcbeb43` | `PASS`, `67c42dc5eac369e9235c99dbe6fa2747e2f2624601438a8688b71b0d859a8dc7` | `evidence/indexes/S0B-01.md` |
| `S0B-02` | `f82ad39f5b0101d54cdd6e8cad43a9b06a707456` | `PASS`, `9886aac84e253ba48e348739d8bbc6e1245b408a5b37481adc5fcd9b4f3b8bd7` | `evidence/indexes/S0B-02.md` |
| `S0B-03` | `4d2e1e2579ec703241a360ae698d2bc3392f7879` | `PASS`, `8b6d80e9d897940bc8263729a8caca32f3271d8f79994208f761772cc5f9a83d` | `evidence/indexes/S0B-03.md` |

Historical blocked Checker reports remain tracked. Each final loop Checker
re-evaluated a corrected successor candidate; no blocked package contributes
to closure.

## Acceptance

| ID | Verdict | Primary evidence | Stage-level fact |
|---|---|---|---|
| `A-SIM-01` | PASS | `evidence/indexes/S0A-01.md` | all seven shared decision vectors plus deterministic/bounded event semantics |
| `A-SIM-02` | PASS | `evidence/indexes/S0A-02.md` | exact 7776-row matrix, three seeds, four shards, verified interruption/resume |
| `A-SIM-03` | PASS | `evidence/indexes/S0A-02.md` | Stage 1/4 profiles and Stage 5 delay points frozen; Stage 4 remains mandatory |
| `A-BENCH-01` | PASS | `evidence/indexes/S0B-02.md` | worst empty/loaded p99 `7.951ms` versus `2.5s` |
| `A-BENCH-02` | PASS | `evidence/indexes/S0B-02.md` | two runs of 100000 replacements at each size, four readers, zero violations |
| `A-BENCH-03` | PASS | `evidence/indexes/S0B-03.md` | worse synchronized rate `39403.448/s` versus `5120/s` |
| `A-BENCH-04` | PASS | `evidence/indexes/S0B-03.md` | worst 16-stream write `0.323808s` versus `5s` |
| `A-ALG-00` | PASS | `evidence/indexes/S0C-02.md` | wrong stale formula rejected; golden weighting/consumption/outer transitions pass |

`plans/ACCEPTANCE_TRACEABILITY.csv` and
`reports/stage0/requirement_to_evidence.csv` contain no unpassed required
Stage 0 acceptance or requirement. Optional `BENCH-05` is explicitly recorded
as not executed and supplies no object-store claim.

## Key runs

- Simulator sweep: S0A-02 corrected GREEN package
  `20260714T145700Z-green3-resume-n7b4a-14126dde2f00`, 7776 raw rows,
  verified partial checkpoints and resume; full simulator/oracle regression
  passes 33 tests.
- Shared-filesystem identity: S0B-01 jobs `2383213` and `2383231` prove two
  distinct-host target-Lustre smoke runs; the node-local negative control is
  rejected.
- Visibility/atomicity: S0B-02 jobs `2383537` and `2383540`; worst p99 below
  8ms, two 100000-replacement sizes/run, zero violations. Unsafe overwrite
  job `2383599` produces 5208 detector hits.
- Corrected capacity: S0B-03 jobs `2383872` and `2383885`; each has 216
  synchronized metadata cells and the complete I/O matrix. Worse metadata
  rate is `39403.448/s`; worst write round is `0.323808s`.
- Latest stage-level regression: `2383871.opbs`, 57/57 tests, exact-qtime
  immutable package SHA-256
  `f027b37211f0eb324a61778edc6ff83f98d500e1a716d7742a54b1a396cf438a`.
- Nine-node baseline/regression: not applicable; base training capability is
  inactive until `S1-13`.
- Model/data profile: scalar/small-vector oracles, synthetic event/count
  simulation, and deterministic byte/JSON storage fixtures only. No model,
  dataset, Torch execution, GPU, training, loss, or quality claim.

## Decisions

- Stage 1 simulation Profile A: `M=8`, `Q=Q_fresh=M`, `S_max=0`, grace `0`,
  `lambda_s=1`, `F=4`, evenly staggered `H=50` steps.
- Stage 1 storage: frozen target Lustre,
  `H=50s_fixed_slot_discovery_no_history_scan`; no capacity mitigation.
- Stage 4 Profile B remains an experiment anchor, not a correctness result;
  Stage 4 and its matched `S_max=0` control remain mandatory.
- Stage 5 retains visibility points `0/1/5/30s` and must recompute delay/H
  from measured runtime step time.
- Simulator recommendations do not establish runtime correctness. Storage
  evidence does not establish production protocol or crash durability.

## Follow-ups and blockers

None for Stage 0 closure. Optional `BENCH-05` is not a blocker.

## Risk change

Closed: reference math/consumption semantics, deterministic simulator and
parameter-scan evidence, target-Lustre identity, cross-node visibility,
atomic replacement detection, bounded metadata discovery, and the frozen I/O
capacity budget.

Remaining for later stages: typed production runtime/config identity,
fragment/model integration, actual learner/syncer correctness, real-model
training/loss, recovery, long-run boundedness, stale-quality conclusions, and
the post-`S1-13` 8+1/50x10 regression gate.

## Next Stage entry conditions

- [x] All `S1-00` Stage 0 dependencies (`S0A-02`, `S0B-03`, `S0C-02`) pass.
- [x] Stage 1 simulator and storage profiles are frozen and evidence-linked.
- [x] Target filesystem is qualified without substituting local/NFS evidence.
- [x] Stage 4 remains mandatory despite the simulator prediction.
- [x] Nine-node execution remains disabled until `S1-13` establishes base
  training capability.
- [x] Fresh independent Stage 0 Checker approves this checkpoint.

## Checker

- Report: `reports/checkers/STAGE_0.md` (SHA-256
  `85d4f1c3e9d4f1cd9c99a3d6d4972980b537211c29b1871c9fcbd915d54aa760`).
- Checked candidate: `6400f1633f6ea0cfd7bfb8dc2eac7177cac91e75`.
- Verdict: `PASS`.
