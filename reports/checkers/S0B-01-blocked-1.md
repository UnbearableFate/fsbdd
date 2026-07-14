# Checker Report：S0B-01/e6ab81a

## Verdict

`BLOCKED`

The storage behavior is promising and the final positive/negative runs are
internally consistent, but the checked result cannot close S0B-01. Two
independent environment-evidence requirements are false or absent in the run
manifests: the recorded `miyabi-development` commit was not the installed
commit when this loop began and ran, and neither candidate manifest contains
the required compute-node type plus actual role mapping (the two manifest
formats also split queue/group and module information). These are explicit
Miyabi/run-manifest gates, not optional reporting polish.

## 独立输入

- source hashes: `RESEARCH_PLAN.md = 2c0db3148110da10ea62f445c189d2d53b405527d8da264b9af9817f5ca9d835`; `STAGE0-4_SPEC.md = fad26a8add51e70ddf50d0130b61d80fe06e1c3945221c29508412a5bcab2feb`.
- loop card: `plans/FS_DILOCO_CODEX_EXECUTION_PLAN/loops/stage0/S0B-01.md`, SHA-256 `96609382fd71c3c14e387b48dc25c04e6b2e1af32b2a3651915cfa9f98b642d6`.
- checker template: SHA-256 `0cf29aa65c6a0d691cdb2661e5cacc3f2beb1619d729e98add082e36539ae297`.
- checked ancestry: loop base `5a8b78c9765c1b13c5b21006a5243a32e2afc19d` -> SPECIFY/RED `6b901c99cd34e90c782234d063c697947d4f7f34` -> checked result `e6ab81ada6306cc5a6f7df0d27d179305c01901f`; both ancestry checks pass and `git diff --check 5a8b78c..e6ab81a` passes.
- candidate artifacts at `e6ab81a`: resolved config `bdb56686ee3ec2d5a44723b0fab89bf9768749eae671bd6ce675777ddf095115`; harness `f4a9e70113588926b8f60b479756fe0cb9516405fe8019867ecdbb8f0af370ba`; PBS wrapper `c13e40bdd3d1587ca3c420228a7db4434197b2c1a0dd952a97e51681a3088cc9`; tests `b0e76f54831c37dc7f7c1ecce8a20a721a51b0315808f6092020fa972fc64e30`; environment draft `0a97cb72e22b2420c659fc61b205d0f42a06960c5d65746e27d6c8ca0ec31dfb`.
- RED: `evidence/raw/S0B-01/20260714T151400Z-red-n3d6a-d6b0d27e3bb2`.
- one-node HARDEN: `evidence/raw/S0B-01/20260714T152628Z-harden-n6e1c-39928d85f6ae`.
- original shared GREEN: `evidence/raw/S0B-01/20260714T152513Z-green2-n4d8b-d510304161dd`.
- corrected node-local negative: `evidence/raw/S0B-01/20260714T152720Z-red5-n3c9a-d27938bf5c40`.
- corrected repeated shared HARDEN: `evidence/raw/S0B-01/20260714T152721Z-harden3-n8d4e-d27938bf5c40`.
- 9N package: not applicable before `S1-13`.

## 重新构建的目标

- Single objective: safely discover the real Miyabi/PBS/module/path context,
  freeze the later protocol `RUN_ROOT`, and provide a repeatable writer/reader
  harness that distinguishes shared Lustre from node-local storage.
- Requirements: `BENCH-01..04` supporting harness evidence, `FS-08`, and the
  two-primitives boundary in `STOR-01`. This loop does not close
  `A-BENCH-01..04`; latency/atomicity are S0B-02 and metadata/bandwidth are
  S0B-03.
- Explicit environment gates: project `AGENTS.md` section 2 requires every run
  manifest to record the actual skill repository/commit, initial hostname,
  node type, PBS job/queue/group, module list, and actual role mapping. The
  evidence policy additionally requires commit/config/filesystem identity.
- Checker execution boundary: initial host `miyabi-g1`, no PBS allocation. I
  used only Git, file/hash inspection, `bash -n`, and PBS history; no project
  Python, tests, benchmark, MPI, or heavy import ran on the login node.

## RED 有效性

- Job `2383084.opbs` ran on `mg0017` and produced three deterministic assertion
  failures: the surrogate admitted a login host, omitted seven required
  manifest fields, and treated an absolute `/tmp` path as shared. Raw unittest
  exit is `1`; the wrapper's expected-failure contract completed normally.
- All failures directly correspond to the loop's manifest, host-routing, and
  node-local counterexamples. They completed in `0.002s`, so this RED is not an
  accidental timeout or environment/import error.
- The RED package passes its full checksum manifest and its config-digest
  preimage independently hashes to
  `d6b0d27e3bb2ec5ca403ea2eea930e346dbbd1f786da669e9c9c741516584465`.

## 代码与协议检查

- Scope/data plane: the candidate only adds the Stage 0 storage config,
  stdlib harness, PBS launcher, tests, RED surrogate, loop orientation, and
  decision draft. MPI launches one rank per allocated node; Python roles do
  not import MPI or exchange application data over it. Payload, visibility
  record, acknowledgement, and per-rank evidence flow through filesystem
  paths.
- Compute guard: both PBS and Python reject `miyabi-g1`, an `mg*` host without
  `PBS_JOBID/PBS_NODEFILE`, and wrong two-rank contexts. Actual smoke roles ran
  on distinct hosts.
- Publication: payload and JSON records use unique same-directory temporary
  files, flush plus `fsync`, `os.replace`, and directory `fsync`. The payload is
  complete before the visibility record; readers validate run ID, byte count,
  digest, and peer host. This is a suitable POSIX instance of the two
  `STOR-01` primitives for the later stress test.
- Timing: all recorded durations use `time.monotonic_ns` within one host. The
  draft correctly refuses to treat the smoke wait interval as the S0B-02
  publication-to-read latency distribution.
- Frozen path: config and both closure jobs use
  `/work/xg24i002/x10041/fsbdd/runtime_runs`; `findmnt` identifies the backing
  mount as Lustre, and stripe, quota, `stat`, create/read/delete behavior, and
  cross-node payload read are preserved. Node-local `/tmp` is kept separate.
- Repeat/identity: evidence directories fail if already present. Run IDs carry
  distinct submission nonces and short config digests; positive and negative
  final runs use new immutable directories and snapshot byte-identical inputs.
- Static safety: candidate `pbs/stage0_storage_smoke.pbs` passes `bash -n`, uses
  `set -eEuo pipefail`, a compute/PBS guard, exact digest verification, an
  in-job module load/list, and `/usr/bin/env` rather than conflicting Open MPI
  `-x` propagation.

### Blocking defect 1: recorded skill commit is demonstrably stale

The candidate hardcodes
`ad1fd34a9de976b4fb26ba47d9a1770430884765` in the resolved config, harness,
tests, YAML/JSON manifests, and report. The installed checkout's reflog instead
shows a fast-forward to `c3ddfa47cadfd9132edf939d359b90b28ab5f8ed` at
`2026-07-14 23:16:15 +0900`. S0B-01 ORIENT was committed at
`2026-07-15 00:12:37 +0900`; RED and closure jobs then ran from approximately
`00:13` through `00:27`. Thus `c3ddfa4`, not `ad1fd34`, was the installed
commit throughout this loop. `ad1fd34` is an ancestor, but the later commit
materially changes the workflow guidance (including interactive/batch
selection). Repeating the configured constant across manifests does not prove
the actual environment.

This contradicts `AGENTS.md` section 2, `docs/05_MIYABI_OPERATIONS.md` section
5.1, and `docs/07_TEST_EVIDENCE_CHECKER.md` section 7.4, all of which require
the actual skill commit. The smallest correction is to inspect/evaluate the
`ad1fd34..c3ddfa4` skill diff, freeze the actual commit in config and code,
and rerun the closure evidence under that recorded instruction set.

### Blocking defect 2: run manifests omit mandatory role/environment fields

At `e6ab81a`, `REQUIRED_MANIFEST_FIELDS` omits compute node type, PBS queue,
PBS group, and actual role mapping. Consequently the candidate
`manifest.json` can validate without all four. `manifest.yaml` contains
queue/group but only points to a module-list file, while `manifest.json`
contains the module list but omits queue/group; neither contains node type or
the actual rank/role/host mapping. The mapping is available only in the
separate `results/smoke-summary.json` and is not bound into either manifest.

The explicit project gate says *every run manifest* records these fields. A
consumer validating only the advertised environment manifest can therefore
accept a package whose actual placement is absent or inconsistent. The
smallest correction is to make one canonical structured manifest contain and
validate all required environment/PBS/code/config fields, finalize it with the
observed rank-role-host-job mapping, add mutation-sensitive tests, and rerun
the positive and node-local closure jobs so the corrected manifests are raw
evidence rather than post-hoc edits.

## Maker 未列出的反例

1. Delete queue, group, node type, and both role-to-host entries from an
   `e6ab81a` JSON manifest. `validate_environment_manifest` still accepts it,
   even though the execution contract requires those facts in each run
   manifest. A separate unbound smoke summary cannot make that manifest
   self-validating.
2. Update the installed `miyabi-development` checkout before ORIENT while
   leaving the source constant unchanged. Every generated manifest then
   agrees with the stale constant and passes candidate validation, falsely
   appearing reproducible. This is the exact observed counterexample, proven
   by the skill reflog and commit/run timestamps.

## 运行证据

- Final checksum manifests: RED
  `0fb47c4af76a9af9caf404bc3f9e5724b33534ee86377ae468bdfaebd74c720f`,
  one-node HARDEN
  `d058b3f8a64301cdb4ce360c93e791b47210c7088f3ce04bd4cad7b44e601a4e`,
  node-local negative
  `0bf86b712a451fc75265a4ef30b3b502d06f46f6e0d58607535cc6df76d4f1b6`,
  and repeated shared HARDEN
  `6c13d30b9a61f29ce94daaaa36f14f158f8108cf8ebab92edfaad940093a1e90`
  all pass every `sha256sum -c` entry.
- Original shared GREEN package
  `20260714T152513Z-green2-n4d8b-d510304161dd` has one expected historical
  evidence defect: `stdout/job.log` was checksummed while empty and then
  received the final completion line. Its other entries verify. Commit
  `451399d` corrected finalization order, and the new shared HARDEN package is
  fully immutable, so the old package is retained as a failed attempt rather
  than used for closure.
- Digest preimages independently reproduce RED `d6b0d27e...`, one-node
  HARDEN `39928d85...`, original GREEN `d5103041...`, and both final two-node
  runs `d27938bf5c40be5193dc72aad64bc6b5b3908e62cf7510c007429cdac0df1eac`.
  Final positive/negative snapshots match commit `91d27c536302c111f2e727eaae96e43174e9814d` and are unchanged through checked
  result `e6ab81a`.
- One-node HARDEN job `2383139.opbs` on `mg0045` passes all 8 focused tests.
  It covers login guard, frozen config, deterministic payload, invalid record
  rejection, PBS static contract, manifest minimum fields, distinct-host
  positive summary, and exact negative failure kinds.
- Node-local job `2383145.opbs` used `mg0017`/`mg0018`, finished in `27s`, and
  scheduler exit is `0` after the wrapper verified MPI exit `2`. Reader failure
  is `visibility_timeout`; writer failure is `acknowledgement_timeout`; the
  summary is `node_local_visibility_rejected`.
- Shared job `2383147.opbs` used `mg0031`/`mg0032`, finished in `3s`, and both
  scheduler/MPI exits are `0`. The reader hash-validates 4096 bytes with digest
  `ff1590d0f858164e49e922c9bdb5ed5a63396550ee276752b11a974bc7dd5f71`;
  writer receives the cross-host acknowledgement. `qstat -f -H` independently
  confirms submit host `miyabi-g1`, queue `debug-g`, group `xg24i002`, two
  allocated hosts, and the recorded PBS identities.
- Filesystem probes identify source
  `172.16.20.201@o2ib200:172.16.20.202@o2ib200:172.16.20.211@o2ib200:172.16.20.212@o2ib200:/lustre/work`,
  type `lustre`, stripe count `1`, stripe size `1048576`, successful group
  quota query, permission mode `640`, uid `30041`, gid `34002`, and a removed
  permission probe. Loaded modules are `nvidia/25.9` and `nv-hpcx/25.9`.
- No loss, global-cycle/local-step, model/data, GPU-compute, or 9-node claim is
  applicable to this Stage 0 harness loop.

## Acceptance 判定

| ID | Evidence | Conclusion |
|---|---|---|
| `BENCH-01..04` supporting harness | frozen resolved axes, fixed payload/record APIs, launcher and target path; actual acceptance measurements intentionally remain S0B-02/03 | `SUPPORTING ONLY` |
| `STOR-01` behavioral smoke | payload-first/visibility-last `os.replace`; two-host hash/ack success; exact node-local failure | `PASS` behavior, not enough to close loop |
| `FS-08` / environment provenance | actual Lustre/PBS/probe data exists, but skill identity is false and mandatory manifest fields are absent | `BLOCKED` |
| S0B-01 single objective | harness behavior passes; required reproducible Miyabi/run manifest does not | `BLOCKED` |

## Follow-ups

None are merely optional. To unblock a fresh Checker:

1. Freeze the actual `miyabi-development@c3ddfa47...` commit after evaluating
   its diff and update all config/code/test/report identities.
2. Produce a canonical manifest that validates node type, queue, group,
   module list, code/config identity, and actual rank/role/host/PBS mapping.
3. Run a new one-node focused HARDEN and new two-node node-local negative plus
   shared positive/repeat packages from the corrected committed source. Do not
   edit prior raw packages.
4. Submit the corrected commit and evidence to a fresh independent Checker.

Normal `CHECK -> PERSIST` progress/index/traceability writes must wait for that
passing verdict.

## 签署

- Checker session/date: fresh independent Checker `/root/s0b01_checker` / 2026-07-15 Asia/Tokyo
- initial host/workflow: `miyabi-g1`, Miyabi login/control-plane static-only review
- checked commit: `e6ab81ada6306cc5a6f7df0d27d179305c01901f`
- report SHA-256: compute externally after this file is persisted; a file cannot contain its own stable digest.
