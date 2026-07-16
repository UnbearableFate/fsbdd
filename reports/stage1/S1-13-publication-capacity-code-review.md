# S1-13 publication-capacity corrective code review

## Scope and boundary

This review follows failed capacity smoke `2393139.opbs` and precedes every
subsequent experiment.  The production FS-Based Decoupled DiLoCo surface is
`src/fsbdd/diloco/`; experiment orchestration, analyzers, contracts, stress
harnesses, and Stage 1 closure support are under `src/fsbdd/auxiliary/`.
`tests/stage1/test_source_architecture.py` asserts that production never imports
the auxiliary tree.  A direct import audit also finds no such dependency.

The reviewed production path covers:

- fixed-slot POSIX publication and bound reads;
- proposal encoding, publication, discovery, and validation;
- global compound-state encoding and publication;
- atomic authority load/commit/cache synchronization;
- readiness observation, selection, and fair scheduling;
- CPU NumPy merge, outer optimizer state, and commit;
- learner snapshot publication and global-fragment adoption.

The reviewed active support path covers the S1-13 learner/syncer roles, gate and
reproduction analyzers, evidence packaging, failure capture, the long-smoke and
formal-long PBS launchers, and the new smallest exact-size capacity preflight.

## Failed-run facts

All 358 Stage 1 tests and all four learner roles passed.  Each learner completed
3,000 optimizer steps, 6,144,000 input tokens, and 60 publication opportunities
per fragment with no pending/in-flight publication or publication error.  The
syncer completed 189 updates but timed out at global vector `[47,46,47,49]`,
below the frozen minimum cycle 59.  Mean update latency was 1.739070100 seconds.

Learner CPU-to-filesystem publication occupied 234.78–244.48 seconds for 240
payloads each.  Global compound payloads were approximately 309–340 MB.  The
production publisher wrote each payload, reopened the same file, reread all
bytes, compared them against the caller bytes, and computed SHA-256 before
visibility.  This second pass occurred for both learner proposals and global
successors and dominated the observed size-proportional latency.

## Findings and corrections

### CR-01: redundant previsibility full-file readback

`PosixStorageBackend.publish` performed a second complete filesystem/page-cache
pass after a successful write.  It did not call `fsync`, so the readback was not
a durability proof; on normal POSIX cache semantics it revalidated bytes just
written by the same process.  This doubled the large-payload data path.

The corrected production mode:

1. hashes the immutable caller bytes once;
2. writes them through an unbuffered exclusive file descriptor, handling
   partial writes until complete;
3. checks the open descriptor is a regular file and its exact `fstat` size;
4. stages and atomically replaces the small visibility record only afterward;
5. retains exact size and SHA-256 validation on every reader and bound reader.

The former bytewise readback remains available only as the explicit
`post_write_readback` diagnostic control.  This makes the optimization directly
measurable without changing algorithm, topology, scheduling, payload format, or
read-side fail-closed behavior.

New regressions cover no production publish-time reopen, the explicit matched
control, invalid verification-mode types, incomplete `fstat` before visibility,
source-derived checksums, same-size corruption, truncation, old-or-complete-new
crash outcomes, and payload-first/visibility-last hook ordering.

### CR-02: insufficient component attribution

The syncer log recorded only total update latency, preventing direct separation
of NumPy merge work from compound commit/publication work.  `ProfileAUpdate`
now records monotonic merge and commit durations, and every durable
`fragment_outer_update` event includes both fields.  This is telemetry only and
does not change selection or update semantics.

### CR-03: prior NumPy correction PBS script did not measure the real update

The old one-node script did not execute the production NumPy merge plus atomic
commit path and used a reusable default output path.  The replacement uses
exclusive shared/result roots, preserves automatic failure capture, applies
bounded timeouts, runs the complete Stage 1 suite, and directly measures one
real NumPy merge and commit at each failed-run fragment size (154,533,888 and
170,115,072 parameter bytes).  The process forbids Torch/CUDA and emits a
machine-readable gate against the directly required 1.355-second update limit.

Per the user-approved recovery simplification, this non-formal preflight records
the actual commit, worktree diff, contract, and PBS script without caller-bound
hashes, per-file checksum inventories, or pre-run Checker authorization.  It
does not execute the obsolete post-write payload-readback mode.  See
`reports/stage1/S1-13-recovery-process-adr.md`.

## Preserved invariants

- payload-first and visibility-last ordering;
- unique immutable payload files and atomic small-record replacement;
- exact record-to-payload byte-count and SHA-256 binding;
- reader-side corruption and truncation rejection;
- old-or-complete-new crash behavior;
- fixed-slot discovery without history scanning;
- compound parameters, outer state, version, and consumption frontier;
- fresh-only Profile A selection and per-fragment version monotonicity;
- no application network data plane and no Torch import in the CPU syncer
  benchmark process.

## Static validation and post-review runtime evidence

Login-node static validation passes: Python bytecode compilation, Ruff, Pyright
with zero errors/warnings, Bash syntax for every S1-13 PBS script and the failure
helper, tracked JSON and current YAML parsing, production-to-auxiliary import
audit, and `git diff --check`.  No project test or model/runtime workload was run
on the login node.

The correction passed lightweight job `2393404.opbs`: all 364 Stage 1 tests
passed, and the real exact-size NumPy merge-plus-commit totals were
0.946533369 and 0.973654636 seconds against the frozen 1.355-second limit.

The following five-node smoke `2393421.opbs` did not exercise learner capacity:
the auxiliary PBS script relocated a resolved config containing a
repository-relative learner-profile path and then used the evidence copy as the
runtime config.  All learners therefore stopped before model loading.  The
production path, capacity result, immutable assets, and prior formal nine-node
result were not implicated.  The reviewed correction keeps the original
runtime config/contract paths and copies them only as provenance.  A static
regression forbids either path from being rebound to the evidence copy.

At 2026-07-16T01:06:03Z the corrected harness passed compileall, Ruff checks,
Pyright with zero errors or warnings, Bash syntax for every active S1-13 PBS
script and the failure helper, JSON/YAML parsing, the production-to-auxiliary
dependency audit, the explicit no-reassignment audit, and `git diff --check`.
No project test or model/runtime workload was run on the login node.  One
replacement five-node smoke is now the next runtime gate; the formal long run
remains contingent on that smoke.

### CR-04: short-horizon terminal margin and missing fail-fast

Replacement job `2393536.opbs` exercised the corrected runtime path. It passed
364 tests, all learners completed 3,000 steps/6,144,000 input tokens, and the
syncer completed 58 updates for every fragment. Mean/max update latency was
1.059704267/1.165041277 seconds, with mean merge/commit components
0.553642429/0.503949820 seconds. This proves that the former publication
capacity bottleneck is corrected.

The job nevertheless failed its 59-of-60 non-formal target. Once all learner
final records existed, fragments 0--2 had only base-57 terminal proposals
against current version 58; fragment 3 had only two of four base-58 proposals.
No fresh `Q=M` update could form, but auxiliary supervision waited about eleven
minutes for the active timeout. The review found two support-layer errors:

1. the 60-opportunity smoke admitted only one unavailable warm-up/terminal
   quorum, making a fixed two-quorum transient disproportionately fail the
   short horizon even though it remains inside the unchanged formal
   2,400-of-approximately-2,441 margin;
2. completion supervision did not fail fast after every learner finalized and
   the latest poll made no progress.

The non-formal threshold is prospectively refrozen at 58 of 60 and documented
in `S1-13-long-schedule-adr.md`; job `2393536` remains failed evidence. Formal
minimum cycle 2,400 is unchanged. The syncer now emits a structured terminal
progress error immediately when the completed run cannot reach its target, and
focused regressions cover both changes. No production protocol, algorithm,
payload, or model/data behavior changes.

At 2026-07-16T01:41:04Z the CR-04 correction passed compileall, Ruff, Pyright
with zero errors/warnings, every active S1-13 PBS/failure-helper Bash syntax
check, JSON/YAML parsing, production dependency and runtime-path audits, exact
58-versus-2,400 contract assertions, and `git diff --check`. No project test or
runtime workload was executed on the login node.

### CR-05: redundant current-to-historical contract identity binding

Job `2394044.opbs` stopped in its pre-workload suite after 364 passes and one
failure. The prospective gate contract changed SHA-256 to `7dd56fab...03ec`,
but the current reproduction contract still bound `8fb3734e...3f84`; the
evidence contract also retained current 59-of-60 prose. The existing regression
failed before any learner or syncer runtime.

Review found that the reproduction contract describes a completed two-node
recovery path and is not consumed by the five-node smoke. Rebinding it whenever
the current gate contract changes adds no runtime or reproducibility guarantee;
Git already records both tracked versions. The active suite therefore no longer
cross-binds those hashes. It checks the current 58-cycle smoke threshold,
unchanged formal 2,400-cycle threshold, workload semantics, topology, schemas,
paths, and the external asset bundle directly. The reproduction contract stays
historical. The evidence contract prose is corrected to 58 of 60. No production
code or runtime behavior is changed by CR-05.

### CR-06: remaining formal-long submission identity flow

The formal-long PBS script formerly required the caller to calculate and pass a
Git commit plus three file hashes, then recalculated all three in the job. This
duplicated Git/shared-filesystem identity without detecting an algorithm or
runtime error and created the same manual synchronization risk seen in job
`2394044`.

The script now accepts repository, config, gate-contract, and external-asset
paths. It requires the repository worktree to be clean, records the actual
commit and worktree status, derives config/gate identities inside the job for
runtime records, and validates only the external asset marker and root against
the resolved config. The final post-run evidence inventory remains because the
untracked result package can be copied after the allocation; it is not a
submission gate or a tracked cross-contract binding.

At `2026-07-16T02:04:28Z`, CR-05/CR-06 passed compileall, Ruff, Pyright with
zero errors/warnings, every active S1-13 PBS/failure-helper Bash syntax check,
tracked JSON/current YAML parsing, semantic 58-versus-2,400 checks, the
formal-long internal-identity/external-asset audit, production dependency and
runtime-path audits, external marker validation, and `git diff --check`. No
project test or runtime ran on the login node.
