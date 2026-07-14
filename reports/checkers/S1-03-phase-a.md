# S1-03 Checker Phase A

## Overall verdict

`ADMISSIBLE`

No blocking, high-severity, or medium-severity precheck finding was found in
the controlling recheck. Checked candidate `e5fc7db5a5f52ec7c407e4867fffc1037d831c5b`
may proceed to exactly one replacement L2 two-node run at frozen root
`runtime_runs/S1-03/formal-l2-r2`. The prior `formal-l2` package is preserved
but is ineligible for selection. This remains a Phase A authorization only;
Phase B must validate the replacement package and observations before closure.

## Recheck cycle 2 — qtime identity correction

### Independent scope

- Same independent Checker context: `/root/s1_03_checker`; Maker chat
  inherited: `false`.
- Prior / checked candidate:
  `a09b7239aeea604b19c3be888a3bba1a077ef5e5` /
  `e5fc7db5a5f52ec7c407e4867fffc1037d831c5b`. The substantive correction is
  commit `31c6f5da8f2c822076d860f197c200fef4d925a7`; `e5fc7db` changes only the
  loop-state record after the corrected preflight. `git diff --check` passed.
- Checker remained on `miyabi-g1` without PBS allocation and performed only
  static control-plane inspection. No project Python, runtime test, training,
  dependency, or Torch change was performed by the Checker.

### Findings, ordered by severity

#### MEDIUM, resolved — the first L2 package has a false UTC timestamp and is rejected

The preserved first package records raw PBS `qtime = Wed Jul 15 08:13:21 2026`
at `runtime_runs/S1-03/formal-l2/env/qstat.txt:29`, but labels it
`2026-07-15T08:13:21Z` in both run and scheduler identity at
`runtime_runs/S1-03/formal-l2/manifest.json:75-92`. Miyabi supplied that
un-zoned value in JST, whose correct UTC instant is
`2026-07-14T23:13:21Z`. Its application summary did pass 500 publications,
1,372 reader observations, zero invalid reads, distinct hosts, and the crash
matrix (`runtime_runs/S1-03/formal-l2/analysis/storage-summary.json:1-58`), and
all 26 checksum entries still verify, but the false time identity makes the
package semantically ineligible even though the generic structural validator
reported 25/25 admissible. The package remains intact at `formal-l2`; no
contract or requirement-matrix row selects it.

#### LOW, resolved — the correction has an effective RED and explicit conversion check

The new regression failed on the old script at the target PBS contract API,
not at import/collection: one semantic failure at
`runtime_runs/S1-03/red-qtime-7a1ca44/stdout/pytest.log:1-20`, SHA-256
`448eec573d3f757a481cb7749e6665bf057c02e9b3143515ffc262e12fe61e7d`.
The corrected script appends the source timezone before asking GNU `date` for
UTC (`pbs/stage1_s1_03_storage.pbs:70-71`). The corrected preflight executes the
representative raw value and records `2026-07-14T23:13:21Z` at
`runtime_runs/S1-03/preflight-qtime-31c6f5d/stdout/qtime-conversion.log:1`,
rather than relying only on the literal-source regression.

#### INFO — replacement identity and evidence destinations are consistently frozen

The PBS default, evidence contract, all nine requirement rows, loop state, and
checked documentation now select only `formal-l2-r2`
(`pbs/stage1_s1_03_storage.pbs:11-14`;
`reports/stage1/S1-03-evidence-contract.json:3-5`;
`reports/stage1/S1-03-requirement-matrix.csv:2-10`;
`plans/FS_DILOCO_CODEX_EXECUTION_PLAN/plans/loop_states/S1-03.yaml:31-38`).
The root did not exist at recheck time, so `fsbdd evidence init` at
`pbs/stage1_s1_03_storage.pbs:56` retains fail-if-exists behavior. The prepared
clean execution worktree was independently observed clean at exact candidate
`e5fc7db5a5f52ec7c407e4867fffc1037d831c5b`.

#### INFO — corrected preflight is internally admissible and matches the candidate inputs

Independent `sha256sum -c` passed manifest plus all 18 payload entries at
`runtime_runs/S1-03/preflight-qtime-31c6f5d/checksums.sha256:1-19`.
Manifest SHA-256
`eaae119c5d549e50f821df2166ab00efcab83a22d7faa0b355c85f3179c3d0e4`
matches `validation-summary.json:5`; listed/actual is 18/18 with zero errors
(`runtime_runs/S1-03/preflight-qtime-31c6f5d/validation-summary.json:2-7`).
It binds clean correction commit `31c6f5d`, canonical config, unchanged source
and skill identities, and the L1 role/path identity
(`runtime_runs/S1-03/preflight-qtime-31c6f5d/manifest.json:24-71`). Its frozen
PBS and evidence-contract copies are byte-identical to the checked files, with
SHA-256 `eac78837...e5d67` and `bbe1595a...77fe`, respectively. The full Stage
1 suite reports 68 passed and the matrix reports exact 9/9 coverage
(`runtime_runs/S1-03/preflight-qtime-31c6f5d/stdout/pytest.log:1-2`;
`runtime_runs/S1-03/preflight-qtime-31c6f5d/analysis/requirements.json:1`).

#### INFO — replacement L2 remains authoritative for the same bounded claims

The correction changes only timestamp identity and destination selection; it
does not alter storage implementation, topology, reader aggregation, crash
matrix, or component-only acceptance boundary. The earlier Phase A protocol
analysis below therefore remains applicable. A replacement package is
selectable only if its raw `env/qstat.txt` qtime, explicitly-JST-converted UTC
manifest timestamps, code commit, two-host role map, mount/module/scheduler
identity, summary, and checksum inventory all agree in Phase B.

## Cycle 1 record — historical initial precheck

- Checker context: `/root/s1_03_checker`; Maker chat inherited: `false`.
- Base / checked commit:
  `754f12d415878bcebba660db967afcfceb852ccf` /
  `a09b7239aeea604b19c3be888a3bba1a077ef5e5`.
- Candidate diff: 10 files, 1,287 insertions and 3 deletions; `git diff
  --check` passed. Commit `a09b723` changes only the S1-03 loop-state handoff
  after implementation commit `e64b6d1`.
- Checker host classification: `miyabi-g1`, with no `PBS_JOBID` or
  `PBS_NODEFILE`; only Git/file/hash/shell/PBS-control-plane inspection was
  performed. No project Python, test, training, or GPU runtime was run by the
  Checker, and no Torch version or dependency changed.

### Findings, ordered by severity

#### LOW — closure is correctly limited to component evidence

The proposed run can authoritatively close the S1-03 storage-primitive portions
of `A-PROP-01` and `A-GLOBAL-01`, but not their later proposal/global business
semantics. The matrix states those boundaries explicitly at
`reports/stage1/S1-03-requirement-matrix.csv:9-10`, consistent with the loop
non-goals and persistence output at
`plans/FS_DILOCO_CODEX_EXECUTION_PLAN/loops/stage1/S1-03.md:18-22,59-65`.
Full proposal closure remains S1-05 and atomic global-state/frontier closure
remains S1-11. Phase B must preserve this component-only wording.

#### INFO — implementation uses the authorized storage protocol

- The public backend surface is exactly fixed-slot `publish` and `read`
  (`src/fsbdd/storage.py:87-106`). The POSIX backend writes a UUID-named payload,
  reads it back and verifies length/SHA-256, stages a small record, and exposes
  it only through `os.replace` (`src/fsbdd/storage.py:245-295`). No lock,
  append-based commit, directory transaction, or history scan is on the
  correctness path.
- The record binds run, fragment-map, fragment, version, sequence, dtype,
  shape, base-content, byte count, checksum, payload identity, and a constrained
  payload path (`src/fsbdd/storage.py:57-72,157-196`). Reads follow one visibility
  record, validate expectations, and return payload bytes only after exact
  size/checksum validation; missing or not-yet-complete payloads are retried
  under an explicit eventual-readability timeout
  (`src/fsbdd/storage.py:297-340`).
- The crash boundary leaves the old record authoritative before replacement
  and the complete new payload authoritative after replacement. The source
  test matrix covers all four frozen interruption points
  (`tests/stage1/test_s1_03_storage.py:73-91`), and concurrent readers plus the
  no-directory-scan guard are at
  `tests/stage1/test_s1_03_storage.py:94-104,158-185`.

#### INFO — RED and focused evidence target semantic failures

The S1-03 RED surrogate rejects a listing-capable interface and acceptance of
partial bytes with a wrong checksum
(`tests/red_cases/s1_03_naive_storage_contract_surrogate.py:6-40`). The frozen
contract reuses the exact S0B-02 unsafe direct-overwrite control rather than
inventing a new detector (`reports/stage1/S1-03-evidence-contract.json:28-31`).
That selected prerequisite package records 5,208 partial/torn observations in
`evidence/raw/S0B-02/20260714T162735Z-unsafefix-unsafefix-n2e5c-006abf2a3b43/results/stress-summary.json`.
The finalized preflight package records the full Stage 1 suite as 68 passed in
`runtime_runs/S1-03/preflight-e64b6d1/stdout/pytest.log:1-2` and requirement-set
equality as 9/9 in
`runtime_runs/S1-03/preflight-e64b6d1/analysis/requirements.json:1`.

#### INFO — frozen identity and package are admissible

- The preflight manifest binds clean code commit `e64b6d1`, canonical config
  digest `dd1dc08d...a3c50`, both authority hashes, skill commit
  `c3ddfa47...d5ed`, initial/compute host, actual role, and absolute project and
  evidence paths
  (`runtime_runs/S1-03/preflight-e64b6d1/manifest.json:24-67`). The difference
  from checked commit `a09b723` is loop-state persistence only; the formal PBS
  script requires the clean execution worktree to equal the supplied
  `EXPECTED_COMMIT` (`pbs/stage1_s1_03_storage.pbs:38-45`). The prepared clean
  worktree was independently observed detached at exact commit `a09b723` and
  clean.
- Independent `sha256sum -c` verification passed every one of the 18 manifest
  and evidence entries in
  `runtime_runs/S1-03/preflight-e64b6d1/checksums.sha256:1-18`. The manifest
  SHA-256 is
  `258ca21cbff18a88cb51dc7328f9338a97f9faa7ceb68d1cd892f05ace396164`,
  matching `validation-summary.json:5`; listed/actual is 17/17 with zero errors
  (`runtime_runs/S1-03/preflight-e64b6d1/validation-summary.json:2-7`). Frozen
  contract and PBS copies are byte-identical to the checked repository files.

#### INFO — the proposed L2 run is fit for the component claims

- The frozen topology is two nodes, two ranks, one rank per node, filesystem-only
  application coordination, and MPI launcher-only use
  (`reports/stage1/S1-03-evidence-contract.json:3-20`). The PBS script requests
  two `debug-g` nodes, freezes group/walltime/module, rejects a dirty or wrong
  commit/skill checkout, records nodefile/module/qstat/mount evidence before
  launch, and maps one role per node
  (`pbs/stage1_s1_03_storage.pbs:2-18,34-69,83-98`). `bash -n` passed; live
  `qstat --rsc/-x` showed `debug-g` enabled for project `xg24i002`, 1–16 nodes,
  and a 30-minute limit, matching the request.
- The run performs 500 payload-first publications on the shared target root
  while the other host continuously reads the fixed slot. The reader's common
  interval starts only after the explicit ready handshake and ends only after
  observing the final sequence (`src/fsbdd/storage_stress.py:113-167`). The
  analyzer fails on same-host roles, a missing final sequence, any invalid
  observation, or zero observations, and adds the four-point old-or-complete-new
  crash matrix (`src/fsbdd/storage_stress.py:185-264`).
- The manifest builder verifies two distinct allocated hosts and exact actual
  writer/reader-to-nodefile equality, then binds scheduler, module, path, code,
  config, source, and skill identities
  (`src/fsbdd/storage_stress.py:267-318`). The PBS tail finalizes and validates
  the package (`pbs/stage1_s1_03_storage.pbs:100-119`). Therefore a completed
  run with `analysis/storage-summary.json` status `pass`, zero invalid reads,
  both distinct roles, the expected crash outcomes, and an independently valid
  checksum inventory is authoritative for the listed component claims.

## Phase B mandatory checks

Phase B must reject closure unless the formal package is fail-if-exists,
finalized, checksum-valid, rooted at `formal-l2-r2`, bound to checked commit
`e5fc7db`, and shows two distinct compute hosts matching the PBS nodefile;
module/scheduler/mount identity; correct explicit JST-to-UTC conversion of the
raw PBS qtime; 500 publications with the final sequence observed; zero invalid
reads; all four crash outcomes; and no test or requirement-validation failure.
The preserved `formal-l2` package must remain rejected for selection. Phase B
must continue to label `A-PROP-01` and `A-GLOBAL-01` as component evidence.
