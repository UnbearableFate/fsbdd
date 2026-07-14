# S1-03 Checker Phase A

## Overall verdict

`ADMISSIBLE`

No blocking, high-severity, or medium-severity precheck finding was found. The
checked candidate may proceed to its single frozen L2 two-node run. This is a
Phase A authorization only; Phase B must validate the finalized L2 package and
the observed results before the loop or either acceptance ID can close.

## Independent scope

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

## Findings, ordered by severity

### LOW — closure is correctly limited to component evidence

The proposed run can authoritatively close the S1-03 storage-primitive portions
of `A-PROP-01` and `A-GLOBAL-01`, but not their later proposal/global business
semantics. The matrix states those boundaries explicitly at
`reports/stage1/S1-03-requirement-matrix.csv:9-10`, consistent with the loop
non-goals and persistence output at
`plans/FS_DILOCO_CODEX_EXECUTION_PLAN/loops/stage1/S1-03.md:18-22,59-65`.
Full proposal closure remains S1-05 and atomic global-state/frontier closure
remains S1-11. Phase B must preserve this component-only wording.

### INFO — implementation uses the authorized storage protocol

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

### INFO — RED and focused evidence target semantic failures

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

### INFO — frozen identity and package are admissible

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

### INFO — the proposed L2 run is fit for the component claims

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
finalized, checksum-valid, bound to checked commit `a09b723`, and shows two
distinct compute hosts matching the PBS nodefile; module/scheduler/mount
identity; 500 publications with the final sequence observed; zero invalid
reads; all four crash outcomes; and no test or requirement-validation failure.
It must continue to label `A-PROP-01` and `A-GLOBAL-01` as component evidence.
