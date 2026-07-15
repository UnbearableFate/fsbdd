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

### CR-03: prior NumPy correction PBS script was not an admissible capacity test

The existing one-node script was not bound to an exact clean commit or frozen
contract, used a non-exclusive default output path, had no matched large-payload
control, and did not finalize a checksum inventory.  It therefore could not
support another five-node allocation.

The rewritten preflight is exact-commit and contract bound, uses exclusive
shared/result roots, preserves automatic failure capture, applies sequential
timeouts within PBS walltime, compares both verification modes at the exact
309,071,808- and 340,239,052-byte failed-run sizes, validates every published
payload outside the timed interval, forbids Torch/CUDA in the benchmark process,
emits a machine-readable gate, and finalizes checksums.

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

## Static validation and remaining runtime gate

Login-node static validation passes: Python bytecode compilation, Ruff, Pyright
with zero errors/warnings, Bash syntax for every S1-13 PBS script and the failure
helper, tracked JSON and current YAML parsing, production-to-auxiliary import
audit, and `git diff --check`.  No project test or model/runtime workload was run
on the login node.

The correction is not runtime-accepted yet.  The next admissible action is the
single one-node exact-size preflight after same-Checker Phase A.  A pass must
return to the same Checker before a separately authorized five-node capacity
retry; formal long-run submission remains unauthorized.
