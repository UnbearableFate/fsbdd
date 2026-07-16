# S1-13 recovery-process ADR

Status: accepted by explicit user direction on 2026-07-16.

## Context

The recovery after capacity smoke `2393139.opbs` accumulated a separate exact
candidate commit, manually supplied script/contract hashes, per-attempt Checker
authorization, and complete evidence-tree checksums before even the smallest
one-node measurement.  Those controls delayed the direct performance question
and did not improve the FS-Based Decoupled DiLoCo algorithm.

The user explicitly requested a less restrictive process and separately
questioned redundant file verification.  The failed run and its cause remain
recorded in `reports/stage1/S1-13-experiment-failures.json`.

## Decision

For the remaining S1-13 recovery ladder and formal long submission:

- the one-node capacity preflight and one five-node non-formal capacity smoke
  do not require a separate authorization commit or Checker Phase A;
- scripts record the actual Git commit, worktree status, worktree diff, PBS
  script, and runtime contract, but do not require caller-supplied file hashes;
- routine preflight evidence is not expanded into a per-file checksum inventory;
- a completed historical reproduction contract is not rebound whenever a
  current gate contract changes; current admission uses semantic threshold,
  topology, workload, path, and schema checks instead of a cross-contract hash;
- the preflight runs the complete Stage 1 suite and directly measures the real
  CPU NumPy merge plus atomic commit at both failed-run fragment sizes;
- the obsolete post-write payload-readback mode is not executed by the
  preflight;
- every failed experiment still receives a raw failure record followed by a
  reviewed cause and expected solution before any next experiment;
- the formal long job accepts paths rather than caller-supplied repository-file
  hashes or a caller-supplied commit. It requires a clean Git worktree, records
  the actual commit, computes runtime identities internally, and validates only
  the external asset marker against the resolved config before training;
- the formal long run and final Stage 1 closure still require an immutable run
  manifest and independent final Checker review, but not repeated Checker
  authorization for every intermediate candidate.

The protocol integrity rule is unchanged: a writer hashes the immutable source,
checks a complete exact-size write, and publishes atomically; a reader validates
the payload during the read it already has to perform.  No writer performs an
additional full-file readback.

The post-run formal evidence inventory is also unchanged. Those results are not
Git-tracked and may be copied after the allocation, so their final checksums
protect retained external evidence; they are not a pre-run admission gate and
are never copied into another tracked contract as a manually synchronized
identity.

## Consequences and rollback

This changes execution governance, not algorithm, topology, thresholds, or
acceptance semantics.  It removes redundant audit latency while preserving
reproducibility from the captured commit/diff/config/script and preserving the
mandatory failure ledger.  Reinstating pre-run Checker authorization or a
manual hash gate requires a new user decision; protocol payload integrity can
be changed only through a separate protocol ADR and regression update.
