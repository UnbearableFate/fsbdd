# Stage 1 storage capability decision

> **BLOCKED:** candidate `4ef5229` does not establish `A-BENCH-03` because its
> aggregate metadata rate combines nonsynchronous rank-local intervals. The
> no-mitigation decision below is withdrawn pending a coordinator-timed rerun.

## Decision

`PASS`: use the frozen Miyabi target Lustre path for Stage 1 with the profile
`H=50s_fixed_slot_discovery_no_history_scan`.

No capacity mitigation is required. This decision is conditional on Stage 1
preserving exactly `M*F` fixed latest-reference slots, bounded control state,
payload-first/visibility-last publication, and no historical proposal or
outer-update scan on the normal discovery path.

## Recomputed gates

| Acceptance | Frozen calculation | Worst of two full runs | Decision |
|---|---|---:|---|
| `A-BENCH-03` | `M*F/poll = 16*32/1s = 512/s`; require `10x = 5120/s` | `45137.772/s` (`88.160x` demand) | PASS |
| `A-BENCH-04` | `10% * H = 0.1*50s = 5s` | `0.325865s` for a 16-stream write round | PASS |

The normal bounded-discovery control inspected 512 references regardless of
whether the separate negative-control history contained 0, 1000, or 10000
objects. Both full runs left fixed/history object counts unchanged, stayed at
the 35-file control limit, and removed every payload and temporary object.

## Stage 1 constraints

- Keep `H=50s` unless a later loop changes it through an explicit, checked
  configuration/ADR; this benchmark does not authorize an implicit change.
- Discovery must remain proportional to the fixed `M*F` surface, not retained
  history.
- Use the S0B-02 proven same-directory, file-fsynced temporary plus
  `os.replace` visibility primitive; this decision does not add a power-loss
  durability claim.
- Continue reporting reference discoveries separately from filesystem
  syscalls, especially for readdir.
- Re-run the capacity gate if Stage 1 materially changes payload sizes,
  concurrency, directory layout, target mount, or synchronization period.

## Evidence and scope

Primary: `20260714T171502Z-green-n2d6c-cdd8a3e5e65a`, job `2383755.opbs`,
checksum-manifest SHA-256
`0aaae840eb301cf3ef6902fe79beb238b93535fe729827aaf6300f8b6033112e`.
Independent repeat: `20260714T171837Z-repeat-n7a4e-cdd8a3e5e65a`, job
`2383771.opbs`, checksum-manifest SHA-256
`c53cb77728b38f80df4218fd51a0e5186fc3075210b72ad80dba51be60625a40`.
Full analysis is in `reports/stage0/storage_capacity.md`.

Optional `BENCH-05` was not executed; no S3/object-storage claim or fallback
decision is made. No model, dataset, Torch, GPU, training, quality, or
nine-node claim is made.
