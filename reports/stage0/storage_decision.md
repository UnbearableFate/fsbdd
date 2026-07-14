# Stage 1 storage capability decision

## Decision

`PASS`: use the frozen Miyabi target Lustre path for Stage 1 with profile
`H=50s_fixed_slot_discovery_no_history_scan`.

No capacity mitigation is required. This decision is conditional on Stage 1
preserving exactly `M*F` fixed latest-reference slots, bounded control state,
payload-first/visibility-last publication, and no history scan on the normal
discovery path.

## Recomputed gates

| Acceptance | Frozen calculation | Worse of two corrected full runs | Decision |
|---|---|---:|---|
| `A-BENCH-03` | `M*F/poll = 16*32/1s = 512/s`; require `10x = 5120/s` | `39403.448/s` (`76.960x` demand, `7.696x` gate) | PASS |
| `A-BENCH-04` | `10% * H = 0.1*50s = 5s` | `0.323808s` for a 16-stream write round | PASS |

Every metadata cell uses a rank-0 monotonic interval from filesystem-barrier
release through all 16 done records. The result therefore includes rank skew
and control-record overhead and does not aggregate disjoint local intervals.
The normal bounded-discovery control inspected 512 references regardless of
whether the separate history contained 0, 1000, or 10000 objects. Both runs
left fixed/history counts unchanged, stayed at the 35-file control limit,
removed every payload/temp object, and removed the run root.

## Stage 1 constraints

- Keep `H=50s` unless a later loop changes it through an explicit, checked
  configuration/ADR.
- Discovery must remain proportional to fixed `M*F`, not retained history.
- Use the S0B-02 proven same-directory, file-fsynced temporary plus
  `os.replace` visibility primitive; this adds no power-loss durability claim.
- Continue reporting reference discoveries separately from filesystem
  syscalls, especially for readdir.
- Re-run the capacity gate if Stage 1 materially changes payload sizes,
  concurrency, directory layout, target mount, or synchronization period.

## Evidence and scope

Primary: `20260714T174115Z-green-sync-n654bce-7b2e1047f86c`, job
`2383872.opbs`, checksum-manifest SHA-256
`677d5b67177b3cde52e719a45022ce2e6257e8feb428a795a3a1608e0efed3bd`.
Independent repeat: `20260714T174444Z-repeat-sync-n90ad24-7b2e1047f86c`,
job `2383885.opbs`, checksum-manifest SHA-256
`ea99f2cfe5da11222033ea4e8d61c9818c6478372627a004edd6605ac401b3e9`.
Full analysis and correction history are in
`reports/stage0/storage_capacity.md`.

Optional `BENCH-05` was not executed; no S3/object-storage claim or fallback
decision is made. No model, dataset, Torch, GPU, training, quality, or
nine-node claim is made.
