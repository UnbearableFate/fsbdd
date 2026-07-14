# Stage 0 metadata and bandwidth capacity result

## Decision

The frozen Miyabi Lustre `RUN_ROOT` passes the S0B-03 capacity gates. At the
largest fixed discovery surface (`M=16`, `F=32`), the slowest sustained warm
profile completed `45137.772` reference operations/s. The frozen demand is
`16 * 32 / 1s = 512` operations/s and the `A-BENCH-03` gate is `5120`
operations/s, so measured capacity is `88.160x` steady demand and `8.816x`
the required gate.

The slowest 16-stream write round was `0.320654s` for 16 concurrent `500MiB`
payloads. This is `0.6413%` of `H=50s`, below the `A-BENCH-04` limit of
`10% H = 5s`. Every read verified the deterministic payload digest. A full
repeat on two different compute hosts also passed, at `45339.331` reference
operations/s and a maximum `0.325865s` write round.

The Stage 1 storage profile is therefore the frozen target Lustre with
`H=50s`, exactly `M*F` fixed discovery slots, and no history scan. No
mitigation is required. Optional `BENCH-05` object-storage comparison was not
run and contributes no claim.

## Measurement contract

- Target: `/work/xg24i002/x10041/fsbdd/runtime_runs` on the Lustre mount frozen
  by S0B-01.
- Runtime source: `6e5508bd3a743e94c5a81146d68a9424c72e8b23`; config digest
  `cdd8a3e5e65a3087ce3593a550f0f2040cc2bb1dd612e436fa8b8fd69eaede85`.
- Miyabi skill: `c3ddfa47cadfd9132edf939d359b90b28ab5f8ed`.
- Topology: two distinct compute hosts, 16 launcher ranks, eight ranks per
  host. MPI is launcher-only; application coordination uses 35 bounded
  control files on the target filesystem.
- Metadata matrix: `M=[4,8,16]`, `F=[8,16,32]`, `stat/read/readdir`,
  first-touch/warm, flat/per-learner fixed-slot layouts, and two repetitions.
- I/O matrix: `50/100/250/500MiB`, one and 16 streams, sequential file-fsynced
  replacement writes and opposite-host verified reads, two repetitions, with
  `8MiB` chunks.
- Timing: coordinator-local monotonic round time; no cross-host timestamps are
  subtracted.

## Metadata capacity

The primary package contains all `3 * 3 * 3 * 2 * 2 * 2 = 216` metadata
matrix cells. Each cell reports reference operations, filesystem syscalls,
elapsed time, reference and syscall rates, and sampled `p50/p95/p99/max`
latency. Readdir reports both reference discoveries and syscalls so a single
directory call that discovers multiple fixed references is not mistaken for
multiple syscalls.

| Run | Hosts | Minimum M=16/F=32 warm reference rate | Steady demand | Margin | Gate |
|---|---|---:|---:|---:|---:|
| primary | `mg0019`, `mg0020` | 45137.772/s | 512/s | 88.160x | PASS |
| repeat | `mg0014`, `mg0016` | 45339.331/s | 512/s | 88.553x | PASS |

The primary minimum is the per-learner `read` profile, not a readdir-derived
reference-rate shortcut. Primary-to-repeat variation of the minimum is
`0.447%`; both remain far above the frozen `5120/s` threshold.

## Sequential I/O capacity

Primary 16-stream results are coordinator-clock round times and aggregate
throughputs. Each row represents 16 complete payloads of the displayed size.

| Payload per stream | Write round repeats | Aggregate write | Verified-read round repeats | Aggregate read |
|---:|---:|---:|---:|---:|
| 50MiB | 0.052636 / 0.054297s | 15198.6 / 14733.9 MiB/s | 0.065433 / 0.067463s | 12226.2 / 11858.4 MiB/s |
| 100MiB | 0.084179 / 0.083242s | 19007.0 / 19221.0 MiB/s | 0.099631 / 0.103829s | 16059.3 / 15410.0 MiB/s |
| 250MiB | 0.171803 / 0.172743s | 23282.5 / 23155.7 MiB/s | 0.206239 / 0.201876s | 19395.0 / 19814.2 MiB/s |
| 500MiB | 0.320654 / 0.318514s | 24949.0 / 25116.7 MiB/s | 0.357382 / 0.358446s | 22385.0 / 22318.5 MiB/s |

The raw matrix also contains both repetitions of the single-stream
write/read profiles at every size, individual stream rates, total bytes, and
digest-verification results. The repeat's slowest 16-stream write round was
`0.325865s`, a `1.626%` increase over the primary maximum and still only
`0.652%` of `H`.

## Bounded discovery and cleanup

The normal path always inspects exactly 512 fixed references at `M=16,F=32`.
As a negative control, a separate history scanner enumerated 0, 1000, and
10000 historical objects. Its primary p50 grew from `0.164ms` to `0.362ms`
to `1.913ms`; the repeat measured `0.160ms`, `0.367ms`, and `1.935ms`.
The bounded path continued to inspect exactly 512 fixed references in every
case and never scanned history.

The primary and repeat both began and ended with 6272 fixed records and
11000 negative-control history records. Each finished with exactly 35 control
files (the frozen limit), zero payload files, and zero temporary files. The
benchmark therefore does not create history-dependent normal state or leave
payload/temp growth behind.

## Immutable evidence

| Phase | Run ID / PBS job | Result | Checksum-manifest SHA-256 |
|---|---|---|---|
| RED | `20260714T170406Z-red-n3069021613-840458639ec1` / `2383721.opbs` | five requirement failures | `2c3a717928cc671ebc47f2c5060923edf66021c83d0cf620b433c96a6d660a1f` |
| focused tests | `20260714T171444Z-unit2-n8b3d-6ccb57f1fbc3` / `2383754.opbs` | 7 tests pass | `2d1e7e1c22897ac9a96cfab4eca519b06c0d05ab48cd1afdd5ec58931737fb27` |
| primary full | `20260714T171502Z-green-n2d6c-cdd8a3e5e65a` / `2383755.opbs` | full matrix passes | `0aaae840eb301cf3ef6902fe79beb238b93535fe729827aaf6300f8b6033112e` |
| full Stage 0 tests | `20260714T171520Z-harden-n5f9b-a8d36b961469` / `2383756.opbs` | 56 tests pass | `ec0b20d49643ac21c3eaf805465b458f13b04ab5e6cd5f1ac1a7701eed9c0905` |
| independent repeat | `20260714T171837Z-repeat-n7a4e-cdd8a3e5e65a` / `2383771.opbs` | second full matrix passes | `c53cb77728b38f80df4218fd51a0e5186fc3075210b72ad80dba51be60625a40` |

Every selected package passes its complete `sha256sum -c` manifest. Each run
prefix equals scheduler UTC qtime, all selected scheduler exits are zero, and
the runtime packages bind commit/config snapshots, both hosts, actual rank
mapping, target filesystem, queue/group/modules, and clean skill provenance.

## Acceptance boundary

- `A-BENCH-03`: pass candidate; the slowest sustained warm M=16/F=32 rate is
  `45137.772/s` against a required `5120/s`.
- `A-BENCH-04`: pass candidate; the worst 16-stream write round is
  `0.325865s` across two full runs against `5s`.
- `FS-05/06`: normal discovery is exactly `M*F`; reference, control, payload,
  and temporary object counts are bounded and cleanup is verified.
- This is a Stage 0 capacity decision only. It makes no production protocol,
  crash-durability, model, training, loss, GPU, S3/object-store, or nine-node
  claim.
