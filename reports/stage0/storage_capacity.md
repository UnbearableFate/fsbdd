# Stage 0 metadata and bandwidth capacity result

## Decision

The frozen Miyabi Lustre `RUN_ROOT` passes the corrected S0B-03 capacity
gates. At the largest fixed discovery surface (`M=16`, `F=32`), the slowest
sustained warm cell completed `40463.131` reference operations/s in the
primary and `39403.448/s` in the independent repeat. The frozen demand is
`16 * 32 / 1s = 512/s` and `A-BENCH-03` requires `5120/s`. The worse repeat
therefore supplies `76.960x` steady demand and `7.696x` the required gate.

The slowest 16-stream write round across both corrected runs was `0.323808s`
for 16 concurrent `500MiB` payloads. This is `0.6476%` of `H=50s`, below the
`A-BENCH-04` limit of `10% H = 5s`. Every read verified the deterministic
payload digest and the opposite-host writer mapping.

The Stage 1 storage profile is the frozen target Lustre with `H=50s`, exactly
`M*F` fixed discovery slots, and no history scan. No mitigation is required.
Optional `BENCH-05` object-storage comparison was not run and contributes no
claim.

## Corrected measurement contract

- Target: `/work/xg24i002/x10041/fsbdd/runtime_runs` on the Lustre mount frozen
  by S0B-01.
- Runtime source: `5424daecc01adfaf633450a1449c154c0784edbe`; config digest
  `7b2e1047f86c5f8f9a32caf100272d72f63892679b0670d771077b3de2fac19b`.
- Miyabi skill: `c3ddfa47cadfd9132edf939d359b90b28ab5f8ed`.
- Topology: two distinct compute hosts, 16 launcher ranks, eight ranks per
  host. MPI is launcher-only; application coordination is through bounded
  target-filesystem records.
- Metadata matrix: `M=[4,8,16]`, `F=[8,16,32]`, `stat/read/readdir`,
  first-touch/warm, flat/per-learner fixed-slot layouts, two repetitions.
- I/O matrix: `50/100/250/500MiB`, one and 16 streams, sequential file-fsynced
  replacement writes and opposite-host verified reads, two repetitions, with
  `8MiB` chunks.

Every one of the 216 metadata cells now has a common coordinator interval.
All 16 ranks publish readiness for the same cell; rank 0 starts its monotonic
timer before publishing the filesystem phase; active ranks perform their
work while inactive ranks remain in the same barrier; all 16 publish done;
rank 0 stops after observing every done record. The aggregate rate divides
the operations completed inside that interval by the full start-to-all-done
time. This includes phase visibility, rank-start skew, and done-record
overhead, so it is a conservative filesystem-coordinated rate. The summarizer
rejects missing/duplicate intervals, incomplete 16-rank barriers, phase or
timing provenance mismatches, and coordinator intervals shorter than any
rank-local workload.

## Metadata capacity

Each of the 216 cells reports reference operations, filesystem syscalls,
coordinator elapsed time, maximum rank-local elapsed time, reference/syscall
rates, and sampled `p50/p95/p99/max` latency. Readdir reports both reference
discoveries and syscalls so one directory call that discovers multiple fixed
references is not misrepresented as multiple syscalls.

| Run | Hosts | Minimum M=16/F=32 warm rate | Steady demand | Margin | Gate |
|---|---|---:|---:|---:|---:|
| primary | `mg0029`, `mg0030` | 40463.131/s | 512/s | 79.030x | PASS |
| repeat | `mg0022`, `mg0026` | 39403.448/s | 512/s | 76.960x | PASS |

Both minima are real `read` cells measured over coordinator intervals longer
than their slowest rank-local workload, not sums divided by unrelated local
times and not readdir-derived shortcuts. Primary-to-repeat variation of the
minimum is `-2.619%`; both remain well above `5120/s`.

## Sequential I/O capacity

Primary 16-stream results use coordinator-clock round time. Each row
represents 16 complete payloads of the displayed size.

| Payload per stream | Write round repeats | Aggregate write | Verified-read round repeats | Aggregate read |
|---:|---:|---:|---:|---:|
| 50MiB | 0.053385 / 0.053733s | 14985.6 / 14888.3 MiB/s | 0.065823 / 0.064334s | 12153.9 / 12435.2 MiB/s |
| 100MiB | 0.082617 / 0.084184s | 19366.5 / 19005.9 MiB/s | 0.097968 / 0.099596s | 16331.8 / 16064.9 MiB/s |
| 250MiB | 0.172863 / 0.173939s | 23139.7 / 22996.6 MiB/s | 0.208070 / 0.205574s | 19224.3 / 19457.7 MiB/s |
| 500MiB | 0.321475 / 0.318608s | 24885.3 / 25109.3 MiB/s | 0.363456 / 0.367620s | 22010.9 / 21761.6 MiB/s |

The raw matrix also contains both single-stream repetitions at every size,
individual stream rates, total bytes, and digest results. The repeat's
slowest 16-stream write was `0.323808s`, `0.725%` above the primary maximum
and still only `0.6476%` of `H`.

## Bounded discovery and cleanup

The normal path always inspects exactly 512 fixed references at `M=16,F=32`.
The separate negative history scanner enumerated 0, 1000, and 10000 objects.
Its primary p50 grew from `0.177ms` to `0.363ms` to `1.972ms`; the repeat
measured `0.164ms`, `0.367ms`, and `2.014ms`. The normal bounded path still
inspected exactly 512 references and never scanned history.

Both corrected runs began and ended with 6272 fixed records and 11000
negative-control history records. Each ended with exactly 35 control files
(the frozen limit), zero payload files, and zero temporary files, and each
run-scoped benchmark root was removed.

## Blocker and correction history

Independent review of candidate `4ef5229` correctly blocked its metadata
claim because ranks skipped inactive cells and the summarizer divided a sum
of disjoint rank-local work by only the largest local elapsed time. That
candidate and its two full runs are retained as immutable debugging history
but are excluded from closure. The corrective RED at commit `7402756`
demonstrated that the old summarizer accepted records with no coordinator
interval. Runtime `5424dae` adds the filesystem barrier, interval provenance,
fail-closed summary validation, and a 16-second serialized-work counterexample
that measures only `640/s` and fails the gate.

## Selected immutable evidence

| Phase | Run ID / PBS job | Result | Checksum-manifest SHA-256 |
|---|---|---|---|
| initial RED | `20260714T170406Z-red-n3069021613-840458639ec1` / `2383721.opbs` | five initial requirement failures | `2c3a717928cc671ebc47f2c5060923edf66021c83d0cf620b433c96a6d660a1f` |
| corrective RED | `20260714T173810Z-red-sync-n773a38-730e81e5729d` / `2383846.opbs` | nonsynchronous aggregate accepted, one failure | `4eb003eafdd98dbc25af3757ccdd5478214442af622e02b910dc6ab86665357e` |
| focused tests | `20260714T174032Z-unit-sync-n80f10e-768517b0cb97` / `2383865.opbs` | 8 tests pass | `c74cb681ebfc72a65b645add2135ddf08e431ad4d0aeb6f55a6cf04c5ee257cc` |
| primary full | `20260714T174115Z-green-sync-n654bce-7b2e1047f86c` / `2383872.opbs` | corrected full matrix passes | `677d5b67177b3cde52e719a45022ce2e6257e8feb428a795a3a1608e0efed3bd` |
| full Stage 0 tests | `20260714T174115Z-harden-sync-n9d6803-8ac93ab4b410` / `2383871.opbs` | 57 tests pass | `f027b37211f0eb324a61778edc6ff83f98d500e1a716d7742a54b1a396cf438a` |
| independent repeat | `20260714T174444Z-repeat-sync-n90ad24-7b2e1047f86c` / `2383885.opbs` | second corrected full matrix passes | `ea99f2cfe5da11222033ea4e8d61c9818c6478372627a004edd6605ac401b3e9` |

Every selected package passes its complete `sha256sum -c` manifest. Every
run prefix equals scheduler UTC qtime, all scheduler exits are zero, and the
runtime packages bind commit/config snapshots, two hosts, actual rank
mapping, target filesystem, queue/group/modules, clean worktrees, and clean
skill provenance.

## Acceptance boundary

- `A-BENCH-03`: pass candidate; the worse of two corrected, synchronized full
  runs is `39403.448/s` against `5120/s`.
- `A-BENCH-04`: pass candidate; the worst 16-stream write round is
  `0.323808s` against `5s`.
- `FS-05/06`: normal discovery is exactly `M*F`; reference, control, payload,
  and temporary object counts are bounded and cleanup is verified.
- This is a Stage 0 capacity decision only. It makes no production protocol,
  crash-durability, model, training, loss, GPU, S3/object-store, or nine-node
  claim.
