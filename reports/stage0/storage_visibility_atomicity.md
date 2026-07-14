# Stage 0 visibility and atomicity result

## Decision

The frozen Miyabi Lustre `RUN_ROOT` passes the S0B-02 visibility and atomic
replacement gates. Empty and loaded writer-side publication-to-reader-ack
`p99` values are `7.939ms` and `7.801ms`, respectively, versus the frozen
`2.5s` limit. Four concurrent readers observed no incomplete record during
`100000` replacements at each of `256` and `4096` bytes. Both writer-kill
publication boundaries passed.

This is a Stage 0 storage-primitive result. It does not qualify metadata
capacity, payload bandwidth, production protocol correctness, training, GPU,
model, loss, or nine-node behavior.

## Measurement contract

- Target: `/work/xg24i002/x10041/fsbdd/runtime_runs` on the Lustre mount frozen
  by S0B-01.
- Runtime source: `78e5992383d3fc6900edeb80f6055451bb08933b` with config digest
  `006abf2a3b433ca2e092fcb5d33914b37eac9875d06b348ba910c786dd7bd28a`.
- Miyabi skill: `c3ddfa47cadfd9132edf939d359b90b28ab5f8ed`.
- Topology: PBS job `2383537.opbs`, writer `mg0030`, reader `mg0031`; MPI is
  launcher-only and all application coordination uses the target filesystem.
- Clock: each sample begins after the writer's `os.replace` completes and ends
  when that same writer observes the reader acknowledgement. This
  single-clock round trip is a conservative visibility upper bound; no
  cross-host timestamps are subtracted.

## Visibility results

| Profile | Samples | p50 | p95 | p99 | max | Limit |
|---|---:|---:|---:|---:|---:|---:|
| empty | 1000 | 5.805ms | 7.130ms | 7.939ms | 163.547ms | 2.5s |
| loaded | 1000 | 5.572ms | 7.010ms | 7.801ms | 8.834ms | 2.5s |

Cold and warm states each contribute `500` samples to each profile. The
loaded profile began only after `1000` background replacements, advanced from
`1000` to `5958` while samples were taken, and recorded `7497` valid reader
observations with zero incomplete background records. The immutable raw trace
contains all `2000` samples.

## Atomicity results

| Record bytes | Replacements | Raw publication records | Readers | Reader observations | Violations |
|---:|---:|---:|---:|---|---:|
| 256 | 100000 | 100000 | 4 | 166406, 165979, 165117, 164840 | 0 |
| 4096 | 100000 | 100000 | 4 | 165990, 165542, 165992, 166207 | 0 |

Each publication transcript records the exact sequence, record checksum,
publication-complete monotonic timestamp, and timing origin. Reader checkpoint
traces preserve observations, sequence progress, timestamps, rolling
transcript digests, and a final observation of sequence `99999`. The raw
violation scanner is empty. Records are written completely to unique
same-directory temporaries, file-fsynced, and made visible with `os.replace`;
this claim concerns atomic visibility rather than directory-entry crash
durability.

Actual `SIGKILL` probes cover both `before_visibility` and `after_visibility`.
The pre-visibility kill exposed no record; the post-visibility kill left the
complete published record readable.

## Detector negative control

PBS job `2383599.opbs` ran the deliberately unsafe direct-overwrite path on
`mg0015` and `mg0027` from the same runtime commit. Four readers detected
`5208` partial or torn reads across `1000` writer iterations, and the raw log
contains exactly `5208` records. This demonstrates that the zero-violation
GREEN result is not caused by a scanner that accepts partial records.

## Immutable evidence

| Phase | Run ID / PBS job | Result | Checksum-manifest SHA-256 |
|---|---|---|---|
| corrected RED provenance | `20260714T164222Z-red-clock-n5a7d-3c37e0b38ef2` / `2383657.opbs` | the same four unsafe/missing contracts fail | `97e41b32b0a8c86d162b62c0b3acdb62a78105ac4ae9ac98a8e4bab6149d8196` |
| focused tests | `20260714T161902Z-unitfix-unitfix-n3b7e-d8d57a0d6aa0` / `2383535.opbs` | 8 tests pass | `bb49d63d9d9906af109ad9ae488b7e503d571f6df552e1cb0f395c6402c9d2bf` |
| full GREEN | `20260714T161909Z-fullfix-fullfix-n6c9a-006abf2a3b43` / `2383537.opbs` | visibility, atomicity, load, and kill matrix pass | `b0da24c0ad6c1d7ad09046143123852aa9fa2fbaf33acf35828b21c2ad891280` |
| independent repeat | `20260714T161927Z-green-startup-n2e9a-006abf2a3b43` / `2383540.opbs` | second full visibility/atomicity matrix passes | `21d3de9427f016df272d89ec3f124d76a3c2365ca5476934f08890f49b6a0794` |
| full Stage 0 tests | `20260714T162750Z-hardenfix-hardenfix-n7a3d-435f3f3bdf1c` / `2383600.opbs` | 49 tests pass | `0f465e517f97f790971d62631bbc39221cd208e6da4bc971630903b816144c92` |
| unsafe negative | `20260714T162735Z-unsafefix-unsafefix-n2e5c-006abf2a3b43` / `2383599.opbs` | 5208 partial/torn reads detected | `0a8055c387eeeb6fa6b0132d0700cc2206942ba4faa0dab8f4482a1a1081d0c1` |

Every listed package passes its complete `sha256sum -c` manifest. Scheduler
history reports exit `0` and a run-ID prefix exactly matching UTC `qtime` for
each package. The full GREEN package also binds both actual rank/role/host
mappings, the target filesystem, config preimage, clean skill checkout, and
input snapshots.

The original pre-implementation RED package
`20260714T155627Z-red-n3f8c-3c37e0b38ef2` remains immutable evidence that the
four contracts failed before implementation, but its prefix is seven seconds
earlier than scheduler UTC `qtime`; it is not part of the closure set. Job
`2383657.opbs` repeats the same four exact failures with a prefix equal to
scheduler UTC `qtime` and repairs that provenance defect.

The independent full repeat used writer `mg0027` and reader `mg0032`. Its
empty/loaded p99 values were `7.497ms` and `7.951ms`; it again completed
`100000` replacements at each record size with four readers and zero
violations.

## Acceptance boundary and next gap

- `A-BENCH-01`: pass candidate; both required profiles remain more than two
  orders of magnitude below `p99 <= min(5s, 0.05 * 50s) = 2.5s`.
- `A-BENCH-02`: pass candidate; `200000` total replacements, four concurrent
  readers per record size, and zero incomplete-record observations.
- S0B-03 remains responsible for `A-BENCH-03/04`: fixed-slot metadata capacity,
  sequential single/concurrent payload bandwidth, and the Stage 1 filesystem
  decision.
