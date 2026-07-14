# Stage 0 cross-node visibility and atomicity result

## Maker verdict

`PASS`, pending the independent S0B-02 Checker.

The frozen Miyabi Lustre target passed both S0B-02 thresholds. The selected
full run is `20260714T161909Z-fullfix-fullfix-n6c9a-006abf2a3b43` (PBS
`2383537.opbs`) at executable commit
`78e5992383d3fc6900edeb80f6055451bb08933b`. Rank 0 ran on `mg0030` and rank 1
on `mg0031`; MPI was launcher-only and all application coordination used the
target filesystem.

## A-BENCH-01

The timing interval begins only after the visibility-record `os.replace`
returns and ends when the writer observes the reader's validated
acknowledgement. Both endpoints therefore use the writer's
`time.monotonic_ns` clock. This is a conservative publication-to-reader-ack
upper bound and does not subtract clocks from different hosts.

| Profile | Samples | p50 | p95 | p99 | max | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| empty | 1000 | 5.805 ms | 7.130 ms | 7.939 ms | 163.547 ms | 2.5 s |
| loaded | 1000 | 5.572 ms | 7.010 ms | 7.801 ms | 8.834 ms | 2.5 s |

Each profile contains 500 cold-path and 500 warm-path samples. The loaded
profile did not merely start a background task: it waited for 1000 completed
atomic background replacements before sampling, advanced to 5958 during the
loaded interval, and the remote background reader made 7497 complete-record
observations with zero violations. Both profile p99 values are far below
`min(5s, 0.05 * H) = 2.5s` for frozen `H=50s`.

## A-BENCH-02

The writer completed 100000 small-record replacements at each of 256 and 4096
bytes. Every publication has a sequence, SHA-256, writer-local completion
timestamp, and timing origin in a 100000-line hashed raw transcript. Four
concurrent remote reader threads per size validated exact length, JSON schema,
run/size/sequence identity, completion marker, and checksum.

| Record | Replacements | Reader observations | Regressions | Torn/partial violations |
|---|---:|---|---:|---:|
| 256 B | 100000 | 166406 / 165979 / 165117 / 164840 | 0 | 0 |
| 4096 B | 100000 | 165990 / 165542 / 165992 / 166207 | 0 | 0 |

The raw violation scanner is empty. Actual `SIGKILL` probes also passed at both
publication boundaries: a writer killed before visibility exposed no record,
while a writer killed after visibility left a complete record and matching
payload. BENCH-02 concerns atomic visibility rather than power-loss durability;
each tested record is fully written and fsynced in a same-directory temporary,
closed, and published with `os.replace`, while directory fsync is done after
each size matrix rather than after every replacement.

The independent unsafe control used 1000 direct two-half overwrites on the
same target with four readers. It detected 5208 partial/torn observations, so
the scanner is demonstrably live rather than a constant-zero summary.

## Reproducibility

| Phase | Run / job | Result | Checksum-manifest SHA-256 |
|---|---|---|---|
| pre-implementation RED | `20260714T155627Z-red-n3f8c-3c37e0b38ef2` / `2383353.opbs` | four required bad claims failed | `0fdfecc3db78a56dcaa9a32a509b5abdc56e6371a6e395c44a94cf70ba1203da` |
| focused GREEN | `20260714T161902Z-unitfix-unitfix-n3b7e-d8d57a0d6aa0` / `2383535.opbs` | 8 focused contracts passed | `bb49d63d9d9906af109ad9ae488b7e503d571f6df552e1cb0f395c6402c9d2bf` |
| full GREEN/HARDEN | `20260714T161909Z-fullfix-fullfix-n6c9a-006abf2a3b43` / `2383537.opbs` | 200000 replacements, both latency profiles, kill probes passed | `b0da24c0ad6c1d7ad09046143123852aa9fa2fbaf33acf35828b21c2ad891280` |
| unsafe runtime control | `20260714T162735Z-unsafefix-unsafefix-n2e5c-006abf2a3b43` / `2383599.opbs` | 5208 detector hits | `0a8055c387eeeb6fa6b0132d0700cc2206942ba4faa0dab8f4482a1a1081d0c1` |
| full Stage 0 tests | `20260714T162750Z-hardenfix-hardenfix-n7a3d-435f3f3bdf1c` / `2383600.opbs` | 49 tests passed | `0f465e517f97f790971d62631bbc39221cd208e6da4bc971630903b816144c92` |

All selected packages pass their complete checksum manifests, record the
actual clean `miyabi-development` checkout at
`c3ddfa47cadfd9132edf939d359b90b28ab5f8ed`, freeze full code/config identities,
and contain the actual PBS nodefile, queue/group, module, filesystem, and role
mapping evidence. Stage 0 does not require the 9-node training gate.

## Boundary and next gap

This result qualifies only A-BENCH-01 and A-BENCH-02 on the frozen target path.
It makes no model, loss, training, power-loss durability, metadata-capacity, or
payload-bandwidth claim. S0B-03 remains the next gap.
