# Checker Report: S0B-03 / `4ef5229`

## Verdict

`BLOCKED`

`A-BENCH-04` is proven, but `A-BENCH-03` is not. The metadata summarizer
combines rank-local measurements that were not concurrent, producing an
aggregate rate that was never observed during a common interval.

## Independent inputs

- Source hashes: `STAGE0-4_SPEC.md =
  fad26a8add51e70ddf50d0130b61d80fe06e1c3945221c29508412a5bcab2feb`;
  `RESEARCH_PLAN.md =
  2c0db3148110da10ea62f445c189d2d53b405527d8da264b9af9817f5ca9d835`.
- Base: `7cd3e6ffcf36343d782b241c9815c57d62f87a69`.
- Runtime: `6e5508bd3a743e94c5a81146d68a9424c72e8b23`.
- Candidate: `4ef52290b607ded61bb8c519a06aedd324a5e665`.
- Resolved config digest:
  `cdd8a3e5e65a3087ce3593a550f0f2040cc2bb1dd612e436fa8b8fd69eaede85`.
- Miyabi skill:
  `c3ddfa47cadfd9132edf939d359b90b28ab5f8ed`, clean and identical to
  recorded provenance.
- Static review ran on `miyabi-g1` without a PBS allocation. Only static
  inspection and read-only scheduler queries were performed.
- The nine-node gate is not applicable before `S1-13`.

All selected manifests passed complete `sha256sum -c` verification and had no
unlisted package files:

| Package | Checksum-manifest SHA-256 | Listed/actual |
|---|---|---:|
| RED | `2c3a717928cc671ebc47f2c5060923edf66021c83d0cf620b433c96a6d660a1f` | 22/22 |
| focused | `2d1e7e1c22897ac9a96cfab4eca519b06c0d05ab48cd1afdd5ec58931737fb27` | 25/25 |
| primary | `0aaae840eb301cf3ef6902fe79beb238b93535fe729827aaf6300f8b6033112e` | 57/57 |
| full Stage 0 | `ec0b20d49643ac21c3eaf805465b458f13b04ab5e6cd5f1ac1a7701eed9c0905` | 35/35 |
| repeat | `c53cb77728b38f80df4218fd51a0e5186fc3075210b72ad80dba51be60625a40` | 57/57 |

## Reconstructed target

- Single objective: measure bounded `M*F` metadata polling and the frozen
  single/16-stream 50--500 MiB I/O matrix, then make the Stage 1 storage
  decision.
- Requirements: `BENCH-03`, `BENCH-04`, `FS-05`, `FS-06`; optional
  `BENCH-05`.
- Acceptance: `A-BENCH-03`, `A-BENCH-04`.
- No 8+1/50x10 gate applies.

## RED validity

The selected RED was committed before implementation and produced five
deterministic requirement failures. Its inner unittest exit was `1`; the
expected-failure wrapper and scheduler exited `0`. The failures cover history
scanning, incomplete metadata profiles, missing threshold math, incomplete
payload/concurrency coverage, and the missing `10% * H` comparison. No
timeout or timing-sensitive assertion was involved.

The excluded `171417Z-unit` failure was a test-regex mismatch: production
correctly raised `metadata concurrency does not match`, while the assertion
required `metadata matrix`. Runtime commit `6e5508b` widened only that
assertion, followed by clean focused and full-suite reruns. Excluding that
debugging run from closure is valid.

## Code and protocol inspection

- Scope is confined to Stage 0 config, benchmark source, PBS launcher, tests,
  reports, and plan state. There are no Torch, model, learner/syncer production
  protocol, or training changes.
- MPI is launcher-only. Application roles communicate through fixed
  filesystem control records and payloads; no MPI collective or application
  data transport was found.
- Normal fixed records remain `6272`; negative-control history remains
  `11000`; control files are exactly the frozen limit `35`; payload and
  temporary files are zero; both run-scoped roots were removed.
- History-scanner p50 grows across 0/1000/10000 objects while the normal
  control inspects exactly 512 references.
- Payload writes use file `fsync` followed by same-directory `os.replace`;
  control records use the atomic JSON primitive. No power-loss durability
  claim is made.
- Source, config, PBS and implementation snapshots match runtime commit
  `6e5508b`; runtime worktrees and skill checkout were clean.
- `git diff --check` passes and the candidate worktree was clean.

### Blocking metadata defect

Ranks do not synchronize around metadata cells. In `_run_metadata_rank`,
inactive ranks skip cells. Consequently ranks 8--15 skip all `M=4` and `M=8`
cells and immediately measure `M=16`, while ranks 0--3 execute roughly 72
one-second warm cells before reaching `M=16`. Ranks 8--15 finish their
`M=16` cells before ranks 0--3 begin theirs.

The summarizer nevertheless groups those disjoint rank records, sums all
reference counts, and divides by only the largest roughly one-second
rank-local elapsed time:

```text
references = sum(per-rank references)
elapsed = max(per-rank elapsed)
rate = references / elapsed
```

No per-cell start timestamps, coordinator intervals, or filesystem barriers
prove overlap. Therefore `45137.772/s` and `45339.331/s` are synthetic sums,
not observed sustained concurrent rates. A filesystem capped below `5120/s`
could falsely pass if ranks execute serially at a few thousand operations/s
each. Tests validate record counts and reuse the same formula, so they do not
detect this counterexample.

## Maker-unlisted counterexamples

1. Sixteen ranks can each perform 3000 operations during disjoint one-second
   intervals. The current summary reports `48000/s`, although observed
   capacity never exceeded `3000/s`.
2. The single-rank bounded full-surface controls already warn against the
   conclusion. Reading all 512 fixed records took `0.126453--0.126856s` in
   the primary and `0.134136--0.134840s` in the repeat, or roughly `4049/s`
   and `3817/s`. These controls are not the frozen sustained threshold
   measurement and do not prove failure, but show why invalid aggregation
   cannot establish pass.

## Runtime evidence

| Job | UTC prefix | Scheduler exit | Inner result |
|---|---|---:|---|
| `2383721` | `20260714T170406Z` | 0 | expected unittest 1 |
| `2383754` | `20260714T171444Z` | 0 | 7/7 |
| `2383755` | `20260714T171502Z` | 0 | mpirun 0 |
| `2383756` | `20260714T171520Z` | 0 | 56/56 |
| `2383771` | `20260714T171837Z` | 0 | mpirun 0 |

Every prefix equals scheduler UTC qtime. The full runs have 16 roles, eight
ranks on each of two distinct hosts, 272 raw bandwidth operations, 32
coordinator rounds, correct configured byte counts, zero digest or
opposite-host mapping errors, and all 32 summary cells.

The worst 16-stream write round is `0.325864872s`; against
`0.1 * 50s = 5s`, it is `0.65173%` of `H`, so `A-BENCH-04` passes. Raw and
generated summaries are arithmetically consistent; the blocker is metadata
aggregation semantics, not transcription.

## Acceptance determination

| ID | Evidence | Conclusion |
|---|---|---|
| `A-BENCH-03` | gate `5120/s`; reported 45k/s is assembled from nonsynchronous intervals | `BLOCKED` |
| `A-BENCH-04` | two coordinator-timed matrices; worst write `0.325864872s <= 5s`; digests and opposite-host reads verified | `PASS` |
| `FS-05` | exactly 512 fixed references independent of history size | supported |
| `FS-06` | fixed/history/control counts stable; zero payload/temp leftovers; run roots removed | supported |
| `BENCH-05` | not executed and no object-store claim made | optional/not applicable |

The report boundary is otherwise appropriate, but the target-Lustre pass and
no-mitigation decision overreach until `A-BENCH-03` is correctly measured.

## Required remediation

Add an actual metadata measurement interval: either one syncer-equivalent
rank sustaining complete `M*F` polls, or per-cell filesystem barriers with
all ranks participating plus coordinator-timed start-to-all-done intervals.
The summarizer must reject records without this timing provenance. Add a
RED/unit counterexample for non-overlapping rank intervals, rerun focused/full
tests and two immutable capacity runs, then regenerate the decision and
candidate reports.

## Follow-ups

None. The remaining item blocks closure.

## Signature

- Checker: fresh independent `/root/s0b03_final_checker`.
- Date: 2026-07-15 Asia/Tokyo.
- Checked base: `7cd3e6ffcf36343d782b241c9815c57d62f87a69`.
- Checked runtime: `6e5508bd3a743e94c5a81146d68a9424c72e8b23`.
- Checked candidate: `4ef52290b607ded61bb8c519a06aedd324a5e665`.
