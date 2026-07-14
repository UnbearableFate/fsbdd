# S1-03 Storage Publication Contract

`StorageBackend` exposes two fixed-slot operations: `publish` and `read`.
The POSIX implementation writes a unique immutable payload, reads it back to
verify size and SHA-256, stages a small complete JSON record, and makes that
record visible with `os.replace`. Normal readers open exactly one visibility
record and then exactly the unique payload named by it. They never list, glob,
walk, replay, or infer latest state from the payload area.

The record binds run, fragment-map, fragment, version, sequence, dtype, shape,
base-content, payload-path, byte-count, checksum, and payload identity. Record
schema/identity errors fail immediately. A referenced payload that has not yet
become complete on another node is retried only until the explicit eventual-
readability timeout; it is never returned partially.

The crash hook matrix covers before/after payload write and before/after the
visibility-record replacement. Before replacement the old record remains
authority, regardless of orphan payload or staged record files. After
replacement the new complete compound payload is authority. Orphan cleanup is
not a correctness prerequisite and remains later GC work.

MPI is used only by the Miyabi batch script to launch one writer and one reader
on two allocated hosts. Their application coordination and data exchange are
shared-filesystem only. The cross-node run injects delay immediately before
visibility replacement and counts every invalid read across the publication
interval. The unsafe direct-overwrite detector and raw control are reused from
S0B-02 rather than reimplemented.

The first two-node package at `runtime_runs/S1-03/formal-l2` is retained as a
failed identity attempt: its application observations passed, but its manifest
labeled the local-JST PBS `qtime` as UTC. A static regression now requires the
explicit JST parse, and the selected replacement package is frozen at
`runtime_runs/S1-03/formal-l2-r2`.

The replacement job `2384655.opbs` ran one writer on `mg0026` and one reader
on `mg0027`. Its selected package is validator-admissible (25/25 files), all 68
Stage 1 tests pass, and 500 publications produced 1,428 valid observations with
zero invalid reads. The reader observed final sequence 500; the crash matrix is
old/old/old/complete-new and normal discovery performed zero directory scans.
