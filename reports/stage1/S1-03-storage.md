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
