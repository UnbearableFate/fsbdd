# Runtime evidence

Raw loop and run evidence is stored below this directory or in the resolved
external/shared `EVIDENCE_ROOT`, but is intentionally not committed to Git.

Commit only the small, human-readable indexes under `indexes/`. Each index must
record the immutable raw evidence location, run ID, code/config/source hashes,
and enough checksums to verify the retained evidence.
