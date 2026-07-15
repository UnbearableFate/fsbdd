# S1-04 Per-Fragment Global-State Bootstrap

`GlobalStateStore` bootstraps one independently authoritative fixed current
slot per frozen fragment. Each slot names one immutable binary compound payload
containing parameters, outer-optimizer state, frozen run/config/model/map and
fragment identities, version and outer-update metadata, and a bounded embedded
base history. There is no shared global head or whole-model checkpoint
authority. A global snapshot is only an ordered read of the frozen fragment
descriptor set and its resulting version vector.

Bootstrap is fail-closed and restartable. It first checks every existing fixed
slot against the exact expected version-zero compound content before publishing
any missing fragment. Missing slots make a snapshot unavailable; a repeated
identical bootstrap is a no-op; and different content under the same frozen
identity is rejected before any current record changes. The successor API does
not accept a caller-selected version. It derives both fragment version and
outer-update count as the previous values plus one and rolls embedded bases to
at most `S_max + 1` entries.

Normal startup reads exactly the `F` fixed current slots. It never lists,
globs, walks, replays history, or reads a global head. Offline
`inspect_live_set` is explicitly separate from the correctness path and may
scan physical payload files to distinguish the `F` referenced authorities from
unreferenced cleanup candidates. Immutable-payload garbage collection remains
later Stage 1 work; it is not used to discover current state.

The Hugging Face freeze adapter lazily imports Torch-backed safetensors,
serializes every uniquely owned trainable parameter exactly once according to
the frozen logical-layer registry and fragment map, and derives identities from
the complete model config and serialized fragment content. The compute test
uses a tiny real GPT-NeoX model initialized from a fixed seed and verifies every
deserialized tensor exactly. Filesystem-only roles do not import Torch.

Phase A rejected early preflights for an incomplete authority-ID grammar,
Miyabi module ordering, and unbound synthetic stress identities. Candidate
`1ff459f3b4fa500ccd9adf2ee850f39a5e4cd818` corrects each issue. The contract
pre-registers the resolved config identity and canonical synthetic model/map
identities; both roles recompute or consume them, report them with the same run
identity, and the analyzer and manifest builder require exact agreement.

The selected formal package is `runtime_runs/S1-04/formal-l2`, PBS job
`2384764.opbs`. It ran the writer on `mg0018` and reader on `mg0019`, exited
zero in 32 seconds, passes all 79 Stage 1 tests, and is validator-admissible
(27/27 files; manifest `1aae5431...d288`). The protocol interrupts after two
of four fragments, rejects the partial snapshot, resumes only the two missing
fragments, rejects conflicting bootstrap content, creates no payload on an
identical repeat, and retains four authoritative records and four base entries
after adding 10,000 unrelated historical objects. Snapshot reads are exactly
four both before and after the history injection; every stderr log is empty.
