# Stage 0 Miyabi storage environment and target-FS decision draft

## Decision boundary

S0B-01 confirms that `/work/xg24i002/x10041/fsbdd/runtime_runs` is a writable
Lustre path with cross-node visibility and is the frozen target `RUN_ROOT` for
the later protocol and storage benchmarks. Node-local `/tmp` is explicitly
rejected for protocol coordination. This is a harness-readiness decision, not
a final filesystem qualification: visibility/atomicity remain S0B-02 and
metadata/bandwidth capacity remain S0B-03.

## Corrected environment identity

The first independent Checker found that the installed `miyabi-development`
checkout had already advanced before S0B-01 ORIENT, while the initial packages
recorded its parent. Those packages remain immutable as blocked history. The
corrected closure binds the installed checkout, snapshots `SKILL.md`, and
rejects a mismatch before any project runtime starts.

| Field | Corrected observed value |
|---|---|
| Initial/control-plane host | `miyabi-g1` (submission and inspection only) |
| Compute allocation | PBS `2383213.opbs`, `debug-g`, hosts `mg0016`, `mg0017` |
| Loaded modules | `nvidia/25.9`, `nv-hpcx/25.9` |
| Launcher | Open MPI `4.1.9a1`, one Python role per node; launcher-only |
| miyabi-development | `https://github.com/UnbearableFate/miyabi-development@c3ddfa47cadfd9132edf939d359b90b28ab5f8ed` |
| Target path | `/work/xg24i002/x10041/fsbdd/runtime_runs` |
| Filesystem | `lustre` mounted read/write at `/work` |
| Filesystem source | `172.16.20.201@o2ib200:172.16.20.202@o2ib200:172.16.20.211@o2ib200:172.16.20.212@o2ib200:/lustre/work` |
| Default stripe observed | count `1`, size `1048576`, offset `-1` |
| Permission probe | create/read/delete passed; probe removed; mode `640`, uid `30041`, gid `34002` |

The JSON and YAML canonical manifests are byte-identical. They include node
type, queue/group, full code/config identities, frozen config paths, PBS
nodefile, filesystem and module identity, permission probe, and the actual
rank/role/host/test-root mapping. The positive run is bound to code commit
`dce034a684993cfd84858ebd09126b4f17863ba1` and config digest
`2018e813ad8e7b5bca742b8e1f51f94e9ed784acad83b6aea1a3965b0f017b39`.

## Corrected cross-node evidence

The payload is a deterministic 4096-byte sequence. Payload and control records
are written to same-directory exclusive temporary files, flushed and fsynced,
published with `os.replace`, and followed by directory fsync. Reader polling
uses `time.monotonic_ns`; MPI only launches the two roles.

| Phase | Evidence / PBS job | Result | Checksum-manifest SHA-256 |
|---|---|---|---|
| RED provenance repair | `20260714T154749Z-red-provenance2-n8c5f-c41187a90807` / `2383254.opbs` | all three unsafe assumptions failed specifically | `a15a193cf00747ee5471d3fd890880887dcd8b6541eef7f94a81ecee20948f8b` |
| focused HARDEN | `20260714T154036Z-green9-skill-n2f6b-785ceb2564a8` / `2383209.opbs` | all 8 storage contracts passed | `356c24f6980791a6d20c844cf9a2738eb6a0afc047c89e14e7ec9a0bb68af7ea` |
| full HARDEN | `20260714T154201Z-harden10-all-n9b4d-5a1a87a340ce` / `2383227.opbs` | all 41 Stage 0 tests passed | `525f8403570050f6c9d0daf609ea778a91f35be77eb8aab28858ee50c4a4091f` |
| node-local negative | `20260714T154111Z-harden9-local-n7e2a-2018e813ad8e` / `2383217.opbs` | reader visibility and writer acknowledgement timed out on distinct hosts | `6facd1d6db87e9310f476d7643362ce04083decabc87c7291b0566330361d4da` |
| shared GREEN | `20260714T154053Z-green10-shared-n4c8e-2018e813ad8e` / `2383213.opbs` | both roles passed on the frozen Lustre root | `cac18b65c3acb3a91161739a1e40eb5bdedb97282fe3ac2219fbb4e208c38a27` |
| independent shared repeat | `20260714T154217Z-harden11-repeat-n5a1f-2018e813ad8e` / `2383231.opbs` | a new immutable run again passed on two distinct hosts | `49f40dea4e0cad779462213a10d157ae4196031f2792a8bfc7e90f61cad3ec60` |

The shared writer on `mg0016` and reader on `mg0017` agreed on 4096 bytes and
SHA-256 `684c60c5704c1bc036325224f7ff200cbc3e21631bfb94b6aa9e6459bcdc2ec2`.
The reader-local visibility wait was `1.448ms`; writer-side publication through
acknowledgement was `64.333ms`. These are smoke-handshake durations, not the
S0B-02 latency distribution or a cross-host clock claim.

The node-local control used the same absolute `/tmp` name on writer `mg0016`
and reader `mg0017`. The reader could not see the visibility record and the
writer could not see an acknowledgement. The two-rank MPI command exited `2`,
and the enclosing expected-negative contract recorded
`node_local_visibility_rejected` without an error-trap false alarm.

The original pre-implementation RED remains at
`20260714T151400Z-red-n3d6a-d6b0d27e3bb2`. The corrected RED repeats its exact
three failures under the actual skill checkout; it repairs environment
provenance rather than replacing the original RED ordering.

## Safety and reproducibility

- Every entrypoint fails closed outside an `mg<number>` PBS allocation.
- Runtime wrappers compare the requested skill commit to the installed Git
  checkout, preserve its clean-status output and `SKILL.md`, and record the
  observed commit in the canonical evidence.
- Shared mode is confined below the frozen target root; local-negative mode is
  confined below `/tmp`.
- Readers validate JSON schema, run identity, peer host, byte count and digest;
  summaries validate both ranks, distinct hosts and reciprocal role identity.
- A run ID contains a fresh nonce and short config digest, and the wrapper
  refuses an existing evidence directory. All six corrected packages pass
  their complete checksum manifests.
- No Torch version, model, dataset, GPU, training-quality, protocol-correctness
  or nine-node-baseline claim is made.

## Next measurement gap

S0B-02 must report empty and loaded p50/p95/p99/max using a conservative
single-clock acknowledgement method and must execute at least 100000 atomic
record replacements with multiple readers and zero torn records. The frozen
visibility threshold is `p99 <= 2.5s`, derived from `min(5s, 0.05 * H)` with
`H=50s`. S0B-03 must separately decide metadata and bandwidth capacity.
