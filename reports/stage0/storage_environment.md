# Stage 0 Miyabi storage environment and target-FS decision draft

## Decision boundary

S0B-01 confirms that `/work/xg24i002/x10041/fsbdd/runtime_runs` is a writable
Lustre path with cross-node visibility and is the frozen target `RUN_ROOT` for
the later protocol and storage benchmarks.  Node-local `/tmp` is explicitly
rejected for protocol coordination.  This is a harness-readiness decision, not
a final filesystem qualification: visibility latency and atomic replacement
remain S0B-02, and metadata/bandwidth capacity remain S0B-03.

## Environment identity

| Field | Observed value |
|---|---|
| Initial/control-plane host | `miyabi-g1` (submission and inspection only) |
| Compute allocation | PBS `2383147.opbs`, `debug-g`, hosts `mg0031`, `mg0032` |
| Loaded modules | `nvidia/25.9`, `nv-hpcx/25.9` |
| Launcher | `mpirun`, one Python role per node; MPI is launcher-only |
| miyabi-development | `https://github.com/UnbearableFate/miyabi-development` at `ad1fd34a9de976b4fb26ba47d9a1770430884765` |
| Target path | `/work/xg24i002/x10041/fsbdd/runtime_runs` |
| Filesystem | `lustre` mounted at `/work` |
| Filesystem source | `172.16.20.201@o2ib200:172.16.20.202@o2ib200:172.16.20.211@o2ib200:172.16.20.212@o2ib200:/lustre/work` |
| Default stripe observed | count `1`, size `1048576`, offset `-1` |
| Permission probe | create/read/delete all passed; probe removed; mode `640`, uid `30041`, gid `34002` |
| Group quota observation | gid `34002`; `26427034500` kbytes used, block hard limit `28991029248`; `12173887` files, inode hard limit `56623104` |

The structured manifest, PBS nodefile, module list, `findmnt`, `df`, `statfs`,
Lustre stripe/quota output, and permission record are preserved under the
shared-smoke evidence package.

## Cross-node smoke result

The fixed 4096-byte deterministic payload was published visibility-last on the
target `RUN_ROOT` by `mg0031` and read and hash-validated by `mg0032`.  Both
roles reported SHA-256
`ff1590d0f858164e49e922c9bdb5ed5a63396550ee276752b11a974bc7dd5f71`.
The reader observed the publication after `1,420,167 ns`; the writer observed
the filesystem acknowledgement after `148,357,277 ns`.  These timings prove
the smoke handshake only and are not substituted for the S0B-02 latency
distribution.

Evidence:

- Shared positive: `evidence/raw/S0B-01/20260714T152721Z-harden3-n8d4e-d27938bf5c40`, checksum-manifest SHA-256 `6c13d30b9a61f29ce94daaaa36f14f158f8108cf8ebab92edfaad940093a1e90`.
- Node-local negative control: `evidence/raw/S0B-01/20260714T152720Z-red5-n3c9a-d27938bf5c40`, checksum-manifest SHA-256 `0bf86b712a451fc75265a4ef30b3b502d06f46f6e0d58607535cc6df76d4f1b6`.
- Storage unit/static hardening: `evidence/raw/S0B-01/20260714T152628Z-harden-n6e1c-39928d85f6ae`, 8/8 tests passed, checksum-manifest SHA-256 `d058b3f8a64301cdb4ce360c93e791b47210c7088f3ce04bd4cad7b44e601a4e`.
- Full Stage 0 regression hardening: `evidence/raw/S0B-01/20260715T004500Z-harden5-all-n2c7f-81d651352a02`, 41/41 tests passed, checksum-manifest SHA-256 `fede5ba64c82e590ba148cf80680d5acf940d7e4eb6566dbb39b774d778135f4`.

The node-local control placed the same absolute `/tmp` path on `mg0017` and
`mg0018`.  The reader timed out waiting for the writer's visibility record and
the writer timed out waiting for the reader acknowledgement, so the harness
classified the path as `node_local_visibility_rejected` for the intended
reason.

## Safety and reproducibility

- Every runtime entry point fails closed unless the short hostname matches a
  Miyabi compute host and both `PBS_JOBID` and `PBS_NODEFILE` are present.
- The structured environment manifest requires the skill commit, initial and
  compute hosts, PBS identity/nodefile, target path, filesystem type/source,
  module list, and successful permission probe.
- Payload publication and all handshake records use same-directory temporary
  files, `fsync`, and `os.replace`; readers validate record schema, byte count,
  digest, run identity, and distinct host identity.
- Run directories are immutable by construction: the PBS wrapper refuses an
  existing evidence path, and each submission identity includes a nonce and
  resolved config digest.
- Application payload and coordination use only the path under test.  MPI is
  not an application-data transport.

## RED provenance

Before the harness existed, the deliberately unsafe surrogate admitted a
login-host benchmark, omitted required environment/FS identity, and assumed an
absolute `/tmp` path was shared.  All three contracts failed in
`evidence/raw/S0B-01/20260714T151400Z-red-n3d6a-d6b0d27e3bb2`;
checksum-manifest SHA-256
`0fb47c4af76a9af9caf404bc3f9e5724b33534ee86377ae468bdfaebd74c720f`.

## Next measurement gap

S0B-02 must measure empty and loaded cross-node visibility distributions,
report p50/p95/p99/max with a conservative cross-host timing method, and run at
least 100000 atomic replacements with multiple readers and zero torn records.
The frozen visibility threshold is `p99 <= 2.5 s` from
`min(5 s, 0.05 * H)` with `H=50 s`.
