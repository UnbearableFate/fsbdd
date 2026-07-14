"""Compute-only, cross-node storage probes for the frozen Stage 0 RUN_ROOT.

The MPI runtime is used only to place one Python role on each host.  All role
coordination and application data flow through the path under test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SKILL_REPOSITORY = "https://github.com/UnbearableFate/miyabi-development"
SKILL_COMMIT = "ad1fd34a9de976b4fb26ba47d9a1770430884765"
COMPUTE_HOST_PATTERN = re.compile(r"^mg[0-9]+$")
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "skill_commit",
        "initial_hostname",
        "compute_hosts",
        "pbs_job_id",
        "pbs_nodefile",
        "target_run_root",
        "filesystem_type",
        "filesystem_source",
        "module_list",
    }
)


class StorageHarnessError(RuntimeError):
    """Raised when the storage probe cannot make a defensible claim."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Publish bytes visibility-last with a same-directory ``os.replace``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{socket.gethostname()}-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (_canonical_json(value) + "\n").encode("utf-8"))


def read_json_record(path: Path, *, required_fields: Sequence[str] = ()) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StorageHarnessError(f"invalid JSON record {path}: {error}") from error
    if not isinstance(value, dict):
        raise StorageHarnessError(f"JSON record must be an object: {path}")
    missing = set(required_fields) - set(value)
    if missing:
        raise StorageHarnessError(f"JSON record {path} is missing {sorted(missing)}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_payload(size: int, *, seed: str = "fsbdd-stage0-storage-smoke") -> bytes:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise StorageHarnessError("payload size must be a positive integer")
    output = bytearray()
    counter = 0
    while len(output) < size:
        output.extend(hashlib.sha256(f"{seed}:{counter}".encode("utf-8")).digest())
        counter += 1
    return bytes(output[:size])


def load_storage_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_storage_config(config)
    return config


def validate_storage_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise StorageHarnessError("storage benchmark schema_version must be 1")
    miyabi = config.get("miyabi")
    smoke = config.get("smoke")
    if not isinstance(miyabi, Mapping) or not isinstance(smoke, Mapping):
        raise StorageHarnessError("miyabi and smoke mappings are required")
    if miyabi.get("required_skill_repository") != SKILL_REPOSITORY:
        raise StorageHarnessError("unexpected miyabi-development repository")
    if miyabi.get("required_skill_commit") != SKILL_COMMIT:
        raise StorageHarnessError("unexpected miyabi-development commit")
    root = Path(str(miyabi.get("target_shared_run_root", "")))
    if not root.is_absolute() or not str(root).startswith("/work/"):
        raise StorageHarnessError("target_shared_run_root must be an absolute /work path")
    if miyabi.get("expected_filesystem_type") != "lustre":
        raise StorageHarnessError("Stage 0 target filesystem type must be frozen as lustre")
    if miyabi.get("compute_nodes") != 2 or miyabi.get("ranks_per_node") != 1:
        raise StorageHarnessError("the smoke topology must be two nodes and one role per node")
    if smoke.get("required_distinct_hosts") != 2:
        raise StorageHarnessError("the smoke must require two distinct hosts")
    if smoke.get("payload_bytes", 0) <= 0 or smoke.get("timeout_seconds", 0) <= 0:
        raise StorageHarnessError("smoke payload and timeout must be positive")


def require_compute_context(
    *,
    hostname: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str, str]:
    host = hostname or socket.gethostname()
    environment = os.environ if environ is None else environ
    job_id = environment.get("PBS_JOBID", "")
    nodefile = environment.get("PBS_NODEFILE", "")
    if not COMPUTE_HOST_PATTERN.fullmatch(host) or not job_id or not nodefile:
        raise StorageHarnessError(
            "refusing storage benchmark outside a confirmed Miyabi PBS compute allocation"
        )
    return host, job_id, nodefile


def _wait_for_record(
    path: Path,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    required_fields: Sequence[str],
) -> tuple[dict[str, Any], int]:
    started_ns = time.monotonic_ns()
    deadline_ns = started_ns + int(timeout_seconds * 1_000_000_000)
    last_invalid: StorageHarnessError | None = None
    while time.monotonic_ns() <= deadline_ns:
        if path.is_file():
            try:
                return (
                    read_json_record(path, required_fields=required_fields),
                    time.monotonic_ns() - started_ns,
                )
            except StorageHarnessError as error:
                last_invalid = error
        time.sleep(poll_interval_seconds)
    detail = f"; last invalid record: {last_invalid}" if last_invalid else ""
    raise StorageHarnessError(f"timed out waiting for {path}{detail}")


def _role_identity() -> tuple[int, int]:
    try:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise StorageHarnessError("smoke-role requires an Open MPI rank context") from error
    if size != 2 or rank not in {0, 1}:
        raise StorageHarnessError("smoke-role requires exactly two ranks")
    return rank, size


def run_smoke_role(
    *,
    config: Mapping[str, Any],
    run_id: str,
    test_root: Path,
    result_root: Path,
    path_mode: str,
) -> dict[str, Any]:
    host, job_id, _ = require_compute_context()
    rank, size = _role_identity()
    smoke = config["smoke"]
    target_root = Path(str(config["miyabi"]["target_shared_run_root"]))
    if path_mode not in {"shared", "local-negative"}:
        raise StorageHarnessError(f"unknown smoke path mode: {path_mode}")
    if path_mode == "shared":
        try:
            test_root.resolve().relative_to(target_root.resolve())
        except ValueError as error:
            raise StorageHarnessError("shared smoke path must be below target RUN_ROOT") from error
    elif not str(test_root).startswith("/tmp/"):
        raise StorageHarnessError("local-negative smoke path must be below /tmp")

    test_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    role = "writer" if rank == 0 else "reader"
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "rank": rank,
        "world_size": size,
        "role": role,
        "hostname": host,
        "pbs_job_id": job_id,
        "path_mode": path_mode,
        "test_root": str(test_root),
        "clock": "time.monotonic_ns",
        "status": "failed",
    }
    payload_path = test_root / "payload.bin"
    visible_path = test_root / "payload.visible.json"
    acknowledgement_path = test_root / "reader.ack.json"
    timeout = float(smoke["timeout_seconds"])
    poll = float(smoke["poll_interval_seconds"])
    payload = deterministic_payload(int(smoke["payload_bytes"]), seed=run_id)
    expected_digest = hashlib.sha256(payload).hexdigest()

    try:
        if rank == 0:
            started_ns = time.monotonic_ns()
            atomic_write_bytes(payload_path, payload)
            record = {
                "schema_version": 1,
                "run_id": run_id,
                "writer_host": host,
                "payload_bytes": len(payload),
                "payload_sha256": expected_digest,
            }
            atomic_write_json(visible_path, record)
            result["publish_elapsed_ns"] = time.monotonic_ns() - started_ns
            try:
                acknowledgement, wait_ns = _wait_for_record(
                    acknowledgement_path,
                    timeout_seconds=timeout,
                    poll_interval_seconds=poll,
                    required_fields=(
                        "run_id",
                        "reader_host",
                        "payload_bytes",
                        "payload_sha256",
                    ),
                )
            except StorageHarnessError as error:
                result["failure_kind"] = "acknowledgement_timeout"
                raise error
            if acknowledgement["run_id"] != run_id:
                result["failure_kind"] = "acknowledgement_mismatch"
                raise StorageHarnessError("reader acknowledgement has a different run_id")
            if acknowledgement["payload_bytes"] != len(payload):
                result["failure_kind"] = "acknowledgement_mismatch"
                raise StorageHarnessError("reader acknowledgement has a different payload size")
            if acknowledgement["payload_sha256"] != expected_digest:
                result["failure_kind"] = "acknowledgement_mismatch"
                raise StorageHarnessError("reader acknowledgement has a different digest")
            result.update(
                {
                    "reader_host": acknowledgement["reader_host"],
                    "acknowledgement_wait_ns": wait_ns,
                    "payload_bytes": len(payload),
                    "payload_sha256": expected_digest,
                    "status": "passed",
                }
            )
        else:
            try:
                record, wait_ns = _wait_for_record(
                    visible_path,
                    timeout_seconds=timeout,
                    poll_interval_seconds=poll,
                    required_fields=(
                        "run_id",
                        "writer_host",
                        "payload_bytes",
                        "payload_sha256",
                    ),
                )
            except StorageHarnessError as error:
                result["failure_kind"] = "visibility_timeout"
                raise error
            if record["run_id"] != run_id:
                result["failure_kind"] = "visible_record_mismatch"
                raise StorageHarnessError("visible record has a different run_id")
            observed_size = payload_path.stat().st_size
            observed_digest = file_sha256(payload_path)
            if observed_size != record["payload_bytes"] or observed_digest != record["payload_sha256"]:
                result["failure_kind"] = "payload_mismatch"
                raise StorageHarnessError("visible payload does not match its publication record")
            acknowledgement = {
                "schema_version": 1,
                "run_id": run_id,
                "reader_host": host,
                "payload_bytes": observed_size,
                "payload_sha256": observed_digest,
            }
            atomic_write_json(acknowledgement_path, acknowledgement)
            result.update(
                {
                    "writer_host": record["writer_host"],
                    "visibility_wait_ns": wait_ns,
                    "payload_bytes": observed_size,
                    "payload_sha256": observed_digest,
                    "status": "passed",
                }
            )
    except (OSError, StorageHarnessError) as error:
        result.setdefault("failure_kind", "storage_error")
        result["error"] = str(error)

    atomic_write_json(result_root / f"smoke-rank-{rank}.json", result)
    return result


def validate_environment_manifest(manifest: Mapping[str, Any]) -> None:
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise StorageHarnessError(f"environment manifest is missing {sorted(missing)}")
    if manifest["skill_commit"] != SKILL_COMMIT:
        raise StorageHarnessError("environment manifest has the wrong skill commit")
    hosts = manifest["compute_hosts"]
    if (
        not isinstance(hosts, list)
        or len(set(hosts)) < 2
        or any(not COMPUTE_HOST_PATTERN.fullmatch(str(host)) for host in hosts)
    ):
        raise StorageHarnessError("environment manifest requires two distinct compute hosts")
    if manifest["filesystem_type"] != "lustre":
        raise StorageHarnessError("target RUN_ROOT was not identified as Lustre")
    if not str(manifest["target_run_root"]).startswith("/work/"):
        raise StorageHarnessError("target RUN_ROOT is not below /work")
    if not manifest["pbs_nodefile"] or not manifest["module_list"]:
        raise StorageHarnessError("nodefile and module list evidence cannot be empty")


def build_environment_manifest(
    *,
    config: Mapping[str, Any],
    run_id: str,
    initial_hostname: str,
    compute_hostname: str,
    pbs_job_id: str,
    pbs_nodefile_path: Path,
    filesystem_type: str,
    filesystem_source: str,
    module_list_path: Path,
    permission_probe: Mapping[str, Any],
) -> dict[str, Any]:
    compute_hosts = sorted(
        set(line.strip() for line in pbs_nodefile_path.read_text(encoding="utf-8").splitlines())
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "skill_repository": SKILL_REPOSITORY,
        "skill_commit": SKILL_COMMIT,
        "initial_hostname": initial_hostname,
        "compute_hostname": compute_hostname,
        "compute_hosts": compute_hosts,
        "pbs_job_id": pbs_job_id,
        "pbs_nodefile_path": str(pbs_nodefile_path),
        "pbs_nodefile": pbs_nodefile_path.read_text(encoding="utf-8").splitlines(),
        "target_run_root": config["miyabi"]["target_shared_run_root"],
        "filesystem_type": filesystem_type,
        "filesystem_source": filesystem_source,
        "module_list": module_list_path.read_text(encoding="utf-8").splitlines(),
        "permission_probe": dict(permission_probe),
        "launcher": config["miyabi"]["launcher"],
        "ranks_per_node": config["miyabi"]["ranks_per_node"],
        "claims_boundary": config["claims_boundary"],
    }
    validate_environment_manifest(manifest)
    return manifest


def summarize_smoke(
    *,
    result_root: Path,
    run_id: str,
    path_mode: str,
    expected: str,
    required_distinct_hosts: int,
) -> dict[str, Any]:
    role_results = [
        read_json_record(result_root / f"smoke-rank-{rank}.json") for rank in (0, 1)
    ]
    if {result.get("rank") for result in role_results} != {0, 1}:
        raise StorageHarnessError("smoke results do not contain both ranks")
    if any(result.get("run_id") != run_id for result in role_results):
        raise StorageHarnessError("smoke result run_id mismatch")
    hosts = sorted({str(result.get("hostname", "")) for result in role_results})
    if len(hosts) < required_distinct_hosts:
        raise StorageHarnessError("smoke roles did not run on distinct hosts")
    statuses = [result.get("status") for result in role_results]
    if expected == "pass":
        if path_mode != "shared" or statuses != ["passed", "passed"]:
            raise StorageHarnessError("shared-path smoke did not pass on both roles")
        outcome = "shared_visibility_confirmed"
    elif expected == "failure":
        failure_kinds = {str(result.get("failure_kind", "")) for result in role_results}
        required_failure = {"acknowledgement_timeout", "visibility_timeout"}
        if path_mode != "local-negative" or statuses != ["failed", "failed"]:
            raise StorageHarnessError("node-local negative control did not fail on both roles")
        if not required_failure.issubset(failure_kinds):
            raise StorageHarnessError("negative control failed for the wrong reason")
        outcome = "node_local_visibility_rejected"
    else:
        raise StorageHarnessError(f"unknown expected smoke result: {expected}")
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "path_mode": path_mode,
        "expected": expected,
        "outcome": outcome,
        "distinct_hosts": hosts,
        "roles": role_results,
    }
    atomic_write_json(result_root / "smoke-summary.json", summary)
    return summary


def _command_manifest(arguments: argparse.Namespace) -> int:
    config = load_storage_config(arguments.config)
    permission_probe = read_json_record(arguments.permission_probe)
    manifest = build_environment_manifest(
        config=config,
        run_id=arguments.run_id,
        initial_hostname=arguments.initial_hostname,
        compute_hostname=arguments.compute_hostname,
        pbs_job_id=arguments.pbs_job_id,
        pbs_nodefile_path=arguments.pbs_nodefile,
        filesystem_type=arguments.filesystem_type,
        filesystem_source=arguments.filesystem_source,
        module_list_path=arguments.module_list,
        permission_probe=permission_probe,
    )
    atomic_write_json(arguments.output, manifest)
    return 0


def _command_smoke_role(arguments: argparse.Namespace) -> int:
    config = load_storage_config(arguments.config)
    result = run_smoke_role(
        config=config,
        run_id=arguments.run_id,
        test_root=arguments.test_root,
        result_root=arguments.result_root,
        path_mode=arguments.path_mode,
    )
    return 0 if result["status"] == "passed" else 2


def _command_summarize(arguments: argparse.Namespace) -> int:
    config = load_storage_config(arguments.config)
    summary = summarize_smoke(
        result_root=arguments.result_root,
        run_id=arguments.run_id,
        path_mode=arguments.path_mode,
        expected=arguments.expected,
        required_distinct_hosts=int(config["smoke"]["required_distinct_hosts"]),
    )
    print(_canonical_json(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--config", type=Path, required=True)
    manifest.add_argument("--run-id", required=True)
    manifest.add_argument("--initial-hostname", required=True)
    manifest.add_argument("--compute-hostname", required=True)
    manifest.add_argument("--pbs-job-id", required=True)
    manifest.add_argument("--pbs-nodefile", type=Path, required=True)
    manifest.add_argument("--filesystem-type", required=True)
    manifest.add_argument("--filesystem-source", required=True)
    manifest.add_argument("--module-list", type=Path, required=True)
    manifest.add_argument("--permission-probe", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(handler=_command_manifest)

    role = subparsers.add_parser("smoke-role")
    role.add_argument("--config", type=Path, required=True)
    role.add_argument("--run-id", required=True)
    role.add_argument("--test-root", type=Path, required=True)
    role.add_argument("--result-root", type=Path, required=True)
    role.add_argument("--path-mode", choices=("shared", "local-negative"), required=True)
    role.set_defaults(handler=_command_smoke_role)

    summarize = subparsers.add_parser("summarize-smoke")
    summarize.add_argument("--config", type=Path, required=True)
    summarize.add_argument("--run-id", required=True)
    summarize.add_argument("--result-root", type=Path, required=True)
    summarize.add_argument("--path-mode", choices=("shared", "local-negative"), required=True)
    summarize.add_argument("--expected", choices=("pass", "failure"), required=True)
    summarize.set_defaults(handler=_command_summarize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except StorageHarnessError as error:
        print(f"storage harness error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
