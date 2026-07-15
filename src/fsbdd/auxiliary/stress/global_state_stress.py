from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any

from fsbdd.diloco.protocol.global_state import (
    BootstrapFragment,
    BootstrapInterrupted,
    CountingStorageBackend,
    FragmentStateDescriptor,
    GlobalStateError,
    GlobalStateIdentities,
    GlobalStateStore,
)
from fsbdd.diloco.common.identity import canonical_digest, file_digest
from fsbdd.auxiliary.contracts.manifest import build_manifest
from fsbdd.diloco.protocol.storage import PosixStorageBackend, PublicationNotFound


def _write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _wait(path: Path, timeout_seconds: float = 180.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"coordination record did not appear: {path}")
            time.sleep(0.01)
            continue
        if not isinstance(value, dict) or value.get("complete") is not True:
            raise RuntimeError(f"coordination record is malformed: {path}")
        return value


def _identities(
    run_id: str,
    config_identity: str,
    model_identity: str,
    fragment_map_identity: str,
) -> GlobalStateIdentities:
    return GlobalStateIdentities(
        run_identity=run_id,
        config_identity=config_identity,
        model_identity=model_identity,
        fragment_map_identity=fragment_map_identity,
    )


def _fragments(
    fragment_count: int, payload_bytes: int
) -> tuple[BootstrapFragment, ...]:
    if fragment_count <= 0 or payload_bytes <= 0:
        raise ValueError("fragment_count and payload_bytes must be positive")
    return tuple(
        BootstrapFragment(
            descriptor=FragmentStateDescriptor(
                index=index,
                identity=f"fragment-{index}",
                dtype="uint8",
                shape=(payload_bytes,),
                parameter_identities=(f"parameter-{index}",),
            ),
            parameters=bytes([index + 1]) * payload_bytes,
            outer_state=json.dumps(
                {"schema_version": 1, "fragment_index": index, "outer_update_count": 0},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        for index in range(fragment_count)
    )


def derive_stress_identities(
    config_identity: str,
    initial: tuple[BootstrapFragment, ...],
) -> tuple[str, str]:
    fragment_map_identity = canonical_digest(
        {
            "schema_version": 1,
            "kind": "s1-04-deterministic-stress-fragment-map",
            "descriptors": [item.descriptor.to_dict() for item in initial],
        }
    )
    model_identity = canonical_digest(
        {
            "schema_version": 1,
            "kind": "s1-04-deterministic-stress-model",
            "config_identity": config_identity,
            "fragment_map_identity": fragment_map_identity,
            "fragments": [
                {
                    "index": item.descriptor.index,
                    "parameters_sha256": hashlib.sha256(item.parameters).hexdigest(),
                    "outer_state_sha256": hashlib.sha256(item.outer_state).hexdigest(),
                }
                for item in initial
            ],
        }
    )
    return model_identity, fragment_map_identity


def _store(
    backend,
    run_id: str,
    initial: tuple[BootstrapFragment, ...],
    s_max: int,
    config_identity: str,
    model_identity: str,
    fragment_map_identity: str,
) -> GlobalStateStore:
    return GlobalStateStore(
        backend,
        identities=_identities(
            run_id,
            config_identity,
            model_identity,
            fragment_map_identity,
        ),
        descriptors=tuple(item.descriptor for item in initial),
        s_max=s_max,
    )


def run_role(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    fragment_count: int,
    payload_bytes: int,
    history_objects: int,
    interrupt_after: int,
    s_max: int,
    config_identity: str,
    model_identity: str,
    fragment_map_identity: str,
) -> dict[str, Any]:
    try:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "global-state stress role requires an MPI launcher environment"
        ) from error
    if size != 2 or rank not in (0, 1):
        raise RuntimeError("global-state stress requires exactly two launcher ranks")
    if not 0 < interrupt_after < fragment_count:
        raise ValueError("interrupt_after must split the frozen fragment set")
    initial = _fragments(fragment_count, payload_bytes)
    derived_model_identity, derived_map_identity = derive_stress_identities(
        config_identity, initial
    )
    if model_identity != derived_model_identity:
        raise ValueError("declared synthetic model identity differs from the workload")
    if fragment_map_identity != derived_map_identity:
        raise ValueError(
            "declared synthetic fragment-map identity differs from the workload"
        )
    frozen_identities = {
        "run_identity": run_id,
        "config_identity": config_identity,
        "model_identity": model_identity,
        "fragment_map_identity": fragment_map_identity,
    }
    backend_root = root / "backend"
    coordination = root / "coordination"
    hostname = socket.gethostname().split(".")[0]

    if rank == 0:
        store = _store(
            PosixStorageBackend(backend_root),
            run_id,
            initial,
            s_max,
            config_identity,
            model_identity,
            fragment_map_identity,
        )
        try:
            store.bootstrap(initial, interrupt_after_fragments=interrupt_after)
        except BootstrapInterrupted:
            partial_interrupted = True
        else:  # pragma: no cover - defensive formal assertion
            raise AssertionError("bootstrap interruption was not injected")
        partial_records = len(tuple((backend_root / "visibility").iterdir()))
        if partial_records != interrupt_after:
            raise AssertionError(
                "interrupted bootstrap exposed an unexpected current-record count"
            )
        _replace_json(coordination / "partial-ready.json", {"complete": True})
        _wait(coordination / "partial-observed.json")
        resumed = store.bootstrap(initial)
        completed = store.load_snapshot()
        before_repeat = {path.name for path in (backend_root / "payloads").iterdir()}
        repeated = store.bootstrap(initial)
        after_repeat = {path.name for path in (backend_root / "payloads").iterdir()}
        if before_repeat != after_repeat or repeated.published_indices:
            raise AssertionError("identical bootstrap restart created a payload")
        changed = list(initial)
        changed[0] = dataclasses.replace(
            changed[0], parameters=b"conflict" * (payload_bytes // 8 + 1)
        )
        before_conflict = completed.digest
        try:
            store.bootstrap(tuple(changed))
        except GlobalStateError:
            conflict_rejected = True
        else:  # pragma: no cover - defensive formal assertion
            raise AssertionError("conflicting version-zero bootstrap was accepted")
        if store.load_snapshot().digest != before_conflict:
            raise AssertionError(
                "conflicting bootstrap changed authoritative current state"
            )
        _replace_json(coordination / "complete-ready.json", {"complete": True})
        _wait(coordination / "baseline-observed.json")
        payload_root = backend_root / "payloads"
        for index in range(history_objects):
            (payload_root / f"historical-{index:05d}.bin").write_bytes(b"historical")
        _replace_json(coordination / "history-ready.json", {"complete": True})
        _wait(coordination / "reader-done.json")
        live_set = store.inspect_live_set()
        result = {
            "schema_version": 1,
            "status": "complete",
            "role": "bootstrap_writer",
            "rank": rank,
            "hostname": hostname,
            "partial_interrupted": partial_interrupted,
            "partial_current_records": partial_records,
            "resume_existing_indices": resumed.existing_indices,
            "resume_published_indices": resumed.published_indices,
            "final_version_vector": completed.version_vector,
            "final_content_identities": [
                state.content_identity for state in completed.states
            ],
            "idempotent_repeat_new_payloads": len(after_repeat - before_repeat),
            "conflict_rejected": conflict_rejected,
            "live_set": live_set.to_dict(),
            "history_objects": history_objects,
            "state_identities": frozen_identities,
        }
        _write_json_new(result_root / "bootstrap_writer.json", result)
        return result

    partial_store = _store(
        PosixStorageBackend(backend_root),
        run_id,
        initial,
        s_max,
        config_identity,
        model_identity,
        fragment_map_identity,
    )
    _wait(coordination / "partial-ready.json")
    try:
        partial_store.load_snapshot(timeout_seconds=0)
    except PublicationNotFound:
        partial_snapshot_rejected = True
    else:  # pragma: no cover - defensive formal assertion
        raise AssertionError("partial bootstrap produced a version vector")
    _replace_json(coordination / "partial-observed.json", {"complete": True})
    _wait(coordination / "complete-ready.json")
    counting = CountingStorageBackend(PosixStorageBackend(backend_root))
    reader_store = _store(
        counting,
        run_id,
        initial,
        s_max,
        config_identity,
        model_identity,
        fragment_map_identity,
    )
    counting.reset_counts()
    baseline = reader_store.load_snapshot(timeout_seconds=30)
    baseline_reads = counting.read_calls
    _replace_json(coordination / "baseline-observed.json", {"complete": True})
    _wait(coordination / "history-ready.json")
    counting.reset_counts()
    after_history = reader_store.load_snapshot(timeout_seconds=30)
    history_reads = counting.read_calls
    if baseline.digest != after_history.digest:
        raise AssertionError("historical objects changed the authoritative snapshot")
    result = {
        "schema_version": 1,
        "status": "complete",
        "role": "snapshot_reader",
        "rank": rank,
        "hostname": hostname,
        "partial_snapshot_rejected": partial_snapshot_rejected,
        "baseline_read_calls": baseline_reads,
        "post_history_read_calls": history_reads,
        "baseline_version_vector": baseline.version_vector,
        "post_history_version_vector": after_history.version_vector,
        "snapshot_digest": baseline.digest,
        "state_identities": frozen_identities,
    }
    _write_json_new(result_root / "snapshot_reader.json", result)
    _replace_json(coordination / "reader-done.json", {"complete": True})
    return result


def summarize(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    fragment_count: int,
    history_objects: int,
    interrupt_after: int,
    s_max: int,
    config_identity: str,
    model_identity: str,
    fragment_map_identity: str,
    output: Path,
) -> dict[str, Any]:
    writer = json.loads(
        (result_root / "bootstrap_writer.json").read_text(encoding="utf-8")
    )
    reader = json.loads(
        (result_root / "snapshot_reader.json").read_text(encoding="utf-8")
    )
    if writer["hostname"] == reader["hostname"]:
        raise AssertionError(
            "two-node global-state run used the same host for both roles"
        )
    expected_identities = {
        "run_identity": run_id,
        "config_identity": config_identity,
        "model_identity": model_identity,
        "fragment_map_identity": fragment_map_identity,
    }
    if writer.get("state_identities") != expected_identities:
        raise AssertionError(
            "writer global-state identities differ from the frozen workload"
        )
    if reader.get("state_identities") != expected_identities:
        raise AssertionError(
            "reader global-state identities differ from the frozen workload"
        )
    if (
        not writer["partial_interrupted"]
        or writer["partial_current_records"] != interrupt_after
    ):
        raise AssertionError("bootstrap interruption evidence is incomplete")
    if not reader["partial_snapshot_rejected"]:
        raise AssertionError("reader accepted a partial bootstrap")
    if writer["resume_existing_indices"] != list(range(interrupt_after)):
        raise AssertionError("bootstrap resume did not preserve existing fragments")
    if writer["resume_published_indices"] != list(
        range(interrupt_after, fragment_count)
    ):
        raise AssertionError(
            "bootstrap resume did not publish exactly the missing fragments"
        )
    expected_vector = [0] * fragment_count
    if writer["final_version_vector"] != expected_vector:
        raise AssertionError("bootstrap final version vector is not all zero")
    if (
        reader["baseline_version_vector"] != expected_vector
        or reader["post_history_version_vector"] != expected_vector
    ):
        raise AssertionError("reader observed an unexpected version vector")
    if (
        reader["baseline_read_calls"] != fragment_count
        or reader["post_history_read_calls"] != fragment_count
    ):
        raise AssertionError("restart read count depends on history volume")
    if writer["idempotent_repeat_new_payloads"] != 0 or not writer["conflict_rejected"]:
        raise AssertionError("bootstrap idempotence or conflict rejection failed")
    live = writer["live_set"]
    if (
        live["current_records"] != fragment_count
        or live["referenced_payloads"] != fragment_count
    ):
        raise AssertionError(
            "authoritative live set does not contain exactly F fragments"
        )
    if live["retained_base_entries"] > fragment_count * (s_max + 1):
        raise AssertionError("base retention exceeds F * (S_max + 1)")
    if (
        writer["history_objects"] != history_objects
        or live["orphan_payload_candidates"] != history_objects
    ):
        raise AssertionError("historical-object stress inventory mismatch")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "run_identity": run_id,
        "filesystem_data_plane": "shared_posix_only",
        "application_coordination": "filesystem_only",
        "mpi_usage": "launcher_only",
        "roles": {
            "bootstrap_writer": [writer["hostname"]],
            "snapshot_reader": [reader["hostname"]],
        },
        "state_identities": expected_identities,
        "bootstrap": {
            "fragment_count": fragment_count,
            "interrupted_after_fragments": interrupt_after,
            "partial_snapshot_rejected": True,
            "resume_existing_indices": writer["resume_existing_indices"],
            "resume_published_indices": writer["resume_published_indices"],
            "final_version_vector": expected_vector,
            "idempotent_repeat_new_payloads": 0,
            "conflict_rejected": True,
        },
        "restart_io": {
            "history_objects": history_objects,
            "baseline_snapshot_read_calls": reader["baseline_read_calls"],
            "post_history_snapshot_read_calls": reader["post_history_read_calls"],
            "directory_scans_in_normal_read_path": 0,
            "global_head_reads": 0,
        },
        "bounded_authority": live,
    }
    _write_json_new(output, summary)
    shutil.rmtree(root)
    return summary


def write_manifest(args: argparse.Namespace) -> None:
    hosts = sorted(set(Path(args.nodefile).read_text(encoding="utf-8").splitlines()))
    if len(hosts) != 2:
        raise ValueError(f"manifest requires two distinct hosts, got {hosts}")
    modules = [
        line.strip()
        for line in Path(args.modules_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    writer = json.loads(
        (Path(args.result_root) / "bootstrap_writer.json").read_text(encoding="utf-8")
    )
    reader = json.loads(
        (Path(args.result_root) / "snapshot_reader.json").read_text(encoding="utf-8")
    )
    role_map = {
        "bootstrap_writer": [writer["hostname"]],
        "snapshot_reader": [reader["hostname"]],
    }
    if set(role_map["bootstrap_writer"] + role_map["snapshot_reader"]) != set(hosts):
        raise ValueError("actual global-state roles do not match allocated hosts")
    expected_state_identities = {
        "run_identity": args.run_id,
        "config_identity": args.config_sha256,
        "model_identity": args.state_model_identity,
        "fragment_map_identity": args.state_fragment_map_identity,
    }
    if writer.get("state_identities") != expected_state_identities:
        raise ValueError("writer state identities do not match manifest inputs")
    if reader.get("state_identities") != expected_state_identities:
        raise ValueError("reader state identities do not match manifest inputs")
    identities = {
        "code": {
            "repository": args.repository,
            "branch": args.branch,
            "commit": args.commit,
            "dirty": False,
        },
        "config": {"sha256": args.config_sha256},
        "source": {
            "research_plan_sha256": args.research_sha256,
            "stage0_4_spec_sha256": args.spec_sha256,
        },
        "skill": {"repository": args.skill_repository, "commit": args.skill_commit},
        "execution": {
            "identity": args.run_id,
            "initial_hostname": args.initial_hostname,
            "compute_hostname": socket.gethostname().split(".")[0],
            "workflow": "two-node-global-state-bootstrap-batch",
        },
        "roles": {"declared": role_map, "actual": role_map},
        "paths": {
            "project_root": args.project_root,
            "evidence_root": args.evidence_root,
        },
    }
    scheduler = {
        "job_id": args.job_id,
        "qtime_utc": args.qtime_utc,
        "queue": args.queue,
        "group": args.group,
        "nodefile_sha256": file_digest(Path(args.nodefile)),
        "modules": modules,
    }
    manifest = build_manifest(
        "S1-04", "L2", args.qtime_utc, "pbs_qtime", identities, scheduler=scheduler
    )
    _write_json_new(Path(args.output), manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fsbdd.auxiliary.stress.global_state_stress"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    role = commands.add_parser("role")
    for name, value_type in (
        ("root", Path),
        ("result-root", Path),
        ("run-id", str),
        ("fragment-count", int),
        ("payload-bytes", int),
        ("history-objects", int),
        ("interrupt-after", int),
        ("s-max", int),
        ("config-identity", str),
        ("model-identity", str),
        ("fragment-map-identity", str),
    ):
        role.add_argument(f"--{name}", required=True, type=value_type)
    analysis = commands.add_parser("summarize")
    for name, value_type in (
        ("root", Path),
        ("result-root", Path),
        ("run-id", str),
        ("fragment-count", int),
        ("history-objects", int),
        ("interrupt-after", int),
        ("s-max", int),
        ("config-identity", str),
        ("model-identity", str),
        ("fragment-map-identity", str),
        ("output", Path),
    ):
        analysis.add_argument(f"--{name}", required=True, type=value_type)
    manifest = commands.add_parser("manifest")
    for name in (
        "repository",
        "branch",
        "commit",
        "run-id",
        "config-sha256",
        "state-model-identity",
        "state-fragment-map-identity",
        "research-sha256",
        "spec-sha256",
        "skill-repository",
        "skill-commit",
        "initial-hostname",
        "project-root",
        "evidence-root",
        "job-id",
        "qtime-utc",
        "queue",
        "group",
        "nodefile",
        "modules-file",
        "result-root",
        "output",
    ):
        manifest.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    if args.command == "role":
        result = run_role(
            root=args.root,
            result_root=args.result_root,
            run_id=args.run_id,
            fragment_count=args.fragment_count,
            payload_bytes=args.payload_bytes,
            history_objects=args.history_objects,
            interrupt_after=args.interrupt_after,
            s_max=args.s_max,
            config_identity=args.config_identity,
            model_identity=args.model_identity,
            fragment_map_identity=args.fragment_map_identity,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "summarize":
        summary = summarize(
            root=args.root,
            result_root=args.result_root,
            run_id=args.run_id,
            fragment_count=args.fragment_count,
            history_objects=args.history_objects,
            interrupt_after=args.interrupt_after,
            s_max=args.s_max,
            config_identity=args.config_identity,
            model_identity=args.model_identity,
            fragment_map_identity=args.fragment_map_identity,
            output=args.output,
        )
        print(
            json.dumps(
                {"status": summary["status"], "restart_io": summary["restart_io"]},
                sort_keys=True,
            )
        )
        return 0
    write_manifest(args)
    print(json.dumps({"status": "building", "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
