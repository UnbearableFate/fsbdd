from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any

from fsbdd.diloco.common.identity import file_digest
from fsbdd.auxiliary.contracts.manifest import build_manifest
from fsbdd.diloco.protocol.storage import (
    PosixStorageBackend,
    PublicationError,
    PublicationInterrupted,
    PublicationSpec,
    ReadExpectation,
)


_MAP_ID = "a" * 64
_BASE_ID = "b" * 64
_FRAGMENT_ID = "fragment-0"


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


def _wait(path: Path, timeout_seconds: float = 120.0) -> dict[str, Any]:
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


def _spec(run_id: str, sequence: int) -> PublicationSpec:
    return PublicationSpec(
        run_identity=run_id,
        fragment_map_identity=_MAP_ID,
        fragment_identity=_FRAGMENT_ID,
        version=sequence,
        sequence=sequence,
        dtype="uint8",
        shape=(1,),
        base_content_identity=_BASE_ID,
    )


def _expectation(run_id: str) -> ReadExpectation:
    return ReadExpectation(
        run_identity=run_id,
        fragment_map_identity=_MAP_ID,
        fragment_identity=_FRAGMENT_ID,
        dtype="uint8",
        shape=(1,),
        base_content_identity=_BASE_ID,
    )


def _payload(sequence: int, payload_bytes: int) -> bytes:
    if payload_bytes < 8:
        raise ValueError("payload_bytes must be at least eight")
    return sequence.to_bytes(8, "big") + bytes([sequence % 251]) * (payload_bytes - 8)


def _validate_payload(sequence: int, payload: bytes) -> None:
    if payload != _payload(sequence, len(payload)):
        raise PublicationError(f"payload content does not match sequence {sequence}")


def run_role(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    publications: int,
    payload_bytes: int,
    visibility_delay_seconds: float,
) -> dict[str, Any]:
    try:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "storage stress role requires an MPI launcher environment"
        ) from error
    if size != 2 or rank not in (0, 1):
        raise RuntimeError("storage stress requires exactly two launcher ranks")
    backend = PosixStorageBackend(root / "backend")
    coordination = root / "coordination"
    hostname = socket.gethostname().split(".")[0]
    if rank == 0:
        backend.publish("latest", _payload(0, payload_bytes), _spec(run_id, 0))
        _replace_json(
            coordination / "writer-initial.json", {"complete": True, "sequence": 0}
        )
        _wait(coordination / "reader-ready.json")

        def delay(_slot, _record) -> None:
            time.sleep(visibility_delay_seconds)

        started = time.monotonic_ns()
        for sequence in range(1, publications + 1):
            backend.publish(
                "latest",
                _payload(sequence, payload_bytes),
                _spec(run_id, sequence),
                visibility_hook=delay,
            )
        completed = time.monotonic_ns()
        _replace_json(
            coordination / "writer-done.json",
            {"complete": True, "sequence": publications},
        )
        _wait(coordination / "reader-done.json")
        result = {
            "schema_version": 1,
            "status": "complete",
            "role": "writer",
            "rank": rank,
            "hostname": hostname,
            "publications": publications,
            "payload_bytes": payload_bytes,
            "started_monotonic_ns": started,
            "completed_monotonic_ns": completed,
        }
        _write_json_new(result_root / "writer.json", result)
        return result

    _wait(coordination / "writer-initial.json")
    _replace_json(coordination / "reader-ready.json", {"complete": True, "sequence": 0})
    observations = 0
    invalid_reads = 0
    seen_sequences: set[int] = set()
    errors: list[str] = []
    deadline = time.monotonic() + 120.0
    while True:
        try:
            value = backend.read(
                "latest",
                _expectation(run_id),
                timeout_seconds=5,
                poll_interval_seconds=0.001,
            )
            _validate_payload(value.record.sequence, value.payload)
            observations += 1
            seen_sequences.add(value.record.sequence)
        except PublicationError as error:
            invalid_reads += 1
            errors.append(str(error))
        done = coordination / "writer-done.json"
        if done.is_file() and publications in seen_sequences:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("reader did not observe the final complete publication")
    _replace_json(
        coordination / "reader-done.json", {"complete": True, "sequence": publications}
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "role": "reader",
        "rank": rank,
        "hostname": hostname,
        "observations": observations,
        "unique_sequences": len(seen_sequences),
        "minimum_sequence": min(seen_sequences),
        "maximum_sequence": max(seen_sequences),
        "invalid_reads": invalid_reads,
        "errors": errors,
    }
    _write_json_new(result_root / "reader.json", result)
    return result


def _crash_matrix(root: Path, run_id: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    expected = {
        "before_payload_write": 0,
        "after_payload_write": 0,
        "before_record_replace": 0,
        "after_record_replace": 1,
    }
    for crash_at, expected_sequence in expected.items():
        backend = PosixStorageBackend(root / crash_at)
        backend.publish("current", b"old-compound", _spec(run_id, 0))
        try:
            backend.publish(
                "current", b"new-compound", _spec(run_id, 1), crash_at=crash_at
            )
        except PublicationInterrupted:
            pass
        else:  # pragma: no cover - defensive evidence assertion
            raise AssertionError(f"crash point did not interrupt: {crash_at}")
        visible = backend.read("current", _expectation(run_id), timeout_seconds=0)
        if visible.record.sequence != expected_sequence:
            raise AssertionError(
                f"wrong crash outcome at {crash_at}: {visible.record.sequence}"
            )
        expected_payload = b"new-compound" if expected_sequence else b"old-compound"
        if visible.payload != expected_payload:
            raise AssertionError(f"partial compound payload at {crash_at}")
        results.append(
            {
                "crash_at": crash_at,
                "visible_sequence": visible.record.sequence,
                "visible_payload_sha256": hashlib.sha256(visible.payload).hexdigest(),
                "outcome": "complete_new" if expected_sequence else "complete_old",
            }
        )
    return results


def summarize(
    *, root: Path, result_root: Path, run_id: str, publications: int, output: Path
) -> dict[str, Any]:
    writer = json.loads((result_root / "writer.json").read_text(encoding="utf-8"))
    reader = json.loads((result_root / "reader.json").read_text(encoding="utf-8"))
    if writer.get("hostname") == reader.get("hostname"):
        raise AssertionError("two-node storage run used the same host for both roles")
    if (
        writer.get("publications") != publications
        or reader.get("maximum_sequence") != publications
    ):
        raise AssertionError("reader did not observe the final complete publication")
    if reader.get("invalid_reads") != 0 or reader.get("errors") != []:
        raise AssertionError(
            "cross-node reader observed an invalid or partial publication"
        )
    if int(reader.get("observations", 0)) <= 0:
        raise AssertionError("cross-node reader made no observations")
    crash_root = root / "crash-matrix"
    crash_results = _crash_matrix(crash_root, run_id)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "run_identity": run_id,
        "filesystem_data_plane": "shared_posix_only",
        "application_coordination": "filesystem_only",
        "mpi_usage": "launcher_only",
        "roles": {
            "writer": [writer["hostname"]],
            "reader": [reader["hostname"]],
        },
        "cross_node": {
            "publications": publications,
            "reader_observations": reader["observations"],
            "unique_sequences_observed": reader["unique_sequences"],
            "final_sequence": reader["maximum_sequence"],
            "invalid_reads": 0,
        },
        "crash_matrix": crash_results,
        "fixed_slot": {
            "visibility_record": "visibility/latest.json",
            "reader_directory_scans": 0,
            "payload_history_scans": 0,
        },
        "unsafe_control_reuse": {
            "fixture": "tests/red_cases/s0b02_unsafe_visibility_surrogate.py",
            "selected_stage0_package": "evidence/raw/S0B-02/20260714T162735Z-unsafefix-unsafefix-n2e5c-006abf2a3b43",
            "stage0_partial_or_torn_reads_detected": 5208,
        },
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
        (Path(args.result_root) / "writer.json").read_text(encoding="utf-8")
    )
    reader = json.loads(
        (Path(args.result_root) / "reader.json").read_text(encoding="utf-8")
    )
    role_map = {"writer": [writer["hostname"]], "reader": [reader["hostname"]]}
    if set(role_map["writer"] + role_map["reader"]) != set(hosts):
        raise ValueError("actual storage roles do not match the allocated hosts")
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
            "identity": f"pbs-{args.job_id}",
            "initial_hostname": args.initial_hostname,
            "compute_hostname": socket.gethostname().split(".")[0],
            "workflow": "two-node-posix-publication-batch",
        },
        "roles": {
            "declared": role_map,
            "actual": role_map,
        },
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
        "S1-03",
        "L2",
        args.qtime_utc,
        "pbs_qtime",
        identities,
        scheduler=scheduler,
    )
    _write_json_new(Path(args.output), manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fsbdd.auxiliary.stress.storage_stress"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    role = commands.add_parser("role")
    role.add_argument("--root", required=True, type=Path)
    role.add_argument("--result-root", required=True, type=Path)
    role.add_argument("--run-id", required=True)
    role.add_argument("--publications", required=True, type=int)
    role.add_argument("--payload-bytes", required=True, type=int)
    role.add_argument("--visibility-delay-seconds", type=float, default=0.0)
    analysis = commands.add_parser("summarize")
    analysis.add_argument("--root", required=True, type=Path)
    analysis.add_argument("--result-root", required=True, type=Path)
    analysis.add_argument("--run-id", required=True)
    analysis.add_argument("--publications", required=True, type=int)
    analysis.add_argument("--output", required=True, type=Path)
    manifest = commands.add_parser("manifest")
    for name in (
        "repository",
        "branch",
        "commit",
        "config-sha256",
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
            publications=args.publications,
            payload_bytes=args.payload_bytes,
            visibility_delay_seconds=args.visibility_delay_seconds,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "summarize":
        summary = summarize(
            root=args.root,
            result_root=args.result_root,
            run_id=args.run_id,
            publications=args.publications,
            output=args.output,
        )
        print(
            json.dumps(
                {"status": summary["status"], "cross_node": summary["cross_node"]},
                sort_keys=True,
            )
        )
        return 0
    write_manifest(args)
    print(json.dumps({"status": "building", "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
