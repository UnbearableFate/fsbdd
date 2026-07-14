"""Compute-only Stage 0 metadata and sequential-I/O capacity benchmark.

MPI is used only to place independent Python ranks.  Coordination, payloads,
and fixed discovery slots all use the frozen shared filesystem path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .storage_bench import (
    StorageHarnessError,
    atomic_write_json,
    load_storage_config,
    read_json_record,
    require_compute_context,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def percentile(values: Sequence[int | float], percent: float) -> float:
    if not values:
        raise StorageHarnessError("cannot compute a percentile of an empty sample")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def deterministic_chunk(size: int, *, identity: str) -> bytes:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise StorageHarnessError("chunk size must be a positive integer")
    seed = hashlib.sha256(identity.encode("utf-8")).digest()
    return (seed * ((size + len(seed) - 1) // len(seed)))[:size]


class Reservoir:
    def __init__(self, limit: int, *, seed: str) -> None:
        self.limit = limit
        self.values: list[int] = []
        self.seen = 0
        self._random = random.Random(seed)

    def add(self, value: int) -> None:
        self.seen += 1
        if len(self.values) < self.limit:
            self.values.append(value)
            return
        position = self._random.randrange(self.seen)
        if position < self.limit:
            self.values[position] = value


def metadata_matrix(config: Mapping[str, Any]) -> list[tuple[int, int, str, int, str]]:
    metadata = config["metadata"]
    return [
        (int(learners), int(fragments), str(layout), repeat, str(profile))
        for learners in metadata["learner_counts"]
        for fragments in metadata["fragment_counts"]
        for layout in metadata["directory_layouts"]
        for repeat in range(int(metadata["repeats"]))
        for profile in metadata["profiles"]
    ]


def bandwidth_matrix(config: Mapping[str, Any]) -> list[tuple[int, int, int]]:
    bandwidth = config["bandwidth"]
    return [
        (int(size_mb), int(streams), repeat)
        for size_mb in bandwidth["payload_sizes_mb"]
        for streams in bandwidth["streams"]
        for repeat in range(int(bandwidth["repeats"]))
    ]


def _rank_identity() -> tuple[int, int]:
    try:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise StorageHarnessError("capacity role requires an Open MPI rank identity") from error
    return rank, world


def _wait_json(path: Path, *, phase: str | None, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "missing"
    while time.monotonic() < deadline:
        try:
            value = read_json_record(path)
            if phase is None or value.get("phase") == phase:
                return value
            last_error = f"stale phase {value.get('phase')!r}"
        except (OSError, StorageHarnessError) as error:
            last_error = str(error)
        time.sleep(0.001)
    raise StorageHarnessError(f"timeout waiting for {path}: {last_error}")


def _wait_rank_records(
    directory: Path,
    *,
    prefix: str,
    ranks: Iterable[int],
    phase: str,
    timeout: float,
) -> list[dict[str, Any]]:
    return [
        _wait_json(directory / f"{prefix}-{rank}.json", phase=phase, timeout=timeout)
        for rank in ranks
    ]


def _dataset_root(
    test_root: Path, *, learners: int, fragments: int, layout: str, repeat: int
) -> Path:
    return test_root / "metadata" / f"m{learners}-f{fragments}-{layout}-r{repeat}"


def _slot_path(root: Path, *, layout: str, learner: int, fragment: int) -> Path:
    if layout == "flat":
        return root / "latest" / f"learner-{learner:02d}-fragment-{fragment:02d}.json"
    return root / "latest" / f"learner-{learner:02d}" / f"fragment-{fragment:02d}.json"


def _prepare_capacity_tree(config: Mapping[str, Any], run_id: str, test_root: Path) -> dict[str, Any]:
    if test_root.exists():
        raise StorageHarnessError(f"capacity test root already exists: {test_root}")
    test_root.mkdir(parents=True)
    fixed_records = 0
    for learners in config["metadata"]["learner_counts"]:
        for fragments in config["metadata"]["fragment_counts"]:
            for layout in config["metadata"]["directory_layouts"]:
                for repeat in range(int(config["metadata"]["repeats"])):
                    root = _dataset_root(
                        test_root,
                        learners=int(learners),
                        fragments=int(fragments),
                        layout=str(layout),
                        repeat=repeat,
                    )
                    for learner in range(int(learners)):
                        for fragment in range(int(fragments)):
                            path = _slot_path(
                                root,
                                layout=str(layout),
                                learner=learner,
                                fragment=fragment,
                            )
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(
                                _canonical_json(
                                    {
                                        "schema_version": 1,
                                        "run_id": run_id,
                                        "learner": learner,
                                        "fragment": fragment,
                                        "sequence": repeat,
                                    }
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                            fixed_records += 1
    history_records = 0
    for size in config["metadata"]["history_sizes_for_negative_control"]:
        directory = test_root / "history-negative" / f"objects-{int(size)}"
        directory.mkdir(parents=True, exist_ok=True)
        for sequence in range(int(size)):
            (directory / f"historical-{sequence:06d}.json").write_text("{}\n", encoding="utf-8")
            history_records += 1
    (test_root / "control").mkdir()
    (test_root / "payloads").mkdir()
    return {"fixed_records": fixed_records, "history_records": history_records}


def _metadata_operation(
    *,
    root: Path,
    profile: str,
    layout: str,
    learner: int,
    learners: int,
    fragments: int,
    run_id: str,
) -> tuple[int, int]:
    started = time.monotonic_ns()
    if profile == "stat":
        path = _slot_path(root, layout=layout, learner=learner, fragment=0)
        path.stat()
        references = 1
    elif profile == "read":
        path = _slot_path(root, layout=layout, learner=learner, fragment=0)
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("run_id") != run_id or record.get("learner") != learner:
            raise StorageHarnessError("fixed-slot read returned the wrong identity")
        references = 1
    elif profile == "readdir":
        if layout == "flat":
            directory = root / "latest"
            expected = learners * fragments
        else:
            directory = root / "latest" / f"learner-{learner:02d}"
            expected = fragments
        references = sum(1 for entry in os.scandir(directory) if entry.name.endswith(".json"))
        if references != expected:
            raise StorageHarnessError(
                f"fixed-slot readdir saw {references} references instead of {expected}"
            )
    else:
        raise StorageHarnessError(f"unknown metadata profile: {profile}")
    return references, time.monotonic_ns() - started


def _run_metadata_state(
    *,
    root: Path,
    profile: str,
    layout: str,
    learner: int,
    learners: int,
    fragments: int,
    run_id: str,
    state: str,
    sustained_seconds: float,
    reservoir_size: int,
    seed: str,
) -> dict[str, Any]:
    sample = Reservoir(reservoir_size, seed=seed)
    references = 0
    syscalls = 0
    started = time.monotonic_ns()
    if state == "first_touch":
        if profile == "readdir":
            count, elapsed = _metadata_operation(
                root=root,
                profile=profile,
                layout=layout,
                learner=learner,
                learners=learners,
                fragments=fragments,
                run_id=run_id,
            )
            references += count
            syscalls += 1
            sample.add(elapsed)
        else:
            for fragment in range(fragments):
                operation_started = time.monotonic_ns()
                path = _slot_path(root, layout=layout, learner=learner, fragment=fragment)
                if profile == "stat":
                    path.stat()
                else:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if record.get("run_id") != run_id or record.get("fragment") != fragment:
                        raise StorageHarnessError("fixed-slot read identity mismatch")
                sample.add(time.monotonic_ns() - operation_started)
                references += 1
                syscalls += 1
    elif state == "warm":
        deadline = time.monotonic() + sustained_seconds
        while time.monotonic() < deadline:
            if profile == "readdir":
                count, elapsed = _metadata_operation(
                    root=root,
                    profile=profile,
                    layout=layout,
                    learner=learner,
                    learners=learners,
                    fragments=fragments,
                    run_id=run_id,
                )
                references += count
                syscalls += 1
                sample.add(elapsed)
            else:
                for fragment in range(fragments):
                    operation_started = time.monotonic_ns()
                    path = _slot_path(root, layout=layout, learner=learner, fragment=fragment)
                    if profile == "stat":
                        path.stat()
                    else:
                        record = json.loads(path.read_text(encoding="utf-8"))
                        if record.get("run_id") != run_id or record.get("fragment") != fragment:
                            raise StorageHarnessError("fixed-slot read identity mismatch")
                    sample.add(time.monotonic_ns() - operation_started)
                    references += 1
                    syscalls += 1
    else:
        raise StorageHarnessError(f"unknown metadata cache state: {state}")
    elapsed_ns = time.monotonic_ns() - started
    return {
        "state": state,
        "reference_operations": references,
        "syscall_operations": syscalls,
        "elapsed_ns": elapsed_ns,
        "latency_samples_ns": sample.values,
        "latency_operations_seen": sample.seen,
        "latency_sampling": "deterministic_reservoir",
    }


def _run_metadata_rank(
    config: Mapping[str, Any], run_id: str, test_root: Path, rank: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metadata = config["metadata"]
    for learners, fragments, layout, repeat, profile in metadata_matrix(config):
        active = rank < learners and (profile != "readdir" or layout != "flat" or rank == 0)
        if not active:
            continue
        root = _dataset_root(
            test_root,
            learners=learners,
            fragments=fragments,
            layout=layout,
            repeat=repeat,
        )
        for state in metadata["cache_states"]:
            result = _run_metadata_state(
                root=root,
                profile=profile,
                layout=layout,
                learner=rank,
                learners=learners,
                fragments=fragments,
                run_id=run_id,
                state=str(state),
                sustained_seconds=float(metadata["sustained_seconds"]),
                reservoir_size=int(metadata["latency_reservoir_size"]),
                seed=f"{run_id}:{rank}:{learners}:{fragments}:{layout}:{repeat}:{profile}:{state}",
            )
            result.update(
                {
                    "learners": learners,
                    "fragments": fragments,
                    "layout": layout,
                    "repeat": repeat,
                    "profile": profile,
                    "rank": rank,
                }
            )
            output.append(result)
    return output


def _history_controls(config: Mapping[str, Any], test_root: Path, run_id: str) -> list[dict[str, Any]]:
    metadata = config["metadata"]
    fixed_root = _dataset_root(
        test_root, learners=16, fragments=32, layout="per_learner", repeat=0
    )
    controls: list[dict[str, Any]] = []
    for size in metadata["history_sizes_for_negative_control"]:
        directory = test_root / "history-negative" / f"objects-{int(size)}"
        scan_latencies: list[int] = []
        scan_counts: list[int] = []
        for _ in range(int(metadata["history_scan_repeats"])):
            started = time.monotonic_ns()
            count = sum(1 for _entry in os.scandir(directory))
            scan_latencies.append(time.monotonic_ns() - started)
            scan_counts.append(count)
        started = time.monotonic_ns()
        bounded_count = 0
        for learner in range(16):
            for fragment in range(32):
                path = _slot_path(
                    fixed_root,
                    layout="per_learner",
                    learner=learner,
                    fragment=fragment,
                )
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("run_id") != run_id:
                    raise StorageHarnessError("bounded discovery read the wrong run")
                bounded_count += 1
        controls.append(
            {
                "history_objects": int(size),
                "history_scan_counts": scan_counts,
                "history_scan_latencies_ns": scan_latencies,
                "bounded_fixed_references": bounded_count,
                "bounded_elapsed_ns": time.monotonic_ns() - started,
            }
        )
    return controls


def _payload_path(test_root: Path, stream: int) -> Path:
    return test_root / "payloads" / f"stream-{stream:02d}.bin"


def _phase_id(size_mb: int, streams: int, repeat: int) -> str:
    return f"size-{size_mb}-streams-{streams}-repeat-{repeat}"


def _write_payload(
    path: Path, *, size_bytes: int, chunk_bytes: int, identity: str
) -> tuple[int, str, int]:
    chunk = deterministic_chunk(min(chunk_bytes, size_bytes), identity=identity)
    temporary = path.with_name(f".{path.name}.tmp-{socket.gethostname()}-{os.getpid()}")
    digest = hashlib.sha256()
    remaining = size_bytes
    started = time.monotonic_ns()
    try:
        with temporary.open("xb") as handle:
            while remaining:
                piece = chunk[: min(len(chunk), remaining)]
                handle.write(piece)
                digest.update(piece)
                remaining -= len(piece)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return size_bytes, digest.hexdigest(), time.monotonic_ns() - started


def _read_payload(path: Path, *, chunk_bytes: int) -> tuple[int, str, int]:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
    digest = hashlib.sha256()
    observed = 0
    started = time.monotonic_ns()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            observed += len(chunk)
    return observed, digest.hexdigest(), time.monotonic_ns() - started


def _run_bandwidth_rank(
    config: Mapping[str, Any],
    run_id: str,
    test_root: Path,
    rank: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bandwidth = config["bandwidth"]
    control = test_root / "control"
    operations: list[dict[str, Any]] = []
    coordinator_rounds: list[dict[str, Any]] = []
    for size_mb, streams, repeat in bandwidth_matrix(config):
        phase = _phase_id(size_mb, streams, repeat)
        writer_ranks = list(range(streams))
        reader_ranks = [8] if streams == 1 else list(range(streams))
        if rank == 0:
            atomic_write_json(control / "write-phase.json", {"phase": phase})
            write_round_start = time.monotonic_ns()
        _wait_json(control / "write-phase.json", phase=phase, timeout=timeout)
        if rank in writer_ranks:
            stream = rank
            size_bytes = size_mb * 1024 * 1024
            observed, digest, elapsed_ns = _write_payload(
                _payload_path(test_root, stream),
                size_bytes=size_bytes,
                chunk_bytes=int(bandwidth["chunk_bytes"]),
                identity=f"{run_id}:{phase}:stream-{stream}",
            )
            operations.append(
                {
                    "phase": phase,
                    "operation": "write",
                    "size_mb": size_mb,
                    "streams": streams,
                    "repeat": repeat,
                    "rank": rank,
                    "stream": stream,
                    "source_writer_rank": rank,
                    "bytes": observed,
                    "sha256": digest,
                    "elapsed_ns": elapsed_ns,
                    "publication": "file_fsync_then_same_directory_replace",
                }
            )
            atomic_write_json(
                control / f"write-done-{rank}.json",
                {"phase": phase, "rank": rank, "sha256": digest},
            )
        _wait_rank_records(
            control,
            prefix="write-done",
            ranks=writer_ranks,
            phase=phase,
            timeout=timeout,
        )
        if rank == 0:
            coordinator_rounds.append(
                {
                    "phase": phase,
                    "operation": "write",
                    "round_elapsed_ns": time.monotonic_ns() - write_round_start,
                }
            )
            atomic_write_json(control / "read-phase.json", {"phase": phase})
            read_round_start = time.monotonic_ns()
        _wait_json(control / "read-phase.json", phase=phase, timeout=timeout)
        if rank in reader_ranks:
            source_stream = 0 if streams == 1 else (rank + int(bandwidth["cross_host_rank_offset"])) % streams
            observed, digest, elapsed_ns = _read_payload(
                _payload_path(test_root, source_stream),
                chunk_bytes=int(bandwidth["chunk_bytes"]),
            )
            operations.append(
                {
                    "phase": phase,
                    "operation": "read",
                    "size_mb": size_mb,
                    "streams": streams,
                    "repeat": repeat,
                    "rank": rank,
                    "stream": rank if streams > 1 else 0,
                    "source_writer_rank": source_stream,
                    "bytes": observed,
                    "sha256": digest,
                    "elapsed_ns": elapsed_ns,
                    "cache_control": "cross_host_reader_plus_posix_fadvise_dontneed_when_available",
                }
            )
            atomic_write_json(
                control / f"read-done-{rank}.json",
                {"phase": phase, "rank": rank, "sha256": digest},
            )
        _wait_rank_records(
            control,
            prefix="read-done",
            ranks=reader_ranks,
            phase=phase,
            timeout=timeout,
        )
        if rank == 0:
            coordinator_rounds.append(
                {
                    "phase": phase,
                    "operation": "read",
                    "round_elapsed_ns": time.monotonic_ns() - read_round_start,
                }
            )
            for stream in range(streams):
                _payload_path(test_root, stream).unlink()
            atomic_write_json(control / "cleanup.json", {"phase": phase})
        _wait_json(control / "cleanup.json", phase=phase, timeout=timeout)
    return operations, coordinator_rounds


def _count_matching(root: Path, predicate: Any) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file() and predicate(path))


def run_capacity_role(
    *, config: Mapping[str, Any], run_id: str, test_root: Path, result_root: Path
) -> dict[str, Any]:
    host, job_id, _ = require_compute_context()
    rank, world = _rank_identity()
    expected_world = int(config["bandwidth"]["launcher_world_size"])
    if world != expected_world:
        raise StorageHarnessError(f"capacity benchmark requires {expected_world} ranks")
    target = Path(str(config["miyabi"]["target_shared_run_root"])).resolve()
    try:
        test_root.resolve().relative_to(target)
    except ValueError as error:
        raise StorageHarnessError("capacity path must be below the frozen RUN_ROOT") from error
    result_root.mkdir(parents=True, exist_ok=True)
    ready = test_root / "ready.json"
    setup: dict[str, Any] | None = None
    if rank == 0:
        setup = _prepare_capacity_tree(config, run_id, test_root)
        setup["fixed_records_before"] = _count_matching(
            test_root / "metadata", lambda path: path.name.endswith(".json")
        )
        setup["history_records_before"] = _count_matching(
            test_root / "history-negative", lambda path: path.name.endswith(".json")
        )
        atomic_write_json(ready, {"run_id": run_id, "phase": "ready"})
    _wait_json(ready, phase="ready", timeout=float(config["smoke"]["timeout_seconds"]) * 10)
    metadata_results = _run_metadata_rank(config, run_id, test_root, rank)
    history = _history_controls(config, test_root, run_id) if rank == 0 else []
    bandwidth_results, coordinator_rounds = _run_bandwidth_rank(
        config,
        run_id,
        test_root,
        rank,
        timeout=float(config["smoke"]["timeout_seconds"]) * 10,
    )
    state: dict[str, Any] | None = None
    if rank == 0:
        assert setup is not None
        state = {
            **setup,
            "fixed_records_after": _count_matching(
                test_root / "metadata", lambda path: path.name.endswith(".json")
            ),
            "history_records_after": _count_matching(
                test_root / "history-negative", lambda path: path.name.endswith(".json")
            ),
            "payload_files_after": _count_matching(test_root / "payloads", lambda _path: True),
            "temporary_files_after": _count_matching(
                test_root, lambda path: ".tmp-" in path.name
            ),
            "control_files_after": _count_matching(test_root / "control", lambda _path: True),
            "control_file_limit": 2 * world + 3,
        }
    role = {
        "schema_version": 1,
        "run_id": run_id,
        "rank": rank,
        "world_size": world,
        "hostname": host,
        "pbs_job_id": job_id,
        "clock": "time.monotonic_ns",
        "metadata": metadata_results,
        "history_controls": history,
        "bandwidth": bandwidth_results,
        "coordinator_rounds": coordinator_rounds,
        "state": state,
        "status": "passed",
    }
    atomic_write_json(result_root / f"capacity-rank-{rank}.json", role)
    return role


def _metadata_summary(roles: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for role in roles:
        for result in role["metadata"]:
            key = (
                result["learners"],
                result["fragments"],
                result["layout"],
                result["repeat"],
                result["profile"],
                result["state"],
            )
            groups[key].append(result)
    expected_keys = {
        (learners, fragments, layout, repeat, profile, state)
        for learners, fragments, layout, repeat, profile in metadata_matrix(config)
        for state in config["metadata"]["cache_states"]
    }
    if set(groups) != expected_keys:
        missing = sorted(expected_keys - set(groups))
        raise StorageHarnessError(f"metadata matrix is incomplete: {missing[:3]}")
    summaries: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: tuple(str(item) for item in value)):
        learners, fragments, layout, repeat, profile, state = key
        records = groups[key]
        expected_ranks = 1 if profile == "readdir" and layout == "flat" else int(learners)
        if len(records) != expected_ranks:
            raise StorageHarnessError("metadata concurrency does not match the frozen matrix")
        references = sum(int(record["reference_operations"]) for record in records)
        syscalls = sum(int(record["syscall_operations"]) for record in records)
        elapsed_ns = max(int(record["elapsed_ns"]) for record in records)
        latency = [
            int(value)
            for record in records
            for value in record["latency_samples_ns"]
        ]
        if not latency or references <= 0 or elapsed_ns <= 0:
            raise StorageHarnessError("metadata result contains no measurable operations")
        summaries.append(
            {
                "learners": learners,
                "fragments": fragments,
                "layout": layout,
                "repeat": repeat,
                "profile": profile,
                "state": state,
                "active_ranks": len(records),
                "reference_operations": references,
                "syscall_operations": syscalls,
                "elapsed_seconds_upper_bound": elapsed_ns / 1e9,
                "reference_ops_per_second": references / (elapsed_ns / 1e9),
                "syscalls_per_second": syscalls / (elapsed_ns / 1e9),
                "latency_sample_count": len(latency),
                "latency_operations_seen": sum(
                    int(record["latency_operations_seen"]) for record in records
                ),
                "p50_seconds": percentile(latency, 50) / 1e9,
                "p95_seconds": percentile(latency, 95) / 1e9,
                "p99_seconds": percentile(latency, 99) / 1e9,
                "max_seconds": max(latency) / 1e9,
            }
        )
    primary = [
        result
        for result in summaries
        if result["learners"] == 16
        and result["fragments"] == 32
        and result["state"] == "warm"
    ]
    if len(primary) != 12:
        raise StorageHarnessError("primary metadata threshold matrix is incomplete")
    measured = min(float(result["reference_ops_per_second"]) for result in primary)
    required = float(config["thresholds"]["metadata_required_ops_per_second"])
    return {
        "matrix": summaries,
        "threshold": {
            "learners": 16,
            "fragments": 32,
            "polling_interval_seconds": float(config["metadata"]["polling_interval_seconds"]),
            "steady_demand_ops_per_second": float(
                config["thresholds"]["metadata_steady_demand_ops_per_second"]
            ),
            "required_ops_per_second": required,
            "measured_minimum_warm_reference_ops_per_second": measured,
            "margin_over_steady_demand": measured
            / float(config["thresholds"]["metadata_steady_demand_ops_per_second"]),
            "passed": measured >= required,
        },
    }


def _history_summary(roles: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    controls = next(role["history_controls"] for role in roles if role["rank"] == 0)
    if [item["history_objects"] for item in controls] != config["metadata"][
        "history_sizes_for_negative_control"
    ]:
        raise StorageHarnessError("history negative-control matrix is incomplete")
    summarized = []
    for control in controls:
        if control["history_scan_counts"] != [control["history_objects"]] * int(
            config["metadata"]["history_scan_repeats"]
        ):
            raise StorageHarnessError("history scan did not observe the expected object count")
        if control["bounded_fixed_references"] != 16 * 32:
            raise StorageHarnessError("bounded discovery did not remain fixed at M by F")
        summarized.append(
            {
                **control,
                "history_scan_p50_seconds": percentile(
                    control["history_scan_latencies_ns"], 50
                )
                / 1e9,
                "bounded_seconds": control["bounded_elapsed_ns"] / 1e9,
            }
        )
    medians = [item["history_scan_p50_seconds"] for item in summarized]
    if not (medians[2] > medians[1] > medians[0]):
        raise StorageHarnessError("history-scanning negative cost did not grow with history")
    return {
        "bounded_discovery_surface": "exactly_M_times_F_fixed_slots",
        "normal_discovery_scans_history": False,
        "negative_history_scanner_cost_grows": True,
        "profiles": summarized,
    }


def _bandwidth_summary(roles: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    operations = [operation for role in roles for operation in role["bandwidth"]]
    rounds = {
        (item["phase"], item["operation"]): item
        for role in roles
        for item in role["coordinator_rounds"]
    }
    summaries: list[dict[str, Any]] = []
    for size_mb, streams, repeat in bandwidth_matrix(config):
        phase = _phase_id(size_mb, streams, repeat)
        writes = [item for item in operations if item["phase"] == phase and item["operation"] == "write"]
        reads = [item for item in operations if item["phase"] == phase and item["operation"] == "read"]
        if len(writes) != streams or len(reads) != streams:
            raise StorageHarnessError("bandwidth matrix has the wrong stream count")
        writer_digests = {int(item["stream"]): item["sha256"] for item in writes}
        for read in reads:
            source = int(read["source_writer_rank"])
            if read["sha256"] != writer_digests.get(source):
                raise StorageHarnessError("cross-host sequential read checksum mismatch")
        for operation_name, records in (("write", writes), ("read", reads)):
            round_record = rounds.get((phase, operation_name))
            if round_record is None:
                raise StorageHarnessError("bandwidth coordinator round is missing")
            round_seconds = int(round_record["round_elapsed_ns"]) / 1e9
            total_bytes = sum(int(item["bytes"]) for item in records)
            per_stream = [
                int(item["bytes"]) / (1024 * 1024) / (int(item["elapsed_ns"]) / 1e9)
                for item in records
            ]
            summaries.append(
                {
                    "size_mb": size_mb,
                    "streams": streams,
                    "repeat": repeat,
                    "operation": operation_name,
                    "round_seconds": round_seconds,
                    "total_bytes": total_bytes,
                    "aggregate_mib_per_second": total_bytes / (1024 * 1024) / round_seconds,
                    "per_stream_mib_per_second": per_stream,
                    "minimum_stream_mib_per_second": min(per_stream),
                    "maximum_stream_mib_per_second": max(per_stream),
                    "checksums_verified": True,
                }
            )
    expected = len(config["bandwidth"]["payload_sizes_mb"]) * len(
        config["bandwidth"]["streams"]
    ) * int(config["bandwidth"]["repeats"]) * 2
    if len(summaries) != expected:
        raise StorageHarnessError("bandwidth summary matrix is incomplete")
    concurrent_writes = [
        item for item in summaries if item["streams"] == 16 and item["operation"] == "write"
    ]
    maximum_round = max(float(item["round_seconds"]) for item in concurrent_writes)
    threshold = float(config["thresholds"]["concurrent_fragment_round_seconds"])
    return {
        "matrix": summaries,
        "threshold": {
            "streams": 16,
            "synchronization_period_seconds": float(
                config["thresholds"]["planned_sync_period_seconds"]
            ),
            "maximum_round_seconds": maximum_round,
            "required_round_seconds": threshold,
            "passed": maximum_round <= threshold,
        },
    }


def summarize_capacity(
    *, config: Mapping[str, Any], run_id: str, result_root: Path
) -> dict[str, Any]:
    world = int(config["bandwidth"]["launcher_world_size"])
    roles = [read_json_record(result_root / f"capacity-rank-{rank}.json") for rank in range(world)]
    if {role.get("rank") for role in roles} != set(range(world)):
        raise StorageHarnessError("capacity role set is incomplete")
    if any(role.get("run_id") != run_id or role.get("status") != "passed" for role in roles):
        raise StorageHarnessError("capacity role identity or status is invalid")
    hosts: dict[str, int] = defaultdict(int)
    for role in roles:
        hosts[str(role["hostname"])] += 1
    expected_per_host = int(config["bandwidth"]["ranks_per_node"])
    if len(hosts) != 2 or sorted(hosts.values()) != [expected_per_host, expected_per_host]:
        raise StorageHarnessError("capacity benchmark did not use eight ranks on each of two hosts")
    first_host = {str(role["hostname"]) for role in roles if int(role["rank"]) < expected_per_host}
    second_host = {str(role["hostname"]) for role in roles if int(role["rank"]) >= expected_per_host}
    if len(first_host) != 1 or len(second_host) != 1 or first_host == second_host:
        raise StorageHarnessError("rank offset does not identify the opposite compute host")
    metadata = _metadata_summary(roles, config)
    history = _history_summary(roles, config)
    bandwidth = _bandwidth_summary(roles, config)
    state = next(role["state"] for role in roles if role["rank"] == 0)
    if state["fixed_records_before"] != state["fixed_records_after"]:
        raise StorageHarnessError("metadata benchmark grew the fixed discovery surface")
    if state["history_records_before"] != state["history_records_after"]:
        raise StorageHarnessError("capacity benchmark grew historical state")
    if state["payload_files_after"] != 0 or state["temporary_files_after"] != 0:
        raise StorageHarnessError("capacity benchmark left payload or temporary objects")
    if state["control_files_after"] > state["control_file_limit"]:
        raise StorageHarnessError("capacity coordination surface is unbounded")
    capacity_passed = bool(metadata["threshold"]["passed"]) and bool(
        bandwidth["threshold"]["passed"]
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "outcome": (
            "target_lustre_capacity_pass"
            if capacity_passed
            else "target_lustre_capacity_mitigation_required"
        ),
        "roles": [
            {
                "rank": role["rank"],
                "hostname": role["hostname"],
                "pbs_job_id": role["pbs_job_id"],
                "status": role["status"],
            }
            for role in roles
        ],
        "topology": {
            "world_size": world,
            "ranks_per_node": expected_per_host,
            "hosts": dict(sorted(hosts.items())),
            "application_coordination": "filesystem_only_fixed_control_slots",
            "mpi_usage": "launcher_only",
        },
        "metadata": metadata,
        "history_negative_control": history,
        "bandwidth": bandwidth,
        "bounded_state": state,
        "decision": {
            "filesystem": "frozen_target_lustre",
            "stage1_profile": (
                "H=50s_fixed_slot_discovery_no_history_scan"
                if capacity_passed
                else "blocked_pending_storage_path_or_H_or_barrier_mitigation"
            ),
            "A-BENCH-03": "PASS" if metadata["threshold"]["passed"] else "FAIL",
            "A-BENCH-04": "PASS" if bandwidth["threshold"]["passed"] else "FAIL",
            "mitigation_required": not capacity_passed,
        },
        "claims_boundary": "Stage 0 metadata and sequential-I/O capacity only; no training, model, loss, GPU, or nine-node claim.",
    }
    atomic_write_json(result_root / "capacity-summary.json", summary)
    return summary


def finalize_manifest(*, manifest_path: Path, summary_path: Path, output: Path) -> dict[str, Any]:
    manifest = read_json_record(manifest_path)
    summary = read_json_record(summary_path)
    manifest.update(
        {
            "capacity_outcome": summary["outcome"],
            "launcher_world_size": summary["topology"]["world_size"],
            "ranks_per_node": summary["topology"]["ranks_per_node"],
            "actual_role_mapping": summary["roles"],
            "capacity_decision": summary["decision"],
        }
    )
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(output, manifest)
    return manifest


def cleanup_capacity(*, config: Mapping[str, Any], test_root: Path) -> dict[str, Any]:
    target = Path(str(config["miyabi"]["target_shared_run_root"])).resolve()
    resolved = test_root.resolve()
    try:
        relative = resolved.relative_to(target)
    except ValueError as error:
        raise StorageHarnessError("cleanup path must be below the frozen RUN_ROOT") from error
    if len(relative.parts) < 2 or relative.parts[0] != "stage0-storage-capacity":
        raise StorageHarnessError("cleanup path is not a run-scoped capacity directory")
    if not resolved.exists():
        raise StorageHarnessError("capacity cleanup path does not exist")
    shutil.rmtree(resolved)
    return {"test_root": str(resolved), "removed": not resolved.exists()}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    role = subparsers.add_parser("role")
    role.add_argument("--config", type=Path, required=True)
    role.add_argument("--run-id", required=True)
    role.add_argument("--test-root", type=Path, required=True)
    role.add_argument("--result-root", type=Path, required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--config", type=Path, required=True)
    summary.add_argument("--run-id", required=True)
    summary.add_argument("--result-root", type=Path, required=True)
    finalize = subparsers.add_parser("finalize-manifest")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--summary", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--config", type=Path, required=True)
    cleanup.add_argument("--test-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "role":
        result = run_capacity_role(
            config=load_storage_config(args.config),
            run_id=args.run_id,
            test_root=args.test_root,
            result_root=args.result_root,
        )
    elif args.command == "summarize":
        result = summarize_capacity(
            config=load_storage_config(args.config),
            run_id=args.run_id,
            result_root=args.result_root,
        )
    elif args.command == "finalize-manifest":
        result = finalize_manifest(
            manifest_path=args.manifest,
            summary_path=args.summary,
            output=args.output,
        )
    else:
        result = cleanup_capacity(
            config=load_storage_config(args.config), test_root=args.test_root
        )
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
