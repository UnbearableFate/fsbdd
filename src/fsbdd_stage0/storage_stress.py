"""Cross-node visibility latency and atomic-replacement stress probes.

MPI is only the process launcher.  The two roles coordinate exclusively through
the frozen Lustre path so the benchmark exercises the same data plane that the
runtime protocol will use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import shutil
import socket
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fsbdd_stage0.storage_bench import (
    StorageHarnessError,
    _canonical_json,
    _fsync_directory,
    _role_identity,
    atomic_write_bytes,
    atomic_write_json,
    file_sha256,
    load_storage_config,
    read_json_record,
    require_compute_context,
    validate_environment_manifest,
)


TIMING_METHOD = "writer_ack_roundtrip_upper_bound"
TIMING_ORIGIN = "visibility_record_replace_complete"


def percentile(values: Sequence[int | float], percent: int | float) -> float:
    """Return a linearly interpolated percentile over a non-empty sample."""

    if not values:
        raise StorageHarnessError("cannot calculate a percentile of an empty sample")
    if isinstance(percent, bool) or not isinstance(percent, (int, float)):
        raise StorageHarnessError("percentile must be numeric")
    if percent < 0 or percent > 100:
        raise StorageHarnessError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percent) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_schedule(config: Mapping[str, Any]) -> list[tuple[int, str, str]]:
    visibility = config["visibility"]
    per_cache = int(visibility["samples_per_cache_state"])
    schedule: list[tuple[int, str, str]] = []
    sequence = 0
    for profile in visibility["profiles"]:
        for cache_state in visibility["cache_states"]:
            for _ in range(per_cache):
                schedule.append((sequence, str(profile), str(cache_state)))
                sequence += 1
    return schedule


def _visibility_path(root: Path, sequence: int, cache_state: str) -> Path:
    if cache_state == "cold":
        return root / "visibility" / f"cold-{sequence:08d}.json"
    return root / "visibility" / "warm.json"


def _ack_path(root: Path, sequence: int, cache_state: str) -> Path:
    if cache_state == "cold":
        return root / "acknowledgements" / f"cold-{sequence:08d}.json"
    return root / "acknowledgements" / "warm.json"


def _wait_for_expected_record(
    path: Path,
    *,
    run_id: str,
    sequence: int,
    timeout_seconds: float,
    poll_interval_seconds: float,
    required_fields: Sequence[str],
) -> tuple[dict[str, Any], int, int]:
    deadline = time.monotonic() + timeout_seconds
    polls = 0
    invalid_reads = 0
    last_error = "record not yet visible"
    while time.monotonic() <= deadline:
        polls += 1
        if path.is_file():
            try:
                record = read_json_record(path, required_fields=required_fields)
            except StorageHarnessError as error:
                invalid_reads += 1
                last_error = str(error)
            else:
                if record.get("run_id") == run_id and record.get("sequence") == sequence:
                    return record, polls, invalid_reads
                last_error = "record belongs to an earlier run or sequence"
        time.sleep(poll_interval_seconds)
    raise StorageHarnessError(
        f"timed out waiting for run={run_id} sequence={sequence} at {path}: {last_error}"
    )


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def replace_visibility_bytes(path: Path, content: bytes) -> None:
    """Close a complete same-directory temporary, then atomically replace.

    BENCH-02 measures the POSIX rename visibility primitive, not persistence
    after power loss.  Per-replacement file and directory fsync would measure
    durability barriers instead and would dominate the 200000-operation
    matrix.  Durable payload/publication and kill probes continue to use
    ``atomic_write_bytes``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.visibility-{socket.gethostname()}-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise StorageHarnessError(f"cannot read JSONL evidence {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise StorageHarnessError(
                f"invalid JSONL evidence {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise StorageHarnessError(f"JSONL evidence {path}:{line_number} is not an object")
        records.append(value)
    return records


def _background_record(run_id: str, sequence: int, size: int) -> bytes:
    prefix = _canonical_json(
        {
            "run_id": run_id,
            "sequence": sequence,
            "sha256": hashlib.sha256(f"{run_id}:background:{sequence}".encode()).hexdigest(),
        }
    ).encode("utf-8")
    if len(prefix) > size:
        raise StorageHarnessError("background record size is too small")
    return prefix + b" " * (size - len(prefix))


def _validate_background_record(content: bytes, run_id: str, size: int) -> int:
    if len(content) != size:
        raise StorageHarnessError(f"background record length {len(content)} != {size}")
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StorageHarnessError(f"invalid background record: {error}") from error
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise StorageHarnessError("background record identity mismatch")
    sequence = value.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise StorageHarnessError("background sequence is not an integer")
    expected = hashlib.sha256(f"{run_id}:background:{sequence}".encode()).hexdigest()
    if value.get("sha256") != expected:
        raise StorageHarnessError("background record checksum mismatch")
    return sequence


def _run_latency_writer(
    *, config: Mapping[str, Any], run_id: str, test_root: Path, result_root: Path
) -> dict[str, Any]:
    visibility = config["visibility"]
    timeout = float(config["smoke"]["timeout_seconds"])
    poll = float(visibility["poll_interval_seconds"])
    background_path = test_root / "background" / "load.record"
    background_done = test_root / "background" / "done.json"
    stop_background = threading.Event()
    background_state: dict[str, Any] = {"replacements": 0, "error": None}

    def background_writer() -> None:
        minimum = int(visibility["background_minimum_replacements"])
        size = int(visibility["background_record_bytes"])
        sequence = 0
        try:
            while not stop_background.is_set() or sequence < minimum:
                replace_visibility_bytes(
                    background_path, _background_record(run_id, sequence, size)
                )
                sequence += 1
                background_state["replacements"] = sequence
        except (OSError, StorageHarnessError) as error:
            background_state["error"] = str(error)

    background_thread: threading.Thread | None = None
    loaded_profile_start_replacements = 0
    loaded_profile_end_replacements = 0
    samples: list[dict[str, Any]] = []
    for sequence, profile, cache_state in _latency_schedule(config):
        if profile == "loaded" and background_thread is None:
            background_thread = threading.Thread(target=background_writer, daemon=True)
            background_thread.start()
            minimum = int(visibility["background_minimum_replacements"])
            warmup_deadline = time.monotonic() + timeout * 75
            while int(background_state["replacements"]) < minimum:
                if background_state["error"]:
                    raise StorageHarnessError(
                        f"background writer failed during warmup: {background_state['error']}"
                    )
                if time.monotonic() > warmup_deadline:
                    raise StorageHarnessError(
                        "background writer did not reach its frozen warmup replacement count"
                    )
                time.sleep(poll)
            loaded_profile_start_replacements = int(background_state["replacements"])
        payload = hashlib.sha256(f"{run_id}:payload:{sequence}".encode()).digest() * 4
        payload_path = test_root / "payloads" / f"payload-{sequence:08d}.bin"
        payload_digest = hashlib.sha256(payload).hexdigest()
        atomic_write_bytes(payload_path, payload)
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "sequence": sequence,
            "profile": profile,
            "cache_state": cache_state,
            "writer_host": socket.gethostname().split(".", 1)[0],
            "payload_path": str(payload_path),
            "payload_bytes": len(payload),
            "payload_sha256": payload_digest,
            "timing_method": TIMING_METHOD,
            "timing_origin": TIMING_ORIGIN,
        }
        visible_path = _visibility_path(test_root, sequence, cache_state)
        atomic_write_json(visible_path, record)
        publish_complete_ns = time.monotonic_ns()
        acknowledgement, writer_polls, invalid_ack_reads = _wait_for_expected_record(
            _ack_path(test_root, sequence, cache_state),
            run_id=run_id,
            sequence=sequence,
            timeout_seconds=timeout,
            poll_interval_seconds=poll,
            required_fields=(
                "run_id",
                "sequence",
                "reader_host",
                "payload_sha256",
                "reader_observed_monotonic_ns",
                "reader_poll_count",
                "reader_invalid_record_reads",
            ),
        )
        acknowledgement_ns = time.monotonic_ns()
        if acknowledgement.get("payload_sha256") != payload_digest:
            raise StorageHarnessError("latency acknowledgement payload digest mismatch")
        if acknowledgement.get("reader_host") == record["writer_host"]:
            raise StorageHarnessError("latency writer and reader were placed on the same host")
        if invalid_ack_reads or acknowledgement["reader_invalid_record_reads"]:
            raise StorageHarnessError("an invalid record was observed during the latency probe")
        samples.append(
            {
                **record,
                "visibility_publish_complete_monotonic_ns": publish_complete_ns,
                "reader_observed_monotonic_ns": acknowledgement[
                    "reader_observed_monotonic_ns"
                ],
                "writer_acknowledgement_monotonic_ns": acknowledgement_ns,
                "writer_ack_roundtrip_ns": acknowledgement_ns - publish_complete_ns,
                "writer_poll_count": writer_polls,
                "reader_poll_count": acknowledgement["reader_poll_count"],
                "invalid_record_reads": 0,
            }
        )

    if background_thread is None:
        raise StorageHarnessError("loaded latency profile did not start its background load")
    loaded_profile_end_replacements = int(background_state["replacements"])
    stop_background.set()
    background_thread.join(timeout=timeout)
    if background_thread.is_alive():
        raise StorageHarnessError("background writer did not terminate")
    if background_state["error"]:
        raise StorageHarnessError(f"background writer failed: {background_state['error']}")
    if background_state["replacements"] < int(visibility["background_minimum_replacements"]):
        raise StorageHarnessError("background writer completed too few replacements")
    atomic_write_json(
        background_done,
        {
            "schema_version": 1,
            "run_id": run_id,
            "replacements": background_state["replacements"],
        },
    )
    _write_jsonl(result_root / "latency_samples.jsonl", samples)
    return {
        "sample_count": len(samples),
        "profile_counts": dict(Counter(sample["profile"] for sample in samples)),
        "cache_state_counts": dict(Counter(sample["cache_state"] for sample in samples)),
        "background_replacements": background_state["replacements"],
        "loaded_profile_start_replacements": loaded_profile_start_replacements,
        "loaded_profile_end_replacements": loaded_profile_end_replacements,
        "timing_method": TIMING_METHOD,
        "timing_origin": TIMING_ORIGIN,
    }


def _run_latency_reader(
    *, config: Mapping[str, Any], run_id: str, test_root: Path
) -> dict[str, Any]:
    visibility = config["visibility"]
    timeout = float(config["smoke"]["timeout_seconds"])
    poll = float(visibility["poll_interval_seconds"])
    background_path = test_root / "background" / "load.record"
    background_done = test_root / "background" / "done.json"
    background_state: dict[str, Any] = {
        "observations": 0,
        "violations": 0,
        "last_sequence": -1,
        "error": None,
    }

    def background_reader() -> None:
        size = int(visibility["background_record_bytes"])
        try:
            while not background_done.is_file():
                if background_path.is_file():
                    try:
                        content = background_path.read_bytes()
                        sequence = _validate_background_record(content, run_id, size)
                    except (OSError, StorageHarnessError):
                        background_state["violations"] += 1
                    else:
                        background_state["observations"] += 1
                        background_state["last_sequence"] = sequence
                time.sleep(poll)
        except OSError as error:
            background_state["error"] = str(error)

    background_thread = threading.Thread(target=background_reader, daemon=True)
    background_thread.start()
    invalid_total = 0
    reader_host = socket.gethostname().split(".", 1)[0]
    for sequence, profile, cache_state in _latency_schedule(config):
        record, reader_polls, invalid_reads = _wait_for_expected_record(
            _visibility_path(test_root, sequence, cache_state),
            run_id=run_id,
            sequence=sequence,
            timeout_seconds=timeout,
            poll_interval_seconds=poll,
            required_fields=(
                "run_id",
                "sequence",
                "profile",
                "cache_state",
                "writer_host",
                "payload_path",
                "payload_bytes",
                "payload_sha256",
                "timing_method",
                "timing_origin",
            ),
        )
        invalid_total += invalid_reads
        if record["profile"] != profile or record["cache_state"] != cache_state:
            raise StorageHarnessError("latency publication profile identity mismatch")
        if record["timing_method"] != TIMING_METHOD or record["timing_origin"] != TIMING_ORIGIN:
            raise StorageHarnessError("latency publication has an invalid timing schema")
        if record["writer_host"] == reader_host:
            raise StorageHarnessError("latency writer and reader were placed on the same host")
        payload_path = Path(str(record["payload_path"]))
        observed_size = payload_path.stat().st_size
        observed_digest = file_sha256(payload_path)
        if observed_size != record["payload_bytes"] or observed_digest != record["payload_sha256"]:
            raise StorageHarnessError("latency payload is incomplete or has the wrong checksum")
        observed_ns = time.monotonic_ns()
        atomic_write_json(
            _ack_path(test_root, sequence, cache_state),
            {
                "schema_version": 1,
                "run_id": run_id,
                "sequence": sequence,
                "reader_host": reader_host,
                "payload_sha256": observed_digest,
                "reader_observed_monotonic_ns": observed_ns,
                "reader_poll_count": reader_polls,
                "reader_invalid_record_reads": invalid_reads,
            },
        )
    background_thread.join(timeout=timeout * 2)
    if background_thread.is_alive():
        raise StorageHarnessError("background reader did not observe its completion record")
    if background_state["error"]:
        raise StorageHarnessError(f"background reader failed: {background_state['error']}")
    if background_state["violations"]:
        raise StorageHarnessError("background load reader observed an incomplete replacement")
    if background_state["observations"] <= 0:
        raise StorageHarnessError("background reader did not observe the loaded profile")
    return {
        "sample_count": len(_latency_schedule(config)),
        "invalid_record_reads": invalid_total,
        "background_observations": background_state["observations"],
        "background_violations": background_state["violations"],
        "background_last_sequence": background_state["last_sequence"],
    }


def _record_checksum(run_id: str, record_size: int, sequence: int, content: str) -> str:
    identity = f"{run_id}\0{record_size}\0{sequence}\0{content}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def build_atomic_record(run_id: str, record_size: int, sequence: int) -> bytes:
    if isinstance(record_size, bool) or not isinstance(record_size, int) or record_size <= 0:
        raise StorageHarnessError("record size must be a positive integer")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise StorageHarnessError("record sequence must be an integer")
    content = hashlib.sha256(f"{run_id}:atomic:{record_size}:{sequence}".encode()).hexdigest()[:16]
    record = {
        "complete": True,
        "content": content,
        "record_size": record_size,
        "run_id": run_id,
        "schema_version": 1,
        "sequence": sequence,
        "sha256": _record_checksum(run_id, record_size, sequence, content),
    }
    encoded = _canonical_json(record).encode("utf-8")
    if len(encoded) > record_size:
        raise StorageHarnessError(f"record metadata does not fit in {record_size} bytes")
    return encoded + b" " * (record_size - len(encoded))


def validate_atomic_record(content: bytes, *, run_id: str, record_size: int) -> int:
    if len(content) != record_size:
        raise StorageHarnessError(f"record length {len(content)} != {record_size}")
    try:
        record = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StorageHarnessError(f"record is not complete JSON: {error}") from error
    if not isinstance(record, dict):
        raise StorageHarnessError("atomic record is not an object")
    required = {
        "complete",
        "content",
        "record_size",
        "run_id",
        "schema_version",
        "sequence",
        "sha256",
    }
    if set(record) != required:
        raise StorageHarnessError("atomic record schema is incomplete or contains extra fields")
    if record["complete"] is not True or record["schema_version"] != 1:
        raise StorageHarnessError("atomic record completion marker is invalid")
    if record["run_id"] != run_id or record["record_size"] != record_size:
        raise StorageHarnessError("atomic record identity mismatch")
    sequence = record["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise StorageHarnessError("atomic record sequence is not an integer")
    expected = _record_checksum(run_id, record_size, sequence, str(record["content"]))
    if record["sha256"] != expected:
        raise StorageHarnessError("atomic record checksum mismatch")
    return sequence


class _ViolationLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.count = 0

    def write(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self._handle.write(_canonical_json(record) + "\n")
            self.count += 1

    def close(self) -> None:
        with self._lock:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()


def _run_atomic_writer(
    *, config: Mapping[str, Any], run_id: str, test_root: Path, result_root: Path
) -> dict[str, Any]:
    atomicity = config["atomicity"]
    timeout = float(config["smoke"]["timeout_seconds"])
    count = int(atomicity["replacements_per_record_size"])
    sizes: list[dict[str, Any]] = []
    for record_size in atomicity["record_sizes_bytes"]:
        record_size = int(record_size)
        record_path = test_root / "atomic" / f"record-{record_size}.json"
        replace_visibility_bytes(record_path, build_atomic_record(run_id, record_size, -1))
        _wait_for_expected_record(
            test_root / "atomic" / f"reader-ready-{record_size}.json",
            run_id=run_id,
            sequence=record_size,
            timeout_seconds=timeout,
            poll_interval_seconds=0.001,
            required_fields=("run_id", "sequence"),
        )
        started_ns = time.monotonic_ns()
        local_transcript = Path("/tmp") / (
            f"fsbdd-{os.getpid()}-{hashlib.sha256(run_id.encode()).hexdigest()[:12]}-"
            f"atomic-writer-{record_size}.jsonl"
        )
        transcript_digest = hashlib.sha256()
        first_publication_ns: int | None = None
        last_publication_ns: int | None = None
        try:
            with local_transcript.open("x", encoding="utf-8") as transcript:
                for sequence in range(count):
                    content = build_atomic_record(run_id, record_size, sequence)
                    content_sha256 = hashlib.sha256(content).hexdigest()
                    replace_visibility_bytes(record_path, content)
                    publication_ns = time.monotonic_ns()
                    if first_publication_ns is None:
                        first_publication_ns = publication_ns
                    last_publication_ns = publication_ns
                    entry = {
                        "schema_version": 1,
                        "run_id": run_id,
                        "record_size": record_size,
                        "sequence": sequence,
                        "record_sha256": content_sha256,
                        "publication_complete_monotonic_ns": publication_ns,
                        "timing_origin": TIMING_ORIGIN,
                    }
                    line = _canonical_json(entry) + "\n"
                    transcript.write(line)
                    transcript_digest.update(line.encode("utf-8"))
                transcript.flush()
                os.fsync(transcript.fileno())
            transcript_name = f"atomic_publications_{record_size}.jsonl"
            shutil.copyfile(local_transcript, result_root / transcript_name)
            _fsync_directory(result_root)
        finally:
            local_transcript.unlink(missing_ok=True)
        completed_ns = time.monotonic_ns()
        _fsync_directory(record_path.parent)
        atomic_write_json(
            test_root / "atomic" / f"writer-done-{record_size}.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "sequence": count - 1,
                "record_size": record_size,
                "replacements": count,
            },
        )
        _wait_for_expected_record(
            test_root / "atomic" / f"reader-done-{record_size}.json",
            run_id=run_id,
            sequence=count - 1,
            timeout_seconds=timeout * 10,
            poll_interval_seconds=0.001,
            required_fields=("run_id", "sequence"),
        )
        sizes.append(
            {
                "record_size": record_size,
                "replacements": count,
                "final_sequence": count - 1,
                "elapsed_ns": completed_ns - started_ns,
                "first_publication_complete_monotonic_ns": first_publication_ns,
                "last_publication_complete_monotonic_ns": last_publication_ns,
                "publication_transcript": transcript_name,
                "publication_transcript_records": count,
                "publication_transcript_sha256": transcript_digest.hexdigest(),
            }
        )
    return {
        "run_id": run_id,
        "record_sizes": sizes,
        "total_replacements": count * len(sizes),
        "primitive": "complete_same_directory_temporary_then_os_replace",
        "claim_boundary": "atomic_visibility_not_crash_durability",
    }


def _run_atomic_reader(
    *, config: Mapping[str, Any], run_id: str, test_root: Path, result_root: Path
) -> dict[str, Any]:
    atomicity = config["atomicity"]
    timeout = float(config["smoke"]["timeout_seconds"])
    poll = float(config["visibility"]["poll_interval_seconds"])
    count = int(atomicity["replacements_per_record_size"])
    reader_count = int(atomicity["concurrent_readers"])
    minimum = int(atomicity["minimum_observations_per_reader"])
    checkpoint_every = int(atomicity["checkpoint_every_observations"])
    violation_log = _ViolationLog(result_root / "atomic_violations.jsonl")
    size_results: list[dict[str, Any]] = []
    try:
        for record_size_value in atomicity["record_sizes_bytes"]:
            record_size = int(record_size_value)
            record_path = test_root / "atomic" / f"record-{record_size}.json"
            states = [
                {
                    "reader_id": reader_id,
                    "polls": 0,
                    "observations": 0,
                    "last_sequence": -2,
                    "first_observed_monotonic_ns": None,
                    "last_observed_monotonic_ns": None,
                    "sequence_regressions": 0,
                    "final_seen": False,
                    "transcript_sha256": "",
                    "violations": 0,
                    "error": None,
                }
                for reader_id in range(reader_count)
            ]

            def scan(state: dict[str, Any]) -> None:
                digest = hashlib.sha256()
                checkpoint_path = result_root / (
                    f"atomic_checkpoints_{record_size}_reader_{state['reader_id']}.jsonl"
                )
                try:
                    with checkpoint_path.open("x", encoding="utf-8") as checkpoint:
                        deadline = time.monotonic() + timeout * 75
                        while time.monotonic() <= deadline:
                            state["polls"] += 1
                            try:
                                content = record_path.read_bytes()
                                sequence = validate_atomic_record(
                                    content, run_id=run_id, record_size=record_size
                                )
                            except (OSError, StorageHarnessError) as error:
                                state["violations"] += 1
                                violation_log.write(
                                    {
                                        "schema_version": 1,
                                        "mode": "atomic",
                                        "run_id": run_id,
                                        "record_size": record_size,
                                        "reader_id": state["reader_id"],
                                        "observation": state["observations"],
                                        "observed_monotonic_ns": time.monotonic_ns(),
                                        "error": str(error),
                                    }
                                )
                            else:
                                observed_ns = time.monotonic_ns()
                                state["observations"] += 1
                                if state["first_observed_monotonic_ns"] is None:
                                    state["first_observed_monotonic_ns"] = observed_ns
                                state["last_observed_monotonic_ns"] = observed_ns
                                if sequence < state["last_sequence"]:
                                    state["sequence_regressions"] += 1
                                state["last_sequence"] = sequence
                                state["final_seen"] = state["final_seen"] or sequence == count - 1
                                digest.update(f"{sequence}\n".encode())
                                if state["observations"] % checkpoint_every == 0:
                                    checkpoint.write(
                                        _canonical_json(
                                            {
                                                "reader_id": state["reader_id"],
                                                "record_size": record_size,
                                                "observations": state["observations"],
                                                "polls": state["polls"],
                                                "last_sequence": sequence,
                                                "observed_monotonic_ns": observed_ns,
                                                "violations": state["violations"],
                                                "transcript_sha256": digest.hexdigest(),
                                            }
                                        )
                                        + "\n"
                                    )
                                    checkpoint.flush()
                            done = (test_root / "atomic" / f"writer-done-{record_size}.json").is_file()
                            if done and state["final_seen"] and state["observations"] >= minimum:
                                break
                            time.sleep(poll)
                        else:
                            raise StorageHarnessError(
                                f"atomic reader {state['reader_id']} timed out for {record_size} bytes"
                            )
                        checkpoint.write(
                            _canonical_json(
                                {
                                    "reader_id": state["reader_id"],
                                    "record_size": record_size,
                                    "observations": state["observations"],
                                    "polls": state["polls"],
                                    "last_sequence": state["last_sequence"],
                                    "observed_monotonic_ns": state[
                                        "last_observed_monotonic_ns"
                                    ],
                                    "violations": state["violations"],
                                    "transcript_sha256": digest.hexdigest(),
                                    "final": True,
                                }
                            )
                            + "\n"
                        )
                        checkpoint.flush()
                        os.fsync(checkpoint.fileno())
                except (OSError, StorageHarnessError) as error:
                    state["error"] = str(error)
                finally:
                    state["transcript_sha256"] = digest.hexdigest()

            threads = [threading.Thread(target=scan, args=(state,)) for state in states]
            for thread in threads:
                thread.start()
            atomic_write_json(
                test_root / "atomic" / f"reader-ready-{record_size}.json",
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "sequence": record_size,
                    "reader_count": reader_count,
                },
            )
            for thread in threads:
                thread.join(timeout=timeout * 76)
            if any(thread.is_alive() for thread in threads):
                raise StorageHarnessError("an atomic reader thread did not terminate")
            if any(state["error"] for state in states):
                raise StorageHarnessError(
                    "atomic reader failed: "
                    + "; ".join(str(state["error"]) for state in states if state["error"])
                )
            atomic_write_json(
                test_root / "atomic" / f"reader-done-{record_size}.json",
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "sequence": count - 1,
                    "record_size": record_size,
                },
            )
            size_results.append(
                {
                    "record_size": record_size,
                    "readers": states,
                    "violations": sum(int(state["violations"]) for state in states),
                }
            )
    finally:
        violation_log.close()
    return {"record_sizes": size_results, "violations": violation_log.count}


def _run_kill_probe_writer(*, run_id: str, test_root: Path, timeout: float) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for position in ("before_visibility", "after_visibility"):
        root = test_root / "kill-probes" / position
        payload_path = root / "payload.bin"
        visibility_path = root / "visibility.json"
        child_ready = root / "child-ready.json"
        killed = root / "writer-killed.json"
        payload = hashlib.sha256(f"{run_id}:kill:{position}".encode()).digest() * 4
        digest = hashlib.sha256(payload).hexdigest()
        child_pid = os.fork()
        if child_pid == 0:
            try:
                atomic_write_bytes(payload_path, payload)
                if position == "after_visibility":
                    atomic_write_json(
                        visibility_path,
                        {
                            "schema_version": 1,
                            "run_id": run_id,
                            "sequence": 1,
                            "payload_bytes": len(payload),
                            "payload_sha256": digest,
                        },
                    )
                atomic_write_json(
                    child_ready,
                    {"schema_version": 1, "run_id": run_id, "sequence": 1},
                )
                while True:
                    time.sleep(1)
            finally:
                os._exit(91)
        _wait_for_expected_record(
            child_ready,
            run_id=run_id,
            sequence=1,
            timeout_seconds=timeout,
            poll_interval_seconds=0.001,
            required_fields=("run_id", "sequence"),
        )
        os.kill(child_pid, signal.SIGKILL)
        _, wait_status = os.waitpid(child_pid, 0)
        if not os.WIFSIGNALED(wait_status) or os.WTERMSIG(wait_status) != signal.SIGKILL:
            raise StorageHarnessError("writer-kill child was not terminated by SIGKILL")
        atomic_write_json(
            killed,
            {
                "schema_version": 1,
                "run_id": run_id,
                "sequence": 1,
                "position": position,
                "signal": "SIGKILL",
            },
        )
        _wait_for_expected_record(
            root / "reader-verified.json",
            run_id=run_id,
            sequence=1,
            timeout_seconds=timeout,
            poll_interval_seconds=0.001,
            required_fields=("run_id", "sequence"),
        )
        results.append({"position": position, "signal": "SIGKILL", "passed": True})
    return {"probes": results}


def _run_kill_probe_reader(*, run_id: str, test_root: Path, timeout: float) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for position in ("before_visibility", "after_visibility"):
        root = test_root / "kill-probes" / position
        _wait_for_expected_record(
            root / "writer-killed.json",
            run_id=run_id,
            sequence=1,
            timeout_seconds=timeout,
            poll_interval_seconds=0.001,
            required_fields=("run_id", "sequence", "position", "signal"),
        )
        visibility_path = root / "visibility.json"
        if position == "before_visibility":
            if visibility_path.exists():
                raise StorageHarnessError("killed pre-publication writer exposed visibility")
        else:
            record = read_json_record(
                visibility_path,
                required_fields=("run_id", "sequence", "payload_bytes", "payload_sha256"),
            )
            payload_path = root / "payload.bin"
            if record["run_id"] != run_id:
                raise StorageHarnessError("post-publication kill record run mismatch")
            if payload_path.stat().st_size != record["payload_bytes"]:
                raise StorageHarnessError("post-publication kill payload length mismatch")
            if file_sha256(payload_path) != record["payload_sha256"]:
                raise StorageHarnessError("post-publication kill payload checksum mismatch")
        atomic_write_json(
            root / "reader-verified.json",
            {"schema_version": 1, "run_id": run_id, "sequence": 1, "position": position},
        )
        results.append({"position": position, "passed": True})
    return {"probes": results}


def _run_unsafe_writer(
    *, config: Mapping[str, Any], run_id: str, test_root: Path
) -> dict[str, Any]:
    atomicity = config["atomicity"]
    record_size = max(int(size) for size in atomicity["record_sizes_bytes"])
    iterations = int(atomicity["unsafe_direct_overwrite_iterations"])
    pause = float(atomicity["unsafe_half_write_pause_seconds"])
    record_path = test_root / "unsafe" / "record.json"
    atomic_write_bytes(record_path, build_atomic_record(run_id, record_size, -1))
    _wait_for_expected_record(
        test_root / "unsafe" / "reader-ready.json",
        run_id=run_id,
        sequence=iterations,
        timeout_seconds=float(config["smoke"]["timeout_seconds"]),
        poll_interval_seconds=0.001,
        required_fields=("run_id", "sequence"),
    )
    for sequence in range(iterations):
        content = build_atomic_record(run_id, record_size, sequence)
        midpoint = len(content) // 2
        with record_path.open("wb") as handle:
            handle.write(content[:midpoint])
            handle.flush()
            os.fsync(handle.fileno())
            time.sleep(pause)
            handle.write(content[midpoint:])
            handle.flush()
            os.fsync(handle.fileno())
    atomic_write_json(
        test_root / "unsafe" / "writer-done.json",
        {"schema_version": 1, "run_id": run_id, "sequence": iterations - 1},
    )
    _wait_for_expected_record(
        test_root / "unsafe" / "reader-done.json",
        run_id=run_id,
        sequence=iterations - 1,
        timeout_seconds=float(config["smoke"]["timeout_seconds"]) * 2,
        poll_interval_seconds=0.001,
        required_fields=("run_id", "sequence"),
    )
    return {"iterations": iterations, "record_size": record_size}


def _run_unsafe_reader(
    *, config: Mapping[str, Any], run_id: str, test_root: Path, result_root: Path
) -> dict[str, Any]:
    atomicity = config["atomicity"]
    record_size = max(int(size) for size in atomicity["record_sizes_bytes"])
    iterations = int(atomicity["unsafe_direct_overwrite_iterations"])
    reader_count = int(atomicity["concurrent_readers"])
    poll = float(config["visibility"]["poll_interval_seconds"])
    record_path = test_root / "unsafe" / "record.json"
    violation_log = _ViolationLog(result_root / "unsafe_violations.jsonl")
    states = [{"reader_id": index, "observations": 0, "violations": 0} for index in range(reader_count)]

    def scan(state: dict[str, Any]) -> None:
        while not (test_root / "unsafe" / "writer-done.json").is_file():
            try:
                content = record_path.read_bytes()
                validate_atomic_record(content, run_id=run_id, record_size=record_size)
            except (OSError, StorageHarnessError) as error:
                state["violations"] += 1
                violation_log.write(
                    {
                        "schema_version": 1,
                        "mode": "unsafe_direct_overwrite",
                        "run_id": run_id,
                        "reader_id": state["reader_id"],
                        "observation": state["observations"],
                        "observed_monotonic_ns": time.monotonic_ns(),
                        "error": str(error),
                    }
                )
            finally:
                state["observations"] += 1
            time.sleep(poll)

    threads = [threading.Thread(target=scan, args=(state,)) for state in states]
    for thread in threads:
        thread.start()
    atomic_write_json(
        test_root / "unsafe" / "reader-ready.json",
        {"schema_version": 1, "run_id": run_id, "sequence": iterations},
    )
    for thread in threads:
        thread.join(timeout=float(config["smoke"]["timeout_seconds"]) * 75)
    try:
        if any(thread.is_alive() for thread in threads):
            raise StorageHarnessError("unsafe detector thread did not terminate")
    finally:
        violation_log.close()
    if violation_log.count <= 0:
        raise StorageHarnessError("unsafe direct overwrite produced no detected partial reads")
    atomic_write_json(
        test_root / "unsafe" / "reader-done.json",
        {"schema_version": 1, "run_id": run_id, "sequence": iterations - 1},
    )
    return {"readers": states, "violations": violation_log.count}


def run_stress_role(
    *,
    config: Mapping[str, Any],
    run_id: str,
    test_root: Path,
    result_root: Path,
    mode: str,
) -> dict[str, Any]:
    host, job_id, _ = require_compute_context()
    rank, size = _role_identity()
    target_root = Path(str(config["miyabi"]["target_shared_run_root"])).resolve()
    try:
        test_root.resolve().relative_to(target_root)
    except ValueError as error:
        raise StorageHarnessError("storage stress path must be below the frozen RUN_ROOT") from error
    if mode not in {"full", "unsafe"}:
        raise StorageHarnessError(f"unknown stress mode: {mode}")
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
        "mode": mode,
        "test_root": str(test_root),
        "clock": "time.monotonic_ns",
        "status": "failed",
    }
    try:
        if mode == "unsafe":
            result["unsafe"] = (
                _run_unsafe_writer(config=config, run_id=run_id, test_root=test_root)
                if rank == 0
                else _run_unsafe_reader(
                    config=config,
                    run_id=run_id,
                    test_root=test_root,
                    result_root=result_root,
                )
            )
        elif rank == 0:
            result["latency"] = _run_latency_writer(
                config=config, run_id=run_id, test_root=test_root, result_root=result_root
            )
            result["atomicity"] = _run_atomic_writer(
                config=config,
                run_id=run_id,
                test_root=test_root,
                result_root=result_root,
            )
            result["writer_kill"] = _run_kill_probe_writer(
                run_id=run_id,
                test_root=test_root,
                timeout=float(config["smoke"]["timeout_seconds"]),
            )
        else:
            result["latency"] = _run_latency_reader(
                config=config, run_id=run_id, test_root=test_root
            )
            result["atomicity"] = _run_atomic_reader(
                config=config,
                run_id=run_id,
                test_root=test_root,
                result_root=result_root,
            )
            result["writer_kill"] = _run_kill_probe_reader(
                run_id=run_id,
                test_root=test_root,
                timeout=float(config["smoke"]["timeout_seconds"]),
            )
        result["status"] = "passed"
    except (OSError, StorageHarnessError) as error:
        result["error"] = str(error)
    atomic_write_json(result_root / f"stress-rank-{rank}.json", result)
    return result


def validate_latency_samples(
    samples: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    visibility = config["visibility"]
    expected_total = int(visibility["samples_per_profile"]) * len(visibility["profiles"])
    if len(samples) != expected_total:
        raise StorageHarnessError(f"latency sample count {len(samples)} != {expected_total}")
    combinations = Counter((sample.get("profile"), sample.get("cache_state")) for sample in samples)
    expected_per_cache = int(visibility["samples_per_cache_state"])
    expected_combinations = {
        (profile, cache): expected_per_cache
        for profile in visibility["profiles"]
        for cache in visibility["cache_states"]
    }
    if dict(combinations) != expected_combinations:
        raise StorageHarnessError("latency evidence is missing a profile or cache-state sample")
    sequences = [sample.get("sequence") for sample in samples]
    if sequences != list(range(expected_total)):
        raise StorageHarnessError("latency sample sequence is not complete and ordered")
    for sample in samples:
        if sample.get("timing_method") != TIMING_METHOD:
            raise StorageHarnessError("latency sample used a non-conservative timing method")
        if sample.get("timing_origin") != TIMING_ORIGIN:
            raise StorageHarnessError("latency sample began before visibility publication completed")
        rtt = sample.get("writer_ack_roundtrip_ns")
        if isinstance(rtt, bool) or not isinstance(rtt, int) or rtt < 0:
            raise StorageHarnessError("latency sample has an invalid writer-side RTT")
        if sample.get("invalid_record_reads") != 0:
            raise StorageHarnessError("latency sample includes an incomplete record observation")
    profiles: dict[str, Any] = {}
    threshold = float(config["thresholds"]["visibility_p99_seconds"])
    for profile in visibility["profiles"]:
        values = [int(sample["writer_ack_roundtrip_ns"]) for sample in samples if sample["profile"] == profile]
        statistics = {
            ("max_seconds" if percent == 100 else f"p{percent:g}_seconds"): (
                percentile(values, percent) / 1_000_000_000
            )
            for percent in visibility["report_percentiles"]
        }
        statistics["sample_count"] = len(values)
        if statistics["p99_seconds"] > threshold:
            raise StorageHarnessError(f"{profile} visibility p99 exceeds {threshold} seconds")
        profiles[str(profile)] = statistics
    return {
        "timing_method": TIMING_METHOD,
        "timing_origin": TIMING_ORIGIN,
        "clock_skew_handling": "writer_clock_elapsed_only_no_cross_host_subtraction",
        "threshold_p99_seconds": threshold,
        "profiles": profiles,
    }


def validate_atomicity_results(
    writer: Mapping[str, Any],
    reader: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    result_root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    atomicity = config["atomicity"]
    expected_count = int(atomicity["replacements_per_record_size"])
    expected_sizes = [int(value) for value in atomicity["record_sizes_bytes"]]
    writer_sizes = {int(item["record_size"]): item for item in writer.get("record_sizes", [])}
    reader_sizes = {int(item["record_size"]): item for item in reader.get("record_sizes", [])}
    if sorted(writer_sizes) != expected_sizes or sorted(reader_sizes) != expected_sizes:
        raise StorageHarnessError("atomicity evidence is missing a frozen record size")
    if reader.get("violations") != int(atomicity["permitted_violations"]):
        raise StorageHarnessError("atomicity evidence contains torn-record violations")
    summary: dict[str, Any] = {}
    for record_size in expected_sizes:
        writer_result = writer_sizes[record_size]
        reader_result = reader_sizes[record_size]
        if writer_result.get("replacements") != expected_count:
            raise StorageHarnessError("atomic writer completed too few replacements")
        if writer_result.get("publication_transcript_records") != expected_count:
            raise StorageHarnessError("atomic writer transcript has the wrong record count")
        transcript_name = str(writer_result.get("publication_transcript", ""))
        transcript_sha256 = str(writer_result.get("publication_transcript_sha256", ""))
        if not transcript_name or len(transcript_sha256) != 64:
            raise StorageHarnessError("atomic writer transcript identity is missing")
        if Path(transcript_name).name != transcript_name:
            raise StorageHarnessError("atomic writer transcript path is not a safe basename")
        if result_root is not None:
            transcript = _read_jsonl(result_root / transcript_name)
            if len(transcript) != expected_count:
                raise StorageHarnessError("raw atomic writer transcript is incomplete")
            digest = hashlib.sha256()
            last_publication_ns: int | None = None
            for sequence, publication in enumerate(transcript):
                publication_run_id = publication.get("run_id")
                if publication_run_id is None or (
                    run_id is not None and publication_run_id != run_id
                ):
                    raise StorageHarnessError("atomic publication run identity is invalid")
                if publication.get("record_size") != record_size:
                    raise StorageHarnessError("atomic publication record size changed")
                if publication.get("sequence") != sequence:
                    raise StorageHarnessError("atomic publication transcript sequence is incomplete")
                if publication.get("timing_origin") != TIMING_ORIGIN:
                    raise StorageHarnessError("atomic publication timing origin is invalid")
                publication_ns = publication.get("publication_complete_monotonic_ns")
                if isinstance(publication_ns, bool) or not isinstance(publication_ns, int):
                    raise StorageHarnessError("atomic publication timestamp is invalid")
                if last_publication_ns is not None and publication_ns < last_publication_ns:
                    raise StorageHarnessError("atomic publication timestamps regressed")
                last_publication_ns = publication_ns
                record_sha256 = str(publication.get("record_sha256", ""))
                if len(record_sha256) != 64 or set(record_sha256) - set("0123456789abcdef"):
                    raise StorageHarnessError("atomic publication checksum is invalid")
                digest.update((_canonical_json(publication) + "\n").encode("utf-8"))
            if digest.hexdigest() != transcript_sha256:
                raise StorageHarnessError("atomic writer transcript digest mismatch")
        readers = reader_result.get("readers")
        if not isinstance(readers, list) or len(readers) != int(atomicity["concurrent_readers"]):
            raise StorageHarnessError("atomicity evidence has the wrong concurrent reader count")
        for state in readers:
            if state.get("violations") != 0:
                raise StorageHarnessError("an atomic reader observed an incomplete record")
            if int(state.get("observations", 0)) < int(atomicity["minimum_observations_per_reader"]):
                raise StorageHarnessError("an atomic reader made too few observations")
            if state.get("final_seen") is not True:
                raise StorageHarnessError("an atomic reader did not observe the final replacement")
            if not str(state.get("transcript_sha256", "")):
                raise StorageHarnessError("atomic reader transcript digest is missing")
            if int(state.get("polls", 0)) < int(state.get("observations", 0)):
                raise StorageHarnessError("atomic reader poll accounting is invalid")
            first_observed = state.get("first_observed_monotonic_ns")
            last_observed = state.get("last_observed_monotonic_ns")
            if (
                isinstance(first_observed, bool)
                or not isinstance(first_observed, int)
                or isinstance(last_observed, bool)
                or not isinstance(last_observed, int)
                or last_observed < first_observed
            ):
                raise StorageHarnessError("atomic reader timestamps are invalid")
        summary[str(record_size)] = {
            "replacements": expected_count,
            "reader_count": len(readers),
            "observations": [state["observations"] for state in readers],
            "violations": 0,
        }
    return {"record_sizes": summary, "violations": 0}


def summarize_stress(
    *, config: Mapping[str, Any], run_id: str, result_root: Path, mode: str
) -> dict[str, Any]:
    roles = [read_json_record(result_root / f"stress-rank-{rank}.json") for rank in (0, 1)]
    if {role.get("rank") for role in roles} != {0, 1}:
        raise StorageHarnessError("stress evidence does not contain both roles")
    if any(role.get("run_id") != run_id or role.get("mode") != mode for role in roles):
        raise StorageHarnessError("stress role identity mismatch")
    if any(role.get("status") != "passed" for role in roles):
        raise StorageHarnessError("one or more storage stress roles failed")
    if len({role.get("hostname") for role in roles}) != 2:
        raise StorageHarnessError("storage stress roles did not run on distinct hosts")
    writer = next(role for role in roles if role["role"] == "writer")
    reader = next(role for role in roles if role["role"] == "reader")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "roles": roles,
    }
    if mode == "unsafe":
        raw_violations = _read_jsonl(result_root / "unsafe_violations.jsonl")
        observed = int(reader["unsafe"]["violations"])
        if observed <= 0 or len(raw_violations) != observed:
            raise StorageHarnessError("unsafe detector summary does not match its raw scanner log")
        summary.update(
            {
                "outcome": "unsafe_direct_overwrite_rejected",
                "unsafe_direct_overwrite": {
                    "iterations": writer["unsafe"]["iterations"],
                    "reader_count": len(reader["unsafe"]["readers"]),
                    "partial_or_torn_reads_detected": observed,
                    "raw_violation_log": "unsafe_violations.jsonl",
                },
            }
        )
    elif mode == "full":
        samples = _read_jsonl(result_root / "latency_samples.jsonl")
        latency = validate_latency_samples(samples, config)
        if writer["latency"].get("background_replacements", 0) < int(
            config["visibility"]["background_minimum_replacements"]
        ):
            raise StorageHarnessError("loaded profile background replacement count is too low")
        loaded_start = int(writer["latency"].get("loaded_profile_start_replacements", 0))
        loaded_end = int(writer["latency"].get("loaded_profile_end_replacements", 0))
        if loaded_start < int(config["visibility"]["background_minimum_replacements"]):
            raise StorageHarnessError("loaded profile began before background-load warmup")
        if loaded_end <= loaded_start:
            raise StorageHarnessError("background load was not active throughout loaded sampling")
        if reader["latency"].get("background_violations") != 0:
            raise StorageHarnessError("loaded profile observed a partial background record")
        raw_violations = _read_jsonl(result_root / "atomic_violations.jsonl")
        if raw_violations:
            raise StorageHarnessError("raw atomicity scanner contains violations")
        atomicity = validate_atomicity_results(
            writer["atomicity"],
            reader["atomicity"],
            config,
            result_root=result_root,
            run_id=run_id,
        )
        expected_positions = set(config["atomicity"]["writer_kill_positions"])
        for role in roles:
            probes = role.get("writer_kill", {}).get("probes", [])
            if {probe.get("position") for probe in probes} != expected_positions:
                raise StorageHarnessError("writer-kill evidence is missing a publication boundary")
            if any(probe.get("passed") is not True for probe in probes):
                raise StorageHarnessError("a writer-kill boundary probe failed")
        summary.update(
            {
                "outcome": "visibility_and_atomicity_confirmed",
                "latency": latency,
                "loaded_background": {
                    "replacements": writer["latency"]["background_replacements"],
                    "replacements_at_profile_start": loaded_start,
                    "replacements_at_profile_end": loaded_end,
                    "reader_observations": reader["latency"]["background_observations"],
                    "violations": 0,
                },
                "atomicity": atomicity,
                "writer_kill_positions": sorted(expected_positions),
                "raw_violation_log": "atomic_violations.jsonl",
            }
        )
    else:
        raise StorageHarnessError(f"unknown stress summary mode: {mode}")
    atomic_write_json(result_root / "stress-summary.json", summary)
    return summary


def finalize_stress_manifest(
    *, manifest_path: Path, summary_path: Path, yaml_output_path: Path
) -> dict[str, Any]:
    manifest = read_json_record(manifest_path)
    summary = read_json_record(summary_path, required_fields=("run_id", "mode", "outcome", "roles"))
    if summary["run_id"] != manifest.get("run_id"):
        raise StorageHarnessError("stress summary and environment manifest run_id differ")
    roles = summary["roles"]
    manifest["actual_role_mapping"] = [
        {
            "rank": role.get("rank"),
            "role": role.get("role"),
            "hostname": role.get("hostname"),
            "pbs_job_id": role.get("pbs_job_id"),
            "status": role.get("status"),
            "test_root": role.get("test_root"),
        }
        for role in sorted(roles, key=lambda item: int(item.get("rank", -1)))
    ]
    manifest["stress_mode"] = summary["mode"]
    manifest["stress_outcome"] = summary["outcome"]
    validate_environment_manifest(manifest)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(yaml_output_path, manifest)
    return manifest


def _command_role(arguments: argparse.Namespace) -> int:
    result = run_stress_role(
        config=load_storage_config(arguments.config),
        run_id=arguments.run_id,
        test_root=arguments.test_root,
        result_root=arguments.result_root,
        mode=arguments.mode,
    )
    return 0 if result["status"] == "passed" else 2


def _command_summarize(arguments: argparse.Namespace) -> int:
    summary = summarize_stress(
        config=load_storage_config(arguments.config),
        run_id=arguments.run_id,
        result_root=arguments.result_root,
        mode=arguments.mode,
    )
    print(_canonical_json(summary))
    return 0


def _command_finalize_manifest(arguments: argparse.Namespace) -> int:
    finalize_stress_manifest(
        manifest_path=arguments.manifest,
        summary_path=arguments.summary,
        yaml_output_path=arguments.yaml_output,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    role = subparsers.add_parser("role")
    role.add_argument("--config", type=Path, required=True)
    role.add_argument("--run-id", required=True)
    role.add_argument("--test-root", type=Path, required=True)
    role.add_argument("--result-root", type=Path, required=True)
    role.add_argument("--mode", choices=("full", "unsafe"), required=True)
    role.set_defaults(handler=_command_role)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--config", type=Path, required=True)
    summarize.add_argument("--run-id", required=True)
    summarize.add_argument("--result-root", type=Path, required=True)
    summarize.add_argument("--mode", choices=("full", "unsafe"), required=True)
    summarize.set_defaults(handler=_command_summarize)
    finalize = subparsers.add_parser("finalize-manifest")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--summary", type=Path, required=True)
    finalize.add_argument("--yaml-output", type=Path, required=True)
    finalize.set_defaults(handler=_command_finalize_manifest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except StorageHarnessError as error:
        print(f"storage stress error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
