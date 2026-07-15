from __future__ import annotations

import dataclasses
import json
import subprocess
import threading
from pathlib import Path

import pytest

from fsbdd.diloco.protocol.storage import (
    PosixStorageBackend,
    PublicationError,
    PublicationInterrupted,
    PublicationSpec,
    ReadExpectation,
    StorageBackend,
)


MAP_ID = "a" * 64
BASE_ID = "b" * 64


def spec(sequence: int, **changes) -> PublicationSpec:
    value = PublicationSpec(
        run_identity="run-a",
        fragment_map_identity=MAP_ID,
        fragment_identity="fragment-0",
        version=sequence,
        sequence=sequence,
        dtype="float32",
        shape=(4,),
        base_content_identity=BASE_ID,
    )
    return dataclasses.replace(value, **changes)


def expectation(**changes) -> ReadExpectation:
    value = ReadExpectation(
        run_identity="run-a",
        fragment_map_identity=MAP_ID,
        fragment_identity="fragment-0",
    )
    return dataclasses.replace(value, **changes)


def test_interface_exposes_fixed_slot_publish_and_read_only() -> None:
    methods = {
        name
        for name, value in StorageBackend.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    assert methods == {"publish", "read"}


def test_payload_first_visibility_last_and_hook(tmp_path: Path) -> None:
    backend = PosixStorageBackend(tmp_path)
    hook_observations = []

    def hook(slot, record) -> None:
        hook_observations.append(
            (
                slot,
                record.sequence,
                (tmp_path / record.payload_relative_path).read_bytes(),
            )
        )
        assert not (tmp_path / "visibility" / "current.json").exists()

    record = backend.publish(
        "current", b"0123456789abcdef", spec(1), visibility_hook=hook
    )
    published = backend.read(
        "current", expectation(sequence=1, version=1, dtype="float32", shape=(4,))
    )
    assert hook_observations == [("current", 1, b"0123456789abcdef")]
    assert published.payload == b"0123456789abcdef"
    assert published.record == record


@pytest.mark.parametrize(
    ("crash_at", "visible_sequence"),
    [
        ("before_payload_write", 0),
        ("after_payload_write", 0),
        ("before_record_replace", 0),
        ("after_record_replace", 1),
    ],
)
def test_crash_points_leave_only_old_or_complete_new(
    tmp_path: Path, crash_at: str, visible_sequence: int
) -> None:
    backend = PosixStorageBackend(tmp_path / crash_at)
    backend.publish("current", b"old-complete", spec(0))
    with pytest.raises(PublicationInterrupted, match=crash_at):
        backend.publish("current", b"new-complete", spec(1), crash_at=crash_at)
    result = backend.read("current", expectation())
    assert result.record.sequence == visible_sequence
    assert result.payload == (b"new-complete" if visible_sequence else b"old-complete")


def test_read_never_scans_payload_directory(tmp_path: Path, monkeypatch) -> None:
    backend = PosixStorageBackend(tmp_path)
    backend.publish("latest", b"complete", spec(1))

    def forbidden(*args, **kwargs):
        raise AssertionError("payload history scan is forbidden")

    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    assert backend.read("latest", expectation()).payload == b"complete"


@pytest.mark.parametrize(
    ("field", "wrong", "message"),
    [
        ("run_identity", "run-b", "run_identity"),
        ("fragment_map_identity", "c" * 64, "fragment_map_identity"),
        ("fragment_identity", "fragment-1", "fragment_identity"),
        ("version", 2, "version"),
        ("sequence", 2, "sequence"),
        ("dtype", "bfloat16", "dtype"),
        ("shape", (2, 2), "shape"),
        ("base_content_identity", "d" * 64, "base_content_identity"),
    ],
)
def test_expected_identity_mismatch_rejects(
    tmp_path: Path, field: str, wrong, message: str
) -> None:
    backend = PosixStorageBackend(tmp_path)
    backend.publish("current", b"0123456789abcdef", spec(1))
    with pytest.raises(PublicationError, match=message):
        backend.read("current", expectation(**{field: wrong}), timeout_seconds=0)


def test_record_schema_corruption_and_payload_truncation_reject(tmp_path: Path) -> None:
    backend = PosixStorageBackend(tmp_path)
    record = backend.publish("current", b"0123456789abcdef", spec(1))
    visibility = tmp_path / "visibility" / "current.json"
    original = visibility.read_bytes()
    visibility.write_bytes(original[: len(original) // 2])
    with pytest.raises(PublicationError, match="canonical JSON"):
        backend.read("current", expectation(), timeout_seconds=0)

    visibility.write_bytes(original)
    payload = tmp_path / record.payload_relative_path
    payload.write_bytes(b"short")
    with pytest.raises(PublicationError, match="complete and readable"):
        backend.read("current", expectation(), timeout_seconds=0)

    value = json.loads(original)
    value["schema_version"] = 2
    visibility.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PublicationError, match="completion marker"):
        backend.read("current", expectation(), timeout_seconds=0)


def test_compound_payload_is_visible_as_one_complete_object(tmp_path: Path) -> None:
    backend = PosixStorageBackend(tmp_path)
    compound = json.dumps(
        {"parameters": [1, 2], "outer_state": [3, 4]}, sort_keys=True
    ).encode()
    backend.publish("fragment-current", compound, spec(7))
    assert backend.read("fragment-current", expectation(sequence=7)).payload == compound


def test_multiple_concurrent_readers_observe_no_invalid_publication(
    tmp_path: Path,
) -> None:
    backend = PosixStorageBackend(tmp_path)
    backend.publish("latest", b"0" * 4096, spec(0))
    finished = threading.Event()
    errors: list[Exception] = []
    observations = [0, 0, 0, 0]

    def reader(index: int) -> None:
        while not finished.is_set():
            try:
                value = backend.read("latest", expectation(), timeout_seconds=0.1)
                assert len(value.payload) == 4096
                observations[index] += 1
            except Exception as error:  # pragma: no cover - asserted below
                errors.append(error)
                finished.set()

    threads = [threading.Thread(target=reader, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for sequence in range(1, 101):
        backend.publish("latest", bytes([sequence % 251]) * 4096, spec(sequence))
    finished.set()
    for thread in threads:
        thread.join(timeout=5)
    assert not errors
    assert all(count > 0 for count in observations)
    assert backend.read("latest", expectation(sequence=100)).record.sequence == 100


def test_eventual_payload_readability_retries(tmp_path: Path, monkeypatch) -> None:
    backend = PosixStorageBackend(tmp_path)
    record = backend.publish("current", b"complete", spec(1))
    payload_path = tmp_path / record.payload_relative_path
    original = Path.read_bytes
    attempts = 0

    def delayed(path: Path) -> bytes:
        nonlocal attempts
        if path == payload_path and attempts < 2:
            attempts += 1
            raise FileNotFoundError(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", delayed)
    assert (
        backend.read(
            "current", expectation(), timeout_seconds=1, poll_interval_seconds=0
        ).payload
        == b"complete"
    )
    assert attempts == 2


def test_slot_path_traversal_and_invalid_spec_reject(tmp_path: Path) -> None:
    backend = PosixStorageBackend(tmp_path)
    with pytest.raises(PublicationError, match="slot"):
        backend.publish("../escape", b"x", spec(0))
    with pytest.raises(PublicationError, match="fragment_map_identity"):
        backend.publish("current", b"x", spec(0, fragment_map_identity="not-a-digest"))


def test_stress_summary_checks_cross_node_roles_and_crash_matrix(
    tmp_path: Path,
) -> None:
    from fsbdd.auxiliary.stress.storage_stress import summarize

    result_root = tmp_path / "results"
    result_root.mkdir()
    (result_root / "writer.json").write_text(
        json.dumps({"hostname": "mg0001", "publications": 3}), encoding="utf-8"
    )
    (result_root / "reader.json").write_text(
        json.dumps(
            {
                "hostname": "mg0002",
                "maximum_sequence": 3,
                "minimum_sequence": 0,
                "unique_sequences": 4,
                "observations": 4,
                "invalid_reads": 0,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    summary = summarize(
        root=tmp_path / "shared",
        result_root=result_root,
        run_id="run-a",
        publications=3,
        output=output,
    )
    assert [item["outcome"] for item in summary["crash_matrix"]] == [
        "complete_old",
        "complete_old",
        "complete_old",
        "complete_new",
    ]
    assert not (tmp_path / "shared").exists()


def test_pbs_contract_is_two_node_batch_and_filesystem_data_plane() -> None:
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "pbs" / "stage1_s1_03_storage.pbs"
    subprocess.run(["bash", "-n", str(script)], check=True)
    content = script.read_text(encoding="utf-8")
    assert "#PBS -l select=2" in content
    assert "--map-by ppr:1:node" in content
    assert "fsbdd.auxiliary.stress.storage_stress role" in content
    assert "fsbdd.cli evidence finalize" in content
    assert 'date -u -d "$QTIME_RAW JST"' in content
    assert "torchrun" not in content
    assert "nccl" not in content.lower()
