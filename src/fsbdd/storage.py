from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable


class PublicationError(RuntimeError):
    pass


class PublicationNotFound(PublicationError):
    """The requested fixed visibility slot was absent at the read deadline."""


class PublicationInterrupted(PublicationError):
    pass


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SLOT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CRASH_POINTS = {
    "before_payload_write",
    "after_payload_write",
    "before_record_replace",
    "after_record_replace",
}


@dataclasses.dataclass(frozen=True, slots=True)
class PublicationSpec:
    run_identity: str
    fragment_map_identity: str
    fragment_identity: str
    version: int
    sequence: int
    dtype: str
    shape: tuple[int, ...]
    base_content_identity: str


@dataclasses.dataclass(frozen=True, slots=True)
class ReadExpectation:
    run_identity: str
    fragment_map_identity: str
    fragment_identity: str
    version: int | None = None
    sequence: int | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    base_content_identity: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class PublicationRecord:
    schema_version: int
    complete: bool
    run_identity: str
    fragment_map_identity: str
    fragment_identity: str
    version: int
    sequence: int
    dtype: str
    shape: tuple[int, ...]
    base_content_identity: str
    payload_relative_path: str
    payload_bytes: int
    payload_sha256: str
    payload_identity: str

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class PublishedPayload:
    record: PublicationRecord
    payload: bytes


VisibilityHook = Callable[[str, PublicationRecord], None]


@runtime_checkable
class StorageBackend(Protocol):
    def publish(
        self,
        slot: str,
        payload: bytes,
        spec: PublicationSpec,
        *,
        visibility_hook: VisibilityHook | None = None,
        crash_at: str | None = None,
    ) -> PublicationRecord: ...

    def read(
        self,
        slot: str,
        expectation: ReadExpectation,
        *,
        timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.01,
    ) -> PublishedPayload: ...


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicationError(f"{field} must be a non-empty string")
    return value


def _require_hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise PublicationError(f"{field} must be a lowercase SHA-256 identity")
    return value


def _require_ordinal(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PublicationError(f"{field} must be a nonnegative integer")
    return value


def _require_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise PublicationError("shape must be an integer sequence")
    shape = tuple(value)
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in shape
    ):
        raise PublicationError("shape must contain only nonnegative integers")
    return shape


def _validate_spec(spec: PublicationSpec) -> None:
    _require_text(spec.run_identity, "run_identity")
    _require_hex64(spec.fragment_map_identity, "fragment_map_identity")
    _require_text(spec.fragment_identity, "fragment_identity")
    _require_ordinal(spec.version, "version")
    _require_ordinal(spec.sequence, "sequence")
    _require_text(spec.dtype, "dtype")
    _require_shape(spec.shape)
    _require_hex64(spec.base_content_identity, "base_content_identity")


def _validate_slot(slot: str) -> str:
    if not isinstance(slot, str) or not _SLOT.fullmatch(slot):
        raise PublicationError("slot must match the fixed-slot identifier grammar")
    return slot


def _canonical_record_bytes(record: PublicationRecord) -> bytes:
    return (
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _parse_record(content: bytes) -> PublicationRecord:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationError(
            "visibility record is not complete canonical JSON"
        ) from error
    expected = {field.name for field in dataclasses.fields(PublicationRecord)}
    if not isinstance(value, dict) or set(value) != expected:
        raise PublicationError("visibility record schema mismatch")
    if value.get("schema_version") != 1 or value.get("complete") is not True:
        raise PublicationError("visibility record completion marker is invalid")
    try:
        record = PublicationRecord(
            schema_version=1,
            complete=True,
            run_identity=_require_text(value["run_identity"], "run_identity"),
            fragment_map_identity=_require_hex64(
                value["fragment_map_identity"], "fragment_map_identity"
            ),
            fragment_identity=_require_text(
                value["fragment_identity"], "fragment_identity"
            ),
            version=_require_ordinal(value["version"], "version"),
            sequence=_require_ordinal(value["sequence"], "sequence"),
            dtype=_require_text(value["dtype"], "dtype"),
            shape=_require_shape(value["shape"]),
            base_content_identity=_require_hex64(
                value["base_content_identity"], "base_content_identity"
            ),
            payload_relative_path=_require_text(
                value["payload_relative_path"], "payload_relative_path"
            ),
            payload_bytes=_require_ordinal(value["payload_bytes"], "payload_bytes"),
            payload_sha256=_require_hex64(value["payload_sha256"], "payload_sha256"),
            payload_identity=_require_hex64(
                value["payload_identity"], "payload_identity"
            ),
        )
    except KeyError as error:
        raise PublicationError(
            "visibility record is missing a required field"
        ) from error
    if record.payload_identity != record.payload_sha256:
        raise PublicationError("payload identity and checksum differ")
    payload_path = PurePosixPath(record.payload_relative_path)
    if (
        payload_path.is_absolute()
        or len(payload_path.parts) != 2
        or payload_path.parts[0] != "payloads"
        or not payload_path.parts[1].endswith(".bin")
    ):
        raise PublicationError(
            "visibility record payload path is outside the immutable payload area"
        )
    return record


def _validate_expectation(
    record: PublicationRecord, expectation: ReadExpectation
) -> None:
    exact = {
        "run_identity": expectation.run_identity,
        "fragment_map_identity": expectation.fragment_map_identity,
        "fragment_identity": expectation.fragment_identity,
    }
    optional = {
        "version": expectation.version,
        "sequence": expectation.sequence,
        "dtype": expectation.dtype,
        "shape": expectation.shape,
        "base_content_identity": expectation.base_content_identity,
    }
    for field, expected in {**exact, **optional}.items():
        if expected is not None and getattr(record, field) != expected:
            raise PublicationError(f"visibility record {field} identity mismatch")


class PosixStorageBackend:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._payload_root = root / "payloads"
        self._visibility_root = root / "visibility"
        self._record_temp_root = root / ".record-tmp"
        self._payload_root.mkdir(parents=True, exist_ok=True)
        self._visibility_root.mkdir(parents=True, exist_ok=True)
        self._record_temp_root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def publish(
        self,
        slot: str,
        payload: bytes,
        spec: PublicationSpec,
        *,
        visibility_hook: VisibilityHook | None = None,
        crash_at: str | None = None,
    ) -> PublicationRecord:
        slot = _validate_slot(slot)
        _validate_spec(spec)
        if not isinstance(payload, bytes):
            raise PublicationError("payload must be immutable bytes")
        if crash_at is not None and crash_at not in _CRASH_POINTS:
            raise PublicationError(f"unknown crash point: {crash_at}")
        if crash_at == "before_payload_write":
            raise PublicationInterrupted(crash_at)

        unique = uuid.uuid4().hex
        payload_relative = f"payloads/{unique}.bin"
        payload_path = self._root / payload_relative
        try:
            with payload_path.open("xb") as stream:
                stream.write(payload)
        except OSError as error:
            raise PublicationError(
                f"failed to write unique payload: {error}"
            ) from error
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        try:
            verified = payload_path.read_bytes()
        except OSError as error:
            raise PublicationError(
                f"completed payload is not readable: {error}"
            ) from error
        if (
            len(verified) != len(payload)
            or hashlib.sha256(verified).hexdigest() != payload_sha256
        ):
            raise PublicationError(
                "completed payload failed size/checksum verification"
            )
        if crash_at == "after_payload_write":
            raise PublicationInterrupted(crash_at)

        record = PublicationRecord(
            schema_version=1,
            complete=True,
            run_identity=spec.run_identity,
            fragment_map_identity=spec.fragment_map_identity,
            fragment_identity=spec.fragment_identity,
            version=spec.version,
            sequence=spec.sequence,
            dtype=spec.dtype,
            shape=spec.shape,
            base_content_identity=spec.base_content_identity,
            payload_relative_path=payload_relative,
            payload_bytes=len(payload),
            payload_sha256=payload_sha256,
            payload_identity=payload_sha256,
        )
        record_temp = self._record_temp_root / f"{slot}-{unique}.json"
        try:
            with record_temp.open("xb") as stream:
                stream.write(_canonical_record_bytes(record))
        except OSError as error:
            raise PublicationError(
                f"failed to stage visibility record: {error}"
            ) from error
        if crash_at == "before_record_replace":
            raise PublicationInterrupted(crash_at)
        if visibility_hook is not None:
            visibility_hook(slot, record)
        try:
            os.replace(record_temp, self._visibility_root / f"{slot}.json")
        except OSError as error:
            raise PublicationError(
                f"atomic visibility record replacement failed: {error}"
            ) from error
        if crash_at == "after_record_replace":
            raise PublicationInterrupted(crash_at)
        return record

    def read(
        self,
        slot: str,
        expectation: ReadExpectation,
        *,
        timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.01,
    ) -> PublishedPayload:
        slot = _validate_slot(slot)
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds < 0
        ):
            raise PublicationError("timeout_seconds must be nonnegative")
        if (
            not isinstance(poll_interval_seconds, (int, float))
            or isinstance(poll_interval_seconds, bool)
            or poll_interval_seconds < 0
        ):
            raise PublicationError("poll_interval_seconds must be nonnegative")
        deadline = time.monotonic() + float(timeout_seconds)
        visibility_path = self._visibility_root / f"{slot}.json"
        while True:
            try:
                record_content = visibility_path.read_bytes()
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise PublicationNotFound(
                        "visibility record did not become readable before timeout"
                    )
                time.sleep(float(poll_interval_seconds))
                continue
            except OSError as error:
                raise PublicationError(
                    f"visibility record read failed: {error}"
                ) from error
            record = _parse_record(record_content)
            _validate_expectation(record, expectation)
            payload_path = self._root / record.payload_relative_path
            try:
                payload = payload_path.read_bytes()
            except FileNotFoundError:
                payload = b""
            except OSError as error:
                raise PublicationError(f"payload read failed: {error}") from error
            valid = (
                len(payload) == record.payload_bytes
                and hashlib.sha256(payload).hexdigest() == record.payload_sha256
            )
            if valid:
                return PublishedPayload(record=record, payload=payload)
            if time.monotonic() >= deadline:
                raise PublicationError(
                    "payload did not become complete and readable before timeout"
                )
            time.sleep(float(poll_interval_seconds))
