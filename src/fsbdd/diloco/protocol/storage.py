from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import stat
import time
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable


class PublicationError(RuntimeError):
    pass


class PublicationNotReady(PublicationError):
    """A fixed slot or its named payload may become readable on a later poll."""


class PublicationNotFound(PublicationNotReady):
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
_MAX_VISIBILITY_RECORD_BYTES = 64 * 1024


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


@runtime_checkable
class RecordReadableStorageBackend(StorageBackend, Protocol):
    """Storage extension for validating visibility without loading payload bytes."""

    def read_record(
        self,
        slot: str,
        expectation: ReadExpectation,
        *,
        timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.01,
    ) -> PublicationRecord: ...


@runtime_checkable
class BoundRecordStorageBackend(RecordReadableStorageBackend, Protocol):
    """Storage extension for loading the immutable payload named by a record."""

    def read_bound_record(self, record: PublicationRecord) -> PublishedPayload: ...


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


def _require_nonnegative_finite_real(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise PublicationError(f"{field} must be a finite nonnegative number")
    return float(value)


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
    if content != _canonical_record_bytes(record):
        raise PublicationError("visibility record is not canonical JSON")
    return record


def decode_publication_record(content: bytes) -> PublicationRecord:
    """Decode one canonical visibility record without touching its payload."""

    if not isinstance(content, bytes):
        raise PublicationError("visibility record content must be immutable bytes")
    if len(content) > _MAX_VISIBILITY_RECORD_BYTES:
        raise PublicationError("visibility record exceeds the bounded metadata size")
    return _parse_record(content)


def _read_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    field: str,
) -> bytes:
    if path.is_symlink():
        raise PublicationError(f"{field} cannot be a symlink")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PublicationError(f"{field} must be a regular file")
        if metadata.st_size > maximum_bytes:
            raise PublicationError(f"{field} exceeds its bounded size")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(metadata.st_size + 1)
        if len(content) > maximum_bytes:
            raise PublicationError(f"{field} exceeds its bounded size")
        return content
    finally:
        os.close(descriptor)


def _read_visibility_record(path: Path) -> PublicationRecord:
    try:
        content = _read_regular_file(
            path,
            maximum_bytes=_MAX_VISIBILITY_RECORD_BYTES,
            field="visibility record",
        )
    except FileNotFoundError:
        raise
    except PublicationError:
        raise
    except OSError as error:
        raise PublicationError(f"visibility record read failed: {error}") from error
    return decode_publication_record(content)


def _validate_read_expectation(expectation: ReadExpectation) -> None:
    if not isinstance(expectation, ReadExpectation):
        raise PublicationError("read expectation must be ReadExpectation")
    _require_text(expectation.run_identity, "expectation.run_identity")
    _require_hex64(
        expectation.fragment_map_identity,
        "expectation.fragment_map_identity",
    )
    _require_text(
        expectation.fragment_identity,
        "expectation.fragment_identity",
    )
    if expectation.version is not None:
        _require_ordinal(expectation.version, "expectation.version")
    if expectation.sequence is not None:
        _require_ordinal(expectation.sequence, "expectation.sequence")
    if expectation.dtype is not None:
        _require_text(expectation.dtype, "expectation.dtype")
    if expectation.shape is not None:
        _require_shape(expectation.shape)
    if expectation.base_content_identity is not None:
        _require_hex64(
            expectation.base_content_identity,
            "expectation.base_content_identity",
        )


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
    def __init__(
        self, root: Path, *, verify_payload_readback: bool = False
    ) -> None:
        if not isinstance(verify_payload_readback, bool):
            raise PublicationError("verify_payload_readback must be boolean")
        self._root = root
        self._verify_payload_readback = verify_payload_readback
        self._payload_root = root / "payloads"
        self._visibility_root = root / "visibility"
        self._record_temp_root = root / ".record-tmp"
        self._retired_root = root / ".retired"
        self._payload_root.mkdir(parents=True, exist_ok=True)
        self._visibility_root.mkdir(parents=True, exist_ok=True)
        self._record_temp_root.mkdir(parents=True, exist_ok=True)
        self._retired_root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def publication_verification_mode(self) -> str:
        return (
            "post_write_readback"
            if self._verify_payload_readback
            else "source_sha256_complete_write"
        )

    def read_record(
        self,
        slot: str,
        expectation: ReadExpectation,
        *,
        timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.01,
    ) -> PublicationRecord:
        """Read and validate one fixed visibility record without its payload."""

        slot = _validate_slot(slot)
        _validate_read_expectation(expectation)
        timeout = _require_nonnegative_finite_real(
            timeout_seconds, "timeout_seconds"
        )
        poll_interval = _require_nonnegative_finite_real(
            poll_interval_seconds, "poll_interval_seconds"
        )
        deadline = time.monotonic() + timeout
        visibility_path = self._visibility_root / f"{slot}.json"
        while True:
            try:
                record = _read_visibility_record(visibility_path)
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise PublicationNotFound(
                        "visibility record did not become readable before timeout"
                    )
                time.sleep(poll_interval)
                continue
            _validate_expectation(record, expectation)
            return record

    def inspect_inventory(self) -> dict[str, int]:
        """Classify every mutable POSIX container without reading tensor bytes."""

        records = []
        for path in self._visibility_root.glob("*.json"):
            try:
                records.append(_read_visibility_record(path))
            except (OSError, PublicationError):
                continue
        referenced = {record.payload_relative_path for record in records}
        payloads = tuple(
            path
            for path in self._payload_root.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        temporary = tuple(
            path
            for path in self._record_temp_root.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        retirement_markers = tuple(
            path
            for path in self._retired_root.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        referenced_bytes = sum(
            path.stat().st_size
            for path in payloads
            if f"payloads/{path.name}" in referenced
        )
        orphan_bytes = sum(
            path.stat().st_size
            for path in payloads
            if f"payloads/{path.name}" not in referenced
        )
        return {
            "visibility_records": len(records),
            "physical_payloads": len(payloads),
            "referenced_payloads": sum(
                f"payloads/{path.name}" in referenced for path in payloads
            ),
            "orphan_payloads": sum(
                f"payloads/{path.name}" not in referenced for path in payloads
            ),
            "referenced_payload_bytes": referenced_bytes,
            "orphan_payload_bytes": orphan_bytes,
            "record_temp_files": len(temporary),
            "record_temp_bytes": sum(path.stat().st_size for path in temporary),
            "retirement_markers": len(retirement_markers),
            "retirement_marker_bytes": sum(
                path.stat().st_size for path in retirement_markers
            ),
        }

    def _retirement_times(self) -> dict[str, int]:
        values: dict[str, int] = {}
        for path in self._retired_root.glob("*.json"):
            try:
                value = json.loads(path.read_bytes())
                relative = value["payload_relative_path"]
                retired_unix_ns = value["retired_unix_ns"]
                if (
                    not isinstance(relative, str)
                    or not relative.startswith("payloads/")
                    or not isinstance(retired_unix_ns, int)
                    or isinstance(retired_unix_ns, bool)
                    or retired_unix_ns < 0
                ):
                    continue
                values[relative] = retired_unix_ns
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
        return values

    def _mark_retiring(self, record: PublicationRecord) -> None:
        """Timestamp loss of visibility before replacing the current record."""

        name = Path(record.payload_relative_path).name
        marker = self._retired_root / f"{name}.json"
        temporary = self._retired_root / f".{name}.{uuid.uuid4().hex}.tmp"
        value = {
            "schema_version": 1,
            "payload_relative_path": record.payload_relative_path,
            "retired_unix_ns": time.time_ns(),
        }
        try:
            content = (
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            with temporary.open("xb") as stream:
                if stream.write(content) != len(content):
                    raise PublicationError(
                        "payload retirement marker write was incomplete"
                    )
            os.replace(temporary, marker)
        except OSError as error:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise PublicationError(
                f"failed to stage payload retirement marker: {error}"
            ) from error

    def reclaim_unreferenced_payloads(
        self,
        *,
        retain_recent: int,
        minimum_age_seconds: float,
    ) -> dict[str, int]:
        """Best-effort bounded reclamation outside the visibility critical path.

        Current payloads are never removed.  A fixed number of the newest
        unreferenced payloads and every payload younger than the frozen safety
        age are retained so readers that observed the immediately previous
        record can finish.
        """

        if (
            not isinstance(retain_recent, int)
            or isinstance(retain_recent, bool)
            or retain_recent < 0
        ):
            raise PublicationError("retain_recent must be a nonnegative integer")
        minimum_age = _require_nonnegative_finite_real(
            minimum_age_seconds, "minimum_age_seconds"
        )
        records = []
        for path in self._visibility_root.glob("*.json"):
            try:
                records.append(_read_visibility_record(path))
            except (OSError, PublicationError):
                continue
        referenced = {record.payload_relative_path for record in records}
        unreferenced = sorted(
            (
                path
                for path in self._payload_root.iterdir()
                if path.is_file()
                and not path.is_symlink()
                and f"payloads/{path.name}" not in referenced
            ),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        now_ns = time.time_ns()
        minimum_age_ns = int(minimum_age * 1_000_000_000)
        retirement_times = self._retirement_times()
        reclaimed_files = 0
        reclaimed_bytes = 0
        for path in unreferenced[retain_recent:]:
            try:
                stat = path.stat()
                relative = f"payloads/{path.name}"
                unreferenced_since = retirement_times.get(relative, stat.st_mtime_ns)
                if now_ns - unreferenced_since < minimum_age_ns:
                    continue
                size = stat.st_size
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                continue
            reclaimed_files += 1
            reclaimed_bytes += size
            try:
                (self._retired_root / f"{path.name}.json").unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        reclaimed_temp_files = 0
        reclaimed_temp_bytes = 0
        for path in self._record_temp_root.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            try:
                stat = path.stat()
                if now_ns - stat.st_mtime_ns < minimum_age_ns:
                    continue
                size = stat.st_size
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                continue
            reclaimed_temp_files += 1
            reclaimed_temp_bytes += size
        return {
            "reclaimed_payload_files": reclaimed_files,
            "reclaimed_payload_bytes": reclaimed_bytes,
            "reclaimed_record_temp_files": reclaimed_temp_files,
            "reclaimed_record_temp_bytes": reclaimed_temp_bytes,
            **self.inspect_inventory(),
        }

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
        # The production path hashes the immutable source bytes once.  A
        # successful unbuffered POSIX write plus an exact fstat size check is the
        # completed-write authority; reopening this same file before visibility
        # only rereads the page cache and doubles the large-payload data path
        # without adding a durability guarantee.  Readers still validate the
        # visible file's exact size and SHA-256 before returning any bytes, so
        # later corruption fails closed.  The explicit diagnostic mode below
        # retains the former readback-derived checksum for matched measurement.
        payload_sha256 = (
            None
            if self._verify_payload_readback
            else hashlib.sha256(payload).hexdigest()
        )
        source = memoryview(payload)
        try:
            with payload_path.open("xb", buffering=0) as stream:
                offset = 0
                while offset < len(source):
                    written = stream.write(source[offset:])
                    if not isinstance(written, int) or written <= 0:
                        raise PublicationError(
                            "unique payload write made no forward progress"
                        )
                    offset += written
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise PublicationError("unique payload is not a regular file")
                if offset != len(payload) or metadata.st_size != len(payload):
                    raise PublicationError(
                        "unique payload write did not consume the complete payload"
                    )
        except OSError as error:
            raise PublicationError(
                f"failed to write unique payload: {error}"
            ) from error
        finally:
            source.release()
        if self._verify_payload_readback:
            # Retain the former two-pass path as an explicit diagnostic control.
            # It is not the production default because rereading a just-written
            # file normally revalidates page-cache bytes rather than durability.
            digest = hashlib.sha256()
            source = memoryview(payload)
            offset = 0
            try:
                with payload_path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        end = offset + len(chunk)
                        if end > len(source) or chunk != source[offset:end]:
                            raise PublicationError(
                                "completed payload failed bytewise verification"
                            )
                        digest.update(chunk)
                        offset = end
            except OSError as error:
                raise PublicationError(
                    f"completed payload is not readable: {error}"
                ) from error
            finally:
                source.release()
            if offset != len(payload):
                raise PublicationError(
                    "completed payload failed size/checksum verification"
                )
            payload_sha256 = digest.hexdigest()
        if payload_sha256 is None:
            raise AssertionError("payload verification produced no SHA-256")
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
            record_content = _canonical_record_bytes(record)
            with record_temp.open("xb") as stream:
                if stream.write(record_content) != len(record_content):
                    raise PublicationError(
                        "visibility record staging write was incomplete"
                    )
        except OSError as error:
            raise PublicationError(
                f"failed to stage visibility record: {error}"
            ) from error
        if crash_at == "before_record_replace":
            raise PublicationInterrupted(crash_at)
        if visibility_hook is not None:
            visibility_hook(slot, record)
        visibility_path = self._visibility_root / f"{slot}.json"
        try:
            previous_record = _read_visibility_record(visibility_path)
        except FileNotFoundError:
            previous_record = None
        if previous_record is not None:
            self._mark_retiring(previous_record)
        try:
            os.replace(record_temp, visibility_path)
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
        timeout = _require_nonnegative_finite_real(
            timeout_seconds, "timeout_seconds"
        )
        poll_interval = _require_nonnegative_finite_real(
            poll_interval_seconds, "poll_interval_seconds"
        )
        deadline = time.monotonic() + timeout
        while True:
            record = self.read_record(
                slot,
                expectation,
                timeout_seconds=max(0.0, deadline - time.monotonic()),
                poll_interval_seconds=poll_interval_seconds,
            )
            payload_path = self._root / record.payload_relative_path
            try:
                payload = _read_regular_file(
                    payload_path,
                    maximum_bytes=record.payload_bytes,
                    field="payload",
                )
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
                raise PublicationNotReady(
                    "payload did not become complete and readable before timeout"
                )
            time.sleep(poll_interval)

    def read_bound_record(self, record: PublicationRecord) -> PublishedPayload:
        """Read the immutable payload named by one already-validated record."""

        if not isinstance(record, PublicationRecord):
            raise PublicationError("bound read requires a PublicationRecord")
        validated = _parse_record(_canonical_record_bytes(record))
        if validated != record:
            raise PublicationError("bound record fields are not canonical")
        relative = Path(record.payload_relative_path)
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != "payloads"
            or ".." in relative.parts
        ):
            raise PublicationError("bound record payload path is unsafe")
        path = self._root / relative
        try:
            payload = _read_regular_file(
                path,
                maximum_bytes=record.payload_bytes,
                field="bound payload",
            )
        except FileNotFoundError as error:
            raise PublicationNotReady("bound payload is no longer readable") from error
        except OSError as error:
            raise PublicationError(f"bound payload read failed: {error}") from error
        if (
            len(payload) != record.payload_bytes
            or hashlib.sha256(payload).hexdigest() != record.payload_sha256
        ):
            raise PublicationNotReady("bound payload failed its immutable checksum")
        return PublishedPayload(record=record, payload=payload)
