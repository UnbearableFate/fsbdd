from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .identity import canonical_bytes, canonical_digest, file_digest
from .storage import (
    PublicationError,
    PublicationNotFound,
    PublicationRecord,
    PublicationSpec,
    PublishedPayload,
    ReadExpectation,
    StorageBackend,
)


class GlobalStateError(PublicationError):
    pass


class BootstrapInterrupted(GlobalStateError):
    pass


_MAGIC = b"FSBDDGS1"
_HEADER_LIMIT = 16 * 1024 * 1024
_ZERO_IDENTITY = "0" * 64
_HEX = frozenset("0123456789abcdef")


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise GlobalStateError(f"{field} must be a non-empty string")
    return value


def _require_hex(value: object, field: str) -> str:
    value = _require_text(value, field)
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise GlobalStateError(f"{field} must be a lowercase SHA-256 identity")
    return value


def _require_ordinal(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise GlobalStateError(f"{field} must be a nonnegative integer")
    return value


def _require_shape(value: object, field: str = "shape") -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise GlobalStateError(f"{field} must be an integer sequence")
    shape = tuple(value)
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in shape
    ):
        raise GlobalStateError(f"{field} must contain nonnegative integers")
    return shape


def _require_bytes(value: object, field: str) -> bytes:
    if not isinstance(value, bytes):
        raise GlobalStateError(f"{field} must be immutable bytes")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class GlobalStateIdentities:
    run_identity: str
    config_identity: str
    model_identity: str
    fragment_map_identity: str

    def __post_init__(self) -> None:
        _require_text(self.run_identity, "run_identity")
        _require_hex(self.config_identity, "config_identity")
        _require_hex(self.model_identity, "model_identity")
        _require_hex(self.fragment_map_identity, "fragment_map_identity")

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentStateDescriptor:
    index: int
    identity: str
    dtype: str
    shape: tuple[int, ...]
    parameter_identities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_ordinal(self.index, "fragment index")
        _require_text(self.identity, "fragment identity")
        _require_text(self.dtype, "fragment dtype")
        _require_shape(self.shape)
        if not isinstance(self.parameter_identities, tuple) or any(
            not isinstance(item, str) or not item for item in self.parameter_identities
        ):
            raise GlobalStateError(
                "parameter_identities must be a tuple of non-empty strings"
            )
        if len(set(self.parameter_identities)) != len(self.parameter_identities):
            raise GlobalStateError(
                "parameter_identities must be unique within a fragment"
            )

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class BootstrapFragment:
    descriptor: FragmentStateDescriptor
    parameters: bytes
    outer_state: bytes

    def __post_init__(self) -> None:
        _require_bytes(self.parameters, "parameters")
        _require_bytes(self.outer_state, "outer_state")


@dataclasses.dataclass(frozen=True, slots=True)
class BaseSnapshot:
    version: int
    parameters: bytes
    parameters_sha256: str

    def __post_init__(self) -> None:
        _require_ordinal(self.version, "base version")
        _require_bytes(self.parameters, "base parameters")
        _require_hex(self.parameters_sha256, "base parameters_sha256")
        if hashlib.sha256(self.parameters).hexdigest() != self.parameters_sha256:
            raise GlobalStateError("base parameter checksum mismatch")


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentGlobalState:
    identities: GlobalStateIdentities
    descriptor: FragmentStateDescriptor
    version: int
    outer_update_count: int
    parameters: bytes
    outer_state: bytes
    base_history: tuple[BaseSnapshot, ...]
    base_content_identity: str
    content_identity: str

    def __post_init__(self) -> None:
        _require_ordinal(self.version, "version")
        _require_ordinal(self.outer_update_count, "outer_update_count")
        _require_bytes(self.parameters, "parameters")
        _require_bytes(self.outer_state, "outer_state")
        if not isinstance(self.base_history, tuple) or not self.base_history:
            raise GlobalStateError("base_history must be a non-empty tuple")
        versions = tuple(item.version for item in self.base_history)
        if versions != tuple(range(versions[0], versions[0] + len(versions))):
            raise GlobalStateError(
                "base_history versions must be contiguous and increasing"
            )
        if versions[-1] != self.version:
            raise GlobalStateError("base_history must end at the current version")
        if self.base_history[-1].parameters != self.parameters:
            raise GlobalStateError(
                "current parameters differ from the newest retained base"
            )
        _require_hex(self.base_content_identity, "base_content_identity")
        if (self.version == 0) != (self.base_content_identity == _ZERO_IDENTITY):
            raise GlobalStateError(
                "only version zero may use the zero base-content identity"
            )
        _require_hex(self.content_identity, "content_identity")


@dataclasses.dataclass(frozen=True, slots=True)
class GlobalSnapshot:
    states: tuple[FragmentGlobalState, ...]
    version_vector: tuple[int, ...]
    digest: str


@dataclasses.dataclass(frozen=True, slots=True)
class BootstrapResult:
    published_indices: tuple[int, ...]
    existing_indices: tuple[int, ...]
    snapshot: GlobalSnapshot


@dataclasses.dataclass(frozen=True, slots=True)
class LiveSetReport:
    current_records: int
    referenced_payloads: int
    retained_base_entries: int
    maximum_base_entries: int
    physical_payloads: int | None
    orphan_payload_candidates: int | None
    shared_global_head_present: bool | None

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def current_slot(index: int) -> str:
    return f"global-current-{_require_ordinal(index, 'fragment index'):06d}"


def _section(
    name: str, content: bytes, *, version: int | None = None
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    if version is not None:
        result["version"] = version
    return result


def _semantic_header(state: FragmentGlobalState) -> dict[str, object]:
    sections = [
        _section("parameters", state.parameters),
        _section("outer_state", state.outer_state),
        *(
            _section(f"base-{position:06d}", item.parameters, version=item.version)
            for position, item in enumerate(state.base_history)
        ),
    ]
    return {
        "schema_version": 1,
        "identities": state.identities.to_dict(),
        "fragment": state.descriptor.to_dict(),
        "version": state.version,
        "outer_update_count": state.outer_update_count,
        "base_content_identity": state.base_content_identity,
        "base_versions": [item.version for item in state.base_history],
        "sections": sections,
    }


def _make_state(
    *,
    identities: GlobalStateIdentities,
    descriptor: FragmentStateDescriptor,
    version: int,
    outer_update_count: int,
    parameters: bytes,
    outer_state: bytes,
    base_history: tuple[BaseSnapshot, ...],
    base_content_identity: str,
) -> FragmentGlobalState:
    provisional = FragmentGlobalState(
        identities=identities,
        descriptor=descriptor,
        version=version,
        outer_update_count=outer_update_count,
        parameters=parameters,
        outer_state=outer_state,
        base_history=base_history,
        base_content_identity=base_content_identity,
        content_identity=_ZERO_IDENTITY,
    )
    return dataclasses.replace(
        provisional, content_identity=canonical_digest(_semantic_header(provisional))
    )


def encode_global_state(state: FragmentGlobalState) -> bytes:
    semantic = _semantic_header(state)
    if canonical_digest(semantic) != state.content_identity:
        raise GlobalStateError("global-state content identity mismatch")
    header = canonical_bytes({**semantic, "content_identity": state.content_identity})
    if len(header) > _HEADER_LIMIT:
        raise GlobalStateError("global-state header exceeds the format limit")
    sections = (
        state.parameters,
        state.outer_state,
        *(item.parameters for item in state.base_history),
    )
    return _MAGIC + struct.pack(">Q", len(header)) + header + b"".join(sections)


def _parse_json_header(payload: bytes) -> tuple[dict[str, Any], bytes]:
    if len(payload) < len(_MAGIC) + 8 or payload[: len(_MAGIC)] != _MAGIC:
        raise GlobalStateError("global-state payload magic mismatch")
    header_bytes = struct.unpack(">Q", payload[len(_MAGIC) : len(_MAGIC) + 8])[0]
    if header_bytes > _HEADER_LIMIT or len(payload) < len(_MAGIC) + 8 + header_bytes:
        raise GlobalStateError("global-state header length is invalid")
    start = len(_MAGIC) + 8
    raw_header = payload[start : start + header_bytes]
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GlobalStateError("global-state header is not complete JSON") from error
    if not isinstance(header, dict) or canonical_bytes(header) != raw_header:
        raise GlobalStateError("global-state header is not canonical")
    return header, payload[start + header_bytes :]


def decode_global_state(payload: bytes) -> FragmentGlobalState:
    _require_bytes(payload, "payload")
    header, body = _parse_json_header(payload)
    expected_header = {
        "schema_version",
        "identities",
        "fragment",
        "version",
        "outer_update_count",
        "base_content_identity",
        "base_versions",
        "sections",
        "content_identity",
    }
    if set(header) != expected_header or header.get("schema_version") != 1:
        raise GlobalStateError("global-state header schema mismatch")
    identities_value = header["identities"]
    fragment_value = header["fragment"]
    if not isinstance(identities_value, dict) or set(identities_value) != {
        "run_identity",
        "config_identity",
        "model_identity",
        "fragment_map_identity",
    }:
        raise GlobalStateError("global-state identity schema mismatch")
    if not isinstance(fragment_value, dict) or set(fragment_value) != {
        "index",
        "identity",
        "dtype",
        "shape",
        "parameter_identities",
    }:
        raise GlobalStateError("global-state fragment schema mismatch")
    if not isinstance(fragment_value["parameter_identities"], list):
        raise GlobalStateError("global-state parameter identities must be a list")
    try:
        identities = GlobalStateIdentities(**identities_value)
        descriptor = FragmentStateDescriptor(
            index=fragment_value["index"],
            identity=fragment_value["identity"],
            dtype=fragment_value["dtype"],
            shape=_require_shape(fragment_value["shape"]),
            parameter_identities=tuple(fragment_value["parameter_identities"]),
        )
    except (KeyError, TypeError) as error:
        raise GlobalStateError(
            "global-state identities or fragment are malformed"
        ) from error
    version = _require_ordinal(header["version"], "version")
    outer_update_count = _require_ordinal(
        header["outer_update_count"], "outer_update_count"
    )
    base_versions_value = header["base_versions"]
    if not isinstance(base_versions_value, list):
        raise GlobalStateError("base_versions must be a list")
    base_versions = tuple(
        _require_ordinal(item, "base version") for item in base_versions_value
    )
    sections = header["sections"]
    if not isinstance(sections, list) or len(sections) != 2 + len(base_versions):
        raise GlobalStateError("global-state section count mismatch")
    expected_names = (
        "parameters",
        "outer_state",
        *(f"base-{index:06d}" for index in range(len(base_versions))),
    )
    contents: list[bytes] = []
    offset = 0
    for index, (section, expected_name) in enumerate(zip(sections, expected_names)):
        expected_fields = {"name", "bytes", "sha256"} | (
            {"version"} if index >= 2 else set()
        )
        if (
            not isinstance(section, dict)
            or set(section) != expected_fields
            or section.get("name") != expected_name
        ):
            raise GlobalStateError("global-state section schema mismatch")
        length = _require_ordinal(section["bytes"], "section bytes")
        checksum = _require_hex(section["sha256"], "section sha256")
        content = body[offset : offset + length]
        if len(content) != length or hashlib.sha256(content).hexdigest() != checksum:
            raise GlobalStateError("global-state section checksum mismatch")
        if index >= 2 and section.get("version") != base_versions[index - 2]:
            raise GlobalStateError("global-state base version metadata mismatch")
        contents.append(content)
        offset += length
    if offset != len(body):
        raise GlobalStateError("global-state payload has trailing bytes")
    bases = tuple(
        BaseSnapshot(
            version=base_version,
            parameters=contents[index + 2],
            parameters_sha256=hashlib.sha256(contents[index + 2]).hexdigest(),
        )
        for index, base_version in enumerate(base_versions)
    )
    content_identity = _require_hex(header["content_identity"], "content_identity")
    state = FragmentGlobalState(
        identities=identities,
        descriptor=descriptor,
        version=version,
        outer_update_count=outer_update_count,
        parameters=contents[0],
        outer_state=contents[1],
        base_history=bases,
        base_content_identity=_require_hex(
            header["base_content_identity"], "base_content_identity"
        ),
        content_identity=content_identity,
    )
    if canonical_digest(_semantic_header(state)) != state.content_identity:
        raise GlobalStateError("global-state content identity mismatch")
    return state


class CountingStorageBackend:
    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend
        self.read_calls = 0
        self.publish_calls = 0

    @property
    def root(self) -> Path | None:
        value = getattr(self.backend, "root", None)
        return value if isinstance(value, Path) else None

    def reset_counts(self) -> None:
        self.read_calls = 0
        self.publish_calls = 0

    def publish(
        self, slot: str, payload: bytes, spec: PublicationSpec, **kwargs: Any
    ) -> PublicationRecord:
        self.publish_calls += 1
        return self.backend.publish(slot, payload, spec, **kwargs)

    def read(
        self, slot: str, expectation: ReadExpectation, **kwargs: Any
    ) -> PublishedPayload:
        self.read_calls += 1
        return self.backend.read(slot, expectation, **kwargs)


class GlobalStateStore:
    def __init__(
        self,
        backend: StorageBackend,
        *,
        identities: GlobalStateIdentities,
        descriptors: tuple[FragmentStateDescriptor, ...],
        s_max: int,
    ) -> None:
        if not isinstance(backend, StorageBackend):
            raise GlobalStateError("backend must implement StorageBackend")
        _require_ordinal(s_max, "s_max")
        if not isinstance(descriptors, tuple) or not descriptors:
            raise GlobalStateError("descriptors must be a non-empty tuple")
        if tuple(item.index for item in descriptors) != tuple(range(len(descriptors))):
            raise GlobalStateError(
                "fragment descriptors must be contiguous and ordered"
            )
        if len({item.identity for item in descriptors}) != len(descriptors):
            raise GlobalStateError("fragment descriptor identities must be unique")
        owned_parameters = tuple(
            identity for item in descriptors for identity in item.parameter_identities
        )
        if len(owned_parameters) != len(set(owned_parameters)):
            raise GlobalStateError(
                "parameter identities must have exactly one fragment owner"
            )
        self._backend = backend
        self.identities = identities
        self.descriptors = descriptors
        self.s_max = s_max

    def _descriptor(self, index: int) -> FragmentStateDescriptor:
        index = _require_ordinal(index, "fragment index")
        try:
            return self.descriptors[index]
        except IndexError as error:
            raise GlobalStateError(f"unknown fragment index: {index}") from error

    def _expectation(self, descriptor: FragmentStateDescriptor) -> ReadExpectation:
        return ReadExpectation(
            run_identity=self.identities.run_identity,
            fragment_map_identity=self.identities.fragment_map_identity,
            fragment_identity=descriptor.identity,
            dtype=descriptor.dtype,
            shape=descriptor.shape,
        )

    def _validate_published(
        self, published: PublishedPayload, descriptor: FragmentStateDescriptor
    ) -> FragmentGlobalState:
        state = decode_global_state(published.payload)
        if state.identities != self.identities:
            raise GlobalStateError("compound state frozen identity mismatch")
        if state.descriptor != descriptor:
            raise GlobalStateError("compound state fragment descriptor mismatch")
        if (
            state.version != published.record.version
            or state.version != published.record.sequence
        ):
            raise GlobalStateError(
                "compound state version differs from its visibility record"
            )
        if len(state.base_history) > self.s_max + 1:
            raise GlobalStateError(
                "compound state exceeds the bounded base-retention window"
            )
        if published.record.base_content_identity != state.base_content_identity:
            raise GlobalStateError(
                "compound state base identity differs from its visibility record"
            )
        return state

    def load_fragment(
        self, index: int, *, timeout_seconds: float = 0
    ) -> FragmentGlobalState:
        descriptor = self._descriptor(index)
        published = self._backend.read(
            current_slot(index),
            self._expectation(descriptor),
            timeout_seconds=timeout_seconds,
        )
        return self._validate_published(published, descriptor)

    def load_snapshot(self, *, timeout_seconds: float = 0) -> GlobalSnapshot:
        states = tuple(
            self.load_fragment(index, timeout_seconds=timeout_seconds)
            for index in range(len(self.descriptors))
        )
        version_vector = tuple(state.version for state in states)
        digest = canonical_digest(
            {
                "identities": self.identities.to_dict(),
                "fragments": [state.content_identity for state in states],
                "version_vector": version_vector,
            }
        )
        return GlobalSnapshot(
            states=states, version_vector=version_vector, digest=digest
        )

    def _bootstrap_state(self, fragment: BootstrapFragment) -> FragmentGlobalState:
        base = BaseSnapshot(
            version=0,
            parameters=fragment.parameters,
            parameters_sha256=hashlib.sha256(fragment.parameters).hexdigest(),
        )
        return _make_state(
            identities=self.identities,
            descriptor=fragment.descriptor,
            version=0,
            outer_update_count=0,
            parameters=fragment.parameters,
            outer_state=fragment.outer_state,
            base_history=(base,),
            base_content_identity=_ZERO_IDENTITY,
        )

    def bootstrap(
        self,
        fragments: tuple[BootstrapFragment, ...],
        *,
        interrupt_after_fragments: int | None = None,
    ) -> BootstrapResult:
        if (
            not isinstance(fragments, tuple)
            or tuple(item.descriptor for item in fragments) != self.descriptors
        ):
            raise GlobalStateError(
                "bootstrap must provide the exact ordered frozen fragment set"
            )
        if interrupt_after_fragments is not None:
            _require_ordinal(interrupt_after_fragments, "interrupt_after_fragments")
        expected = tuple(self._bootstrap_state(fragment) for fragment in fragments)
        existing: list[int] = []
        missing: list[int] = []
        for index, state in enumerate(expected):
            try:
                current = self.load_fragment(index, timeout_seconds=0)
            except PublicationNotFound:
                missing.append(index)
                continue
            if current != state:
                raise GlobalStateError(
                    f"conflicting bootstrap state already exists for fragment {index}"
                )
            existing.append(index)
        published_indices: list[int] = []
        for index in missing:
            state = expected[index]

            def refuse_overwrite(
                _slot: str,
                _record: PublicationRecord,
                *,
                current_index: int = index,
                expected_state: FragmentGlobalState = state,
            ) -> None:
                try:
                    visible = self.load_fragment(current_index, timeout_seconds=0)
                except PublicationNotFound:
                    return
                if visible == expected_state:
                    raise GlobalStateError(
                        f"bootstrap publication raced with identical fragment {current_index}"
                    )
                raise GlobalStateError(
                    f"bootstrap publication raced with conflicting fragment {current_index}"
                )

            self._backend.publish(
                current_slot(index),
                encode_global_state(state),
                PublicationSpec(
                    run_identity=self.identities.run_identity,
                    fragment_map_identity=self.identities.fragment_map_identity,
                    fragment_identity=state.descriptor.identity,
                    version=0,
                    sequence=0,
                    dtype=state.descriptor.dtype,
                    shape=state.descriptor.shape,
                    base_content_identity=_ZERO_IDENTITY,
                ),
                visibility_hook=refuse_overwrite,
            )
            published_indices.append(index)
            if (
                interrupt_after_fragments is not None
                and len(published_indices) >= interrupt_after_fragments
            ):
                raise BootstrapInterrupted(
                    f"bootstrap interrupted after {len(published_indices)} fragments"
                )
        return BootstrapResult(
            published_indices=tuple(published_indices),
            existing_indices=tuple(existing),
            snapshot=self.load_snapshot(timeout_seconds=0),
        )

    def publish_successor(
        self,
        index: int,
        *,
        parameters: bytes,
        outer_state: bytes,
        crash_at: str | None = None,
        before_visibility: Callable[[FragmentGlobalState], None] | None = None,
    ) -> FragmentGlobalState:
        _require_bytes(parameters, "parameters")
        _require_bytes(outer_state, "outer_state")
        descriptor = self._descriptor(index)
        visible = self._backend.read(
            current_slot(index), self._expectation(descriptor), timeout_seconds=0
        )
        current = self._validate_published(visible, descriptor)
        version = current.version + 1
        next_base = BaseSnapshot(
            version=version,
            parameters=parameters,
            parameters_sha256=hashlib.sha256(parameters).hexdigest(),
        )
        history = (*current.base_history, next_base)[-(self.s_max + 1) :]
        successor = _make_state(
            identities=self.identities,
            descriptor=descriptor,
            version=version,
            outer_update_count=current.outer_update_count + 1,
            parameters=parameters,
            outer_state=outer_state,
            base_history=history,
            base_content_identity=current.content_identity,
        )

        def require_unchanged_base(_slot: str, _record: PublicationRecord) -> None:
            latest = self._backend.read(
                current_slot(index), self._expectation(descriptor), timeout_seconds=0
            )
            if latest.record.payload_identity != visible.record.payload_identity:
                raise GlobalStateError(
                    f"fragment {index} changed during successor publication"
                )
            if before_visibility is not None:
                before_visibility(successor)
                latest = self._backend.read(
                    current_slot(index),
                    self._expectation(descriptor),
                    timeout_seconds=0,
                )
                if latest.record.payload_identity != visible.record.payload_identity:
                    raise GlobalStateError(
                        f"fragment {index} changed during successor publication hook"
                    )

        self._backend.publish(
            current_slot(index),
            encode_global_state(successor),
            PublicationSpec(
                run_identity=self.identities.run_identity,
                fragment_map_identity=self.identities.fragment_map_identity,
                fragment_identity=descriptor.identity,
                version=version,
                sequence=version,
                dtype=descriptor.dtype,
                shape=descriptor.shape,
                base_content_identity=current.content_identity,
            ),
            visibility_hook=require_unchanged_base,
            crash_at=crash_at,
        )
        return self.load_fragment(index, timeout_seconds=0)

    def inspect_live_set(self) -> LiveSetReport:
        states: list[FragmentGlobalState] = []
        referenced_paths: set[str] = set()
        for index, descriptor in enumerate(self.descriptors):
            published = self._backend.read(
                current_slot(index), self._expectation(descriptor), timeout_seconds=0
            )
            states.append(self._validate_published(published, descriptor))
            referenced_paths.add(published.record.payload_relative_path)
        root = getattr(self._backend, "root", None)
        physical_payloads: int | None = None
        orphan_payloads: int | None = None
        global_head: bool | None = None
        if isinstance(root, Path):
            payload_root = root / "payloads"
            physical = {
                f"payloads/{path.name}"
                for path in payload_root.iterdir()
                if path.is_file()
            }
            physical_payloads = len(physical)
            orphan_payloads = len(physical - referenced_paths)
            global_head = (root / "visibility" / "global-head.json").exists()
        return LiveSetReport(
            current_records=len(states),
            referenced_payloads=len(referenced_paths),
            retained_base_entries=sum(len(state.base_history) for state in states),
            maximum_base_entries=len(states) * (self.s_max + 1),
            physical_payloads=physical_payloads,
            orphan_payload_candidates=orphan_payloads,
            shared_global_head_present=global_head,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class LoadedBootstrapPlan:
    identities: GlobalStateIdentities
    descriptors: tuple[FragmentStateDescriptor, ...]
    fragments: tuple[BootstrapFragment, ...]
    s_max: int
    digest: str


def load_bootstrap_plan(path: Path) -> LoadedBootstrapPlan:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GlobalStateError(
            f"bootstrap plan is not readable JSON: {error}"
        ) from error
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "identities",
        "s_max",
        "fragments",
    }:
        raise GlobalStateError("bootstrap plan schema mismatch")
    if (
        value["schema_version"] != 1
        or not isinstance(value["identities"], dict)
        or not isinstance(value["fragments"], list)
    ):
        raise GlobalStateError("bootstrap plan top-level values are malformed")
    try:
        identities = GlobalStateIdentities(**value["identities"])
    except TypeError as error:
        raise GlobalStateError("bootstrap plan identities are malformed") from error
    s_max = _require_ordinal(value["s_max"], "s_max")
    descriptors: list[FragmentStateDescriptor] = []
    fragments: list[BootstrapFragment] = []
    base = path.resolve().parent
    expected_fragment_fields = {
        "index",
        "identity",
        "dtype",
        "shape",
        "parameter_identities",
        "parameters_path",
        "parameters_sha256",
        "outer_state_path",
        "outer_state_sha256",
    }
    for item in value["fragments"]:
        if not isinstance(item, dict) or set(item) != expected_fragment_fields:
            raise GlobalStateError("bootstrap plan fragment schema mismatch")
        if not isinstance(item["parameter_identities"], list):
            raise GlobalStateError("bootstrap plan parameter identities must be a list")
        descriptor = FragmentStateDescriptor(
            index=item["index"],
            identity=item["identity"],
            dtype=item["dtype"],
            shape=_require_shape(item["shape"]),
            parameter_identities=tuple(item["parameter_identities"]),
        )
        parameters_path = (
            base / _require_text(item["parameters_path"], "parameters_path")
        ).resolve()
        outer_path = (
            base / _require_text(item["outer_state_path"], "outer_state_path")
        ).resolve()
        expected_parameter_sha = _require_hex(
            item["parameters_sha256"], "parameters_sha256"
        )
        expected_outer_sha = _require_hex(
            item["outer_state_sha256"], "outer_state_sha256"
        )
        if (
            file_digest(parameters_path) != expected_parameter_sha
            or file_digest(outer_path) != expected_outer_sha
        ):
            raise GlobalStateError(
                f"bootstrap plan input checksum mismatch for fragment {descriptor.index}"
            )
        descriptors.append(descriptor)
        fragments.append(
            BootstrapFragment(
                descriptor=descriptor,
                parameters=parameters_path.read_bytes(),
                outer_state=outer_path.read_bytes(),
            )
        )
    descriptors_tuple = tuple(descriptors)
    fragments_tuple = tuple(fragments)
    if not descriptors_tuple or tuple(
        item.index for item in descriptors_tuple
    ) != tuple(range(len(descriptors_tuple))):
        raise GlobalStateError(
            "bootstrap plan fragments must be non-empty, contiguous, and ordered"
        )
    return LoadedBootstrapPlan(
        identities=identities,
        descriptors=descriptors_tuple,
        fragments=fragments_tuple,
        s_max=s_max,
        digest=canonical_digest(value),
    )
