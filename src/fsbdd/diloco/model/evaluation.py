from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any

from fsbdd.diloco.protocol.global_commit import (
    AtomicFragmentAuthority,
    AtomicGlobalCommitStore,
    decode_commit_envelope,
)
from fsbdd.diloco.protocol.global_state import (
    FragmentGlobalState,
    GlobalStateIdentities,
    decode_global_state,
    encode_global_state,
)
from fsbdd.diloco.common.identity import canonical_bytes, canonical_digest


class EvaluationSnapshotError(RuntimeError):
    """A frozen evaluation vector or one of its exact payloads is invalid."""


_HEX = frozenset("0123456789abcdef")


@dataclasses.dataclass(slots=True)
class EvaluationAccessAudit:
    """Record every restart-loader filesystem read and reject authority lookup paths."""

    root: Path
    operations: list[dict[str, str]] = dataclasses.field(default_factory=list)
    manifest_reads: int = 0
    frozen_payload_reads: int = 0
    current_authority_reads: int = 0
    latest_resolution_reads: int = 0
    unauthorized_reads: int = 0

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    def read_bytes(self, path: Path, *, purpose: str) -> bytes:
        candidate = Path(path).resolve()
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as error:
            self.unauthorized_reads += 1
            raise EvaluationSnapshotError(
                "evaluation restart loader escaped its frozen root"
            ) from error
        lowered = tuple(part.lower() for part in relative.parts)
        if any("current" in part for part in lowered):
            self.current_authority_reads += 1
            raise EvaluationSnapshotError(
                "evaluation restart loader attempted a current-authority read"
            )
        if any("latest" in part or part == "visibility" for part in lowered):
            self.latest_resolution_reads += 1
            raise EvaluationSnapshotError(
                "evaluation restart loader attempted latest resolution"
            )
        expected_manifest = purpose == "manifest" and relative == Path("manifest.json")
        expected_state = (
            purpose == "frozen_state"
            and len(relative.parts) == 2
            and relative.parts[0] == "fragments"
            and relative.parts[1].endswith(".state")
        )
        if not expected_manifest and not expected_state:
            self.unauthorized_reads += 1
            raise EvaluationSnapshotError(
                "evaluation restart loader attempted an undeclared filesystem read"
            )
        payload = candidate.read_bytes()
        self.operations.append(
            {"purpose": purpose, "relative_path": relative.as_posix()}
        )
        if expected_manifest:
            self.manifest_reads += 1
        else:
            self.frozen_payload_reads += 1
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "instrumentation": "all_restart_loader_reads_through_path_audit",
            "operations": [dict(item) for item in self.operations],
            "manifest_reads": self.manifest_reads,
            "frozen_payload_reads": self.frozen_payload_reads,
            "current_authority_reads": self.current_authority_reads,
            "latest_resolution_reads": self.latest_resolution_reads,
            "unauthorized_reads": self.unauthorized_reads,
        }


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationSnapshotError(f"{field} must be a non-empty string")
    return value


def _hex(value: object, field: str) -> str:
    value = _text(value, field)
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise EvaluationSnapshotError(f"{field} must be a lowercase SHA-256 identity")
    return value


def _ordinal(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EvaluationSnapshotError(f"{field} must be a nonnegative integer")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenEvaluationFragment:
    fragment_index: int
    fragment_identity: str
    version: int
    state_content_identity: str
    authority_identity: str
    state_relative_path: str
    state_bytes: int
    state_sha256: str
    parameters_bytes: int
    parameters_sha256: str

    def __post_init__(self) -> None:
        _ordinal(self.fragment_index, "fragment_index")
        _hex(self.fragment_identity, "fragment_identity")
        _ordinal(self.version, "version")
        _hex(self.state_content_identity, "state_content_identity")
        _hex(self.authority_identity, "authority_identity")
        relative = PurePosixPath(_text(self.state_relative_path, "state_relative_path"))
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != "fragments"
            or not relative.parts[1].endswith(".state")
        ):
            raise EvaluationSnapshotError(
                "state_relative_path must name one immutable fragments/*.state payload"
            )
        if _ordinal(self.state_bytes, "state_bytes") <= 0:
            raise EvaluationSnapshotError("state_bytes must be positive")
        _hex(self.state_sha256, "state_sha256")
        if _ordinal(self.parameters_bytes, "parameters_bytes") <= 0:
            raise EvaluationSnapshotError("parameters_bytes must be positive")
        _hex(self.parameters_sha256, "parameters_sha256")

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _manifest_semantic(
    *,
    identities: GlobalStateIdentities,
    learner_ids: tuple[str, ...],
    policy_identity: str,
    fragments: tuple[FrozenEvaluationFragment, ...],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "frozen_evaluation_snapshot",
        "identities": identities.to_dict(),
        "learner_ids": list(learner_ids),
        "policy_identity": policy_identity,
        "version_vector": [item.version for item in fragments],
        "content_identities": [item.state_content_identity for item in fragments],
        "authority_identities": [item.authority_identity for item in fragments],
        "fragments": [item.to_dict() for item in fragments],
        "load_contract": "manifest_first_exact_named_payloads_no_latest_resolution",
        "steady_state": False,
        "purpose": "evaluation",
    }


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenEvaluationManifest:
    identities: GlobalStateIdentities
    learner_ids: tuple[str, ...]
    policy_identity: str
    fragments: tuple[FrozenEvaluationFragment, ...]
    snapshot_identity: str
    created_unix_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.identities, GlobalStateIdentities):
            raise EvaluationSnapshotError("identities must be GlobalStateIdentities")
        if (
            not isinstance(self.learner_ids, tuple)
            or not self.learner_ids
            or any(not isinstance(item, str) or not item for item in self.learner_ids)
            or len(set(self.learner_ids)) != len(self.learner_ids)
        ):
            raise EvaluationSnapshotError("learner_ids must be a unique nonempty tuple")
        _hex(self.policy_identity, "policy_identity")
        if (
            not isinstance(self.fragments, tuple)
            or not self.fragments
            or any(
                not isinstance(item, FrozenEvaluationFragment)
                for item in self.fragments
            )
            or tuple(item.fragment_index for item in self.fragments)
            != tuple(range(len(self.fragments)))
        ):
            raise EvaluationSnapshotError(
                "fragments must be a complete contiguous ordered tuple"
            )
        _hex(self.snapshot_identity, "snapshot_identity")
        _ordinal(self.created_unix_ns, "created_unix_ns")
        if canonical_digest(self.semantic()) != self.snapshot_identity:
            raise EvaluationSnapshotError("evaluation snapshot identity mismatch")

    @property
    def version_vector(self) -> tuple[int, ...]:
        return tuple(item.version for item in self.fragments)

    @property
    def content_identities(self) -> tuple[str, ...]:
        return tuple(item.state_content_identity for item in self.fragments)

    def semantic(self) -> dict[str, object]:
        return _manifest_semantic(
            identities=self.identities,
            learner_ids=self.learner_ids,
            policy_identity=self.policy_identity,
            fragments=self.fragments,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.semantic(),
            "snapshot_identity": self.snapshot_identity,
            "created_unix_ns": self.created_unix_ns,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenEvaluationPlan:
    """One in-memory capture of every authority before materialization begins."""

    identities: GlobalStateIdentities
    learner_ids: tuple[str, ...]
    policy_identity: str
    authorities: tuple[AtomicFragmentAuthority, ...]
    captured_unix_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.identities, GlobalStateIdentities):
            raise EvaluationSnapshotError("plan identities are invalid")
        if not isinstance(self.learner_ids, tuple) or not self.learner_ids:
            raise EvaluationSnapshotError("plan learner_ids are invalid")
        _hex(self.policy_identity, "plan policy_identity")
        _ordinal(self.captured_unix_ns, "captured_unix_ns")
        if (
            not isinstance(self.authorities, tuple)
            or not self.authorities
            or any(
                not isinstance(item, AtomicFragmentAuthority)
                for item in self.authorities
            )
            or tuple(item.state.descriptor.index for item in self.authorities)
            != tuple(range(len(self.authorities)))
        ):
            raise EvaluationSnapshotError(
                "plan authorities must be a complete contiguous ordered tuple"
            )
        for authority in self.authorities:
            if (
                authority.state.identities != self.identities
                or authority.frontiers.learner_ids != self.learner_ids
                or authority.envelope.policy_identity != self.policy_identity
            ):
                raise EvaluationSnapshotError("captured authority identities differ")

    @classmethod
    def capture(
        cls,
        store: AtomicGlobalCommitStore,
        *,
        clock_ns: Any = time.time_ns,
    ) -> FrozenEvaluationPlan:
        if not isinstance(store, AtomicGlobalCommitStore):
            raise EvaluationSnapshotError("capture requires AtomicGlobalCommitStore")
        captured = store.load_snapshot()
        timestamp = clock_ns()
        return cls(
            identities=store.store.identities,
            learner_ids=store.learner_ids,
            policy_identity=store.policy_identity,
            authorities=captured.authorities,
            captured_unix_ns=_ordinal(timestamp, "captured_unix_ns"),
        )

    @property
    def version_vector(self) -> tuple[int, ...]:
        return tuple(item.version for item in self.authorities)

    @property
    def content_identities(self) -> tuple[str, ...]:
        return tuple(item.content_identity for item in self.authorities)


@dataclasses.dataclass(frozen=True, slots=True)
class LoadedEvaluationSnapshot:
    manifest: FrozenEvaluationManifest
    authorities: tuple[AtomicFragmentAuthority, ...]
    state_payload_bytes: int
    parameter_payload_bytes: int
    access_audit: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            tuple(item.version for item in self.authorities)
            != self.manifest.version_vector
        ):
            raise EvaluationSnapshotError("loaded evaluation version vector differs")
        if (
            tuple(item.content_identity for item in self.authorities)
            != self.manifest.content_identities
        ):
            raise EvaluationSnapshotError("loaded evaluation content identities differ")
        if (
            self.access_audit.get("manifest_reads") != 1
            or self.access_audit.get("frozen_payload_reads")
            != len(self.manifest.fragments)
            or self.access_audit.get("current_authority_reads") != 0
            or self.access_audit.get("latest_resolution_reads") != 0
            or self.access_audit.get("unauthorized_reads") != 0
        ):
            raise EvaluationSnapshotError("evaluation loader access audit failed")


def _fragment_record(
    authority: AtomicFragmentAuthority,
    encoded: bytes,
) -> FrozenEvaluationFragment:
    index = authority.state.descriptor.index
    return FrozenEvaluationFragment(
        fragment_index=index,
        fragment_identity=authority.state.descriptor.identity,
        version=authority.version,
        state_content_identity=authority.content_identity,
        authority_identity=authority.authority_identity,
        state_relative_path=f"fragments/fragment-{index:06d}.state",
        state_bytes=len(encoded),
        state_sha256=hashlib.sha256(encoded).hexdigest(),
        parameters_bytes=len(authority.parameters),
        parameters_sha256=hashlib.sha256(authority.parameters).hexdigest(),
    )


def materialize_evaluation_snapshot(
    plan: FrozenEvaluationPlan,
    destination: Path,
) -> FrozenEvaluationManifest:
    if not isinstance(plan, FrozenEvaluationPlan):
        raise EvaluationSnapshotError("materialization requires FrozenEvaluationPlan")
    root = Path(destination)
    try:
        root.mkdir(parents=True, exist_ok=False)
        (root / "fragments").mkdir()
    except OSError as error:
        raise EvaluationSnapshotError(
            f"evaluation destination must be new: {root}"
        ) from error

    encoded_states = tuple(encode_global_state(item.state) for item in plan.authorities)
    fragments = tuple(
        _fragment_record(authority, encoded)
        for authority, encoded in zip(plan.authorities, encoded_states, strict=True)
    )
    for record, encoded in zip(fragments, encoded_states, strict=True):
        path = root / record.state_relative_path
        try:
            with path.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise EvaluationSnapshotError(
                f"cannot write frozen evaluation state: {path}"
            ) from error

    semantic = _manifest_semantic(
        identities=plan.identities,
        learner_ids=plan.learner_ids,
        policy_identity=plan.policy_identity,
        fragments=fragments,
    )
    manifest = FrozenEvaluationManifest(
        identities=plan.identities,
        learner_ids=plan.learner_ids,
        policy_identity=plan.policy_identity,
        fragments=fragments,
        snapshot_identity=canonical_digest(semantic),
        created_unix_ns=plan.captured_unix_ns,
    )
    manifest_path = root / "manifest.json"
    try:
        with manifest_path.open("xb") as stream:
            stream.write(canonical_bytes(manifest.to_dict()) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise EvaluationSnapshotError(
            f"cannot publish evaluation manifest: {manifest_path}"
        ) from error
    return manifest


def _manifest_from_dict(value: object) -> FrozenEvaluationManifest:
    if not isinstance(value, dict):
        raise EvaluationSnapshotError("evaluation manifest must be an object")
    expected = {
        "schema_version",
        "kind",
        "identities",
        "learner_ids",
        "policy_identity",
        "version_vector",
        "content_identities",
        "authority_identities",
        "fragments",
        "load_contract",
        "steady_state",
        "purpose",
        "snapshot_identity",
        "created_unix_ns",
    }
    if set(value) != expected or value.get("schema_version") != 1:
        raise EvaluationSnapshotError("evaluation manifest schema mismatch")
    if (
        value.get("kind") != "frozen_evaluation_snapshot"
        or value.get("load_contract")
        != "manifest_first_exact_named_payloads_no_latest_resolution"
        or value.get("steady_state") is not False
        or value.get("purpose") != "evaluation"
    ):
        raise EvaluationSnapshotError("evaluation manifest contract mismatch")
    identities_value = value["identities"]
    if not isinstance(identities_value, dict):
        raise EvaluationSnapshotError("evaluation identities must be an object")
    try:
        identities = GlobalStateIdentities(**identities_value)
        fragments = tuple(FrozenEvaluationFragment(**row) for row in value["fragments"])
        manifest = FrozenEvaluationManifest(
            identities=identities,
            learner_ids=tuple(value["learner_ids"]),
            policy_identity=value["policy_identity"],
            fragments=fragments,
            snapshot_identity=value["snapshot_identity"],
            created_unix_ns=value["created_unix_ns"],
        )
    except (KeyError, TypeError) as error:
        raise EvaluationSnapshotError("evaluation manifest is malformed") from error
    if (
        list(manifest.version_vector) != value["version_vector"]
        or list(manifest.content_identities) != value["content_identities"]
        or [item.authority_identity for item in fragments]
        != value["authority_identities"]
    ):
        raise EvaluationSnapshotError("evaluation manifest redundant vectors differ")
    return manifest


def load_evaluation_manifest(
    root: Path,
    *,
    access_audit: EvaluationAccessAudit | None = None,
) -> FrozenEvaluationManifest:
    path = Path(root) / "manifest.json"
    audit = access_audit or EvaluationAccessAudit(Path(root))
    try:
        raw = audit.read_bytes(path, purpose="manifest")
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationSnapshotError(
            f"cannot read evaluation manifest: {path}"
        ) from error
    if canonical_bytes(value) + b"\n" != raw:
        raise EvaluationSnapshotError("evaluation manifest is not canonical")
    return _manifest_from_dict(value)


def load_evaluation_snapshot(
    root: Path,
    *,
    access_audit: EvaluationAccessAudit | None = None,
) -> LoadedEvaluationSnapshot:
    """Freeze the manifest facts first, then load only its exact named states."""

    base = Path(root)
    audit = access_audit or EvaluationAccessAudit(base)
    if audit.root != base.resolve():
        raise EvaluationSnapshotError("evaluation access audit root differs")
    manifest = load_evaluation_manifest(base, access_audit=audit)
    frozen_fragments = manifest.fragments
    authorities: list[AtomicFragmentAuthority] = []
    state_bytes = 0
    parameter_bytes = 0
    for record in frozen_fragments:
        path = base / record.state_relative_path
        try:
            payload = audit.read_bytes(path, purpose="frozen_state")
        except OSError as error:
            raise EvaluationSnapshotError(
                f"cannot read frozen evaluation state: {path}"
            ) from error
        if (
            len(payload) != record.state_bytes
            or hashlib.sha256(payload).hexdigest() != record.state_sha256
        ):
            raise EvaluationSnapshotError("frozen evaluation state integrity mismatch")
        try:
            state: FragmentGlobalState = decode_global_state(payload)
            envelope = decode_commit_envelope(
                state.outer_state,
                identities=state.identities,
                descriptor=state.descriptor,
            )
            authority = AtomicFragmentAuthority(state, envelope)
        except Exception as error:
            raise EvaluationSnapshotError(
                "frozen evaluation compound authority is invalid"
            ) from error
        if (
            state.identities != manifest.identities
            or state.descriptor.index != record.fragment_index
            or state.descriptor.identity != record.fragment_identity
            or state.version != record.version
            or state.content_identity != record.state_content_identity
            or authority.authority_identity != record.authority_identity
            or envelope.frontiers.learner_ids != manifest.learner_ids
            or envelope.policy_identity != manifest.policy_identity
            or len(state.parameters) != record.parameters_bytes
            or hashlib.sha256(state.parameters).hexdigest() != record.parameters_sha256
        ):
            raise EvaluationSnapshotError(
                "frozen evaluation state differs from manifest facts"
            )
        authorities.append(authority)
        state_bytes += len(payload)
        parameter_bytes += len(state.parameters)
    return LoadedEvaluationSnapshot(
        manifest=manifest,
        authorities=tuple(authorities),
        state_payload_bytes=state_bytes,
        parameter_payload_bytes=parameter_bytes,
        access_audit=audit.to_dict(),
    )
