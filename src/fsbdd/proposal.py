from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
import threading
from collections.abc import Callable, Sequence
from fractions import Fraction

from .global_state import FragmentStateDescriptor, GlobalStateIdentities
from .identity import canonical_bytes, canonical_digest
from .storage import (
    PublicationError,
    PublicationNotFound,
    PublicationRecord,
    PublicationSpec,
    PublishedPayload,
    ReadExpectation,
    StorageBackend,
)


class ProposalError(PublicationError):
    """A proposal violates its immutable identity or selection contract."""


_MAGIC = b"FSBDDPR1"
_HEADER_LIMIT = 16 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProposalError(f"{field} must be a non-empty string")
    return value


def _require_hex(value: object, field: str) -> str:
    value = _require_text(value, field)
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise ProposalError(f"{field} must be a lowercase SHA-256 identity")
    return value


def _require_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProposalError(f"{field} must be an integer")
    return value


def _require_ordinal(value: object, field: str) -> int:
    value = _require_integer(value, field)
    if value < 0:
        raise ProposalError(f"{field} must be nonnegative")
    return value


def _require_positive(value: object, field: str) -> int:
    value = _require_integer(value, field)
    if value <= 0:
        raise ProposalError(f"{field} must be positive")
    return value


def _require_bytes(value: object, field: str) -> bytes:
    if not isinstance(value, bytes):
        raise ProposalError(f"{field} must be immutable bytes")
    return value


def _f32(value: float) -> float:
    try:
        rounded = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, TypeError, ValueError) as error:
        raise ProposalError(f"value is not representable as float32: {value!r}") from error
    if not math.isfinite(rounded):
        raise ProposalError(f"value must be finite: {value!r}")
    return rounded


@dataclasses.dataclass(frozen=True, slots=True)
class Proposal:
    proposal_id: str
    identities: GlobalStateIdentities
    learner_id: str
    descriptor: FragmentStateDescriptor
    sequence: int
    base_version: int
    base_content_identity: str
    local_steps: int
    processed_tokens: int
    snapshot_local_step: int
    parameters: bytes
    parameters_sha256: str
    content_identity: str

    def __post_init__(self) -> None:
        _require_text(self.proposal_id, "proposal_id")
        if not isinstance(self.identities, GlobalStateIdentities):
            raise ProposalError("identities must be GlobalStateIdentities")
        _require_text(self.learner_id, "learner_id")
        if not isinstance(self.descriptor, FragmentStateDescriptor):
            raise ProposalError("descriptor must be FragmentStateDescriptor")
        _require_ordinal(self.sequence, "sequence")
        _require_ordinal(self.base_version, "base_version")
        _require_hex(self.base_content_identity, "base_content_identity")
        _require_integer(self.local_steps, "local_steps")
        _require_integer(self.processed_tokens, "processed_tokens")
        _require_integer(self.snapshot_local_step, "snapshot_local_step")
        parameters = _require_bytes(self.parameters, "parameters")
        _require_hex(self.parameters_sha256, "parameters_sha256")
        if hashlib.sha256(parameters).hexdigest() != self.parameters_sha256:
            raise ProposalError("parameter payload checksum mismatch")
        _require_hex(self.content_identity, "content_identity")
        if canonical_digest(_proposal_semantic(self)) != self.content_identity:
            raise ProposalError("proposal content identity mismatch")

    @property
    def payload_bytes(self) -> int:
        return len(self.parameters)

    @property
    def payload_identity(self) -> str:
        return self.parameters_sha256

    @classmethod
    def create(
        cls,
        *,
        proposal_id: str,
        identities: GlobalStateIdentities,
        learner_id: str,
        descriptor: FragmentStateDescriptor,
        sequence: int,
        base_version: int,
        base_content_identity: str,
        local_steps: int,
        processed_tokens: int,
        snapshot_local_step: int,
        parameters: bytes,
    ) -> Proposal:
        parameters = _require_bytes(parameters, "parameters")
        if not isinstance(identities, GlobalStateIdentities):
            raise ProposalError("identities must be GlobalStateIdentities")
        if not isinstance(descriptor, FragmentStateDescriptor):
            raise ProposalError("descriptor must be FragmentStateDescriptor")
        values = {
            "proposal_id": proposal_id,
            "identities": identities,
            "learner_id": learner_id,
            "descriptor": descriptor,
            "sequence": sequence,
            "base_version": base_version,
            "base_content_identity": base_content_identity,
            "local_steps": local_steps,
            "processed_tokens": processed_tokens,
            "snapshot_local_step": snapshot_local_step,
            "parameters": parameters,
            "parameters_sha256": hashlib.sha256(parameters).hexdigest(),
        }
        content_identity = canonical_digest(_proposal_semantic_values(**values))
        return cls(**values, content_identity=content_identity)


def _proposal_semantic_values(
    *,
    proposal_id: str,
    identities: GlobalStateIdentities,
    learner_id: str,
    descriptor: FragmentStateDescriptor,
    sequence: int,
    base_version: int,
    base_content_identity: str,
    local_steps: int,
    processed_tokens: int,
    snapshot_local_step: int,
    parameters: bytes,
    parameters_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "proposal_id": proposal_id,
        "identities": identities.to_dict(),
        "learner_id": learner_id,
        "fragment": descriptor.to_dict(),
        "sequence": sequence,
        "base_version": base_version,
        "base_content_identity": base_content_identity,
        "local_steps": local_steps,
        "processed_tokens": processed_tokens,
        "snapshot_local_step": snapshot_local_step,
        "payload": {
            "kind": "complete_local_parameters",
            "dtype": descriptor.dtype,
            "shape": descriptor.shape,
            "bytes": len(parameters),
            "sha256": parameters_sha256,
        },
    }


def _proposal_semantic(proposal: Proposal) -> dict[str, object]:
    return _proposal_semantic_values(
        proposal_id=proposal.proposal_id,
        identities=proposal.identities,
        learner_id=proposal.learner_id,
        descriptor=proposal.descriptor,
        sequence=proposal.sequence,
        base_version=proposal.base_version,
        base_content_identity=proposal.base_content_identity,
        local_steps=proposal.local_steps,
        processed_tokens=proposal.processed_tokens,
        snapshot_local_step=proposal.snapshot_local_step,
        parameters=proposal.parameters,
        parameters_sha256=proposal.parameters_sha256,
    )


def encode_proposal(proposal: Proposal) -> bytes:
    if not isinstance(proposal, Proposal):
        raise ProposalError("proposal must be a Proposal")
    semantic = _proposal_semantic(proposal)
    if canonical_digest(semantic) != proposal.content_identity:
        raise ProposalError("proposal content identity mismatch")
    header = canonical_bytes({**semantic, "content_identity": proposal.content_identity})
    if len(header) > _HEADER_LIMIT:
        raise ProposalError("proposal header exceeds the format limit")
    return _MAGIC + struct.pack(">Q", len(header)) + header + proposal.parameters


def decode_proposal(payload: bytes) -> Proposal:
    payload = _require_bytes(payload, "payload")
    prefix = len(_MAGIC) + 8
    if len(payload) < prefix or payload[: len(_MAGIC)] != _MAGIC:
        raise ProposalError("proposal payload magic mismatch")
    header_bytes = struct.unpack(">Q", payload[len(_MAGIC) : prefix])[0]
    if header_bytes > _HEADER_LIMIT or len(payload) < prefix + header_bytes:
        raise ProposalError("proposal header length is invalid")
    raw_header = payload[prefix : prefix + header_bytes]
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProposalError("proposal header is not complete JSON") from error
    if not isinstance(header, dict) or canonical_bytes(header) != raw_header:
        raise ProposalError("proposal header is not canonical")
    expected = {
        "schema_version",
        "proposal_id",
        "identities",
        "learner_id",
        "fragment",
        "sequence",
        "base_version",
        "base_content_identity",
        "local_steps",
        "processed_tokens",
        "snapshot_local_step",
        "payload",
        "content_identity",
    }
    if set(header) != expected or header.get("schema_version") != 1:
        raise ProposalError("proposal header schema mismatch")
    identities_value = header["identities"]
    fragment_value = header["fragment"]
    payload_value = header["payload"]
    if not isinstance(identities_value, dict) or set(identities_value) != {
        "run_identity",
        "config_identity",
        "model_identity",
        "fragment_map_identity",
    }:
        raise ProposalError("proposal identity schema mismatch")
    if not isinstance(fragment_value, dict) or set(fragment_value) != {
        "index",
        "identity",
        "dtype",
        "shape",
        "parameter_identities",
    }:
        raise ProposalError("proposal fragment schema mismatch")
    if not isinstance(fragment_value["shape"], list) or not isinstance(
        fragment_value["parameter_identities"], list
    ):
        raise ProposalError("proposal fragment tuple fields must be JSON lists")
    if not isinstance(payload_value, dict) or set(payload_value) != {
        "kind",
        "dtype",
        "shape",
        "bytes",
        "sha256",
    }:
        raise ProposalError("proposal parameter payload schema mismatch")
    if payload_value["kind"] != "complete_local_parameters":
        raise ProposalError("proposal payload is not a complete local parameter value")
    try:
        identities = GlobalStateIdentities(**identities_value)
        descriptor = FragmentStateDescriptor(
            index=fragment_value["index"],
            identity=fragment_value["identity"],
            dtype=fragment_value["dtype"],
            shape=tuple(fragment_value["shape"]),
            parameter_identities=tuple(fragment_value["parameter_identities"]),
        )
    except (KeyError, TypeError, ValueError, PublicationError) as error:
        raise ProposalError("proposal identities or fragment are malformed") from error
    parameters = payload[prefix + header_bytes :]
    if (
        payload_value["dtype"] != descriptor.dtype
        or payload_value["shape"] != list(descriptor.shape)
        or payload_value["bytes"] != len(parameters)
        or payload_value["sha256"] != hashlib.sha256(parameters).hexdigest()
    ):
        raise ProposalError("proposal parameter payload integrity mismatch")
    try:
        return Proposal(
            proposal_id=header["proposal_id"],
            identities=identities,
            learner_id=header["learner_id"],
            descriptor=descriptor,
            sequence=header["sequence"],
            base_version=header["base_version"],
            base_content_identity=header["base_content_identity"],
            local_steps=header["local_steps"],
            processed_tokens=header["processed_tokens"],
            snapshot_local_step=header["snapshot_local_step"],
            parameters=parameters,
            parameters_sha256=payload_value["sha256"],
            content_identity=header["content_identity"],
        )
    except (TypeError, ValueError) as error:
        raise ProposalError("proposal fields are malformed") from error


def proposal_slot(learner_id: str, fragment_index: int) -> str:
    learner_id = _require_text(learner_id, "learner_id")
    fragment_index = _require_ordinal(fragment_index, "fragment_index")
    learner_identity = hashlib.sha256(learner_id.encode("utf-8")).hexdigest()
    return f"proposal-{learner_identity}-{fragment_index:06d}"


BeforeVisibility = Callable[[str, Proposal], None]


class ProposalStore:
    def __init__(
        self,
        backend: StorageBackend,
        *,
        identities: GlobalStateIdentities,
        descriptors: tuple[FragmentStateDescriptor, ...],
        learner_ids: tuple[str, ...],
        maximum_local_steps: int,
        maximum_processed_tokens: int,
    ) -> None:
        if not isinstance(backend, StorageBackend):
            raise ProposalError("backend must implement StorageBackend")
        if not isinstance(descriptors, tuple) or not descriptors:
            raise ProposalError("descriptors must be a non-empty tuple")
        if tuple(item.index for item in descriptors) != tuple(range(len(descriptors))):
            raise ProposalError("fragment descriptors must be contiguous and ordered")
        if len({item.identity for item in descriptors}) != len(descriptors):
            raise ProposalError("fragment descriptor identities must be unique")
        if not isinstance(learner_ids, tuple) or not learner_ids:
            raise ProposalError("learner_ids must be a non-empty tuple")
        for learner_id in learner_ids:
            _require_text(learner_id, "learner_id")
        if len(set(learner_ids)) != len(learner_ids):
            raise ProposalError("learner_ids must be unique")
        self._backend = backend
        self.identities = identities
        self.descriptors = descriptors
        self.learner_ids = learner_ids
        self.maximum_local_steps = _require_positive(
            maximum_local_steps, "maximum_local_steps"
        )
        self.maximum_processed_tokens = _require_positive(
            maximum_processed_tokens, "maximum_processed_tokens"
        )
        self._in_flight = tuple(
            tuple(threading.Lock() for _ in descriptors) for _ in learner_ids
        )

    def _descriptor(self, index: int) -> FragmentStateDescriptor:
        index = _require_ordinal(index, "fragment_index")
        try:
            return self.descriptors[index]
        except IndexError as error:
            raise ProposalError(f"unknown fragment index: {index}") from error

    def _validate_address(
        self, learner_id: str, fragment_index: int
    ) -> FragmentStateDescriptor:
        if learner_id not in self.learner_ids:
            raise ProposalError(f"unknown learner identity: {learner_id}")
        return self._descriptor(fragment_index)

    def _expectation(self, descriptor: FragmentStateDescriptor) -> ReadExpectation:
        return ReadExpectation(
            run_identity=self.identities.run_identity,
            fragment_map_identity=self.identities.fragment_map_identity,
            fragment_identity=descriptor.identity,
            dtype=descriptor.dtype,
            shape=descriptor.shape,
        )

    def _validate_published(
        self,
        published: PublishedPayload,
        learner_id: str,
        descriptor: FragmentStateDescriptor,
    ) -> Proposal:
        proposal = decode_proposal(published.payload)
        record = published.record
        if proposal.identities != self.identities:
            raise ProposalError("compound proposal frozen identity mismatch")
        if proposal.learner_id != learner_id:
            raise ProposalError("compound proposal learner identity mismatch")
        if proposal.descriptor != descriptor:
            raise ProposalError("compound proposal fragment descriptor mismatch")
        if proposal.sequence != record.sequence:
            raise ProposalError("proposal sequence differs from visibility record")
        if proposal.base_version != record.version:
            raise ProposalError("proposal base version differs from visibility record")
        if proposal.base_content_identity != record.base_content_identity:
            raise ProposalError("proposal base identity differs from visibility record")
        return proposal

    def load_latest(
        self,
        learner_id: str,
        fragment_index: int,
        *,
        timeout_seconds: float = 0,
    ) -> Proposal:
        descriptor = self._validate_address(learner_id, fragment_index)
        published = self._backend.read(
            proposal_slot(learner_id, fragment_index),
            self._expectation(descriptor),
            timeout_seconds=timeout_seconds,
        )
        return self._validate_published(published, learner_id, descriptor)

    def discover_latest(self, *, timeout_seconds: float = 0) -> tuple[Proposal, ...]:
        discovered: list[Proposal] = []
        for learner_id in self.learner_ids:
            for descriptor in self.descriptors:
                try:
                    proposal = self.load_latest(
                        learner_id,
                        descriptor.index,
                        timeout_seconds=timeout_seconds,
                    )
                except PublicationNotFound:
                    continue
                discovered.append(proposal)
        return tuple(discovered)

    def _validate_for_publish(self, proposal: Proposal) -> None:
        if not isinstance(proposal, Proposal):
            raise ProposalError("proposal must be a Proposal")
        descriptor = self._validate_address(
            proposal.learner_id, proposal.descriptor.index
        )
        if proposal.identities != self.identities:
            raise ProposalError("proposal frozen identity mismatch")
        if proposal.descriptor != descriptor:
            raise ProposalError("proposal fragment descriptor mismatch")
        if proposal.local_steps <= 0:
            raise ProposalError("proposal local_steps must be positive")
        if proposal.processed_tokens <= 0:
            raise ProposalError("proposal processed_tokens must be positive")
        if proposal.snapshot_local_step <= 0:
            raise ProposalError("proposal snapshot_local_step must be positive")
        if proposal.local_steps > self.maximum_local_steps:
            raise ProposalError("proposal local_steps exceeds the frozen maximum")
        if proposal.processed_tokens > self.maximum_processed_tokens:
            raise ProposalError("proposal processed_tokens exceeds the frozen maximum")

    def _in_flight_lock(self, learner_id: str, fragment_index: int) -> threading.Lock:
        return self._in_flight[
            self.learner_ids.index(learner_id)
        ][fragment_index]

    def publish(
        self,
        proposal: Proposal,
        *,
        before_visibility: BeforeVisibility | None = None,
        crash_at: str | None = None,
    ) -> Proposal:
        self._validate_for_publish(proposal)
        slot = proposal_slot(proposal.learner_id, proposal.descriptor.index)
        in_flight = self._in_flight_lock(
            proposal.learner_id, proposal.descriptor.index
        )
        if not in_flight.acquire(blocking=False):
            raise ProposalError(
                "a proposal publication is already in flight for this learner and fragment"
            )
        try:
            try:
                current = self.load_latest(
                    proposal.learner_id,
                    proposal.descriptor.index,
                    timeout_seconds=0,
                )
            except PublicationNotFound:
                current = None
            if current is not None:
                if current.sequence > proposal.sequence:
                    raise ProposalError("proposal sequence would regress latest")
                if current.sequence == proposal.sequence:
                    if current.content_identity == proposal.content_identity:
                        return current
                    raise ProposalError(
                        "conflicting proposal content at the same sequence"
                    )

            encoded = encode_proposal(proposal)

            def visibility_hook(
                visibility_slot: str, _record: PublicationRecord
            ) -> None:
                if before_visibility is not None:
                    before_visibility(visibility_slot, proposal)

            record = self._backend.publish(
                slot,
                encoded,
                PublicationSpec(
                    run_identity=self.identities.run_identity,
                    fragment_map_identity=self.identities.fragment_map_identity,
                    fragment_identity=proposal.descriptor.identity,
                    version=proposal.base_version,
                    sequence=proposal.sequence,
                    dtype=proposal.descriptor.dtype,
                    shape=proposal.descriptor.shape,
                    base_content_identity=proposal.base_content_identity,
                ),
                visibility_hook=visibility_hook,
                crash_at=crash_at,
            )
            if record.sequence != proposal.sequence:
                raise ProposalError("storage returned the wrong proposal sequence")
            return proposal
        finally:
            in_flight.release()


@dataclasses.dataclass(frozen=True, slots=True)
class ConsumptionFrontier:
    last_sequence: int = -1
    last_base_version: int = -1

    def __post_init__(self) -> None:
        for field, value in (
            ("last_sequence", self.last_sequence),
            ("last_base_version", self.last_base_version),
        ):
            if _require_integer(value, field) < -1:
                raise ProposalError(f"{field} must be at least -1")


@dataclasses.dataclass(frozen=True, slots=True)
class ConsumptionFrontiers:
    identities: GlobalStateIdentities
    descriptor: FragmentStateDescriptor
    learner_ids: tuple[str, ...]
    entries: tuple[ConsumptionFrontier, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identities, GlobalStateIdentities):
            raise ProposalError("frontier identities must be GlobalStateIdentities")
        if not isinstance(self.descriptor, FragmentStateDescriptor):
            raise ProposalError("frontier descriptor must be FragmentStateDescriptor")
        if not isinstance(self.learner_ids, tuple) or not self.learner_ids:
            raise ProposalError("frontier learner_ids must be a non-empty tuple")
        for learner_id in self.learner_ids:
            _require_text(learner_id, "frontier learner_id")
        if len(set(self.learner_ids)) != len(self.learner_ids):
            raise ProposalError("frontier learner_ids must be unique")
        if len(self.entries) != len(self.learner_ids) or any(
            not isinstance(item, ConsumptionFrontier) for item in self.entries
        ):
            raise ProposalError("frontier entries must match the frozen learners")

    @classmethod
    def empty(
        cls,
        *,
        identities: GlobalStateIdentities,
        descriptor: FragmentStateDescriptor,
        learner_ids: tuple[str, ...],
    ) -> ConsumptionFrontiers:
        return cls(
            identities=identities,
            descriptor=descriptor,
            learner_ids=learner_ids,
            entries=tuple(ConsumptionFrontier() for _ in learner_ids),
        )

    def for_learner(self, learner_id: str) -> ConsumptionFrontier:
        try:
            return self.entries[self.learner_ids.index(learner_id)]
        except ValueError as error:
            raise ProposalError(f"unknown frontier learner: {learner_id}") from error


@dataclasses.dataclass(frozen=True, slots=True)
class RetainedBaseIdentity:
    version: int
    content_identity: str

    def __post_init__(self) -> None:
        _require_ordinal(self.version, "retained base version")
        _require_hex(self.content_identity, "retained base content_identity")


@dataclasses.dataclass(frozen=True, slots=True)
class EligibilityPolicy:
    identities: GlobalStateIdentities
    descriptor: FragmentStateDescriptor
    learner_ids: tuple[str, ...]
    current_version: int
    retained_bases: tuple[RetainedBaseIdentity, ...]
    s_max: int
    q: int
    q_fresh: int
    max_contributors: int
    maximum_local_steps: int
    maximum_processed_tokens: int
    lambda_s: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.identities, GlobalStateIdentities):
            raise ProposalError("policy identities must be GlobalStateIdentities")
        if not isinstance(self.descriptor, FragmentStateDescriptor):
            raise ProposalError("policy descriptor must be FragmentStateDescriptor")
        if not isinstance(self.learner_ids, tuple) or not self.learner_ids:
            raise ProposalError("policy learner_ids must be a non-empty tuple")
        for learner_id in self.learner_ids:
            _require_text(learner_id, "policy learner_id")
        if len(set(self.learner_ids)) != len(self.learner_ids):
            raise ProposalError("policy learner_ids must be unique")
        current_version = _require_ordinal(self.current_version, "current_version")
        s_max = _require_ordinal(self.s_max, "s_max")
        q = _require_positive(self.q, "q")
        q_fresh = _require_ordinal(self.q_fresh, "q_fresh")
        max_contributors = _require_positive(
            self.max_contributors, "max_contributors"
        )
        if not 0 <= q_fresh <= q <= max_contributors <= len(self.learner_ids):
            raise ProposalError(
                "require 0 <= q_fresh <= q <= max_contributors <= learner count"
            )
        _require_positive(self.maximum_local_steps, "maximum_local_steps")
        _require_positive(self.maximum_processed_tokens, "maximum_processed_tokens")
        if not isinstance(self.retained_bases, tuple) or not self.retained_bases:
            raise ProposalError("retained_bases must be a non-empty tuple")
        if any(
            not isinstance(item, RetainedBaseIdentity) for item in self.retained_bases
        ):
            raise ProposalError("retained_bases must contain RetainedBaseIdentity values")
        versions = tuple(item.version for item in self.retained_bases)
        if len(set(versions)) != len(versions) or tuple(sorted(versions)) != versions:
            raise ProposalError("retained base versions must be unique and ordered")
        if any(version > current_version for version in versions):
            raise ProposalError("retained base version cannot be in the future")
        if len(versions) > s_max + 1:
            raise ProposalError("retained base window exceeds s_max + 1")
        if not math.isfinite(self.lambda_s) or self.lambda_s < 0:
            raise ProposalError("lambda_s must be finite and nonnegative")

    def retained_identity(self, version: int) -> str | None:
        for item in self.retained_bases:
            if item.version == version:
                return item.content_identity
        return None


@dataclasses.dataclass(frozen=True, slots=True)
class SelectionResult:
    ready: bool
    selected: tuple[Proposal, ...]
    rejections: dict[str, str]


def _identity_reason(proposal: Proposal, policy: EligibilityPolicy) -> str | None:
    if proposal.identities.run_identity != policy.identities.run_identity:
        return "run_identity_mismatch"
    if proposal.identities.config_identity != policy.identities.config_identity:
        return "config_identity_mismatch"
    if proposal.identities.model_identity != policy.identities.model_identity:
        return "model_identity_mismatch"
    if (
        proposal.identities.fragment_map_identity
        != policy.identities.fragment_map_identity
    ):
        return "fragment_map_identity_mismatch"
    if proposal.descriptor.index != policy.descriptor.index:
        return "fragment_identity_mismatch"
    if proposal.descriptor.identity != policy.descriptor.identity:
        return "fragment_identity_mismatch"
    if proposal.descriptor.dtype != policy.descriptor.dtype:
        return "dtype_mismatch"
    if proposal.descriptor.shape != policy.descriptor.shape:
        return "shape_mismatch"
    if (
        proposal.descriptor.parameter_identities
        != policy.descriptor.parameter_identities
    ):
        return "parameter_identity_mismatch"
    if proposal.learner_id not in policy.learner_ids:
        return "learner_identity_mismatch"
    if (
        hashlib.sha256(proposal.parameters).hexdigest()
        != proposal.parameters_sha256
        or canonical_digest(_proposal_semantic(proposal)) != proposal.content_identity
    ):
        return "integrity_mismatch"
    return None


def _eligibility_reason(
    proposal: Proposal,
    frontier: ConsumptionFrontier,
    policy: EligibilityPolicy,
) -> str | None:
    reason = _identity_reason(proposal, policy)
    if reason is not None:
        return reason
    if proposal.sequence <= frontier.last_sequence:
        return "consumed_sequence"
    if proposal.base_version <= frontier.last_base_version:
        return "consumed_base"
    if proposal.local_steps <= 0:
        return "nonpositive_local_steps"
    if proposal.processed_tokens <= 0:
        return "nonpositive_tokens"
    if proposal.snapshot_local_step <= 0:
        return "nonpositive_snapshot_step"
    if proposal.local_steps > policy.maximum_local_steps:
        return "local_steps_exceeded"
    if proposal.processed_tokens > policy.maximum_processed_tokens:
        return "processed_tokens_exceeded"
    staleness = policy.current_version - proposal.base_version
    if staleness < 0:
        return "future_base"
    if staleness > policy.s_max:
        return "too_stale"
    retained_identity = policy.retained_identity(proposal.base_version)
    if retained_identity is None:
        return "base_not_retained"
    if proposal.base_content_identity != retained_identity:
        return "base_identity_mismatch"
    return None


def _candidate_key(
    proposal: Proposal, policy: EligibilityPolicy
) -> tuple[int | Fraction | str, ...]:
    staleness = policy.current_version - proposal.base_version
    effective_tokens = Fraction(proposal.processed_tokens, 1) / (
        Fraction(1, 1) + Fraction.from_float(policy.lambda_s) * staleness
    )
    return (
        staleness,
        -effective_tokens,
        -proposal.sequence,
        proposal.learner_id,
        proposal.proposal_id,
    )


def select_candidates(
    proposals: Sequence[Proposal],
    frontiers: ConsumptionFrontiers,
    policy: EligibilityPolicy,
) -> SelectionResult:
    if isinstance(proposals, (str, bytes)) or not isinstance(proposals, Sequence):
        raise ProposalError("proposals must be a sequence")
    if not isinstance(frontiers, ConsumptionFrontiers):
        raise ProposalError("frontiers must be ConsumptionFrontiers")
    if frontiers.identities != policy.identities:
        raise ProposalError("frontier and policy frozen identities differ")
    if frontiers.descriptor != policy.descriptor:
        raise ProposalError("frontier and policy fragment descriptors differ")
    if frontiers.learner_ids != policy.learner_ids:
        raise ProposalError("frontier and policy learner identities differ")
    rejections: dict[str, str] = {}
    eligible_by_learner: dict[str, list[Proposal]] = {}
    seen_ids: set[str] = set()
    for proposal in proposals:
        if not isinstance(proposal, Proposal):
            raise ProposalError("candidate must be a Proposal")
        if proposal.proposal_id in seen_ids:
            raise ProposalError(f"duplicate proposal identity: {proposal.proposal_id}")
        seen_ids.add(proposal.proposal_id)
        frontier = (
            frontiers.for_learner(proposal.learner_id)
            if proposal.learner_id in frontiers.learner_ids
            else ConsumptionFrontier()
        )
        reason = _eligibility_reason(proposal, frontier, policy)
        if reason is not None:
            rejections[proposal.proposal_id] = reason
            continue
        eligible_by_learner.setdefault(proposal.learner_id, []).append(proposal)

    distinct: list[Proposal] = []
    for learner_id in sorted(eligible_by_learner):
        ordered = sorted(
            eligible_by_learner[learner_id],
            key=lambda item: _candidate_key(item, policy),
        )
        distinct.append(ordered[0])
        for duplicate in ordered[1:]:
            rejections[duplicate.proposal_id] = "duplicate_learner"
    ordered_distinct = sorted(
        distinct, key=lambda item: _candidate_key(item, policy)
    )
    fresh_count = sum(
        item.base_version == policy.current_version for item in ordered_distinct
    )
    ready = len(ordered_distinct) >= policy.q and fresh_count >= policy.q_fresh
    selected = (
        tuple(ordered_distinct[: policy.max_contributors]) if ready else tuple()
    )
    return SelectionResult(ready=ready, selected=selected, rejections=rejections)


def commit_consumption(
    frontiers: ConsumptionFrontiers,
    selected: Sequence[Proposal],
    publication_succeeded: bool,
) -> ConsumptionFrontiers:
    if not isinstance(publication_succeeded, bool):
        raise ProposalError("publication_succeeded must be boolean")
    if not publication_succeeded:
        return frontiers
    entries = list(frontiers.entries)
    seen_learners: set[str] = set()
    for proposal in selected:
        if proposal.identities != frontiers.identities:
            raise ProposalError("selected proposal and frontier identities differ")
        if proposal.descriptor != frontiers.descriptor:
            raise ProposalError("selected proposal and frontier fragment descriptors differ")
        if proposal.learner_id in seen_learners:
            raise ProposalError("selected proposals contain a duplicate learner")
        seen_learners.add(proposal.learner_id)
        try:
            index = frontiers.learner_ids.index(proposal.learner_id)
        except ValueError as error:
            raise ProposalError(
                f"selected proposal has unknown learner: {proposal.learner_id}"
            ) from error
        previous = entries[index]
        if proposal.sequence <= previous.last_sequence:
            raise ProposalError("cannot commit a consumed proposal sequence")
        if proposal.base_version <= previous.last_base_version:
            raise ProposalError("cannot commit an already consumed base")
        entries[index] = ConsumptionFrontier(
            last_sequence=proposal.sequence,
            last_base_version=proposal.base_version,
        )
    return ConsumptionFrontiers(
        frontiers.identities,
        frontiers.descriptor,
        frontiers.learner_ids,
        tuple(entries),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateWeight:
    proposal_id: str
    learner_id: str
    processed_tokens: int
    staleness: int
    raw_weight: float
    normalized_weight: float

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def compute_candidate_weights(
    proposals: Sequence[Proposal],
    *,
    current_version: int,
    lambda_s: float = 1.0,
) -> tuple[CandidateWeight, ...]:
    current_version = _require_ordinal(current_version, "current_version")
    if not proposals:
        raise ProposalError("weight candidates must not be empty")
    lambda_value = _f32(lambda_s)
    if lambda_value < 0:
        raise ProposalError("lambda_s must be nonnegative")
    facts: list[tuple[Proposal, int, float]] = []
    for proposal in proposals:
        if proposal.processed_tokens <= 0:
            raise ProposalError("weight candidate tokens must be positive")
        staleness = current_version - proposal.base_version
        if staleness < 0:
            raise ProposalError("weight candidate has a future base")
        tokens = _f32(proposal.processed_tokens)
        denominator = _f32(1.0 + _f32(lambda_value * staleness))
        facts.append((proposal, staleness, _f32(tokens / denominator)))
    total = _f32(0.0)
    for _, _, raw_weight in facts:
        total = _f32(total + raw_weight)
    if total <= 0:
        raise ProposalError("weight mass must be positive")
    return tuple(
        CandidateWeight(
            proposal_id=proposal.proposal_id,
            learner_id=proposal.learner_id,
            processed_tokens=proposal.processed_tokens,
            staleness=staleness,
            raw_weight=raw_weight,
            normalized_weight=_f32(raw_weight / total),
        )
        for proposal, staleness, raw_weight in facts
    )
