from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from collections.abc import Callable

from fsbdd.diloco.protocol.global_state import (
    BootstrapFragment,
    FragmentGlobalState,
    FragmentStateDescriptor,
    GlobalStateIdentities,
    GlobalStateStore,
)
from fsbdd.diloco.common.identity import canonical_bytes, canonical_digest
from fsbdd.diloco.protocol.proposal import (
    CandidateWeight,
    ConsumptionFrontier,
    ConsumptionFrontiers,
    Proposal,
    ProposalError,
    commit_consumption,
)
from fsbdd.diloco.syncer.readiness import FrozenSelection
from fsbdd.diloco.protocol.storage import PublicationRecord


class AtomicCommitError(RuntimeError):
    """A compound authority commit violates version or consumption semantics."""


_MAGIC = b"FSBDDCM1"
_HEADER_LIMIT = 16 * 1024 * 1024
_ZERO_IDENTITY = "0" * 64
_HEX = frozenset("0123456789abcdef")


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AtomicCommitError(f"{field} must be a non-empty string")
    return value


def _require_hex(value: object, field: str) -> str:
    value = _require_text(value, field)
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise AtomicCommitError(f"{field} must be a lowercase SHA-256 identity")
    return value


def _require_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AtomicCommitError(f"{field} must be an integer")
    return value


def _require_nonnegative(value: object, field: str) -> int:
    value = _require_integer(value, field)
    if value < 0:
        raise AtomicCommitError(f"{field} must be nonnegative")
    return value


def _require_bytes(value: object, field: str) -> bytes:
    if not isinstance(value, bytes):
        raise AtomicCommitError(f"{field} must be immutable bytes")
    return value


def _f32(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AtomicCommitError(f"{field} must be numeric")
    try:
        result = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, ValueError) as error:
        raise AtomicCommitError(f"{field} is not representable as float32") from error
    if not math.isfinite(result):
        raise AtomicCommitError(f"{field} must be finite")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class CommittedContribution:
    proposal_id: str
    content_identity: str
    learner_id: str
    sequence: int
    base_version: int
    base_content_identity: str
    processed_tokens: int
    staleness: int
    normalized_weight: float
    parameters_sha256: str
    payload_bytes: int

    def __post_init__(self) -> None:
        _require_text(self.proposal_id, "proposal_id")
        _require_hex(self.content_identity, "proposal content identity")
        _require_text(self.learner_id, "learner_id")
        _require_nonnegative(self.sequence, "proposal sequence")
        _require_nonnegative(self.base_version, "proposal base version")
        _require_hex(self.base_content_identity, "proposal base content identity")
        if _require_integer(self.processed_tokens, "processed_tokens") <= 0:
            raise AtomicCommitError("processed_tokens must be positive")
        _require_nonnegative(self.staleness, "staleness")
        if not isinstance(self.normalized_weight, float) or _f32(
            self.normalized_weight, "normalized_weight"
        ) != self.normalized_weight:
            raise AtomicCommitError(
                "normalized_weight must be a canonical float32 value"
            )
        if self.normalized_weight < 0:
            raise AtomicCommitError("normalized_weight must be nonnegative")
        _require_hex(self.parameters_sha256, "proposal parameters sha256")
        if _require_integer(self.payload_bytes, "proposal payload_bytes") <= 0:
            raise AtomicCommitError("proposal payload_bytes must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            **dataclasses.asdict(self),
            "normalized_weight": _f32(self.normalized_weight, "normalized_weight"),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class CommitEnvelope:
    outer_optimizer_state: bytes
    outer_optimizer_state_sha256: str
    frontiers: ConsumptionFrontiers
    selected: tuple[CommittedContribution, ...]
    policy_identity: str
    previous_authority_identity: str
    selection_identity: str
    update_identity: str
    envelope_identity: str

    def __post_init__(self) -> None:
        self._validate_structure()
        if (
            hashlib.sha256(self.outer_optimizer_state).hexdigest()
            != self.outer_optimizer_state_sha256
        ):
            raise AtomicCommitError("outer optimizer state checksum mismatch")

    def _validate_structure(self) -> None:
        _require_bytes(self.outer_optimizer_state, "outer_optimizer_state")
        _require_hex(
            self.outer_optimizer_state_sha256,
            "outer_optimizer_state_sha256",
        )
        if not isinstance(self.frontiers, ConsumptionFrontiers):
            raise AtomicCommitError("frontiers must be ConsumptionFrontiers")
        if not isinstance(self.selected, tuple) or any(
            not isinstance(item, CommittedContribution) for item in self.selected
        ):
            raise AtomicCommitError(
                "selected must be a tuple of CommittedContribution values"
            )
        _require_hex(self.policy_identity, "policy_identity")
        _require_hex(self.previous_authority_identity, "previous_authority_identity")
        _require_hex(self.selection_identity, "selection_identity")
        _require_hex(self.update_identity, "update_identity")
        _require_hex(self.envelope_identity, "envelope_identity")
        learners = [item.learner_id for item in self.selected]
        proposals = [item.proposal_id for item in self.selected]
        if len(set(learners)) != len(learners):
            raise AtomicCommitError(
                "selected contributions contain a duplicate learner"
            )
        if len(set(proposals)) != len(proposals):
            raise AtomicCommitError(
                "selected contributions contain a duplicate proposal"
            )
        if self.selected:
            if self.previous_authority_identity == _ZERO_IDENTITY:
                raise AtomicCommitError(
                    "a nonempty commit requires previous authority identity"
                )
            if self.selection_identity == _ZERO_IDENTITY:
                raise AtomicCommitError("a nonempty commit requires selection identity")
            if self.update_identity == _ZERO_IDENTITY:
                raise AtomicCommitError("a nonempty commit requires update identity")
            weight_sum = _f32(0.0, "weight sum")
            for item in self.selected:
                try:
                    frontier = self.frontiers.for_learner(item.learner_id)
                except ProposalError as error:
                    raise AtomicCommitError(
                        "selected learner is absent from the committed frontier"
                    ) from error
                if (
                    frontier.last_sequence != item.sequence
                    or frontier.last_base_version != item.base_version
                ):
                    raise AtomicCommitError(
                        "selected contribution differs from committed frontier"
                    )
                weight_sum = _f32(
                    weight_sum + _f32(item.normalized_weight, "normalized_weight"),
                    "weight sum",
                )
            if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=2e-6):
                raise AtomicCommitError("selected float32 weights must sum to one")
        elif (
            self.previous_authority_identity != _ZERO_IDENTITY
            or self.selection_identity != _ZERO_IDENTITY
            or self.update_identity != _ZERO_IDENTITY
        ):
            raise AtomicCommitError(
                "an empty bootstrap envelope requires zero previous selection and update identities"
            )
        if canonical_digest(_envelope_semantic(self)) != self.envelope_identity:
            raise AtomicCommitError("commit envelope identity mismatch")


def _verified_commit_envelope(
    *,
    outer_optimizer_state: bytes,
    outer_optimizer_state_sha256: str,
    frontiers: ConsumptionFrontiers,
    selected: tuple[CommittedContribution, ...],
    policy_identity: str,
    previous_authority_identity: str,
    selection_identity: str,
    update_identity: str,
) -> CommitEnvelope:
    semantic = _envelope_semantic_values(
        outer_optimizer_state=outer_optimizer_state,
        frontiers=frontiers,
        selected=selected,
        policy_identity=policy_identity,
        previous_authority_identity=previous_authority_identity,
        selection_identity=selection_identity,
        update_identity=update_identity,
        outer_optimizer_state_sha256=outer_optimizer_state_sha256,
    )
    value = object.__new__(CommitEnvelope)
    fields: dict[str, object] = {
        "outer_optimizer_state": outer_optimizer_state,
        "outer_optimizer_state_sha256": outer_optimizer_state_sha256,
        "frontiers": frontiers,
        "selected": selected,
        "policy_identity": policy_identity,
        "previous_authority_identity": previous_authority_identity,
        "selection_identity": selection_identity,
        "update_identity": update_identity,
        "envelope_identity": canonical_digest(semantic),
    }
    for name, field_value in fields.items():
        object.__setattr__(value, name, field_value)
    value._validate_structure()
    return value


def _frontier_rows(frontiers: ConsumptionFrontiers) -> list[dict[str, object]]:
    return [
        {
            "learner_id": learner_id,
            "last_sequence": frontier.last_sequence,
            "last_base_version": frontier.last_base_version,
        }
        for learner_id, frontier in zip(
            frontiers.learner_ids, frontiers.entries, strict=True
        )
    ]


def _envelope_semantic_values(
    *,
    outer_optimizer_state: bytes,
    frontiers: ConsumptionFrontiers,
    selected: tuple[CommittedContribution, ...],
    policy_identity: str,
    previous_authority_identity: str,
    selection_identity: str,
    update_identity: str,
    outer_optimizer_state_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "policy_identity": policy_identity,
        "previous_authority_identity": previous_authority_identity,
        "selection_identity": selection_identity,
        "update_identity": update_identity,
        "frontiers": _frontier_rows(frontiers),
        "selected": [item.to_dict() for item in selected],
        "outer_optimizer_state": {
            "bytes": len(outer_optimizer_state),
            "sha256": (
                hashlib.sha256(outer_optimizer_state).hexdigest()
                if outer_optimizer_state_sha256 is None
                else outer_optimizer_state_sha256
            ),
        },
    }


def _envelope_semantic(envelope: CommitEnvelope) -> dict[str, object]:
    return _envelope_semantic_values(
        outer_optimizer_state=envelope.outer_optimizer_state,
        frontiers=envelope.frontiers,
        selected=envelope.selected,
        policy_identity=envelope.policy_identity,
        previous_authority_identity=envelope.previous_authority_identity,
        selection_identity=envelope.selection_identity,
        update_identity=envelope.update_identity,
        outer_optimizer_state_sha256=envelope.outer_optimizer_state_sha256,
    )


def make_commit_envelope(
    *,
    outer_optimizer_state: bytes,
    frontiers: ConsumptionFrontiers,
    selected: tuple[CommittedContribution, ...],
    policy_identity: str,
    previous_authority_identity: str,
    selection_identity: str,
    update_identity: str,
) -> CommitEnvelope:
    outer_optimizer_state_sha256 = hashlib.sha256(outer_optimizer_state).hexdigest()
    return _verified_commit_envelope(
        outer_optimizer_state=outer_optimizer_state,
        outer_optimizer_state_sha256=outer_optimizer_state_sha256,
        frontiers=frontiers,
        selected=selected,
        policy_identity=policy_identity,
        previous_authority_identity=previous_authority_identity,
        selection_identity=selection_identity,
        update_identity=update_identity,
    )


def encode_commit_envelope(envelope: CommitEnvelope) -> bytes:
    if not isinstance(envelope, CommitEnvelope):
        raise AtomicCommitError("envelope must be CommitEnvelope")
    semantic = _envelope_semantic(envelope)
    if canonical_digest(semantic) != envelope.envelope_identity:
        raise AtomicCommitError("commit envelope identity mismatch")
    header = canonical_bytes(
        {**semantic, "envelope_identity": envelope.envelope_identity}
    )
    if len(header) > _HEADER_LIMIT:
        raise AtomicCommitError("commit envelope header exceeds the format limit")
    return (
        _MAGIC
        + struct.pack(">Q", len(header))
        + header
        + envelope.outer_optimizer_state
    )


def decode_commit_envelope(
    payload: bytes,
    *,
    identities: GlobalStateIdentities,
    descriptor: FragmentStateDescriptor,
) -> CommitEnvelope:
    payload = _require_bytes(payload, "commit envelope payload")
    if len(payload) < len(_MAGIC) + 8 or payload[: len(_MAGIC)] != _MAGIC:
        raise AtomicCommitError("commit envelope magic mismatch")
    header_length = struct.unpack(">Q", payload[len(_MAGIC) : len(_MAGIC) + 8])[0]
    if header_length > _HEADER_LIMIT or len(payload) < len(_MAGIC) + 8 + header_length:
        raise AtomicCommitError("commit envelope header length is invalid")
    start = len(_MAGIC) + 8
    header_bytes = payload[start : start + header_length]
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AtomicCommitError(
            "commit envelope header is not complete JSON"
        ) from error
    if not isinstance(header, dict) or canonical_bytes(header) != header_bytes:
        raise AtomicCommitError("commit envelope header is not canonical")
    expected = {
        "schema_version",
        "policy_identity",
        "previous_authority_identity",
        "selection_identity",
        "update_identity",
        "frontiers",
        "selected",
        "outer_optimizer_state",
        "envelope_identity",
    }
    if set(header) != expected or header.get("schema_version") != 1:
        raise AtomicCommitError("commit envelope header schema mismatch")
    body = payload[start + header_length :]
    outer = header["outer_optimizer_state"]
    if not isinstance(outer, dict) or set(outer) != {"bytes", "sha256"}:
        raise AtomicCommitError("outer optimizer section schema mismatch")
    outer_sha256 = _require_hex(outer["sha256"], "outer optimizer sha256")
    if (
        _require_nonnegative(outer["bytes"], "outer optimizer bytes") != len(body)
        or outer_sha256 != hashlib.sha256(body).hexdigest()
    ):
        raise AtomicCommitError("outer optimizer state integrity mismatch")
    rows = header["frontiers"]
    if not isinstance(rows, list) or not rows:
        raise AtomicCommitError("commit frontiers must be a nonempty list")
    expected_frontier_fields = {
        "learner_id",
        "last_sequence",
        "last_base_version",
    }
    learner_ids: list[str] = []
    entries = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_frontier_fields:
            raise AtomicCommitError("commit frontier row schema mismatch")
        learner_ids.append(_require_text(row["learner_id"], "frontier learner_id"))
        entries.append(
            ConsumptionFrontier(
                _require_integer(row["last_sequence"], "last_sequence"),
                _require_integer(row["last_base_version"], "last_base_version"),
            )
        )
    frontiers = ConsumptionFrontiers(
        identities=identities,
        descriptor=descriptor,
        learner_ids=tuple(learner_ids),
        entries=tuple(entries),
    )
    selected_rows = header["selected"]
    fields = {field.name for field in dataclasses.fields(CommittedContribution)}
    if not isinstance(selected_rows, list) or any(
        not isinstance(row, dict) or set(row) != fields for row in selected_rows
    ):
        raise AtomicCommitError("selected contribution schema mismatch")
    try:
        selected = tuple(CommittedContribution(**row) for row in selected_rows)
    except TypeError as error:
        raise AtomicCommitError("selected contribution is malformed") from error
    envelope = _verified_commit_envelope(
        outer_optimizer_state=body,
        outer_optimizer_state_sha256=outer_sha256,
        frontiers=frontiers,
        selected=selected,
        policy_identity=_require_hex(header["policy_identity"], "policy_identity"),
        previous_authority_identity=_require_hex(
            header["previous_authority_identity"],
            "previous_authority_identity",
        ),
        selection_identity=_require_hex(
            header["selection_identity"], "selection_identity"
        ),
        update_identity=_require_hex(header["update_identity"], "update_identity"),
    )
    if envelope.envelope_identity != _require_hex(
        header["envelope_identity"], "envelope_identity"
    ):
        raise AtomicCommitError("commit envelope identity mismatch")
    return envelope


@dataclasses.dataclass(frozen=True, slots=True)
class AtomicFragmentAuthority:
    state: FragmentGlobalState
    envelope: CommitEnvelope

    def __post_init__(self) -> None:
        if not isinstance(self.state, FragmentGlobalState):
            raise AtomicCommitError("state must be FragmentGlobalState")
        if not isinstance(self.envelope, CommitEnvelope):
            raise AtomicCommitError("envelope must be CommitEnvelope")
        if self.envelope.frontiers.identities != self.state.identities:
            raise AtomicCommitError("authority and frontier identities differ")
        if self.envelope.frontiers.descriptor != self.state.descriptor:
            raise AtomicCommitError("authority and frontier descriptors differ")
        if self.state.outer_update_count != self.state.version:
            raise AtomicCommitError(
                "Stage 1 authority outer update count must equal its version"
            )
        if (self.state.version == 0) != (not self.envelope.selected):
            raise AtomicCommitError(
                "only the bootstrap authority may have an empty selection"
            )
        if any(
            item.payload_bytes != len(self.state.parameters)
            for item in self.envelope.selected
        ):
            raise AtomicCommitError(
                "committed proposal size differs from the fragment authority"
            )

    @property
    def authority_identity(self) -> str:
        return canonical_digest(
            {
                "schema_version": 1,
                "state_content_identity": self.state.content_identity,
                "state_version": self.state.version,
                "frontiers": _frontier_rows(self.envelope.frontiers),
            }
        )

    @property
    def version(self) -> int:
        return self.state.version

    @property
    def content_identity(self) -> str:
        return self.state.content_identity

    @property
    def parameters(self) -> bytes:
        return self.state.parameters

    @property
    def outer_optimizer_state(self) -> bytes:
        return self.envelope.outer_optimizer_state

    @property
    def frontiers(self) -> ConsumptionFrontiers:
        return self.envelope.frontiers


@dataclasses.dataclass(frozen=True, slots=True)
class AtomicGlobalSnapshot:
    authorities: tuple[AtomicFragmentAuthority, ...]
    version_vector: tuple[int, ...]
    digest: str


@dataclasses.dataclass(frozen=True, slots=True)
class AtomicBootstrapResult:
    published_indices: tuple[int, ...]
    existing_indices: tuple[int, ...]
    snapshot: AtomicGlobalSnapshot


@dataclasses.dataclass(frozen=True, slots=True)
class AtomicCommitRequest:
    fragment_index: int
    expected_current_version: int
    expected_current_content_identity: str
    next_version: int
    parameters: bytes
    outer_optimizer_state: bytes
    selection: FrozenSelection
    policy_identity: str
    update_identity: str

    def __post_init__(self) -> None:
        fragment_index = _require_nonnegative(self.fragment_index, "fragment_index")
        current_version = _require_nonnegative(
            self.expected_current_version, "expected_current_version"
        )
        _require_hex(
            self.expected_current_content_identity,
            "expected_current_content_identity",
        )
        next_version = _require_nonnegative(self.next_version, "next_version")
        if next_version != current_version + 1:
            raise AtomicCommitError("next_version must equal current version plus one")
        _require_bytes(self.parameters, "parameters")
        _require_bytes(self.outer_optimizer_state, "outer_optimizer_state")
        if not isinstance(self.selection, FrozenSelection):
            raise AtomicCommitError("selection must be FrozenSelection")
        if (
            self.selection.fragment_index != fragment_index
            or self.selection.current_version != current_version
        ):
            raise AtomicCommitError(
                "selection fragment or version differs from request"
            )
        if not self.selection.proposals:
            raise AtomicCommitError("atomic commit requires a nonempty selection")
        if len(self.selection.proposals) != len(self.selection.weights):
            raise AtomicCommitError("selection proposal and weight counts differ")
        for proposal, weight in zip(
            self.selection.proposals, self.selection.weights, strict=True
        ):
            if not isinstance(proposal, Proposal) or not isinstance(
                weight, CandidateWeight
            ):
                raise AtomicCommitError("selection contains malformed facts")
            if (
                weight.proposal_id != proposal.proposal_id
                or weight.learner_id != proposal.learner_id
                or weight.processed_tokens != proposal.processed_tokens
                or weight.staleness != current_version - proposal.base_version
            ):
                raise AtomicCommitError("selection weight facts differ from proposal")
            if _f32(weight.normalized_weight, "normalized_weight") < 0:
                raise AtomicCommitError("selection weight must be nonnegative")
        weight_sum = _f32(0.0, "weight sum")
        for weight in self.selection.weights:
            weight_sum = _f32(
                weight_sum + _f32(weight.normalized_weight, "normalized_weight"),
                "weight sum",
            )
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=2e-6):
            raise AtomicCommitError("selection float32 weights must sum to one")
        _require_hex(self.policy_identity, "policy_identity")
        _require_hex(self.update_identity, "update_identity")


@dataclasses.dataclass(frozen=True, slots=True)
class AtomicCommitResult:
    authority: AtomicFragmentAuthority
    published: bool
    duplicate_retry: bool


BeforeVisibility = Callable[[FragmentGlobalState], None]


class AtomicGlobalCommitStore:
    """Publish global fragment state and consumption as one current authority."""

    def __init__(
        self,
        store: GlobalStateStore,
        *,
        learner_ids: tuple[str, ...],
        policy_identity: str,
    ) -> None:
        if not isinstance(store, GlobalStateStore):
            raise AtomicCommitError("store must be GlobalStateStore")
        if not isinstance(learner_ids, tuple) or not learner_ids:
            raise AtomicCommitError("learner_ids must be a nonempty tuple")
        if any(not isinstance(item, str) or not item for item in learner_ids):
            raise AtomicCommitError("learner_ids must contain nonempty strings")
        if len(set(learner_ids)) != len(learner_ids):
            raise AtomicCommitError("learner_ids must be unique")
        if store.s_max != 0:
            raise AtomicCommitError("Stage 1 atomic commit requires s_max=0")
        self.store = store
        self.learner_ids = learner_ids
        self.policy_identity = _require_hex(policy_identity, "policy_identity")
        self._authority_cache: dict[
            int, tuple[PublicationRecord, AtomicFragmentAuthority]
        ] = {}

    def _empty_envelope(
        self, descriptor: FragmentStateDescriptor, outer_optimizer_state: bytes
    ) -> CommitEnvelope:
        return make_commit_envelope(
            outer_optimizer_state=outer_optimizer_state,
            frontiers=ConsumptionFrontiers.empty(
                identities=self.store.identities,
                descriptor=descriptor,
                learner_ids=self.learner_ids,
            ),
            selected=(),
            policy_identity=self.policy_identity,
            previous_authority_identity=_ZERO_IDENTITY,
            selection_identity=_ZERO_IDENTITY,
            update_identity=_ZERO_IDENTITY,
        )

    def bootstrap(
        self, fragments: tuple[BootstrapFragment, ...]
    ) -> AtomicBootstrapResult:
        if not isinstance(fragments, tuple):
            raise AtomicCommitError("fragments must be a tuple")
        wrapped = tuple(
            dataclasses.replace(
                fragment,
                outer_state=encode_commit_envelope(
                    self._empty_envelope(fragment.descriptor, fragment.outer_state)
                ),
            )
            for fragment in fragments
        )
        result = self.store.bootstrap(wrapped)
        return AtomicBootstrapResult(
            published_indices=result.published_indices,
            existing_indices=result.existing_indices,
            snapshot=self.load_snapshot(),
        )

    def load_fragment(self, index: int) -> AtomicFragmentAuthority:
        record = self.store.peek_fragment_record(index, timeout_seconds=0)
        cached = self._authority_cache.get(index)
        if cached is not None and cached[0] == record:
            return cached[1]
        if self.store.supports_bound_record_reads:
            state = self.store.load_bound_fragment_record(index, record)
            loaded_record = record
        else:
            state, loaded_record = self.store.load_fragment_publication(
                index, timeout_seconds=0
            )
        envelope = decode_commit_envelope(
            state.outer_state,
            identities=state.identities,
            descriptor=state.descriptor,
        )
        if envelope.frontiers.learner_ids != self.learner_ids:
            raise AtomicCommitError("committed learner frontier identities differ")
        if envelope.policy_identity != self.policy_identity:
            raise AtomicCommitError("committed policy identity differs")
        authority = AtomicFragmentAuthority(state, envelope)
        self._authority_cache[index] = (loaded_record, authority)
        return authority

    def load_snapshot(self) -> AtomicGlobalSnapshot:
        authorities = tuple(
            self.load_fragment(index) for index in range(len(self.store.descriptors))
        )
        versions = tuple(item.version for item in authorities)
        return AtomicGlobalSnapshot(
            authorities=authorities,
            version_vector=versions,
            digest=canonical_digest(
                {
                    "identities": self.store.identities.to_dict(),
                    "authorities": [item.authority_identity for item in authorities],
                    "version_vector": versions,
                }
            ),
        )

    @staticmethod
    def _committed_contributions(
        selection: FrozenSelection,
    ) -> tuple[CommittedContribution, ...]:
        return tuple(
            CommittedContribution(
                proposal_id=proposal.proposal_id,
                content_identity=proposal.content_identity,
                learner_id=proposal.learner_id,
                sequence=proposal.sequence,
                base_version=proposal.base_version,
                base_content_identity=proposal.base_content_identity,
                processed_tokens=proposal.processed_tokens,
                staleness=weight.staleness,
                normalized_weight=weight.normalized_weight,
                parameters_sha256=proposal.parameters_sha256,
                payload_bytes=proposal.payload_bytes,
            )
            for proposal, weight in zip(
                selection.proposals, selection.weights, strict=True
            )
        )

    def _matches_retry(
        self,
        authority: AtomicFragmentAuthority,
        request: AtomicCommitRequest,
    ) -> bool:
        return (
            authority.version == request.next_version
            and authority.state.base_content_identity
            == request.expected_current_content_identity
            and authority.parameters == request.parameters
            and authority.outer_optimizer_state == request.outer_optimizer_state
            and authority.envelope.policy_identity == request.policy_identity
            and authority.envelope.previous_authority_identity
            == request.selection.authority_identity
            and authority.envelope.selection_identity
            == request.selection.selection_identity
            and authority.envelope.update_identity == request.update_identity
            and authority.envelope.selected
            == self._committed_contributions(request.selection)
        )

    def _validate_selection_against_authority(
        self,
        authority: AtomicFragmentAuthority,
        selection: FrozenSelection,
    ) -> None:
        if selection.authority_identity != authority.authority_identity:
            raise AtomicCommitError("frozen selection authority identity differs")
        if selection.fragment_index != authority.state.descriptor.index:
            raise AtomicCommitError("frozen selection fragment differs from authority")
        if selection.current_version != authority.version:
            raise AtomicCommitError("frozen selection version differs from authority")
        learners: set[str] = set()
        proposals: set[str] = set()
        for proposal in selection.proposals:
            if proposal.identities != authority.state.identities:
                raise AtomicCommitError("selected proposal identities differ")
            if proposal.descriptor != authority.state.descriptor:
                raise AtomicCommitError("selected proposal fragment differs")
            if proposal.learner_id not in self.learner_ids:
                raise AtomicCommitError("selected proposal learner is not frozen")
            if proposal.learner_id in learners:
                raise AtomicCommitError(
                    "selected proposals contain a duplicate learner"
                )
            if proposal.proposal_id in proposals:
                raise AtomicCommitError(
                    "selected proposals contain a duplicate identity"
                )
            learners.add(proposal.learner_id)
            proposals.add(proposal.proposal_id)
            frontier = authority.frontiers.for_learner(proposal.learner_id)
            if proposal.sequence <= frontier.last_sequence:
                raise AtomicCommitError(
                    "selected proposal sequence is already consumed"
                )
            if proposal.base_version <= frontier.last_base_version:
                raise AtomicCommitError("selected proposal base is already consumed")
            if proposal.base_version != authority.version:
                raise AtomicCommitError("selected proposal is not fresh")
            if proposal.base_content_identity != authority.content_identity:
                raise AtomicCommitError("selected proposal base identity differs")
            if (
                proposal.local_steps <= 0
                or proposal.processed_tokens <= 0
                or proposal.snapshot_local_step <= 0
            ):
                raise AtomicCommitError("selected proposal progress must be positive")

    def commit(
        self,
        request: AtomicCommitRequest,
        *,
        crash_at: str | None = None,
        before_visibility: BeforeVisibility | None = None,
    ) -> AtomicCommitResult:
        if not isinstance(request, AtomicCommitRequest):
            raise AtomicCommitError("request must be AtomicCommitRequest")
        if request.policy_identity != self.policy_identity:
            raise AtomicCommitError("request policy identity differs from the store")
        current = self.load_fragment(request.fragment_index)
        if current.version == request.next_version:
            if self._matches_retry(current, request):
                return AtomicCommitResult(
                    current, published=False, duplicate_retry=True
                )
            raise AtomicCommitError(
                "same next version is already committed with different content"
            )
        if current.version != request.expected_current_version:
            raise AtomicCommitError("expected current version differs from authority")
        if current.content_identity != request.expected_current_content_identity:
            raise AtomicCommitError("expected current content identity differs")
        if len(request.parameters) != len(current.parameters):
            raise AtomicCommitError(
                "successor parameter byte count differs from current authority"
            )
        self._validate_selection_against_authority(current, request.selection)
        frontiers = commit_consumption(
            current.frontiers,
            request.selection.proposals,
            True,
        )
        envelope = make_commit_envelope(
            outer_optimizer_state=request.outer_optimizer_state,
            frontiers=frontiers,
            selected=self._committed_contributions(request.selection),
            policy_identity=request.policy_identity,
            previous_authority_identity=current.authority_identity,
            selection_identity=request.selection.selection_identity,
            update_identity=request.update_identity,
        )
        cached = self._authority_cache.get(request.fragment_index)
        if cached is None or cached[1] is not current:
            raise AtomicCommitError("current authority cache is unavailable for commit")
        successor_state, successor_record = self.store.publish_successor_from_current(
            current.state,
            expected_record=cached[0],
            parameters=request.parameters,
            outer_state=encode_commit_envelope(envelope),
            crash_at=crash_at,
            before_visibility=before_visibility,
        )
        committed = AtomicFragmentAuthority(successor_state, envelope)
        self._authority_cache[request.fragment_index] = (
            successor_record,
            committed,
        )
        if not self._matches_retry(committed, request):
            raise AtomicCommitError("visible successor differs from immutable request")
        return AtomicCommitResult(committed, published=True, duplicate_retry=False)
