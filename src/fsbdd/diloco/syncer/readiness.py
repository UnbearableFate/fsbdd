from __future__ import annotations

import dataclasses
import enum
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Sequence

from fsbdd.diloco.protocol.global_state import FragmentGlobalState
from fsbdd.diloco.common.identity import canonical_digest
from fsbdd.diloco.protocol.proposal import (
    CandidateWeight,
    ConsumptionFrontiers,
    EligibilityPolicy,
    Proposal,
    ProposalError,
    ProposalStore,
    RetainedBaseIdentity,
    compute_candidate_weights,
    select_candidates,
)
from fsbdd.diloco.protocol.storage import (
    PublicationNotFound,
    PublicationNotReady,
    PublicationRecord,
)


class ReadinessError(ProposalError):
    """The syncer readiness or scheduling contract was violated."""


def _require_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReadinessError(f"{field} must be an integer")
    return value


def _require_nonnegative(value: object, field: str) -> int:
    value = _require_integer(value, field)
    if value < 0:
        raise ReadinessError(f"{field} must be nonnegative")
    return value


def _require_positive(value: object, field: str) -> int:
    value = _require_integer(value, field)
    if value <= 0:
        raise ReadinessError(f"{field} must be positive")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class ReadinessConfig:
    logical_syncer_id: str
    q: int
    q_fresh: int
    max_contributors: int
    grace_period_ns: int
    s_max: int = 0
    lambda_s: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.logical_syncer_id, str) or not self.logical_syncer_id:
            raise ReadinessError("logical_syncer_id must be a non-empty string")
        q = _require_positive(self.q, "q")
        q_fresh = _require_nonnegative(self.q_fresh, "q_fresh")
        maximum = _require_positive(self.max_contributors, "max_contributors")
        if not 0 <= q_fresh <= q <= maximum:
            raise ReadinessError("require 0 <= q_fresh <= q <= max_contributors")
        _require_nonnegative(self.grace_period_ns, "grace_period_ns")
        if _require_nonnegative(self.s_max, "s_max") != 0:
            raise ReadinessError("Stage 1 syncer readiness requires s_max=0")
        if not isinstance(self.lambda_s, (int, float)) or isinstance(
            self.lambda_s, bool
        ):
            raise ReadinessError("lambda_s must be numeric")
        if not math.isfinite(float(self.lambda_s)) or self.lambda_s < 0:
            raise ReadinessError("lambda_s must be finite and nonnegative")


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentReadinessAuthority:
    state: FragmentGlobalState
    frontiers: ConsumptionFrontiers

    def __post_init__(self) -> None:
        if not isinstance(self.state, FragmentGlobalState):
            raise ReadinessError("authority state must be FragmentGlobalState")
        if not isinstance(self.frontiers, ConsumptionFrontiers):
            raise ReadinessError("authority frontiers must be ConsumptionFrontiers")
        if self.frontiers.identities != self.state.identities:
            raise ReadinessError("authority frozen identities differ")
        if self.frontiers.descriptor != self.state.descriptor:
            raise ReadinessError("authority fragment descriptors differ")

    @property
    def identity(self) -> str:
        return canonical_digest(
            {
                "schema_version": 1,
                "state_content_identity": self.state.content_identity,
                "state_version": self.state.version,
                "frontiers": [
                    {
                        "learner_id": learner_id,
                        "last_sequence": frontier.last_sequence,
                        "last_base_version": frontier.last_base_version,
                    }
                    for learner_id, frontier in zip(
                        self.frontiers.learner_ids,
                        self.frontiers.entries,
                        strict=True,
                    )
                ],
            }
        )


class ReadinessPhase(str, enum.Enum):
    WAITING = "waiting"
    GRACE = "grace"
    FROZEN = "frozen"
    ACTIVE = "active"


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenSelection:
    logical_syncer_id: str
    fragment_index: int
    current_version: int
    authority_identity: str
    generation: int
    grace_started_ns: int
    frozen_observed_ns: int
    proposals: tuple[Proposal, ...]
    weights: tuple[CandidateWeight, ...]
    selection_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.logical_syncer_id, str) or not self.logical_syncer_id:
            raise ReadinessError("logical_syncer_id must be a non-empty string")
        fragment_index = _require_nonnegative(self.fragment_index, "fragment_index")
        current_version = _require_nonnegative(
            self.current_version, "current_version"
        )
        if (
            not isinstance(self.authority_identity, str)
            or len(self.authority_identity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.authority_identity
            )
        ):
            raise ReadinessError(
                "authority_identity must be a lowercase SHA-256 identity"
            )
        _require_positive(self.generation, "generation")
        grace_started_ns = _require_nonnegative(
            self.grace_started_ns, "grace_started_ns"
        )
        frozen_observed_ns = _require_nonnegative(
            self.frozen_observed_ns, "frozen_observed_ns"
        )
        if frozen_observed_ns < grace_started_ns:
            raise ReadinessError("frozen observation precedes grace start")
        if not isinstance(self.proposals, tuple) or not self.proposals:
            raise ReadinessError("frozen proposals must be a nonempty tuple")
        if not isinstance(self.weights, tuple) or len(self.weights) != len(
            self.proposals
        ):
            raise ReadinessError("frozen proposal and weight counts differ")
        learners: set[str] = set()
        proposal_ids: set[str] = set()
        weight_sum = 0.0
        for proposal, weight in zip(self.proposals, self.weights, strict=True):
            if not isinstance(proposal, Proposal) or not isinstance(
                weight, CandidateWeight
            ):
                raise ReadinessError("frozen selection contains malformed facts")
            if proposal.descriptor.index != fragment_index:
                raise ReadinessError("frozen proposal fragment differs")
            staleness = current_version - proposal.base_version
            if staleness < 0:
                raise ReadinessError("frozen proposal has a future base")
            if (
                weight.proposal_id != proposal.proposal_id
                or weight.learner_id != proposal.learner_id
                or weight.processed_tokens != proposal.processed_tokens
                or weight.staleness != staleness
            ):
                raise ReadinessError("frozen weight facts differ from proposal")
            if proposal.learner_id in learners:
                raise ReadinessError("frozen selection contains a duplicate learner")
            if proposal.proposal_id in proposal_ids:
                raise ReadinessError("frozen selection contains a duplicate proposal")
            learners.add(proposal.learner_id)
            proposal_ids.add(proposal.proposal_id)
            weight_sum += weight.normalized_weight
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=2e-6):
            raise ReadinessError("frozen normalized weights must sum to one")
        if (
            not isinstance(self.selection_identity, str)
            or len(self.selection_identity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.selection_identity
            )
        ):
            raise ReadinessError(
                "selection_identity must be a lowercase SHA-256 identity"
            )
        semantic = {
            "schema_version": 1,
            "logical_syncer_id": self.logical_syncer_id,
            "fragment_index": fragment_index,
            "current_version": current_version,
            "authority_identity": self.authority_identity,
            "proposal_content_identities": [
                item.content_identity for item in self.proposals
            ],
            "weights": [item.to_dict() for item in self.weights],
        }
        if canonical_digest(semantic) != self.selection_identity:
            raise ReadinessError("frozen selection identity mismatch")

    @property
    def learner_ids(self) -> tuple[str, ...]:
        return tuple(item.learner_id for item in self.proposals)

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_syncer_id": self.logical_syncer_id,
            "fragment_index": self.fragment_index,
            "current_version": self.current_version,
            "authority_identity": self.authority_identity,
            "generation": self.generation,
            "grace_started_ns": self.grace_started_ns,
            "frozen_observed_ns": self.frozen_observed_ns,
            "proposal_ids": [item.proposal_id for item in self.proposals],
            "proposal_content_identities": [
                item.content_identity for item in self.proposals
            ],
            "learner_ids": list(self.learner_ids),
            "proposal_facts": [
                {
                    "proposal_id": item.proposal_id,
                    "content_identity": item.content_identity,
                    "learner_id": item.learner_id,
                    "sequence": item.sequence,
                    "base_version": item.base_version,
                    "processed_tokens": item.processed_tokens,
                }
                for item in self.proposals
            ],
            "weights": [item.to_dict() for item in self.weights],
            "selection_identity": self.selection_identity,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class UpdateLease:
    logical_syncer_id: str
    claim_ordinal: int
    selection: FrozenSelection


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentReadinessView:
    fragment_index: int
    current_version: int
    phase: ReadinessPhase
    eligible_distinct: int
    fresh_distinct: int
    rejected: int
    duplicate_observations: int
    grace_started_ns: int | None
    grace_deadline_ns: int | None
    selected_count: int
    selection_identity: str | None
    generation: int
    claims: int


@dataclasses.dataclass(frozen=True, slots=True)
class PollReport:
    observed_ns: int
    fixed_slot_reads: int
    missing_slots: int
    transient_unavailable_slots: int
    fragments: tuple[FragmentReadinessView, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ReadinessSnapshot:
    logical_syncer_id: str
    fragment_count: int
    round_robin_cursor: int
    active_fragment: int | None
    poll_cycles: int
    fixed_slot_reads: int
    claims: int
    resident_selected_proposals: int
    resident_latest_proposals: int
    proposal_payload_cache_hits: int
    proposal_payload_cache_misses: int
    transition_counts: tuple[tuple[str, int], ...]
    fragments: tuple[FragmentReadinessView, ...]


@dataclasses.dataclass(slots=True)
class _FragmentRound:
    authority: FragmentReadinessAuthority
    phase: ReadinessPhase = ReadinessPhase.WAITING
    eligible_distinct: int = 0
    fresh_distinct: int = 0
    rejected: int = 0
    duplicate_observations: int = 0
    grace_started_ns: int | None = None
    grace_deadline_ns: int | None = None
    selection: FrozenSelection | None = None
    generation: int = 0
    claims: int = 0


ClockNs = Callable[[], int]


class SyncerReadinessMachine:
    """One bounded readiness and fair-scheduling authority for every fragment."""

    def __init__(
        self,
        store: ProposalStore,
        *,
        authorities: Sequence[FragmentReadinessAuthority],
        config: ReadinessConfig,
        clock_ns: ClockNs = time.monotonic_ns,
    ) -> None:
        if not isinstance(store, ProposalStore):
            raise ReadinessError("readiness machine requires a ProposalStore")
        if not isinstance(config, ReadinessConfig):
            raise ReadinessError("config must be ReadinessConfig")
        if isinstance(authorities, (str, bytes)) or not isinstance(
            authorities, Sequence
        ):
            raise ReadinessError("authorities must be a sequence")
        if len(authorities) != len(store.descriptors):
            raise ReadinessError("authorities must cover every frozen fragment")
        if config.max_contributors > len(store.learner_ids):
            raise ReadinessError("max_contributors exceeds learner count")
        if not callable(clock_ns):
            raise ReadinessError("clock_ns must be callable")
        self.store = store
        self.config = config
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        rounds: list[_FragmentRound] = []
        for index, authority in enumerate(authorities):
            self._validate_authority(index, authority)
            rounds.append(_FragmentRound(authority=authority))
        self._rounds = rounds
        self._cursor = 0
        self._active: UpdateLease | None = None
        self._last_observed_ns = -1
        self._poll_cycles = 0
        self._fixed_slot_reads = 0
        self._claim_ordinal = 0
        self._proposal_cache: dict[tuple[str, int], Proposal] = {}
        self._proposal_records: dict[tuple[str, int], PublicationRecord] = {}
        self._proposal_payload_identities: dict[tuple[str, int], str] = {}
        self._payload_cache_misses = 0
        self._payload_cache_hits = 0
        self._transition_counts = {
            "waiting_to_grace": 0,
            "grace_to_waiting": 0,
            "grace_to_frozen": 0,
            "waiting_to_frozen": 0,
            "frozen_to_active": 0,
            "active_to_frozen": 0,
            "active_to_waiting": 0,
            "authority_to_waiting": 0,
        }

    def _install_proposal_observation(
        self,
        key: tuple[str, int],
        record: PublicationRecord,
        proposal: Proposal | None,
    ) -> bool:
        """Install one monotonic fixed-slot observation.

        A payload checksum alone is not the whole visibility authority: sequence,
        base version, and base identity are record metadata.  Treating an equal
        payload identity as an unconditional cache hit would therefore accept a
        same-payload record rewrite without revalidating the compound proposal.
        """

        previous_record = self._proposal_records.get(key)
        previous_proposal = self._proposal_cache.get(key)
        if previous_record is not None:
            if record.sequence < previous_record.sequence:
                raise ReadinessError("proposal visibility sequence regressed")
            if record.version < previous_record.version:
                raise ReadinessError("proposal visibility base version regressed")
            if record.sequence == previous_record.sequence:
                if record != previous_record:
                    raise ReadinessError(
                        "proposal visibility changed at the same sequence"
                    )
                if proposal is not None and proposal != previous_proposal:
                    raise ReadinessError(
                        "proposal content changed at the same sequence"
                    )
                self._payload_cache_hits += 1
                return False
        if proposal is None:
            raise ReadinessError(
                "changed proposal visibility was not validated with its payload"
            )
        if (
            proposal.sequence != record.sequence
            or proposal.base_version != record.version
            or proposal.base_content_identity != record.base_content_identity
        ):
            raise ReadinessError("proposal differs from its visibility record")
        self._proposal_cache[key] = proposal
        self._proposal_records[key] = record
        self._proposal_payload_identities[key] = record.payload_identity
        self._payload_cache_misses += 1
        return True

    def _validate_authority(
        self, index: int, authority: FragmentReadinessAuthority
    ) -> None:
        if not isinstance(authority, FragmentReadinessAuthority):
            raise ReadinessError("authority must be FragmentReadinessAuthority")
        if authority.state.identities != self.store.identities:
            raise ReadinessError("authority and proposal-store identities differ")
        if authority.state.descriptor != self.store.descriptors[index]:
            raise ReadinessError("authority does not match its fragment position")
        if authority.frontiers.learner_ids != self.store.learner_ids:
            raise ReadinessError("authority and proposal-store learners differ")

    def _now(self, observed_ns: int | None) -> int:
        value = self._clock_ns() if observed_ns is None else observed_ns
        value = _require_nonnegative(value, "observed_ns")
        if value < self._last_observed_ns:
            raise ReadinessError("monotonic observation time regressed")
        self._last_observed_ns = value
        return value

    def _policy(
        self, round_state: _FragmentRound, *, permissive: bool
    ) -> EligibilityPolicy:
        state = round_state.authority.state
        return EligibilityPolicy(
            identities=state.identities,
            descriptor=state.descriptor,
            learner_ids=self.store.learner_ids,
            current_version=state.version,
            retained_bases=(
                RetainedBaseIdentity(state.version, state.content_identity),
            ),
            s_max=0,
            q=1 if permissive else self.config.q,
            q_fresh=0 if permissive else self.config.q_fresh,
            max_contributors=(
                len(self.store.learner_ids)
                if permissive
                else self.config.max_contributors
            ),
            maximum_local_steps=self.store.maximum_local_steps,
            maximum_processed_tokens=self.store.maximum_processed_tokens,
            lambda_s=float(self.config.lambda_s),
        )

    def _set_phase(self, round_state: _FragmentRound, phase: ReadinessPhase) -> None:
        previous = round_state.phase
        if previous == phase:
            return
        key = f"{previous.value}_to_{phase.value}"
        if key in self._transition_counts:
            self._transition_counts[key] += 1
        round_state.phase = phase

    def _freeze(
        self,
        index: int,
        round_state: _FragmentRound,
        eligible: tuple[Proposal, ...],
        observed_ns: int,
    ) -> None:
        selected = eligible[: self.config.max_contributors]
        if len(selected) < self.config.q:
            raise ReadinessError("freeze attempted without selected quorum")
        weights = compute_candidate_weights(
            selected,
            current_version=round_state.authority.state.version,
            lambda_s=float(self.config.lambda_s),
        )
        round_state.generation += 1
        semantic = {
            "schema_version": 1,
            "logical_syncer_id": self.config.logical_syncer_id,
            "fragment_index": index,
            "current_version": round_state.authority.state.version,
            "authority_identity": round_state.authority.identity,
            "proposal_content_identities": [item.content_identity for item in selected],
            "weights": [item.to_dict() for item in weights],
        }
        round_state.selection = FrozenSelection(
            logical_syncer_id=self.config.logical_syncer_id,
            fragment_index=index,
            current_version=round_state.authority.state.version,
            authority_identity=round_state.authority.identity,
            generation=round_state.generation,
            grace_started_ns=(
                observed_ns
                if round_state.grace_started_ns is None
                else round_state.grace_started_ns
            ),
            frozen_observed_ns=observed_ns,
            proposals=selected,
            weights=weights,
            selection_identity=canonical_digest(semantic),
        )
        self._set_phase(round_state, ReadinessPhase.FROZEN)

    @staticmethod
    def _normalize_candidates(
        proposals: Sequence[Proposal], fragment_count: int
    ) -> tuple[tuple[tuple[Proposal, ...], ...], tuple[int, ...]]:
        by_fragment: list[dict[str, Proposal]] = [{} for _ in range(fragment_count)]
        duplicates = [0] * fragment_count
        for proposal in proposals:
            if not isinstance(proposal, Proposal):
                raise ReadinessError("observed candidate must be a Proposal")
            index = proposal.descriptor.index
            if not 0 <= index < fragment_count:
                raise ReadinessError(f"candidate has unknown fragment index: {index}")
            previous = by_fragment[index].get(proposal.proposal_id)
            if previous is None:
                by_fragment[index][proposal.proposal_id] = proposal
                continue
            if previous.content_identity != proposal.content_identity:
                raise ReadinessError(
                    f"conflicting duplicate proposal identity: {proposal.proposal_id}"
                )
            duplicates[index] += 1
        return (
            tuple(tuple(items.values()) for items in by_fragment),
            tuple(duplicates),
        )

    def _view(self, index: int, round_state: _FragmentRound) -> FragmentReadinessView:
        selection = round_state.selection
        return FragmentReadinessView(
            fragment_index=index,
            current_version=round_state.authority.state.version,
            phase=round_state.phase,
            eligible_distinct=round_state.eligible_distinct,
            fresh_distinct=round_state.fresh_distinct,
            rejected=round_state.rejected,
            duplicate_observations=round_state.duplicate_observations,
            grace_started_ns=round_state.grace_started_ns,
            grace_deadline_ns=round_state.grace_deadline_ns,
            selected_count=0 if selection is None else len(selection.proposals),
            selection_identity=(
                None if selection is None else selection.selection_identity
            ),
            generation=round_state.generation,
            claims=round_state.claims,
        )

    def observe(
        self,
        proposals: Sequence[Proposal],
        *,
        observed_ns: int | None = None,
    ) -> PollReport:
        if isinstance(proposals, (str, bytes)) or not isinstance(proposals, Sequence):
            raise ReadinessError("proposals must be a sequence")
        with self._lock:
            now_ns = self._now(observed_ns)
            candidates, duplicate_counts = self._normalize_candidates(
                proposals, len(self._rounds)
            )
            self._poll_cycles += 1
            for index, round_state in enumerate(self._rounds):
                if round_state.phase in (
                    ReadinessPhase.FROZEN,
                    ReadinessPhase.ACTIVE,
                ):
                    continue
                selection = select_candidates(
                    candidates[index],
                    round_state.authority.frontiers,
                    self._policy(round_state, permissive=True),
                )
                eligible = selection.selected if selection.ready else tuple()
                eligible_count = len(eligible)
                fresh_count = sum(
                    item.base_version == round_state.authority.state.version
                    for item in eligible
                )
                has_quorum = (
                    eligible_count >= self.config.q
                    and fresh_count >= self.config.q_fresh
                )
                round_state.eligible_distinct = eligible_count
                round_state.fresh_distinct = fresh_count
                round_state.rejected = len(selection.rejections)
                round_state.duplicate_observations = duplicate_counts[index]
                if round_state.phase == ReadinessPhase.WAITING:
                    if not has_quorum:
                        continue
                    round_state.grace_started_ns = now_ns
                    round_state.grace_deadline_ns = now_ns + self.config.grace_period_ns
                    if self.config.grace_period_ns == 0:
                        self._freeze(index, round_state, eligible, now_ns)
                    else:
                        self._set_phase(round_state, ReadinessPhase.GRACE)
                    continue
                if not has_quorum:
                    self._set_phase(round_state, ReadinessPhase.WAITING)
                    round_state.grace_started_ns = None
                    round_state.grace_deadline_ns = None
                    continue
                deadline_ns = round_state.grace_deadline_ns
                if deadline_ns is None:
                    raise ReadinessError("grace phase has no deadline")
                if now_ns >= deadline_ns:
                    self._freeze(index, round_state, eligible, now_ns)
            return PollReport(
                observed_ns=now_ns,
                fixed_slot_reads=0,
                missing_slots=0,
                transient_unavailable_slots=0,
                fragments=tuple(
                    self._view(index, item) for index, item in enumerate(self._rounds)
                ),
            )

    def poll_store(self, *, observed_ns: int | None = None) -> PollReport:
        missing = 0
        transient = 0
        reads = 0
        if not self.store.supports_bound_record_reads:
            for learner_id in self.store.learner_ids:
                for descriptor in self.store.descriptors:
                    reads += 1
                    key = (learner_id, descriptor.index)
                    try:
                        record, proposal = self.store.load_latest_if_changed(
                            learner_id,
                            descriptor.index,
                            known_payload_identity=self._proposal_payload_identities.get(
                                key
                            ),
                            timeout_seconds=0,
                        )
                    except PublicationNotFound:
                        missing += 1
                        continue
                    except PublicationNotReady:
                        transient += 1
                        continue
                    self._install_proposal_observation(key, record, proposal)
            with self._lock:
                report = self.observe(
                    tuple(self._proposal_cache.values()), observed_ns=observed_ns
                )
                self._fixed_slot_reads += reads
                return dataclasses.replace(
                    report,
                    fixed_slot_reads=reads,
                    missing_slots=missing,
                    transient_unavailable_slots=transient,
                )
        changed: list[tuple[str, int, PublicationRecord]] = []
        for learner_id in self.store.learner_ids:
            for descriptor in self.store.descriptors:
                reads += 1
                key = (learner_id, descriptor.index)
                try:
                    record = self.store.peek_latest_record(
                        learner_id,
                        descriptor.index,
                        timeout_seconds=0,
                    )
                except PublicationNotFound:
                    missing += 1
                    continue
                except PublicationNotReady:
                    transient += 1
                    continue
                known = self._proposal_payload_identities.get(key)
                if record.payload_identity == known:
                    self._install_proposal_observation(key, record, None)
                    continue
                previous_record = self._proposal_records.get(key)
                if previous_record is not None:
                    if record.sequence < previous_record.sequence:
                        raise ReadinessError("proposal visibility sequence regressed")
                    if record.version < previous_record.version:
                        raise ReadinessError(
                            "proposal visibility base version regressed"
                        )
                    if record.sequence == previous_record.sequence:
                        raise ReadinessError(
                            "proposal visibility changed at the same sequence"
                        )
                changed.append((learner_id, descriptor.index, record))
        if changed:
            # Payload decoding includes two large SHA-256 authorities.  Changed
            # fixed slots are independent, so decode them concurrently while
            # retaining deterministic cache-install order below.
            with ThreadPoolExecutor(max_workers=len(changed)) as workers:
                futures = [
                    workers.submit(
                        self.store.load_bound_record,
                        learner_id,
                        fragment_index,
                        record,
                    )
                    for learner_id, fragment_index, record in changed
                ]
                for (learner_id, fragment_index, record), future in zip(
                    changed, futures, strict=True
                ):
                    try:
                        proposal = future.result()
                    except PublicationNotFound:
                        missing += 1
                        continue
                    except PublicationNotReady:
                        transient += 1
                        continue
                    key = (learner_id, fragment_index)
                    self._install_proposal_observation(key, record, proposal)
        with self._lock:
            report = self.observe(
                tuple(self._proposal_cache.values()), observed_ns=observed_ns
            )
            self._fixed_slot_reads += reads
            return dataclasses.replace(
                report,
                fixed_slot_reads=reads,
                missing_slots=missing,
                transient_unavailable_slots=transient,
            )

    def claim_next(self) -> UpdateLease | None:
        with self._lock:
            if self._active is not None:
                return None
            fragment_count = len(self._rounds)
            for distance in range(fragment_count):
                index = (self._cursor + distance) % fragment_count
                round_state = self._rounds[index]
                if round_state.phase != ReadinessPhase.FROZEN:
                    continue
                if round_state.selection is None:
                    raise ReadinessError("frozen fragment has no selection")
                self._claim_ordinal += 1
                lease = UpdateLease(
                    logical_syncer_id=self.config.logical_syncer_id,
                    claim_ordinal=self._claim_ordinal,
                    selection=round_state.selection,
                )
                self._active = lease
                self._cursor = (index + 1) % fragment_count
                round_state.claims += 1
                self._set_phase(round_state, ReadinessPhase.ACTIVE)
                return lease
            return None

    def _require_active(self, lease: UpdateLease) -> _FragmentRound:
        if not isinstance(lease, UpdateLease):
            raise ReadinessError("lease must be UpdateLease")
        if self._active != lease:
            raise ReadinessError("lease is not the active syncer update")
        return self._rounds[lease.selection.fragment_index]

    def release(self, lease: UpdateLease) -> None:
        with self._lock:
            round_state = self._require_active(lease)
            self._active = None
            self._set_phase(round_state, ReadinessPhase.FROZEN)

    @staticmethod
    def _validate_authority_progress(
        previous: FragmentReadinessAuthority,
        current: FragmentReadinessAuthority,
    ) -> None:
        if current.state.version < previous.state.version:
            raise ReadinessError("authority version regressed")
        if (
            current.state.version == previous.state.version
            and current.state.content_identity != previous.state.content_identity
        ):
            raise ReadinessError("authority content changed at the same version")
        for before, after in zip(
            previous.frontiers.entries, current.frontiers.entries, strict=True
        ):
            if after.last_sequence < before.last_sequence:
                raise ReadinessError("authority consumed sequence regressed")
            if after.last_base_version < before.last_base_version:
                raise ReadinessError("authority consumed base regressed")

    def _install_authority_unlocked(
        self, index: int, authority: FragmentReadinessAuthority
    ) -> bool:
        self._validate_authority(index, authority)
        round_state = self._rounds[index]
        self._validate_authority_progress(round_state.authority, authority)
        if round_state.authority.identity == authority.identity:
            return False
        round_state.authority = authority
        round_state.eligible_distinct = 0
        round_state.fresh_distinct = 0
        round_state.rejected = 0
        round_state.duplicate_observations = 0
        round_state.grace_started_ns = None
        round_state.grace_deadline_ns = None
        round_state.selection = None
        previous = round_state.phase
        round_state.phase = ReadinessPhase.WAITING
        if previous == ReadinessPhase.ACTIVE:
            self._transition_counts["active_to_waiting"] += 1
        else:
            self._transition_counts["authority_to_waiting"] += 1
        return True

    def install_authority(
        self, index: int, authority: FragmentReadinessAuthority
    ) -> bool:
        index = _require_nonnegative(index, "fragment_index")
        with self._lock:
            if index >= len(self._rounds):
                raise ReadinessError(f"unknown fragment index: {index}")
            if (
                self._active is not None
                and self._active.selection.fragment_index == index
            ):
                raise ReadinessError("active fragment authority requires complete")
            return self._install_authority_unlocked(index, authority)

    def complete(
        self, lease: UpdateLease, authority: FragmentReadinessAuthority
    ) -> None:
        with self._lock:
            round_state = self._require_active(lease)
            if round_state.authority.identity == authority.identity:
                raise ReadinessError("completed update did not advance authority")
            self._validate_authority(lease.selection.fragment_index, authority)
            self._validate_authority_progress(round_state.authority, authority)
            self._active = None
            self._install_authority_unlocked(lease.selection.fragment_index, authority)

    def selection_for(self, index: int) -> FrozenSelection | None:
        index = _require_nonnegative(index, "fragment_index")
        with self._lock:
            try:
                return self._rounds[index].selection
            except IndexError as error:
                raise ReadinessError(f"unknown fragment index: {index}") from error

    @property
    def active_lease(self) -> UpdateLease | None:
        with self._lock:
            return self._active

    def snapshot(self) -> ReadinessSnapshot:
        with self._lock:
            fragments = tuple(
                self._view(index, item) for index, item in enumerate(self._rounds)
            )
            return ReadinessSnapshot(
                logical_syncer_id=self.config.logical_syncer_id,
                fragment_count=len(self._rounds),
                round_robin_cursor=self._cursor,
                active_fragment=(
                    None
                    if self._active is None
                    else self._active.selection.fragment_index
                ),
                poll_cycles=self._poll_cycles,
                fixed_slot_reads=self._fixed_slot_reads,
                claims=self._claim_ordinal,
                resident_selected_proposals=sum(
                    0 if item.selection is None else len(item.selection.proposals)
                    for item in self._rounds
                ),
                resident_latest_proposals=len(self._proposal_cache),
                proposal_payload_cache_hits=self._payload_cache_hits,
                proposal_payload_cache_misses=self._payload_cache_misses,
                transition_counts=tuple(sorted(self._transition_counts.items())),
                fragments=fragments,
            )
