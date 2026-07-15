from __future__ import annotations

import contextlib
import dataclasses
import math
from collections.abc import Iterator, Sequence

from fsbdd.diloco.protocol.global_commit import (
    AtomicCommitRequest,
    AtomicFragmentAuthority,
    AtomicGlobalCommitStore,
)
from fsbdd.diloco.common.identity import canonical_digest
from fsbdd.diloco.protocol.proposal import Proposal, ProposalStore
from fsbdd.diloco.syncer.merge import (
    ContributionFact,
    FragmentMergeRequest,
    FragmentOuterState,
    FragmentUpdateResult,
    OuterSGDPolicy,
    SourceMetrics,
    execute_numpy_streaming_fragment_update,
    execute_streaming_fragment_update,
)
from fsbdd.diloco.syncer.readiness import (
    FragmentReadinessAuthority,
    FrozenSelection,
    PollReport,
    ReadinessConfig,
    SyncerReadinessMachine,
)


class ProfileAError(RuntimeError):
    """The frozen fresh-reference profile or integrated transition is invalid."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileAError(f"{field} must be a non-empty string")
    return value


def _ordinal(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProfileAError(f"{field} must be a nonnegative integer")
    return value


def _positive(value: object, field: str) -> int:
    value = _ordinal(value, field)
    if value == 0:
        raise ProfileAError(f"{field} must be positive")
    return value


def _finite_nonnegative(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ProfileAError(f"{field} must be finite and nonnegative")
    return float(value)


def profile_a_policy_identity(policy: OuterSGDPolicy) -> str:
    if not isinstance(policy, OuterSGDPolicy):
        raise ProfileAError("policy must be OuterSGDPolicy")
    return canonical_digest(
        {
            "merge": "direct_weighted_average",
            "outer_optimizer": policy.name,
            "learning_rate": policy.f32_learning_rate,
            "momentum": policy.f32_momentum,
            "nesterov": policy.nesterov,
            "accumulation_dtype": "float32",
        }
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileAConfig:
    learner_count: int
    fragment_count: int
    q: int
    q_fresh: int
    s_max: int
    grace_period_ns: int
    lambda_s: float
    schedule: str = "round_robin"

    def __post_init__(self) -> None:
        learners = _positive(self.learner_count, "learner_count")
        _positive(self.fragment_count, "fragment_count")
        if self.q != learners or self.q_fresh != learners:
            raise ProfileAError("Profile A requires q == q_fresh == learner_count")
        if self.s_max != 0:
            raise ProfileAError("Profile A requires s_max=0")
        if self.grace_period_ns != 0:
            raise ProfileAError("Profile A requires zero grace")
        _finite_nonnegative(self.lambda_s, "lambda_s")
        if self.schedule != "round_robin":
            raise ProfileAError("Profile A requires deterministic round_robin schedule")


@dataclasses.dataclass(frozen=True, slots=True)
class ComparisonBasis:
    matched_compute: bool
    matched_communication: bool
    matched_tokens: bool
    claim: str

    def __post_init__(self) -> None:
        for name in ("matched_compute", "matched_communication", "matched_tokens"):
            if not isinstance(getattr(self, name), bool):
                raise ProfileAError(f"{name} must be boolean")
        claim = _text(self.claim, "comparison claim")
        if (
            not self.matched_compute
            and not self.matched_communication
            and not self.matched_tokens
            and claim
            != "protocol_and_runtime_acceptance_only_no_quality_or_efficiency_comparison"
        ):
            raise ProfileAError(
                "unmatched evidence may claim only protocol and runtime acceptance"
            )

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class LearnerProgressRecord:
    learner_id: str
    local_optimizer_steps: int
    processed_input_tokens: int
    loss_bearing_target_tokens: int

    def __post_init__(self) -> None:
        _text(self.learner_id, "learner_id")
        _ordinal(self.local_optimizer_steps, "local_optimizer_steps")
        _ordinal(self.processed_input_tokens, "processed_input_tokens")
        _ordinal(self.loss_bearing_target_tokens, "loss_bearing_target_tokens")

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentProgressRecord:
    fragment_index: int
    outer_update_count: int
    accepted_tokens: int
    fresh_accepted_contributions: int
    stale_accepted_contributions: int

    def __post_init__(self) -> None:
        _ordinal(self.fragment_index, "fragment_index")
        _ordinal(self.outer_update_count, "outer_update_count")
        _ordinal(self.accepted_tokens, "accepted_tokens")
        _ordinal(self.fresh_accepted_contributions, "fresh accepted contributions")
        _ordinal(self.stale_accepted_contributions, "stale accepted contributions")

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileAProgressReport:
    learners: tuple[LearnerProgressRecord, ...]
    fragments: tuple[FragmentProgressRecord, ...]
    global_cycle: int
    comparison: ComparisonBasis

    def __post_init__(self) -> None:
        if not self.learners or not self.fragments:
            raise ProfileAError("progress report requires learners and fragments")
        if tuple(item.fragment_index for item in self.fragments) != tuple(
            range(len(self.fragments))
        ):
            raise ProfileAError("fragment progress must be contiguous and ordered")
        expected = min(item.outer_update_count for item in self.fragments)
        if _ordinal(self.global_cycle, "global_cycle") != expected:
            raise ProfileAError("global_cycle must equal the minimum fragment count")
        if not isinstance(self.comparison, ComparisonBasis):
            raise ProfileAError("comparison basis is invalid")

    @property
    def total_compute_steps(self) -> int:
        return sum(item.local_optimizer_steps for item in self.learners)

    @property
    def total_accepted_tokens(self) -> int:
        return sum(item.accepted_tokens for item in self.fragments)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "learners": [item.to_dict() for item in self.learners],
            "fragments": [item.to_dict() for item in self.fragments],
            "global_cycle": self.global_cycle,
            "comparison": self.comparison.to_dict(),
            "counter_semantics": {
                "learner_local_optimizer_steps": "rank-local inner optimizer steps",
                "learner_processed_input_tokens": "rank-local input tokens",
                "fragment_outer_update_count": "successful per-fragment publications",
                "global_cycle": "minimum fragment outer update count",
                "accepted_tokens": "sum of selected proposal processed tokens",
                "fresh_stale": "selected contribution counts classified at freeze",
            },
        }


class ProfileAProgressTracker:
    def __init__(
        self,
        learner_ids: Sequence[str],
        fragment_outer_counts: Sequence[int],
        *,
        comparison: ComparisonBasis,
    ) -> None:
        learners = tuple(_text(item, "learner_id") for item in learner_ids)
        if not learners or len(set(learners)) != len(learners):
            raise ProfileAError("learner_ids must be unique and nonempty")
        counts = tuple(
            _ordinal(item, "initial fragment outer count")
            for item in fragment_outer_counts
        )
        if not counts:
            raise ProfileAError("fragment count vector must be nonempty")
        if not isinstance(comparison, ComparisonBasis):
            raise ProfileAError("comparison is invalid")
        self._learner_ids = learners
        self._learners = {
            learner_id: LearnerProgressRecord(learner_id, 0, 0, 0)
            for learner_id in learners
        }
        self._fragments = [
            FragmentProgressRecord(index, count, 0, 0, 0)
            for index, count in enumerate(counts)
        ]
        self._comparison = comparison

    def update_learner(
        self,
        learner_id: str,
        *,
        local_optimizer_steps: int,
        processed_input_tokens: int,
        loss_bearing_target_tokens: int,
    ) -> None:
        try:
            previous = self._learners[learner_id]
        except KeyError as error:
            raise ProfileAError(f"unknown learner: {learner_id}") from error
        current = LearnerProgressRecord(
            learner_id,
            local_optimizer_steps,
            processed_input_tokens,
            loss_bearing_target_tokens,
        )
        if (
            current.local_optimizer_steps < previous.local_optimizer_steps
            or current.processed_input_tokens < previous.processed_input_tokens
            or current.loss_bearing_target_tokens < previous.loss_bearing_target_tokens
        ):
            raise ProfileAError("learner progress cannot regress")
        self._learners[learner_id] = current

    def record_fragment_commit(
        self,
        fragment_index: int,
        *,
        next_outer_update_count: int,
        accepted_tokens: int,
        fresh_accepted_contributions: int,
        stale_accepted_contributions: int,
    ) -> None:
        index = _ordinal(fragment_index, "fragment_index")
        try:
            previous = self._fragments[index]
        except IndexError as error:
            raise ProfileAError(f"unknown fragment: {index}") from error
        next_count = _ordinal(next_outer_update_count, "next outer update count")
        if next_count != previous.outer_update_count + 1:
            raise ProfileAError("one fragment commit must advance exactly one")
        self._fragments[index] = FragmentProgressRecord(
            fragment_index=index,
            outer_update_count=next_count,
            accepted_tokens=previous.accepted_tokens
            + _positive(accepted_tokens, "accepted_tokens"),
            fresh_accepted_contributions=previous.fresh_accepted_contributions
            + _positive(fresh_accepted_contributions, "fresh accepted contributions"),
            stale_accepted_contributions=previous.stale_accepted_contributions
            + _ordinal(stale_accepted_contributions, "stale accepted contributions"),
        )

    def report(self) -> ProfileAProgressReport:
        fragments = tuple(self._fragments)
        return ProfileAProgressReport(
            learners=tuple(self._learners[item] for item in self._learner_ids),
            fragments=fragments,
            global_cycle=min(item.outer_update_count for item in fragments),
            comparison=self._comparison,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class StopDecision:
    stop: bool
    reason: str | None
    policy_identity: str
    observed_global_cycle: int
    observed_walltime_seconds: float
    observed_compute_steps: int
    observed_accepted_tokens: int
    observed_flops: int | None

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenStopPolicy:
    target_global_cycles: int | None = None
    maximum_walltime_seconds: float | None = None
    maximum_compute_steps: int | None = None
    maximum_accepted_tokens: int | None = None
    maximum_flops: int | None = None

    def __post_init__(self) -> None:
        if all(
            value is None
            for value in (
                self.target_global_cycles,
                self.maximum_walltime_seconds,
                self.maximum_compute_steps,
                self.maximum_accepted_tokens,
                self.maximum_flops,
            )
        ):
            raise ProfileAError("stop policy requires at least one frozen budget")
        for name in (
            "target_global_cycles",
            "maximum_compute_steps",
            "maximum_accepted_tokens",
            "maximum_flops",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive(value, name)
        if self.maximum_walltime_seconds is not None:
            if (
                _finite_nonnegative(
                    self.maximum_walltime_seconds, "maximum_walltime_seconds"
                )
                <= 0
            ):
                raise ProfileAError("maximum_walltime_seconds must be positive")

    @property
    def identity(self) -> str:
        return canonical_digest({"schema_version": 1, **dataclasses.asdict(self)})

    def evaluate(
        self,
        report: ProfileAProgressReport,
        *,
        walltime_seconds: float,
        observed_flops: int | None = None,
    ) -> StopDecision:
        if not isinstance(report, ProfileAProgressReport):
            raise ProfileAError("stop evaluation requires ProfileAProgressReport")
        walltime = _finite_nonnegative(walltime_seconds, "walltime_seconds")
        if observed_flops is not None:
            _ordinal(observed_flops, "observed_flops")
        reason = None
        if (
            self.target_global_cycles is not None
            and report.global_cycle >= self.target_global_cycles
        ):
            reason = "target_global_cycles"
        elif (
            self.maximum_walltime_seconds is not None
            and walltime >= self.maximum_walltime_seconds
        ):
            reason = "maximum_walltime_seconds"
        elif (
            self.maximum_compute_steps is not None
            and report.total_compute_steps >= self.maximum_compute_steps
        ):
            reason = "maximum_compute_steps"
        elif (
            self.maximum_accepted_tokens is not None
            and report.total_accepted_tokens >= self.maximum_accepted_tokens
        ):
            reason = "maximum_accepted_tokens"
        elif (
            self.maximum_flops is not None
            and observed_flops is not None
            and observed_flops >= self.maximum_flops
        ):
            reason = "maximum_flops"
        return StopDecision(
            stop=reason is not None,
            reason=reason,
            policy_identity=self.identity,
            observed_global_cycle=report.global_cycle,
            observed_walltime_seconds=walltime,
            observed_compute_steps=report.total_compute_steps,
            observed_accepted_tokens=report.total_accepted_tokens,
            observed_flops=observed_flops,
        )


class _SelectedProposalSource:
    prevalidated_payload_integrity = True

    def __init__(self, proposals: Sequence[Proposal]) -> None:
        self._payloads = {item.proposal_id: item.parameters for item in proposals}
        if len(self._payloads) != len(proposals):
            raise ProfileAError("selected proposal identities must be unique")
        self._opens = 0
        self._bytes = 0
        self._active = 0
        self._maximum_active = 0

    @contextlib.contextmanager
    def open_payload(self, contribution: ContributionFact) -> Iterator[bytes]:
        try:
            payload = self._payloads[contribution.proposal_id]
        except KeyError as error:
            raise ProfileAError("selected proposal payload is absent") from error
        self._opens += 1
        self._bytes += len(payload)
        self._active += 1
        self._maximum_active = max(self._maximum_active, self._active)
        try:
            yield payload
        finally:
            self._active -= 1

    def metrics(self) -> SourceMetrics:
        return SourceMetrics(
            opens=self._opens,
            bytes_read=self._bytes,
            active_payloads=self._active,
            maximum_active_payloads=self._maximum_active,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileAUpdate:
    previous: AtomicFragmentAuthority
    successor: AtomicFragmentAuthority
    result: FragmentUpdateResult
    selection: FrozenSelection
    source_metrics: SourceMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.selection, FrozenSelection):
            raise ProfileAError("Profile A update selection is invalid")
        if self.successor.version != self.previous.version + 1:
            raise ProfileAError("Profile A update did not advance exactly one")
        if self.result.parameters != self.successor.parameters:
            raise ProfileAError("merge result differs from committed successor")


class ProfileAFragmentExecutor:
    """Integrate discovery, Profile A readiness, merge, commit, and progress."""

    def __init__(
        self,
        *,
        atomic_store: AtomicGlobalCommitStore,
        proposal_store: ProposalStore,
        profile: ProfileAConfig,
        outer_policy: OuterSGDPolicy,
        progress: ProfileAProgressTracker,
        merge_backend: str = "torch",
    ) -> None:
        if not isinstance(atomic_store, AtomicGlobalCommitStore):
            raise ProfileAError("atomic_store is invalid")
        if not isinstance(proposal_store, ProposalStore):
            raise ProfileAError("proposal_store is invalid")
        if not isinstance(profile, ProfileAConfig):
            raise ProfileAError("profile is invalid")
        if not isinstance(outer_policy, OuterSGDPolicy):
            raise ProfileAError("outer_policy is invalid")
        if not isinstance(progress, ProfileAProgressTracker):
            raise ProfileAError("progress tracker is invalid")
        if merge_backend not in {"torch", "numpy"}:
            raise ProfileAError("merge_backend must be torch or numpy")
        if (
            profile.learner_count != len(atomic_store.learner_ids)
            or profile.fragment_count != len(atomic_store.store.descriptors)
            or atomic_store.learner_ids != proposal_store.learner_ids
            or atomic_store.store.identities != proposal_store.identities
            or atomic_store.store.descriptors != proposal_store.descriptors
            or atomic_store.store.s_max != 0
        ):
            raise ProfileAError("Profile A stores and frozen topology differ")
        expected_policy = profile_a_policy_identity(outer_policy)
        if atomic_store.policy_identity != expected_policy:
            raise ProfileAError("atomic policy identity differs from Profile A policy")
        snapshot = atomic_store.load_snapshot()
        authorities = tuple(
            FragmentReadinessAuthority(item.state, item.frontiers)
            for item in snapshot.authorities
        )
        self.atomic_store = atomic_store
        self.proposal_store = proposal_store
        self.profile = profile
        self.outer_policy = outer_policy
        self.progress = progress
        self.merge_backend = merge_backend
        self.readiness = SyncerReadinessMachine(
            proposal_store,
            authorities=authorities,
            config=ReadinessConfig(
                logical_syncer_id="profile-a-syncer",
                q=profile.q,
                q_fresh=profile.q_fresh,
                max_contributors=profile.learner_count,
                grace_period_ns=profile.grace_period_ns,
                s_max=profile.s_max,
                lambda_s=profile.lambda_s,
            ),
        )

    @staticmethod
    def _contributions(selection: FrozenSelection) -> tuple[ContributionFact, ...]:
        return tuple(
            ContributionFact(
                proposal_id=proposal.proposal_id,
                content_identity=proposal.content_identity,
                learner_id=proposal.learner_id,
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

    def poll(self, *, observed_ns: int | None = None) -> PollReport:
        return self.readiness.poll_store(observed_ns=observed_ns)

    def execute_next(
        self,
        *,
        observed_ns: int | None = None,
        poll_store: bool = True,
    ) -> ProfileAUpdate | None:
        if poll_store:
            self.poll(observed_ns=observed_ns)
        lease = self.readiness.claim_next()
        if lease is None:
            return None
        selection = lease.selection
        index = selection.fragment_index
        previous = self.atomic_store.load_fragment(index)
        if (
            previous.authority_identity != selection.authority_identity
            or previous.version != selection.current_version
        ):
            self.readiness.release(lease)
            raise ProfileAError("claimed selection differs from current authority")
        contributions = self._contributions(selection)
        source = _SelectedProposalSource(selection.proposals)
        outer_state = FragmentOuterState(
            update_count=previous.state.outer_update_count,
            momentum_buffer=(
                previous.outer_optimizer_state
                if previous.outer_optimizer_state
                else None
            ),
        )
        try:
            merge = (
                execute_numpy_streaming_fragment_update
                if self.merge_backend == "numpy"
                else execute_streaming_fragment_update
            )
            result = merge(
                FragmentMergeRequest(
                    descriptor=previous.state.descriptor,
                    fragment_map_identity=previous.state.identities.fragment_map_identity,
                    current_version=previous.version,
                    current_content_identity=previous.content_identity,
                    current_parameters=previous.parameters,
                    outer_state=outer_state,
                    selection_identity=selection.selection_identity,
                    contributions=contributions,
                ),
                source,
                self.outer_policy,
            )
            committed = self.atomic_store.commit(
                AtomicCommitRequest(
                    fragment_index=index,
                    expected_current_version=previous.version,
                    expected_current_content_identity=previous.content_identity,
                    next_version=previous.version + 1,
                    parameters=result.parameters,
                    outer_optimizer_state=(
                        b""
                        if result.outer_state.momentum_buffer is None
                        else result.outer_state.momentum_buffer
                    ),
                    selection=selection,
                    policy_identity=self.atomic_store.policy_identity,
                    update_identity=result.update_identity,
                )
            )
        except BaseException:
            self.readiness.release(lease)
            raise
        successor = committed.authority
        self.readiness.complete(
            lease,
            FragmentReadinessAuthority(successor.state, successor.frontiers),
        )
        fresh = sum(item.staleness == 0 for item in contributions)
        stale = len(contributions) - fresh
        self.progress.record_fragment_commit(
            index,
            next_outer_update_count=successor.state.outer_update_count,
            accepted_tokens=sum(item.processed_tokens for item in contributions),
            fresh_accepted_contributions=fresh,
            stale_accepted_contributions=stale,
        )
        return ProfileAUpdate(
            previous=previous,
            successor=successor,
            result=result,
            selection=selection,
            source_metrics=source.metrics(),
        )
