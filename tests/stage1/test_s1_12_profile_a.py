from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from fsbdd.global_commit import AtomicGlobalCommitStore
from fsbdd.global_state import (
    BootstrapFragment,
    FragmentStateDescriptor,
    GlobalStateIdentities,
    GlobalStateStore,
)
from fsbdd.profile_a import (
    ComparisonBasis,
    FrozenStopPolicy,
    ProfileAConfig,
    ProfileAError,
    ProfileAFragmentExecutor,
    ProfileAProgressTracker,
    profile_a_policy_identity,
)
from fsbdd.proposal import Proposal, ProposalStore
from fsbdd.storage import PosixStorageBackend
from fsbdd.syncer_merge import OuterSGDPolicy
from fsbdd_stage0.oracle import (
    OuterSGDState,
    inverse_staleness_weights,
    outer_sgd_step,
    weighted_direct_merge,
)


LEARNERS = tuple(f"learner-{index:02d}" for index in range(4))
POLICY = OuterSGDPolicy(learning_rate=0.1, momentum=0.9, nesterov=True)
COMPARISON = ComparisonBasis(
    matched_compute=False,
    matched_communication=False,
    matched_tokens=False,
    claim="protocol_and_runtime_acceptance_only_no_quality_or_efficiency_comparison",
)


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _payload(values: list[float]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _values(payload: bytes) -> list[float]:
    return np.frombuffer(payload, dtype="<f4").astype(np.float64).tolist()


def _relative_l2(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(float(np.linalg.norm(right_array)), 1e-12)
    )


def make_system(
    root: Path, *, elements: int = 16
) -> tuple[AtomicGlobalCommitStore, ProposalStore, ProfileAFragmentExecutor]:
    identities = GlobalStateIdentities(
        run_identity="s1-12-unit",
        config_identity=_identity("s1-12-config"),
        model_identity=_identity("s1-12-model"),
        fragment_map_identity=_identity("s1-12-map"),
    )
    descriptors = tuple(
        FragmentStateDescriptor(
            index=index,
            identity=_identity(f"fragment-{index}"),
            dtype="float32",
            shape=(elements,),
            parameter_identities=(f"parameter-{index}",),
        )
        for index in range(2)
    )
    atomic = AtomicGlobalCommitStore(
        GlobalStateStore(
            PosixStorageBackend(root / "global"),
            identities=identities,
            descriptors=descriptors,
            s_max=0,
        ),
        learner_ids=LEARNERS,
        policy_identity=profile_a_policy_identity(POLICY),
    )
    atomic.bootstrap(
        tuple(
            BootstrapFragment(
                descriptor=descriptor,
                parameters=_payload(
                    [
                        float(np.float32((index + 1) * 0.25 + position * 0.01))
                        for position in range(elements)
                    ]
                ),
                outer_state=b"",
            )
            for index, descriptor in enumerate(descriptors)
        )
    )
    proposals = ProposalStore(
        PosixStorageBackend(root / "proposals"),
        identities=identities,
        descriptors=descriptors,
        learner_ids=LEARNERS,
        maximum_local_steps=1_000_000,
        maximum_processed_tokens=1_000_000_000,
    )
    progress = ProfileAProgressTracker(
        LEARNERS, (0, 0), comparison=COMPARISON
    )
    executor = ProfileAFragmentExecutor(
        atomic_store=atomic,
        proposal_store=proposals,
        profile=ProfileAConfig(4, 2, 4, 4, 0, 0, 1.0),
        outer_policy=POLICY,
        progress=progress,
    )
    return atomic, proposals, executor


def publish_round(
    atomic: AtomicGlobalCommitStore,
    proposals: ProposalStore,
    fragment_index: int,
) -> tuple[list[list[float]], list[int]]:
    authority = atomic.load_fragment(fragment_index)
    current = _values(authority.parameters)
    locals_: list[list[float]] = []
    tokens: list[int] = []
    for learner_index, learner_id in enumerate(LEARNERS):
        tokens.append((learner_index + 1) * 7 + authority.version)
        gradient = [
            float(
                np.float32(
                    (((authority.version + 1) * (learner_index + 1) * (position + 3)) % 23 - 11)
                    * 0.0005
                )
            )
            for position in range(len(current))
        ]
        local = [
            float(np.float32(value - delta))
            for value, delta in zip(current, gradient, strict=True)
        ]
        locals_.append(local)
        proposals.publish(
            Proposal.create(
                proposal_id=(
                    f"{learner_id}-fragment-{fragment_index}-"
                    f"version-{authority.version}-sequence-{authority.version + 1}"
                ),
                identities=authority.state.identities,
                learner_id=learner_id,
                descriptor=authority.state.descriptor,
                sequence=authority.version + 1,
                base_version=authority.version,
                base_content_identity=authority.content_identity,
                local_steps=2,
                processed_tokens=tokens[-1],
                snapshot_local_step=2 * (authority.version + 1),
                parameters=_payload(local),
            )
        )
    return locals_, tokens


def test_profile_a_rejects_relaxed_quorum_staleness_or_schedule() -> None:
    with pytest.raises(ProfileAError, match="q =="):
        ProfileAConfig(4, 2, 3, 3, 0, 0, 1.0)
    with pytest.raises(ProfileAError, match="s_max"):
        ProfileAConfig(4, 2, 4, 4, 1, 0, 1.0)
    with pytest.raises(ProfileAError, match="zero grace"):
        ProfileAConfig(4, 2, 4, 4, 0, 1, 1.0)
    with pytest.raises(ProfileAError, match="round_robin"):
        ProfileAConfig(4, 2, 4, 4, 0, 0, 1.0, "random")


def test_progress_is_typed_per_learner_and_per_fragment() -> None:
    tracker = ProfileAProgressTracker(LEARNERS, (3, 7), comparison=COMPARISON)
    tracker.update_learner(
        LEARNERS[0],
        local_optimizer_steps=100,
        processed_input_tokens=6400,
        loss_bearing_target_tokens=6300,
    )
    before = tracker.report()
    assert before.global_cycle == 3
    tracker.record_fragment_commit(
        0,
        next_outer_update_count=4,
        accepted_tokens=1000,
        fresh_accepted_contributions=4,
        stale_accepted_contributions=0,
    )
    after = tracker.report()
    assert [item.outer_update_count for item in after.fragments] == [4, 7]
    assert after.global_cycle == 4
    assert after.learners[0].local_optimizer_steps == 100
    assert after.fragments[0].accepted_tokens == 1000
    assert after.fragments[0].fresh_accepted_contributions == 4
    assert after.fragments[0].stale_accepted_contributions == 0
    assert "global_step" not in after.to_dict()


def test_stop_policy_does_not_use_fastest_learner_step() -> None:
    tracker = ProfileAProgressTracker(LEARNERS, (2, 2), comparison=COMPARISON)
    tracker.update_learner(
        LEARNERS[0],
        local_optimizer_steps=1_000_000,
        processed_input_tokens=64_000_000,
        loss_bearing_target_tokens=63_000_000,
    )
    policy = FrozenStopPolicy(
        target_global_cycles=3,
        maximum_walltime_seconds=100,
        maximum_compute_steps=2_000_000,
        maximum_accepted_tokens=1_000_000,
    )
    decision = policy.evaluate(tracker.report(), walltime_seconds=1)
    assert not decision.stop
    tracker.record_fragment_commit(
        0,
        next_outer_update_count=3,
        accepted_tokens=1,
        fresh_accepted_contributions=4,
        stale_accepted_contributions=0,
    )
    tracker.record_fragment_commit(
        1,
        next_outer_update_count=3,
        accepted_tokens=1,
        fresh_accepted_contributions=4,
        stale_accepted_contributions=0,
    )
    decision = policy.evaluate(tracker.report(), walltime_seconds=2)
    assert decision.stop and decision.reason == "target_global_cycles"


def test_integrated_single_update_matches_stage0_oracle(tmp_path: Path) -> None:
    atomic, proposals, executor = make_system(tmp_path)
    before = atomic.load_fragment(0)
    current = _values(before.parameters)
    locals_, tokens = publish_round(atomic, proposals, 0)
    update = executor.execute_next(observed_ns=1)
    assert update is not None and update.successor.state.descriptor.index == 0
    weights = inverse_staleness_weights(tokens, [0] * 4, 1.0)
    merged = weighted_direct_merge(
        current=current,
        bases=[current] * 4,
        locals_=locals_,
        weights=weights,
    )
    expected, state = outer_sgd_step(
        current,
        merged,
        OuterSGDState(None),
        learning_rate=0.1,
        momentum=0.9,
        nesterov=True,
    )
    assert _relative_l2(_values(update.successor.parameters), expected) <= 1e-6
    assert update.successor.outer_optimizer_state
    assert _relative_l2(_values(update.successor.outer_optimizer_state), state.momentum_buffer) <= 1e-6
    assert update.source_metrics.opens == 4
    assert update.source_metrics.maximum_active_payloads == 1
    assert update.result.byte_accounting.full_model_operations == 0
    assert executor.progress.report().global_cycle == 0
    assert [
        item.outer_update_count for item in executor.progress.report().fragments
    ] == [1, 0]


def test_fifty_integrated_updates_per_fragment_do_not_amplify_drift(
    tmp_path: Path,
) -> None:
    atomic, proposals, executor = make_system(tmp_path, elements=32)
    oracle_parameters = [
        _values(atomic.load_fragment(index).parameters) for index in range(2)
    ]
    oracle_states = [OuterSGDState(None), OuterSGDState(None)]
    errors: list[list[float]] = [[], []]
    for cycle in range(50):
        expected_by_fragment = []
        for index in range(2):
            locals_, tokens = publish_round(atomic, proposals, index)
            weights = inverse_staleness_weights(tokens, [0] * 4, 1.0)
            merged = weighted_direct_merge(
                current=oracle_parameters[index],
                bases=[oracle_parameters[index]] * 4,
                locals_=locals_,
                weights=weights,
            )
            expected, state = outer_sgd_step(
                oracle_parameters[index],
                merged,
                oracle_states[index],
                learning_rate=0.1,
                momentum=0.9,
                nesterov=True,
            )
            expected_by_fragment.append((expected, state))
        for expected_index in range(2):
            update = executor.execute_next(observed_ns=cycle * 2 + expected_index + 1)
            assert update is not None
            index = update.successor.state.descriptor.index
            expected, state = expected_by_fragment[index]
            error = _relative_l2(_values(update.successor.parameters), expected)
            errors[index].append(error)
            oracle_parameters[index] = expected
            oracle_states[index] = state
    report = executor.progress.report()
    assert [item.outer_update_count for item in report.fragments] == [50, 50]
    assert report.global_cycle == 50
    assert all(item.stale_accepted_contributions == 0 for item in report.fragments)
    for trace in errors:
        assert len(trace) == 50
        assert max(trace) <= 1e-6
        assert max(trace[-10:]) <= max(max(trace[:10]), 1e-7)


def test_unmatched_comparison_cannot_claim_benefit() -> None:
    with pytest.raises(ProfileAError, match="acceptance"):
        ComparisonBasis(False, False, False, "faster_and_better")
