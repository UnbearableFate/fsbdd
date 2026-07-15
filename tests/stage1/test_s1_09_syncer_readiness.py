from __future__ import annotations

import dataclasses
import hashlib
import itertools
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from fsbdd.global_state import (
    BootstrapFragment,
    CountingStorageBackend,
    FragmentStateDescriptor,
    GlobalStateIdentities,
    GlobalStateStore,
)
from fsbdd.proposal import (
    ConsumptionFrontiers,
    Proposal,
    ProposalStore,
    commit_consumption,
)
from fsbdd.storage import PosixStorageBackend, PublicationNotReady
from fsbdd.syncer_readiness import (
    FragmentReadinessAuthority,
    ReadinessConfig,
    ReadinessError,
    ReadinessPhase,
    SyncerReadinessMachine,
)
from fsbdd.syncer_readiness_stress import _validate_frozen_selection


IDENTITIES = GlobalStateIdentities(
    run_identity="s1-09-unit",
    config_identity="a" * 64,
    model_identity="b" * 64,
    fragment_map_identity="c" * 64,
)


def _descriptors(fragment_count: int) -> tuple[FragmentStateDescriptor, ...]:
    return tuple(
        FragmentStateDescriptor(
            index=index,
            identity=f"fragment-{index}",
            dtype="uint8",
            shape=(16,),
            parameter_identities=(f"parameter-{index}",),
        )
        for index in range(fragment_count)
    )


def _system(
    tmp_path: Path,
    *,
    learner_count: int = 4,
    fragment_count: int = 2,
    q: int = 2,
    q_fresh: int | None = None,
    max_contributors: int | None = None,
    grace_period_ns: int = 0,
    counting: bool = False,
):
    learner_ids = tuple(f"learner-{index:02d}" for index in range(learner_count))
    descriptors = _descriptors(fragment_count)
    global_store = GlobalStateStore(
        PosixStorageBackend(tmp_path / "global"),
        identities=IDENTITIES,
        descriptors=descriptors,
        s_max=0,
    )
    states = global_store.bootstrap(
        tuple(
            BootstrapFragment(
                descriptor=descriptor,
                parameters=bytes([descriptor.index]) * 16,
                outer_state=b"outer",
            )
            for descriptor in descriptors
        )
    ).snapshot.states
    backend = PosixStorageBackend(tmp_path / "proposals")
    proposal_backend = CountingStorageBackend(backend) if counting else backend
    proposal_store = ProposalStore(
        proposal_backend,
        identities=IDENTITIES,
        descriptors=descriptors,
        learner_ids=learner_ids,
        maximum_local_steps=1000,
        maximum_processed_tokens=1_000_000,
    )
    authorities = tuple(
        FragmentReadinessAuthority(
            state,
            ConsumptionFrontiers.empty(
                identities=IDENTITIES,
                descriptor=state.descriptor,
                learner_ids=learner_ids,
            ),
        )
        for state in states
    )
    config = ReadinessConfig(
        logical_syncer_id="syncer-0",
        q=q,
        q_fresh=q if q_fresh is None else q_fresh,
        max_contributors=(
            learner_count if max_contributors is None else max_contributors
        ),
        grace_period_ns=grace_period_ns,
    )
    machine = SyncerReadinessMachine(
        proposal_store, authorities=authorities, config=config
    )
    return machine, proposal_store, authorities, proposal_backend


def _proposal(
    store: ProposalStore,
    authorities: tuple[FragmentReadinessAuthority, ...],
    learner_index: int,
    fragment_index: int,
    *,
    sequence: int = 1,
    proposal_id: str | None = None,
    tokens: int | None = None,
) -> Proposal:
    authority = authorities[fragment_index]
    learner_id = store.learner_ids[learner_index]
    return Proposal.create(
        proposal_id=(
            proposal_id
            or f"{learner_id}-fragment-{fragment_index}-sequence-{sequence}"
        ),
        identities=store.identities,
        learner_id=learner_id,
        descriptor=store.descriptors[fragment_index],
        sequence=sequence,
        base_version=authority.state.version,
        base_content_identity=authority.state.content_identity,
        local_steps=sequence,
        processed_tokens=(
            (learner_index + 1) * 100 + sequence if tokens is None else tokens
        ),
        snapshot_local_step=sequence,
        parameters=bytes([learner_index + fragment_index + sequence]) * 16,
    )


def _all(
    store: ProposalStore,
    authorities: tuple[FragmentReadinessAuthority, ...],
) -> tuple[Proposal, ...]:
    return tuple(
        _proposal(store, authorities, learner, fragment)
        for learner in range(len(store.learner_ids))
        for fragment in range(len(store.descriptors))
    )


def test_config_freezes_stage1_smax_and_valid_thresholds() -> None:
    with pytest.raises(ReadinessError, match="s_max=0"):
        ReadinessConfig("syncer", 1, 1, 1, 0, s_max=1)
    with pytest.raises(ReadinessError, match="q_fresh"):
        ReadinessConfig("syncer", 2, 3, 3, 0)
    with pytest.raises(ReadinessError, match="non-empty"):
        ReadinessConfig("", 1, 1, 1, 0)


def test_distinct_learner_quorum_ignores_duplicate_files(tmp_path: Path) -> None:
    machine, store, authorities, _ = _system(
        tmp_path, learner_count=3, fragment_count=1, q=3
    )
    one = _proposal(store, authorities, 0, 0, sequence=1)
    replacement = _proposal(store, authorities, 0, 0, sequence=2)
    two = _proposal(store, authorities, 1, 0)
    report = machine.observe((one, replacement, two), observed_ns=0)
    view = report.fragments[0]
    assert view.phase == ReadinessPhase.WAITING
    assert view.eligible_distinct == view.fresh_distinct == 2
    assert view.rejected == 1
    report = machine.observe(
        (one, replacement, two, _proposal(store, authorities, 2, 0)),
        observed_ns=0,
    )
    assert report.fragments[0].phase == ReadinessPhase.FROZEN
    selection = machine.selection_for(0)
    assert selection is not None
    assert len(selection.proposals) == len(set(selection.learner_ids)) == 3
    assert replacement.proposal_id in {
        item.proposal_id for item in selection.proposals
    }


def test_exact_duplicate_observation_is_one_and_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    machine, store, authorities, _ = _system(
        tmp_path, learner_count=2, fragment_count=1, q=2
    )
    first = _proposal(store, authorities, 0, 0, proposal_id="same")
    report = machine.observe((first, first), observed_ns=0)
    assert report.fragments[0].eligible_distinct == 1
    assert report.fragments[0].duplicate_observations == 1
    conflicting = _proposal(
        store, authorities, 0, 0, sequence=2, proposal_id="same"
    )
    with pytest.raises(ReadinessError, match="conflicting duplicate"):
        machine.observe((first, conflicting), observed_ns=1)


def test_cpu_poll_count_never_substitutes_for_quorum(tmp_path: Path) -> None:
    machine, _, _, _ = _system(
        tmp_path, learner_count=4, fragment_count=1, q=3
    )
    for index in range(1000):
        report = machine.observe((), observed_ns=index)
    assert report.fragments[0].phase == ReadinessPhase.WAITING
    assert machine.claim_next() is None
    assert machine.snapshot().poll_cycles == 1000


@pytest.mark.parametrize("q_fresh", [0, 1, 2, 3])
def test_smax_zero_makes_every_eligible_distinct_learner_fresh(
    tmp_path: Path, q_fresh: int
) -> None:
    machine, store, authorities, _ = _system(
        tmp_path,
        learner_count=3,
        fragment_count=1,
        q=3,
        q_fresh=q_fresh,
    )
    fresh = tuple(_proposal(store, authorities, index, 0) for index in range(3))
    below_quorum = machine.observe(fresh[:2], observed_ns=0).fragments[0]
    assert below_quorum.phase == ReadinessPhase.WAITING
    assert below_quorum.eligible_distinct == below_quorum.fresh_distinct == 2
    ready = machine.observe(fresh, observed_ns=1).fragments[0]
    assert ready.phase == ReadinessPhase.FROZEN
    assert ready.eligible_distinct == ready.fresh_distinct == 3
    assert machine.config.s_max == 0


def test_profile_a_freezes_every_learner_immediately_and_is_order_independent(
    tmp_path: Path,
) -> None:
    machine, store, authorities, _ = _system(
        tmp_path, learner_count=4, fragment_count=1, q=4
    )
    proposals = _all(store, authorities)
    identities = set()
    orders = itertools.permutations(proposals)
    for order_index, ordering in enumerate(orders):
        replay, _, replay_authorities, _ = _system(
            tmp_path / f"replay-{order_index}",
            learner_count=4,
            fragment_count=1,
            q=4,
        )
        replay_proposals = tuple(
            _proposal(
                replay.store,
                replay_authorities,
                store.learner_ids.index(item.learner_id),
                0,
                tokens=item.processed_tokens,
            )
            for item in ordering
        )
        replay.observe(replay_proposals, observed_ns=100 + order_index)
        frozen = replay.selection_for(0)
        assert frozen is not None
        identities.add(frozen.selection_identity)
        assert len(frozen.proposals) == 4
        assert abs(sum(item.normalized_weight for item in frozen.weights) - 1.0) < 2e-6
        _validate_frozen_selection(
            frozen.to_dict(),
            profile={"learner_count": 4, "logical_syncer_id": "syncer-0"},
            fragment_index=0,
            expected_count=4,
        )
    assert len(identities) == 1
    machine.observe(proposals, observed_ns=0)
    assert machine.selection_for(0) is not None


def test_fixed_grace_includes_pre_freeze_and_excludes_post_freeze_arrivals(
    tmp_path: Path,
) -> None:
    machine, store, authorities, _ = _system(
        tmp_path,
        learner_count=6,
        fragment_count=1,
        q=3,
        grace_period_ns=10,
    )
    first = tuple(_proposal(store, authorities, index, 0) for index in range(3))
    assert machine.observe(first, observed_ns=100).fragments[0].phase == ReadinessPhase.GRACE
    before_freeze = (*first, _proposal(store, authorities, 3, 0))
    machine.observe(before_freeze, observed_ns=109)
    assert machine.selection_for(0) is None
    machine.observe(before_freeze, observed_ns=110)
    frozen = machine.selection_for(0)
    assert frozen is not None and len(frozen.proposals) == 4
    original = dataclasses.asdict(frozen)
    after_freeze = (*before_freeze, _proposal(store, authorities, 4, 0, sequence=9))
    machine.observe(after_freeze, observed_ns=10_000)
    assert dataclasses.asdict(machine.selection_for(0)) == original


def test_quorum_loss_cancels_grace_and_recovery_gets_full_interval(
    tmp_path: Path,
) -> None:
    machine, store, authorities, _ = _system(
        tmp_path,
        learner_count=4,
        fragment_count=1,
        q=3,
        grace_period_ns=10,
    )
    quorum = tuple(_proposal(store, authorities, index, 0) for index in range(3))
    machine.observe(quorum, observed_ns=0)
    lost = machine.observe(quorum[:2], observed_ns=5).fragments[0]
    assert lost.phase == ReadinessPhase.WAITING
    assert lost.grace_started_ns is None
    recovered = machine.observe(quorum, observed_ns=8).fragments[0]
    assert recovered.grace_deadline_ns == 18
    assert machine.observe(quorum, observed_ns=17).fragments[0].phase == ReadinessPhase.GRACE
    assert machine.observe(quorum, observed_ns=18).fragments[0].phase == ReadinessPhase.FROZEN


def test_round_robin_bounds_continuously_ready_service_window(
    tmp_path: Path,
) -> None:
    machine, store, authorities, _ = _system(
        tmp_path, learner_count=2, fragment_count=4, q=2
    )
    machine.observe(_all(store, authorities), observed_ns=0)
    claims = []
    for _ in range(32):
        lease = machine.claim_next()
        assert lease is not None
        claims.append(lease.selection.fragment_index)
        assert machine.claim_next() is None
        machine.release(lease)
    assert claims == [0, 1, 2, 3] * 8
    for start in range(len(claims) - 3):
        assert set(claims[start : start + 4]) == {0, 1, 2, 3}


def test_concurrent_claim_attempts_create_exactly_one_active_lease(
    tmp_path: Path,
) -> None:
    machine, store, authorities, _ = _system(
        tmp_path, learner_count=2, fragment_count=2, q=2
    )
    machine.observe(_all(store, authorities), observed_ns=0)
    with ThreadPoolExecutor(max_workers=16) as workers:
        leases = list(workers.map(lambda _index: machine.claim_next(), range(64)))
    claimed = [item for item in leases if item is not None]
    assert len(claimed) == 1
    assert machine.active_lease == claimed[0]
    machine.release(claimed[0])


def test_mixed_fragment_authority_progress_is_independent_and_once_only(
    tmp_path: Path,
) -> None:
    machine, store, authorities, _ = _system(
        tmp_path, learner_count=2, fragment_count=2, q=2
    )
    proposals = _all(store, authorities)
    machine.observe(proposals, observed_ns=0)
    fragment_one_identity = machine.selection_for(1).selection_identity  # type: ignore[union-attr]
    lease = machine.claim_next()
    assert lease is not None and lease.selection.fragment_index == 0
    consumed = commit_consumption(
        authorities[0].frontiers, lease.selection.proposals, True
    )
    advanced = FragmentReadinessAuthority(authorities[0].state, consumed)
    machine.complete(lease, advanced)
    snapshot = machine.snapshot()
    assert snapshot.fragments[0].phase == ReadinessPhase.WAITING
    assert snapshot.fragments[1].phase == ReadinessPhase.FROZEN
    assert machine.selection_for(1).selection_identity == fragment_one_identity  # type: ignore[union-attr]
    machine.observe(proposals, observed_ns=1)
    assert machine.selection_for(0) is None
    assert machine.snapshot().fragments[0].eligible_distinct == 0
    assert len(advanced.frontiers.entries) == 2


def test_mixed_global_versions_share_no_atomic_version_gate(tmp_path: Path) -> None:
    machine, store, authorities, _ = _system(
        tmp_path, learner_count=2, fragment_count=2, q=2
    )
    global_store = GlobalStateStore(
        PosixStorageBackend(tmp_path / "global"),
        identities=IDENTITIES,
        descriptors=store.descriptors,
        s_max=0,
    )
    successor = global_store.publish_successor(
        0, parameters=b"new-fragment-zero", outer_state=b"outer-one"
    )
    advanced = FragmentReadinessAuthority(
        successor,
        ConsumptionFrontiers.empty(
            identities=IDENTITIES,
            descriptor=successor.descriptor,
            learner_ids=store.learner_ids,
        ),
    )
    assert machine.install_authority(0, advanced) is True
    updated = (advanced, authorities[1])
    proposals = tuple(
        _proposal(store, updated, learner, fragment)
        for learner in range(2)
        for fragment in range(2)
    )
    report = machine.observe(proposals, observed_ns=0)
    assert tuple(item.current_version for item in report.fragments) == (1, 0)
    assert tuple(item.phase for item in report.fragments) == (
        ReadinessPhase.FROZEN,
        ReadinessPhase.FROZEN,
    )


def test_restart_rebuilds_from_authority_and_conservatively_restarts_grace(
    tmp_path: Path,
) -> None:
    machine, store, authorities, _ = _system(
        tmp_path,
        learner_count=3,
        fragment_count=1,
        q=2,
        grace_period_ns=10,
    )
    proposals = tuple(_proposal(store, authorities, index, 0) for index in range(2))
    machine.observe(proposals, observed_ns=5)
    restarted = SyncerReadinessMachine(
        store, authorities=authorities, config=machine.config
    )
    view = restarted.observe(proposals, observed_ns=100).fragments[0]
    assert view.phase == ReadinessPhase.GRACE
    assert view.grace_started_ns == 100 and view.grace_deadline_ns == 110
    assert restarted.observe(proposals, observed_ns=109).fragments[0].phase == ReadinessPhase.GRACE
    assert restarted.observe(proposals, observed_ns=110).fragments[0].phase == ReadinessPhase.FROZEN


def test_equal_clock_ties_are_deterministic_and_regression_is_rejected(
    tmp_path: Path,
) -> None:
    machine, store, authorities, _ = _system(
        tmp_path, learner_count=2, fragment_count=3, q=2
    )
    proposals = _all(store, authorities)
    machine.observe(proposals, observed_ns=7)
    machine.observe(tuple(reversed(proposals)), observed_ns=7)
    assert [machine.claim_next().selection.fragment_index] == [0]  # type: ignore[union-attr]
    with pytest.raises(ReadinessError, match="regressed"):
        machine.observe(proposals, observed_ns=6)


def test_authority_regressions_and_wrong_active_completion_fail_closed(
    tmp_path: Path,
) -> None:
    machine, store, authorities, _ = _system(
        tmp_path, learner_count=2, fragment_count=1, q=2
    )
    proposals = _all(store, authorities)
    machine.observe(proposals, observed_ns=0)
    lease = machine.claim_next()
    assert lease is not None
    with pytest.raises(ReadinessError, match="did not advance"):
        machine.complete(lease, authorities[0])
    assert machine.active_lease == lease
    machine.release(lease)
    consumed = FragmentReadinessAuthority(
        authorities[0].state,
        commit_consumption(authorities[0].frontiers, proposals, True),
    )
    assert machine.install_authority(0, consumed) is True
    with pytest.raises(ReadinessError, match="sequence regressed"):
        machine.install_authority(0, authorities[0])


def test_poll_store_reads_exact_fixed_slots_independent_of_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    machine, store, authorities, backend = _system(
        tmp_path,
        learner_count=3,
        fragment_count=2,
        q=3,
        counting=True,
    )
    for proposal in _all(store, authorities):
        store.publish(proposal)
    backend.reset_counts()
    first = machine.poll_store(observed_ns=0)
    assert first.fixed_slot_reads == backend.read_calls == 6
    payload_root = tmp_path / "proposals" / "payloads"
    for index in range(10_000):
        (payload_root / f"unrelated-{index:05d}.bin").write_bytes(b"history")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("readiness polling must not scan directories")

    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(os, "listdir", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    monkeypatch.setattr(os, "walk", forbidden)
    backend.reset_counts()
    second = machine.poll_store(observed_ns=1)
    assert second.fixed_slot_reads == backend.read_calls == 6
    assert machine.snapshot().resident_selected_proposals <= 6
    assert machine.snapshot().fixed_slot_reads == 12


def test_transiently_unavailable_slot_is_retried_without_stopping_polling(
    tmp_path: Path,
) -> None:
    _, store, authorities, backend = _system(
        tmp_path, learner_count=2, fragment_count=1, q=2
    )
    for proposal in _all(store, authorities):
        store.publish(proposal)

    class OnceUnavailable:
        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.injected = False

        def publish(self, *args, **kwargs):
            return self.delegate.publish(*args, **kwargs)

        def read(self, *args, **kwargs):
            if not self.injected:
                self.injected = True
                raise PublicationNotReady("injected eventually-readable slot")
            return self.delegate.read(*args, **kwargs)

    retry_store = ProposalStore(
        OnceUnavailable(backend),
        identities=store.identities,
        descriptors=store.descriptors,
        learner_ids=store.learner_ids,
        maximum_local_steps=store.maximum_local_steps,
        maximum_processed_tokens=store.maximum_processed_tokens,
    )
    machine = SyncerReadinessMachine(
        retry_store, authorities=authorities, config=ReadinessConfig("syncer", 2, 2, 2, 0)
    )
    first = machine.poll_store(observed_ns=0)
    assert first.fixed_slot_reads == 2
    assert first.transient_unavailable_slots == 1
    assert first.fragments[0].phase == ReadinessPhase.WAITING
    second = machine.poll_store(observed_ns=1)
    assert second.transient_unavailable_slots == 0
    assert second.fragments[0].phase == ReadinessPhase.FROZEN


def test_selection_identity_excludes_wall_clock_and_restart_history(
    tmp_path: Path,
) -> None:
    first, store, authorities, _ = _system(
        tmp_path, learner_count=3, fragment_count=1, q=3
    )
    proposals = _all(store, authorities)
    first.observe(proposals, observed_ns=1)
    first_selection = first.selection_for(0)
    restarted = SyncerReadinessMachine(
        store, authorities=authorities, config=first.config
    )
    restarted.observe(tuple(reversed(proposals)), observed_ns=10**12)
    second_selection = restarted.selection_for(0)
    assert first_selection is not None and second_selection is not None
    assert first_selection.selection_identity == second_selection.selection_identity
    assert first_selection.proposals == second_selection.proposals
    assert first_selection.weights == second_selection.weights


def test_authority_identity_binds_fixed_frontiers(tmp_path: Path) -> None:
    _, store, authorities, _ = _system(
        tmp_path, learner_count=2, fragment_count=1, q=2
    )
    proposals = _all(store, authorities)
    consumed = commit_consumption(authorities[0].frontiers, proposals, True)
    changed = FragmentReadinessAuthority(authorities[0].state, consumed)
    assert changed.identity != authorities[0].identity
    assert hashlib.sha256(changed.identity.encode()).hexdigest()
