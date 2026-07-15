from __future__ import annotations

import dataclasses
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from fsbdd.global_commit import (
    AtomicCommitError,
    AtomicCommitRequest,
    AtomicGlobalCommitStore,
    decode_commit_envelope,
    encode_commit_envelope,
)
from fsbdd.global_state import (
    BootstrapFragment,
    FragmentStateDescriptor,
    GlobalStateIdentities,
    GlobalStateStore,
)
from fsbdd.identity import canonical_digest
from fsbdd.proposal import (
    EligibilityPolicy,
    Proposal,
    RetainedBaseIdentity,
    compute_candidate_weights,
    select_candidates,
)
from fsbdd.storage import PosixStorageBackend, PublicationInterrupted
from fsbdd.syncer_readiness import FrozenSelection


IDENTITIES = GlobalStateIdentities(
    run_identity="s1-11-unit",
    config_identity=hashlib.sha256(b"s1-11-config").hexdigest(),
    model_identity=hashlib.sha256(b"s1-11-model").hexdigest(),
    fragment_map_identity=hashlib.sha256(b"s1-11-map").hexdigest(),
)
LEARNERS = tuple(f"learner-{index}" for index in range(4))
POLICY_IDENTITY = canonical_digest(
    {
        "merge": "direct_weighted_average",
        "outer_optimizer": "sgd",
        "learning_rate": 0.1,
        "momentum": 0.9,
        "nesterov": True,
    }
)


def _descriptor(index: int = 0) -> FragmentStateDescriptor:
    return FragmentStateDescriptor(
        index=index,
        identity=hashlib.sha256(f"fragment-{index}".encode()).hexdigest(),
        dtype="uint8",
        shape=(16,),
        parameter_identities=(f"parameter-{index}",),
    )


def _system(
    root: Path, *, fragment_count: int = 1
) -> tuple[AtomicGlobalCommitStore, PosixStorageBackend]:
    descriptors = tuple(_descriptor(index) for index in range(fragment_count))
    backend = PosixStorageBackend(root)
    store = AtomicGlobalCommitStore(
        GlobalStateStore(
            backend,
            identities=IDENTITIES,
            descriptors=descriptors,
            s_max=0,
        ),
        learner_ids=LEARNERS,
        policy_identity=POLICY_IDENTITY,
    )
    result = store.bootstrap(
        tuple(
            BootstrapFragment(
                descriptor=descriptor,
                parameters=bytes([descriptor.index]) * 16,
                outer_state=f"outer-{descriptor.index}-0".encode(),
            )
            for descriptor in descriptors
        )
    )
    assert result.published_indices == tuple(range(fragment_count))
    return store, backend


def _proposal(
    authority,
    learner_id: str,
    *,
    sequence: int | None = None,
    proposal_id: str | None = None,
    tokens: int = 100,
) -> Proposal:
    resolved_sequence = authority.version if sequence is None else sequence
    return Proposal.create(
        proposal_id=(
            proposal_id
            or f"{learner_id}-v{authority.version}-s{resolved_sequence}"
        ),
        identities=authority.state.identities,
        learner_id=learner_id,
        descriptor=authority.state.descriptor,
        sequence=resolved_sequence,
        base_version=authority.version,
        base_content_identity=authority.content_identity,
        local_steps=1,
        processed_tokens=tokens,
        snapshot_local_step=1,
        parameters=bytes([tokens % 251]) * len(authority.parameters),
    )


def _selection(
    authority,
    proposals: tuple[Proposal, ...],
    *,
    generation: int | None = None,
) -> FrozenSelection:
    weights = compute_candidate_weights(
        proposals, current_version=authority.version, lambda_s=1.0
    )
    semantic = {
        "schema_version": 1,
        "logical_syncer_id": "syncer-0",
        "fragment_index": authority.state.descriptor.index,
        "current_version": authority.version,
        "authority_identity": authority.authority_identity,
        "proposal_content_identities": [item.content_identity for item in proposals],
        "weights": [item.to_dict() for item in weights],
    }
    return FrozenSelection(
        fragment_index=authority.state.descriptor.index,
        current_version=authority.version,
        authority_identity=authority.authority_identity,
        generation=authority.version + 1 if generation is None else generation,
        grace_started_ns=authority.version,
        frozen_observed_ns=authority.version,
        proposals=proposals,
        weights=weights,
        selection_identity=canonical_digest(semantic),
    )


def _request(
    authority,
    *,
    learners: tuple[str, ...] = LEARNERS,
    marker: str | None = None,
) -> AtomicCommitRequest:
    tag = marker or f"version-{authority.version + 1}"
    proposals = tuple(
        _proposal(
            authority,
            learner_id,
            sequence=authority.version,
            tokens=100 + index,
        )
        for index, learner_id in enumerate(learners)
    )
    selection = _selection(authority, proposals)
    return AtomicCommitRequest(
        fragment_index=authority.state.descriptor.index,
        expected_current_version=authority.version,
        expected_current_content_identity=authority.content_identity,
        next_version=authority.version + 1,
        parameters=hashlib.sha256(f"parameters:{tag}".encode()).digest()[:16],
        outer_optimizer_state=f"outer:{tag}".encode(),
        selection=selection,
        policy_identity=POLICY_IDENTITY,
        update_identity=hashlib.sha256(f"update:{tag}".encode()).hexdigest(),
    )


def _eligibility(authority, proposal: Proposal) -> tuple[bool, dict[str, str]]:
    result = select_candidates(
        (proposal,),
        authority.frontiers,
        EligibilityPolicy(
            identities=authority.state.identities,
            descriptor=authority.state.descriptor,
            learner_ids=LEARNERS,
            current_version=authority.version,
            retained_bases=(
                RetainedBaseIdentity(
                    authority.version, authority.content_identity
                ),
            ),
            s_max=0,
            q=1,
            q_fresh=1,
            max_contributors=1,
            maximum_local_steps=100,
            maximum_processed_tokens=1_000_000,
        ),
    )
    return result.ready, result.rejections


def test_bootstrap_exposes_one_compound_authority_per_fragment(tmp_path: Path) -> None:
    store, backend = _system(tmp_path / "global", fragment_count=2)
    snapshot = store.load_snapshot()
    assert snapshot.version_vector == (0, 0)
    assert len(snapshot.digest) == 64
    for index, authority in enumerate(snapshot.authorities):
        assert authority.parameters == bytes([index]) * 16
        assert authority.outer_optimizer_state == f"outer-{index}-0".encode()
        assert authority.state.outer_update_count == authority.version == 0
        assert authority.envelope.policy_identity == POLICY_IDENTITY
        assert authority.envelope.selected == ()
        assert authority.frontiers.learner_ids == LEARNERS
        assert all(
            (item.last_sequence, item.last_base_version) == (-1, -1)
            for item in authority.frontiers.entries
        )
    visibility = sorted(path.name for path in (backend.root / "visibility").iterdir())
    assert visibility == ["global-current-000000.json", "global-current-000001.json"]
    assert not any("latest" in item or "head" in item for item in visibility)


@pytest.mark.parametrize(
    ("crash_at", "new_is_visible"),
    (
        ("before_payload_write", False),
        ("after_payload_write", False),
        ("before_record_replace", False),
        ("after_record_replace", True),
    ),
)
def test_every_publication_interruption_replays_from_exact_authority(
    tmp_path: Path, crash_at: str, new_is_visible: bool
) -> None:
    store, _ = _system(tmp_path / crash_at)
    old = store.load_fragment(0)
    request = _request(old)
    selected = request.selection.proposals
    assert all(_eligibility(old, item)[0] for item in selected)

    with pytest.raises(PublicationInterrupted, match=crash_at):
        store.commit(request, crash_at=crash_at)

    observed = store.load_fragment(0)
    if not new_is_visible:
        assert observed == old
        assert observed.frontiers == old.frontiers
        assert all(_eligibility(observed, item)[0] for item in selected)
        result = store.commit(request)
        assert result.published and not result.duplicate_retry
    else:
        assert observed.version == 1
        assert observed.envelope.update_identity == request.update_identity
        assert all(not _eligibility(observed, item)[0] for item in selected)
        result = store.commit(request)
        assert not result.published and result.duplicate_retry

    committed = store.load_fragment(0)
    assert committed.version == committed.state.outer_update_count == 1
    assert committed.parameters == request.parameters
    assert committed.outer_optimizer_state == request.outer_optimizer_state
    assert committed.envelope.selection_identity == request.selection.selection_identity
    assert committed.envelope.update_identity == request.update_identity
    assert len(committed.frontiers.entries) == len(LEARNERS)
    for proposal in selected:
        frontier = committed.frontiers.for_learner(proposal.learner_id)
        assert frontier.last_sequence == proposal.sequence
        assert frontier.last_base_version == proposal.base_version


def test_version_skip_stale_base_and_conflicting_same_version_fail_closed(
    tmp_path: Path,
) -> None:
    store, _ = _system(tmp_path / "global")
    old = store.load_fragment(0)
    request = _request(old)
    with pytest.raises(AtomicCommitError, match="plus one"):
        dataclasses.replace(request, next_version=2)
    result = store.commit(request)
    assert result.authority.version == 1

    exact = store.commit(request)
    assert exact.duplicate_retry and exact.authority == result.authority
    conflicting = dataclasses.replace(
        request,
        parameters=b"different-content",
        update_identity=hashlib.sha256(b"different-update").hexdigest(),
    )
    with pytest.raises(AtomicCommitError, match="same next version"):
        store.commit(conflicting)

    current = store.load_fragment(0)
    valid_current = _request(current)
    stale_selection = dataclasses.replace(
        valid_current.selection,
        authority_identity=hashlib.sha256(b"wrong-authority").hexdigest(),
    )
    stale = dataclasses.replace(
        valid_current,
        selection=stale_selection,
    )
    with pytest.raises(AtomicCommitError, match="authority identity"):
        store.commit(stale)
    assert store.load_fragment(0) == result.authority


def test_same_base_is_consumed_once_and_refreshed_late_arrival_enters_next_round(
    tmp_path: Path,
) -> None:
    store, _ = _system(tmp_path / "global")
    old = store.load_fragment(0)
    selected = tuple(_proposal(old, learner) for learner in LEARNERS[:2])
    request = dataclasses.replace(
        _request(old, learners=LEARNERS[:2]),
        selection=_selection(old, selected),
    )
    late: list[Proposal] = []

    def arrive_after_freeze(_successor) -> None:
        late.append(
            _proposal(old, LEARNERS[2], proposal_id="late-old-base", tokens=999)
        )

    committed = store.commit(request, before_visibility=arrive_after_freeze).authority
    assert len(late) == 1
    assert {item.learner_id for item in committed.envelope.selected} == set(
        LEARNERS[:2]
    )
    assert committed.frontiers.for_learner(LEARNERS[2]).last_sequence == -1
    ready, rejected = _eligibility(committed, late[0])
    assert not ready and rejected[late[0].proposal_id] == "too_stale"

    repeated = _proposal(
        old,
        LEARNERS[0],
        sequence=old.version + 1,
        proposal_id="same-consumed-base-later-sequence",
    )
    ready, rejected = _eligibility(committed, repeated)
    assert not ready and rejected[repeated.proposal_id] == "consumed_base"

    refreshed = _proposal(
        committed,
        LEARNERS[2],
        sequence=committed.version,
        proposal_id="late-refreshed-on-new-base",
    )
    ready, rejected = _eligibility(committed, refreshed)
    assert ready and rejected == {}
    next_request = dataclasses.replace(
        _request(committed, learners=(LEARNERS[2],)),
        selection=_selection(committed, (refreshed,)),
    )
    next_authority = store.commit(next_request).authority
    assert next_authority.version == 2
    assert next_authority.frontiers.for_learner(LEARNERS[2]).last_base_version == 1


def test_concurrent_readers_observe_only_complete_committed_authorities(
    tmp_path: Path,
) -> None:
    store, _ = _system(tmp_path / "global")
    complete: set[
        tuple[int, str, bytes, bytes, str, str, tuple[tuple[int, int], ...]]
    ] = set()
    observations: list[
        tuple[int, str, bytes, bytes, str, str, tuple[tuple[int, int], ...]]
    ] = []
    lock = threading.Lock()
    started = threading.Event()
    finished = threading.Event()

    def facts(authority):
        return (
            authority.version,
            authority.content_identity,
            authority.parameters,
            authority.outer_optimizer_state,
            authority.envelope.policy_identity,
            authority.envelope.update_identity,
            tuple(
                (item.last_sequence, item.last_base_version)
                for item in authority.frontiers.entries
            ),
        )

    initial = store.load_fragment(0)
    complete.add(facts(initial))

    def reader() -> None:
        started.wait()
        local = []
        while not finished.is_set() or len(local) < 50:
            local.append(facts(store.load_fragment(0)))
        with lock:
            observations.extend(local)

    with ThreadPoolExecutor(max_workers=9) as pool:
        readers = [pool.submit(reader) for _ in range(8)]
        started.set()
        for _ in range(20):
            current = store.load_fragment(0)
            result = store.commit(_request(current))
            complete.add(facts(result.authority))
        finished.set()
        for future in readers:
            future.result()

    assert len(observations) >= 400
    assert set(observations) <= complete
    assert {item[0] for item in observations} <= set(range(21))
    assert store.load_fragment(0).version == 20


def test_envelope_integrity_schema_and_policy_fail_closed(tmp_path: Path) -> None:
    store, _ = _system(tmp_path / "global")
    authority = store.load_fragment(0)
    encoded = encode_commit_envelope(authority.envelope)
    with pytest.raises(AtomicCommitError, match="integrity"):
        decode_commit_envelope(
            encoded[:-1] + bytes([encoded[-1] ^ 1]),
            identities=IDENTITIES,
            descriptor=authority.state.descriptor,
        )
    with pytest.raises(AtomicCommitError, match="magic"):
        decode_commit_envelope(
            b"BADMAGIC" + encoded[8:],
            identities=IDENTITIES,
            descriptor=authority.state.descriptor,
        )
    wrong = AtomicGlobalCommitStore(
        store.store,
        learner_ids=LEARNERS,
        policy_identity=hashlib.sha256(b"wrong-policy").hexdigest(),
    )
    with pytest.raises(AtomicCommitError, match="policy identity"):
        wrong.load_fragment(0)


def test_frontier_shape_remains_fixed_at_large_sequence_ordinals(tmp_path: Path) -> None:
    store, _ = _system(tmp_path / "global")
    authority = store.load_fragment(0)
    request = _request(authority)
    huge = 10_000
    proposals = tuple(
        _proposal(
            authority,
            item.learner_id,
            sequence=huge,
            proposal_id=f"{item.learner_id}-huge",
            tokens=item.processed_tokens,
        )
        for item in request.selection.proposals
    )
    request = dataclasses.replace(request, selection=_selection(authority, proposals))
    committed = store.commit(request).authority
    assert len(committed.frontiers.entries) == len(LEARNERS) == 4
    assert all(item.last_sequence == huge for item in committed.frontiers.entries)
    encoded = encode_commit_envelope(committed.envelope)
    decoded = decode_commit_envelope(
        encoded,
        identities=IDENTITIES,
        descriptor=authority.state.descriptor,
    )
    assert decoded == committed.envelope
