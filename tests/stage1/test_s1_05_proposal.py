from __future__ import annotations

import dataclasses
import hashlib
import inspect
import itertools
import json
import os
import threading
from pathlib import Path

import pytest

from fsbdd.diloco.protocol.global_state import (
    CountingStorageBackend,
    FragmentStateDescriptor,
    GlobalStateIdentities,
)
from fsbdd.diloco.protocol.proposal import (
    ConsumptionFrontier,
    ConsumptionFrontiers,
    EligibilityPolicy,
    Proposal,
    ProposalError,
    ProposalStore,
    RetainedBaseIdentity,
    commit_consumption,
    compute_candidate_weights,
    decode_proposal,
    encode_proposal,
    proposal_slot,
    select_candidates,
)
from fsbdd.diloco.protocol.storage import (
    PosixStorageBackend,
    PublicationInterrupted,
    PublicationNotFound,
)
from fsbdd.auxiliary.stage0.oracle import inverse_staleness_weights


IDENTITIES = GlobalStateIdentities(
    run_identity="run-s1-05",
    config_identity="a" * 64,
    model_identity="b" * 64,
    fragment_map_identity="c" * 64,
)
DESCRIPTORS = tuple(
    FragmentStateDescriptor(
        index=index,
        identity=f"fragment-{index}",
        dtype="uint8",
        shape=(16,),
        parameter_identities=(f"parameter-{index}",),
    )
    for index in range(2)
)
LEARNERS = ("a", "b", "c")


def base_identity(version: int) -> str:
    return hashlib.sha256(f"base-{version}".encode()).hexdigest()


def make_proposal(
    proposal_id: str = "a-seq1",
    *,
    learner_id: str = "a",
    descriptor: FragmentStateDescriptor = DESCRIPTORS[0],
    sequence: int = 1,
    base_version: int = 3,
    base_content_identity: str | None = None,
    local_steps: int = 5,
    processed_tokens: int = 50,
    snapshot_local_step: int = 100,
    parameters: bytes = b"0123456789abcdef",
    identities: GlobalStateIdentities = IDENTITIES,
) -> Proposal:
    return Proposal.create(
        proposal_id=proposal_id,
        identities=identities,
        learner_id=learner_id,
        descriptor=descriptor,
        sequence=sequence,
        base_version=base_version,
        base_content_identity=(
            base_identity(base_version)
            if base_content_identity is None
            else base_content_identity
        ),
        local_steps=local_steps,
        processed_tokens=processed_tokens,
        snapshot_local_step=snapshot_local_step,
        parameters=parameters,
    )


def make_store(tmp_path: Path, backend=None) -> ProposalStore:
    return ProposalStore(
        backend or PosixStorageBackend(tmp_path),
        identities=IDENTITIES,
        descriptors=DESCRIPTORS,
        learner_ids=LEARNERS,
        maximum_local_steps=100,
        maximum_processed_tokens=1_000_000,
    )


def make_policy(
    *,
    current_version: int = 3,
    s_max: int = 1,
    q: int = 1,
    q_fresh: int = 0,
    max_contributors: int = 3,
    learner_ids: tuple[str, ...] = LEARNERS,
    descriptor: FragmentStateDescriptor = DESCRIPTORS[0],
) -> EligibilityPolicy:
    first = max(0, current_version - s_max)
    return EligibilityPolicy(
        identities=IDENTITIES,
        descriptor=descriptor,
        learner_ids=learner_ids,
        current_version=current_version,
        retained_bases=tuple(
            RetainedBaseIdentity(version, base_identity(version))
            for version in range(first, current_version + 1)
        ),
        s_max=s_max,
        q=q,
        q_fresh=q_fresh,
        max_contributors=max_contributors,
        maximum_local_steps=100,
        maximum_processed_tokens=1_000_000,
        lambda_s=1.0,
    )


def test_compound_codec_is_canonical_complete_and_immutable() -> None:
    proposal = make_proposal()
    encoded = encode_proposal(proposal)
    assert decode_proposal(encoded) == proposal
    assert decode_proposal(encoded).parameters == b"0123456789abcdef"
    assert proposal.parameters_sha256 == hashlib.sha256(proposal.parameters).hexdigest()
    assert proposal.payload_bytes == 16
    assert proposal.payload_identity == proposal.parameters_sha256
    with pytest.raises(dataclasses.FrozenInstanceError):
        proposal.sequence = 2  # type: ignore[misc]
    with pytest.raises(ProposalError, match="integrity"):
        decode_proposal(encoded + b"trailing")
    corrupted = encoded[:-1] + bytes([encoded[-1] ^ 1])
    with pytest.raises(ProposalError, match="integrity"):
        decode_proposal(corrupted)


def test_partial_publication_is_invisible_until_complete_retry(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    proposal = make_proposal()
    with pytest.raises(PublicationInterrupted, match="after_payload_write"):
        store.publish(proposal, crash_at="after_payload_write")
    with pytest.raises(PublicationNotFound):
        store.load_latest("a", 0)
    assert store.publish(proposal) == proposal
    assert store.load_latest("a", 0) == proposal


def test_latest_is_idempotent_monotonic_and_conflict_safe(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    sequence_one = make_proposal()
    store.publish(sequence_one)
    payload_root = tmp_path / "payloads"
    before = {path.name for path in payload_root.iterdir()}
    assert store.publish(sequence_one) == sequence_one
    assert {path.name for path in payload_root.iterdir()} == before
    with pytest.raises(ProposalError, match="same sequence"):
        store.publish(make_proposal(parameters=b"fedcba9876543210"))
    assert {path.name for path in payload_root.iterdir()} == before
    sequence_three = make_proposal("a-seq3", sequence=3)
    store.publish(sequence_three)
    with pytest.raises(ProposalError, match="regress"):
        store.publish(make_proposal("a-seq2", sequence=2))
    assert store.load_latest("a", 0) == sequence_three


def test_delayed_old_completion_cannot_regress_latest(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.publish(make_proposal(sequence=1))
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def publish_old() -> None:
        entered.set()
        assert release.wait(timeout=10)
        try:
            store.publish(make_proposal("a-seq2", sequence=2))
        except BaseException as error:  # pragma: no branch - asserted below
            errors.append(error)

    thread = threading.Thread(target=publish_old)
    thread.start()
    assert entered.wait(timeout=10)
    newest = make_proposal("a-seq3", sequence=3)
    store.publish(newest)
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ProposalError)
    assert "regress" in str(errors[0])
    assert store.load_latest("a", 0) == newest


def test_single_writer_rejects_a_second_in_flight_publication(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.publish(make_proposal(sequence=1))
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def delay(_slot: str, _proposal: Proposal) -> None:
        entered.set()
        assert release.wait(timeout=10)

    def publish_old() -> None:
        try:
            store.publish(make_proposal("a-seq2", sequence=2), before_visibility=delay)
        except ProposalError as error:
            errors.append(error)

    thread = threading.Thread(target=publish_old)
    thread.start()
    assert entered.wait(timeout=10)
    with pytest.raises(ProposalError, match="already in flight"):
        store.publish(make_proposal("a-seq3", sequence=3))
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert not errors
    assert store.load_latest("a", 0).sequence == 2


def test_rootless_distinct_backend_views_reject_a_delayed_stale_task(
    tmp_path: Path,
) -> None:
    class RootlessBackend:
        def __init__(self, backend: PosixStorageBackend) -> None:
            self.backend = backend

        def publish(self, *args, **kwargs):
            return self.backend.publish(*args, **kwargs)

        def read(self, *args, **kwargs):
            return self.backend.read(*args, **kwargs)

    backend = PosixStorageBackend(tmp_path)
    older_view = make_store(tmp_path, RootlessBackend(backend))
    current_view = make_store(tmp_path, RootlessBackend(backend))
    current_view.publish(make_proposal(sequence=1))
    delayed = make_proposal("a-seq2", sequence=2)
    current_view.publish(make_proposal("a-seq3", sequence=3))
    with pytest.raises(ProposalError, match="regress"):
        older_view.publish(delayed)
    assert current_view.load_latest("a", 0).sequence == 3


def test_discovery_reads_exact_fixed_slots_without_history_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = PosixStorageBackend(tmp_path)
    counting = CountingStorageBackend(backend)
    store = make_store(tmp_path, counting)
    for learner in LEARNERS:
        for descriptor in DESCRIPTORS:
            store.publish(
                make_proposal(
                    f"{learner}-{descriptor.index}",
                    learner_id=learner,
                    descriptor=descriptor,
                )
            )
    counting.reset_counts()
    assert len(store.discover_latest()) == len(LEARNERS) * len(DESCRIPTORS)
    assert counting.read_calls == len(LEARNERS) * len(DESCRIPTORS)
    for index in range(10_000):
        (tmp_path / "payloads" / f"history-{index:05d}.bin").write_bytes(b"history")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("normal proposal discovery must not scan directories")

    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(os, "listdir", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    monkeypatch.setattr(os, "walk", forbidden)
    counting.reset_counts()
    assert len(store.discover_latest()) == len(LEARNERS) * len(DESCRIPTORS)
    assert counting.read_calls == len(LEARNERS) * len(DESCRIPTORS)


def test_fixed_slot_depends_only_on_learner_and_fragment_identity() -> None:
    assert proposal_slot("a", 2) == proposal_slot("a", 2)
    assert proposal_slot("a", 2) != proposal_slot("a", 3)
    assert proposal_slot("a", 2) != proposal_slot("b", 2)
    assert len(proposal_slot("learner-with/path-like text", 999)) < 128


def test_fixed_slot_record_is_bound_to_body_learner_sequence_and_base(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.publish(make_proposal())
    visibility = tmp_path / "visibility"
    source = visibility / f"{proposal_slot('a', 0)}.json"
    copied = visibility / f"{proposal_slot('b', 0)}.json"
    copied.write_bytes(source.read_bytes())
    with pytest.raises(ProposalError, match="learner identity"):
        store.load_latest("b", 0)
    copied.unlink()
    record = json.loads(source.read_text(encoding="utf-8"))
    record["sequence"] = 99
    source.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProposalError, match="sequence"):
        store.load_latest("a", 0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"learner_id": "unknown"}, "unknown learner"),
        (
            {"identities": dataclasses.replace(IDENTITIES, model_identity="d" * 64)},
            "frozen identity",
        ),
        (
            {"descriptor": dataclasses.replace(DESCRIPTORS[0], dtype="float32")},
            "descriptor",
        ),
        ({"local_steps": 0}, "local_steps"),
        ({"processed_tokens": 0}, "processed_tokens"),
        ({"snapshot_local_step": 0}, "snapshot_local_step"),
        ({"local_steps": 101}, "local_steps"),
        ({"processed_tokens": 1_000_001}, "processed_tokens"),
    ],
)
def test_publish_rejects_wrong_identity_and_progress_before_visibility(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ProposalError, match=message):
        store.publish(make_proposal(**changes))
    assert (
        not (tmp_path / "visibility").joinpath(proposal_slot("a", 0) + ".json").exists()
    )


def _shared_case(
    case: dict[str, object],
) -> tuple[list[Proposal], ConsumptionFrontiers, EligibilityPolicy]:
    learner_ids = sorted(
        {
            item["learner_id"]
            for item in case["proposals"]  # type: ignore[union-attr]
        }
        | set(case["frontiers"])  # type: ignore[arg-type]
    )
    while len(learner_ids) < case["max_contributors"]:  # type: ignore[operator]
        learner_ids.append(f"unused-{len(learner_ids)}")
    frozen_learners = tuple(learner_ids)
    current_version = case["current_version"]
    s_max = case["s_max"]
    policy = make_policy(
        current_version=current_version,  # type: ignore[arg-type]
        s_max=s_max,  # type: ignore[arg-type]
        q=case["q"],  # type: ignore[arg-type]
        q_fresh=case["q_fresh"],  # type: ignore[arg-type]
        max_contributors=case["max_contributors"],  # type: ignore[arg-type]
        learner_ids=frozen_learners,
    )
    frontier_values = case["frontiers"]
    frontiers = ConsumptionFrontiers(
        IDENTITIES,
        DESCRIPTORS[0],
        frozen_learners,
        tuple(
            ConsumptionFrontier(**frontier_values.get(learner, {}))  # type: ignore[union-attr]
            for learner in frozen_learners
        ),
    )
    proposals = []
    for value in case["proposals"]:  # type: ignore[union-attr]
        identity = (
            base_identity(value["base_version"])
            if value["base_identity_matches"]
            else "f" * 64
        )
        proposals.append(
            make_proposal(
                value["proposal_id"],
                learner_id=value["learner_id"],
                sequence=value["sequence"],
                base_version=value["base_version"],
                base_content_identity=identity,
                processed_tokens=value["tokens"],
                local_steps=value["local_steps"],
            )
        )
    return proposals, frontiers, policy


def test_shared_stage0_decision_vectors_match_for_every_permutation() -> None:
    fixture = Path("tests/fixtures/stage0_decision_vectors.json")
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == (
        "0dd9fe699bce3eef6b3da97b67d12a755ac227a0d36132e23e6feda128cbd5a8"
    )
    cases = json.loads(fixture.read_text(encoding="utf-8"))["decision_cases"]
    for case in cases:
        proposals, frontiers, policy = _shared_case(case)
        for ordering in itertools.permutations(proposals):
            result = select_candidates(ordering, frontiers, policy)
            assert result.ready is case["expected_ready"], case["id"]
            assert [item.proposal_id for item in result.selected] == case[
                "expected_selected"
            ], case["id"]
            assert result.rejections == case["expected_rejections"], case["id"]
            assert len(result.selected) == len(
                {item.learner_id for item in result.selected}
            )


@pytest.mark.parametrize(
    ("proposal", "reason"),
    [
        (make_proposal("future", base_version=4), "future_base"),
        (make_proposal("stale", base_version=1), "too_stale"),
        (
            make_proposal("wrong-base", base_content_identity="f" * 64),
            "base_identity_mismatch",
        ),
        (make_proposal("steps-zero", local_steps=0), "nonpositive_local_steps"),
        (make_proposal("tokens-zero", processed_tokens=0), "nonpositive_tokens"),
        (
            make_proposal("snapshot-zero", snapshot_local_step=0),
            "nonpositive_snapshot_step",
        ),
        (make_proposal("steps-high", local_steps=101), "local_steps_exceeded"),
        (
            make_proposal("tokens-high", processed_tokens=1_000_001),
            "processed_tokens_exceeded",
        ),
        (
            make_proposal(
                "wrong-run",
                identities=dataclasses.replace(IDENTITIES, run_identity="other"),
            ),
            "run_identity_mismatch",
        ),
        (
            make_proposal(
                "wrong-model",
                identities=dataclasses.replace(IDENTITIES, model_identity="d" * 64),
            ),
            "model_identity_mismatch",
        ),
        (
            make_proposal(
                "wrong-config",
                identities=dataclasses.replace(IDENTITIES, config_identity="d" * 64),
            ),
            "config_identity_mismatch",
        ),
        (
            make_proposal(
                "wrong-map",
                identities=dataclasses.replace(
                    IDENTITIES, fragment_map_identity="d" * 64
                ),
            ),
            "fragment_map_identity_mismatch",
        ),
        (
            make_proposal(
                "wrong-fragment",
                descriptor=dataclasses.replace(
                    DESCRIPTORS[0], identity="other-fragment"
                ),
            ),
            "fragment_identity_mismatch",
        ),
        (
            make_proposal(
                "wrong-dtype",
                descriptor=dataclasses.replace(DESCRIPTORS[0], dtype="float32"),
            ),
            "dtype_mismatch",
        ),
        (
            make_proposal(
                "wrong-shape",
                descriptor=dataclasses.replace(DESCRIPTORS[0], shape=(8, 2)),
            ),
            "shape_mismatch",
        ),
        (
            make_proposal(
                "wrong-parameters",
                descriptor=dataclasses.replace(
                    DESCRIPTORS[0], parameter_identities=("other-parameter",)
                ),
            ),
            "parameter_identity_mismatch",
        ),
    ],
)
def test_eligibility_boundary_has_deterministic_rejection(
    proposal: Proposal, reason: str
) -> None:
    result = select_candidates(
        [proposal],
        ConsumptionFrontiers.empty(
            identities=IDENTITIES,
            descriptor=DESCRIPTORS[0],
            learner_ids=LEARNERS,
        ),
        make_policy(),
    )
    assert result.selected == ()
    assert result.rejections == {proposal.proposal_id: reason}


def test_in_window_missing_retained_base_is_rejected_explicitly() -> None:
    policy = dataclasses.replace(
        make_policy(current_version=3, s_max=1),
        retained_bases=(RetainedBaseIdentity(3, base_identity(3)),),
    )
    proposal = make_proposal("missing-base", base_version=2)
    result = select_candidates(
        [proposal],
        ConsumptionFrontiers.empty(
            identities=IDENTITIES,
            descriptor=DESCRIPTORS[0],
            learner_ids=LEARNERS,
        ),
        policy,
    )
    assert result.rejections == {"missing-base": "base_not_retained"}


def test_smax_zero_and_positive_use_the_same_generic_eligibility_function() -> None:
    fresh = make_proposal("fresh", base_version=3)
    stale = make_proposal("stale", base_version=2)
    frontiers = ConsumptionFrontiers.empty(
        identities=IDENTITIES,
        descriptor=DESCRIPTORS[0],
        learner_ids=LEARNERS,
    )
    zero = select_candidates(
        [fresh, stale], frontiers, make_policy(s_max=0, current_version=3)
    )
    positive = select_candidates(
        [fresh, stale], frontiers, make_policy(s_max=1, current_version=3)
    )
    assert zero.rejections == {"stale": "too_stale"}
    assert [item.proposal_id for item in zero.selected] == ["fresh"]
    assert positive.rejections == {"stale": "duplicate_learner"}
    assert [item.proposal_id for item in positive.selected] == ["fresh"]
    source = inspect.getsource(
        __import__(
            "fsbdd.diloco.protocol.proposal", fromlist=["_eligibility_reason"]
        )._eligibility_reason
    )
    assert "s_max == 0" not in source
    assert "base_version == policy.current_version" not in source


def test_consumption_frontier_is_fixed_size_and_advances_only_after_publish() -> None:
    frontiers = ConsumptionFrontiers.empty(
        identities=IDENTITIES,
        descriptor=DESCRIPTORS[0],
        learner_ids=LEARNERS,
    )
    proposal = make_proposal()
    assert commit_consumption(frontiers, [proposal], False) is frontiers
    committed = commit_consumption(frontiers, [proposal], True)
    assert len(committed.entries) == len(LEARNERS)
    assert committed.for_learner("a") == ConsumptionFrontier(1, 3)
    for sequence in range(2, 10_002):
        committed = commit_consumption(
            committed,
            [
                make_proposal(
                    f"a-seq{sequence}",
                    sequence=sequence,
                    base_version=sequence + 2,
                )
            ],
            True,
        )
    assert len(committed.entries) == len(LEARNERS)
    same_base = make_proposal(
        "a-new-sequence-same-base", sequence=20_000, base_version=10_003
    )
    result = select_candidates(
        [same_base],
        committed,
        make_policy(current_version=10_003, s_max=0),
    )
    assert result.rejections == {same_base.proposal_id: "consumed_base"}


def test_consumption_frontier_rejects_cross_fragment_selection_and_commit() -> None:
    fragment_zero = ConsumptionFrontiers.empty(
        identities=IDENTITIES,
        descriptor=DESCRIPTORS[0],
        learner_ids=LEARNERS,
    )
    fragment_one_proposal = make_proposal(
        "fragment-one",
        descriptor=DESCRIPTORS[1],
    )
    with pytest.raises(ProposalError, match="fragment descriptors differ"):
        select_candidates(
            [fragment_one_proposal],
            fragment_zero,
            make_policy(descriptor=DESCRIPTORS[1]),
        )
    with pytest.raises(ProposalError, match="fragment descriptors differ"):
        commit_consumption(fragment_zero, [fragment_one_proposal], True)


def test_float32_weights_match_shared_oracle_and_report_telemetry() -> None:
    proposals = [
        make_proposal("a", processed_tokens=10, base_version=3),
        make_proposal("b", learner_id="b", processed_tokens=20, base_version=2),
        make_proposal("c", learner_id="c", processed_tokens=30, base_version=1),
    ]
    weights = compute_candidate_weights(proposals, current_version=3, lambda_s=1.0)
    expected = inverse_staleness_weights([10, 20, 30], [0, 1, 2], 1.0)
    assert [item.normalized_weight for item in weights] == expected
    assert [item.staleness for item in weights] == [0, 1, 2]
    assert [item.processed_tokens for item in weights] == [10, 20, 30]
    assert [item.raw_weight for item in weights] == [10.0, 10.0, 10.0]
    assert sum(item.normalized_weight for item in weights) == pytest.approx(1.0)
    monotonic = compute_candidate_weights(
        [
            make_proposal("fresh", processed_tokens=100, base_version=3),
            make_proposal(
                "stale", learner_id="b", processed_tokens=100, base_version=2
            ),
        ],
        current_version=3,
    )
    assert monotonic[0].raw_weight >= monotonic[1].raw_weight


def test_s1_05_pbs_is_two_node_clean_and_runs_production_stress() -> None:
    script = Path("pbs/stage1_s1_05_proposal.pbs").read_text(encoding="utf-8")
    assert "#PBS -l select=2" in script
    assert "--map-by ppr:1:node" in script
    assert "fsbdd.auxiliary.stress.proposal_stress role" in script
    assert "tests/stage1" in script
    assert "EXPECTED_COMMIT" in script
    assert "status --porcelain" in script
    assert "formal-l2" in script
