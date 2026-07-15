from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from fsbdd.fragment_map import build_fragment_map  # noqa: E402
from fsbdd.global_state import (  # noqa: E402
    FragmentStateDescriptor,
    GlobalStateIdentities,
)
from fsbdd.learner import (  # noqa: E402
    ConstantStepScheduler,
    LearnerProgress,
    LearnerRng,
    LearnerRuntime,
    SafeBoundaryEvent,
)
from fsbdd.learner_publish import (  # noqa: E402
    AdoptedFragmentBase,
    BoundedProposalPublisher,
    FragmentPublicationSchedule,
    FragmentSnapshot,
    FragmentSnapshotCoordinator,
    SnapshotPublishError,
    build_fragment_descriptors,
    build_fragment_parameter_groups,
    serialize_fragment_parameters,
)
from fsbdd.model_registry import build_logical_layer_registry  # noqa: E402
from fsbdd.learner_publish_stress import (  # noqa: E402
    SnapshotStressError,
    write_manifest,
)
from fsbdd.proposal import Proposal, ProposalStore  # noqa: E402
from fsbdd.storage import PosixStorageBackend  # noqa: E402


IDENTITIES = GlobalStateIdentities(
    run_identity="run-s1-07",
    config_identity="a" * 64,
    model_identity="b" * 64,
    fragment_map_identity="c" * 64,
)


def _base(version: int, fragment_index: int = 0) -> AdoptedFragmentBase:
    return AdoptedFragmentBase(
        version,
        hashlib.sha256(f"base-{fragment_index}-{version}".encode()).hexdigest(),
    )


def _descriptor(index: int, numel: int) -> FragmentStateDescriptor:
    return FragmentStateDescriptor(
        index=index,
        identity=f"fragment-{index}",
        dtype="float32",
        shape=(numel,),
        parameter_identities=(f"parameter-{index}",),
    )


def _store(
    tmp_path: Path,
    descriptors: tuple[FragmentStateDescriptor, ...],
) -> ProposalStore:
    return ProposalStore(
        PosixStorageBackend(tmp_path),
        identities=IDENTITIES,
        descriptors=descriptors,
        learner_ids=("learner-00",),
        maximum_local_steps=100,
        maximum_processed_tokens=1_000_000,
    )


def _event(progress: LearnerProgress) -> SafeBoundaryEvent:
    step = progress.local_optimizer_steps
    return SafeBoundaryEvent(
        learner_id=progress.learner_id,
        local_optimizer_step=step,
        microbatches=1,
        processed_input_tokens_step=10,
        processed_input_tokens_total=progress.processed_input_tokens,
        loss_bearing_target_tokens_step=8,
        loss_bearing_target_tokens_total=progress.loss_bearing_target_tokens,
        token_weighted_loss=1.0,
        loss_numerator=8.0,
        gradient_norm=1.0,
        learning_rates=(0.001,),
        lr_schedule_basis="local_optimizer_steps",
        step_start_monotonic_ns=step * 1_000_000,
        safe_boundary_monotonic_ns=step * 1_000_000 + 500_000,
        step_latency_seconds=0.0005,
        step_input_tokens_per_second=20_000.0,
        fragment_global_versions=tuple(
            fragment.global_version for fragment in progress.fragments
        ),
        fragment_local_steps=tuple(
            fragment.local_steps_since_adoption for fragment in progress.fragments
        ),
        fragment_processed_input_tokens=tuple(
            fragment.processed_input_tokens_since_adoption
            for fragment in progress.fragments
        ),
        fragment_update_norms=tuple(0.1 for _ in progress.fragments),
        distributed_initialized=False,
        comparison={"claim": "test"},
        inactive_metrics={},
    )


def _advance(progress: LearnerProgress, tokens: int = 10) -> None:
    progress.local_optimizer_steps += 1
    progress.processed_input_tokens += tokens
    progress.loss_bearing_target_tokens += tokens - 2
    for fragment in progress.fragments:
        fragment.local_steps_since_adoption += 1
        fragment.processed_input_tokens_since_adoption += tokens


def _proposal(
    descriptor: FragmentStateDescriptor,
    sequence: int,
    *,
    parameters: bytes,
) -> Proposal:
    return Proposal.create(
        proposal_id=f"proposal-{descriptor.index}-{sequence}",
        identities=IDENTITIES,
        learner_id="learner-00",
        descriptor=descriptor,
        sequence=sequence,
        base_version=0,
        base_content_identity=_base(0, descriptor.index).content_identity,
        local_steps=min(100, max(1, sequence)),
        processed_tokens=max(1, sequence) * 10,
        snapshot_local_step=max(1, sequence),
        parameters=parameters,
    )


def _snapshot(proposal: Proposal) -> FragmentSnapshot:
    return FragmentSnapshot(
        proposal=proposal,
        safe_boundary_monotonic_ns=1,
        staging_started_monotonic_ns=2,
        staging_completed_monotonic_ns=3,
        gpu_to_cpu_seconds=1e-9,
    )


def test_weighted_offsets_are_deterministic_spread_and_byte_accounted() -> None:
    pythia_bytes = (154_533_888, 170_108_928, 170_115_072, 154_533_888)
    schedule = FragmentPublicationSchedule.unified(pythia_bytes, 50)
    assert schedule.offsets == (6, 18, 32, 44)
    assert schedule == FragmentPublicationSchedule.unified(pythia_bytes, 50)
    assert len(set(schedule.offsets)) == 4
    assert [
        sum(index in schedule.due_fragments(step) for step in range(1, 51))
        for index in range(4)
    ] == [1, 1, 1, 1]
    budget = schedule.frequency_budget()
    assert budget["exact_numerator"] / budget["exact_denominator"] == sum(
        pythia_bytes
    ) / 50
    assert budget["maximum_due_bytes"] == max(pythia_bytes)

    unequal = FragmentPublicationSchedule.unified((90, 5, 5), 8)
    assert unequal.offsets == (4, 7, 0)
    assert len(set(unequal.offsets)) == 3
    future_hf = FragmentPublicationSchedule(
        fragment_bytes=(120, 80),
        intervals=(12, 4),
        offsets=(3, 1),
        offset_algorithm="explicit_test_offsets",
    )
    future_budget = future_hf.frequency_budget()
    assert "unified_h" not in future_budget
    assert future_budget["exact_numerator"] == 30
    assert future_budget["exact_denominator"] == 1


def test_snapshot_rejects_optimizer_mutation_before_sequence_reservation(
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    parameter.grad = torch.ones_like(parameter)
    descriptor = _descriptor(0, parameter.numel())
    progress = LearnerProgress.initialize("learner-00", (0,))
    coordinator = FragmentSnapshotCoordinator(
        identities=IDENTITIES,
        progress=progress,
        descriptors=(descriptor,),
        fragment_parameters=((parameter,),),
        adopted_bases=(_base(0),),
        schedule=FragmentPublicationSchedule.unified((parameter.numel() * 4,), 1),
        store=_store(tmp_path, (descriptor,)),
    )
    _advance(progress)
    with pytest.raises(SnapshotPublishError, match="gradients remain active"):
        coordinator.on_safe_boundary(_event(progress))
    assert progress.fragments[0].proposal_sequence == 0
    parameter.grad = None
    coordinator.close()


def test_bounded_publisher_replaces_pending_but_never_inflight(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor(0, 4)
    store = _store(tmp_path, (descriptor,))
    entered = threading.Event()
    release = threading.Event()
    traces: list[dict] = []

    def slow_first(proposal: Proposal) -> None:
        if proposal.sequence == 1:
            entered.set()
            assert release.wait(5)

    publisher = BoundedProposalPublisher(
        store,
        learner_id="learner-00",
        before_publish=slow_first,
        trace_sink=traces.append,
    )
    first = _snapshot(_proposal(descriptor, 1, parameters=b"a" * 16))
    second = _snapshot(_proposal(descriptor, 2, parameters=b"b" * 16))
    third = _snapshot(_proposal(descriptor, 3, parameters=b"c" * 16))
    old = _snapshot(_proposal(descriptor, 0, parameters=b"z" * 16))
    assert publisher.submit(first) == "pending"
    assert entered.wait(5)
    assert publisher.submit(second) == "pending"
    assert publisher.submit(third) == "pending"
    assert publisher.submit(old) == "skipped_nonmonotonic"
    while_slow = publisher.summary()
    assert while_slow["per_fragment_in_flight"] == [1]
    assert while_slow["per_fragment_pending"] == [1]
    assert while_slow["snapshot_replacement_count"] == 1
    assert while_slow["snapshot_skip_count"] == 1
    release.set()
    publisher.drain()
    assert store.load_latest("learner-00", 0).sequence == 3
    final = publisher.summary()
    assert final["maximum_in_flight_per_fragment"] == [1]
    assert final["maximum_pending_per_fragment"] == [1]
    assert [trace["outcome"] for trace in sorted(traces, key=lambda row: row["sequence"])] == [
        "skipped_nonmonotonic",
        "published",
        "replaced_before_publish",
        "published",
    ]
    assert final["resident_trace_records"] <= final["resident_trace_record_bound"] == 3
    publisher.close()


def test_many_publications_keep_constant_resident_state(tmp_path: Path) -> None:
    descriptor = _descriptor(0, 4)
    entered = threading.Event()
    release = threading.Event()

    def gate_first(proposal: Proposal) -> None:
        if proposal.sequence == 1:
            entered.set()
            assert release.wait(5)

    publisher = BoundedProposalPublisher(
        _store(tmp_path, (descriptor,)),
        learner_id="learner-00",
        before_publish=gate_first,
    )
    publisher.submit(_snapshot(_proposal(descriptor, 1, parameters=b"a" * 16)))
    assert entered.wait(5)
    for sequence in range(2, 1002):
        byte = bytes([sequence % 251])
        publisher.submit(
            _snapshot(_proposal(descriptor, sequence, parameters=byte * 16))
        )
    during = publisher.summary()
    assert during["captured_snapshot_count"] == 1001
    assert during["snapshot_replacement_count"] == 999
    assert during["resident_trace_records"] <= 3
    assert "traces" not in during
    assert len(json.dumps(during, sort_keys=True)) < 10_000
    release.set()
    publisher.drain()
    after = publisher.summary()
    assert after["published_snapshot_count"] == 2
    assert after["resident_trace_records"] <= 3
    assert len(json.dumps(after, sort_keys=True)) < 10_000
    publisher.close()


def test_inflight_old_base_and_pending_new_base_coexist_immutably(
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    descriptor = _descriptor(0, parameter.numel())
    store = _store(tmp_path, (descriptor,))
    progress = LearnerProgress.initialize("learner-00", (3,))
    entered = threading.Event()
    release = threading.Event()
    traces: list[dict] = []

    def gate(_proposal_value: Proposal) -> None:
        entered.set()
        assert release.wait(5)

    coordinator = FragmentSnapshotCoordinator(
        identities=IDENTITIES,
        progress=progress,
        descriptors=(descriptor,),
        fragment_parameters=((parameter,),),
        adopted_bases=(_base(3),),
        schedule=FragmentPublicationSchedule.unified((parameter.numel() * 4,), 1),
        store=store,
        before_publish=gate,
        trace_sink=traces.append,
    )
    _advance(progress)
    coordinator.on_safe_boundary(_event(progress))
    assert entered.wait(5)
    coordinator.update_adopted_base(
        0,
        version=4,
        content_identity=_base(4).content_identity,
    )
    with torch.no_grad():
        parameter.add_(10)
    _advance(progress)
    coordinator.on_safe_boundary(_event(progress))
    coexist = coordinator.summary()["publication"]
    assert coexist["active_in_flight"][0]["base_version"] == 3
    assert coexist["active_pending"][0]["base_version"] == 4
    assert coexist["active_in_flight"][0]["sequence"] == 1
    assert coexist["active_pending"][0]["sequence"] == 2
    release.set()
    coordinator.drain()
    latest = store.load_latest("learner-00", 0)
    assert (latest.sequence, latest.base_version, latest.base_content_identity) == (
        2,
        4,
        _base(4).content_identity,
    )
    by_sequence = {trace["sequence"]: trace for trace in traces}
    assert by_sequence[1]["base_version"] == 3
    assert by_sequence[1]["base_content_identity"] == _base(3).content_identity
    assert by_sequence[2]["base_version"] == 4
    assert by_sequence[2]["base_content_identity"] == _base(4).content_identity
    coordinator.close()


def test_restart_uses_the_next_reserved_sequence(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    descriptor = _descriptor(0, parameter.numel())
    store = _store(tmp_path, (descriptor,))
    progress = LearnerProgress.initialize("learner-00", (3,))
    first = FragmentSnapshotCoordinator(
        identities=IDENTITIES,
        progress=progress,
        descriptors=(descriptor,),
        fragment_parameters=((parameter,),),
        adopted_bases=(_base(3),),
        schedule=FragmentPublicationSchedule.unified((parameter.numel() * 4,), 1),
        store=store,
    )
    _advance(progress)
    first.on_safe_boundary(_event(progress))
    reserved = LearnerProgress.from_dict(progress.to_dict())
    assert reserved.fragments[0].proposal_sequence == 1
    first.drain()
    first.close()

    restored_parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    restored = reserved
    restarted = FragmentSnapshotCoordinator(
        identities=IDENTITIES,
        progress=restored,
        descriptors=(descriptor,),
        fragment_parameters=((restored_parameter,),),
        adopted_bases=(_base(3),),
        schedule=FragmentPublicationSchedule.unified((restored_parameter.numel() * 4,), 1),
        store=store,
    )
    _advance(restored)
    restarted.on_safe_boundary(_event(restored))
    restarted.drain()
    assert restored.fragments[0].proposal_sequence == 2
    assert store.load_latest("learner-00", 0).sequence == 2
    restarted.close()


def test_snapshot_payload_stays_bytewise_immutable_during_later_mutation(
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    descriptor = _descriptor(0, parameter.numel())
    store = _store(tmp_path, (descriptor,))
    progress = LearnerProgress.initialize("learner-00", (0,))
    entered = threading.Event()
    release = threading.Event()

    def gate(_proposal_value: Proposal) -> None:
        entered.set()
        assert release.wait(5)

    coordinator = FragmentSnapshotCoordinator(
        identities=IDENTITIES,
        progress=progress,
        descriptors=(descriptor,),
        fragment_parameters=((parameter,),),
        adopted_bases=(_base(0),),
        schedule=FragmentPublicationSchedule.unified((parameter.numel() * 4,), 1),
        store=store,
        before_publish=gate,
    )
    _advance(progress)
    expected = serialize_fragment_parameters((parameter,), parameter.numel() * 4)
    coordinator.on_safe_boundary(_event(progress))
    assert entered.wait(5)
    with torch.no_grad():
        parameter.mul_(0).add_(99)
    current = serialize_fragment_parameters((parameter,), parameter.numel() * 4)
    assert current != expected
    release.set()
    coordinator.drain()
    published = store.load_latest("learner-00", 0)
    assert published.parameters == expected
    assert published.parameters_sha256 == hashlib.sha256(expected).hexdigest()
    coordinator.close()


def test_background_failure_clears_slots_abandons_pending_and_surfaces(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor(0, 4)
    entered = threading.Event()
    release = threading.Event()
    traces: list[dict] = []

    def fail_first(_proposal_value: Proposal) -> None:
        entered.set()
        assert release.wait(5)
        raise OSError("injected publication failure")

    publisher = BoundedProposalPublisher(
        _store(tmp_path, (descriptor,)),
        learner_id="learner-00",
        before_publish=fail_first,
        trace_sink=traces.append,
    )
    publisher.submit(_snapshot(_proposal(descriptor, 1, parameters=b"a" * 16)))
    assert entered.wait(5)
    publisher.submit(_snapshot(_proposal(descriptor, 2, parameters=b"b" * 16)))
    release.set()
    with pytest.raises(SnapshotPublishError, match="injected publication failure"):
        publisher.drain()
    failed = publisher.summary()
    assert failed["per_fragment_in_flight"] == [0]
    assert failed["per_fragment_pending"] == [0]
    assert failed["maximum_in_flight_per_fragment"] == [1]
    assert failed["maximum_pending_per_fragment"] == [1]
    assert failed["resident_trace_records"] <= 3
    assert sorted(trace["outcome"] for trace in traces) == [
        "abandoned_after_failure",
        "failed",
    ]
    with pytest.raises(SnapshotPublishError, match="injected publication failure"):
        publisher.close()


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(1707)
    return transformers.GPTNeoXForCausalLM(
        transformers.GPTNeoXConfig(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=16,
            use_cache=False,
        )
    )


def _batches(count: int) -> list[dict[str, torch.Tensor]]:
    result = []
    for step in range(count):
        input_ids = (torch.arange(32).reshape(4, 8) + step) % 64
        result.append(
            {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "attention_mask": torch.ones_like(input_ids),
            }
        )
    return result


def test_real_learner_boundary_stages_only_due_target_fragments(
    tmp_path: Path,
) -> None:
    model = _tiny_model()
    registry = build_logical_layer_registry(model)
    fragment_map = build_fragment_map(registry, 2)
    descriptors = build_fragment_descriptors(fragment_map)
    groups = build_fragment_parameter_groups(model, registry, fragment_map)
    identities = dataclasses.replace(
        IDENTITIES,
        model_identity=registry.digest,
        fragment_map_identity=fragment_map.digest,
    )
    store = ProposalStore(
        PosixStorageBackend(tmp_path),
        identities=identities,
        descriptors=descriptors,
        learner_ids=("learner-00",),
        maximum_local_steps=100,
        maximum_processed_tokens=1_000_000,
    )
    progress = LearnerProgress.initialize("learner-00", (0, 0))
    schedule = FragmentPublicationSchedule.unified(
        tuple(fragment.sync_bytes for fragment in fragment_map.fragments),
        2,
    )
    coordinator = FragmentSnapshotCoordinator(
        identities=identities,
        progress=progress,
        descriptors=descriptors,
        fragment_parameters=groups,
        adopted_bases=(_base(0, 0), _base(0, 1)),
        schedule=schedule,
        store=store,
    )
    runtime = LearnerRuntime(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.001),
        scheduler=ConstantStepScheduler(),
        progress=progress,
        rng=LearnerRng.initialize(progress.learner_id, 1707, torch.device("cpu")),
        fragment_parameters=groups,
        device=torch.device("cpu"),
        precision="fp32",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        comparison={"claim": "test"},
        safe_boundary_observers=(coordinator.on_safe_boundary,),
    )
    run = runtime.run(_batches(4), optimizer_steps=4)
    coordinator.drain()
    assert [event.active_metrics["snapshot_due_fragments"] for event in run.events] == [
        list(schedule.due_fragments(step)) for step in range(1, 5)
    ]
    assert all("gpu_to_cpu_seconds" in event.active_metrics for event in run.events)
    full_model_bytes = sum(parameter.numel() for parameter in model.parameters()) * 4
    for descriptor in descriptors:
        latest = store.load_latest("learner-00", descriptor.index)
        assert latest.snapshot_local_step in range(1, 5)
        assert schedule.is_due(latest.snapshot_local_step, descriptor.index)
        assert latest.payload_bytes == descriptor.shape[0] * 4
        assert latest.payload_bytes < full_model_bytes
        assert latest.local_steps == latest.snapshot_local_step
        assert latest.processed_tokens == latest.snapshot_local_step * 32
    summary = coordinator.summary()
    assert summary["publication"]["captured_snapshot_count"] == 4
    assert summary["publication"]["published_snapshot_count"] >= 2
    assert summary["publication"]["errors"] == []
    coordinator.close()


def test_manifest_rejects_role_rank_and_state_identity_forgery(tmp_path: Path) -> None:
    result_root = tmp_path / "roles"
    result_root.mkdir()
    state = IDENTITIES.to_dict()
    state["run_identity"] = "s1-07-123.opbs"
    writer = {
        "status": "pass",
        "role": "snapshot_writer",
        "rank": 0,
        "hostname": "node-a",
        "state_identities": state,
        "torch": {"distributed_initialized": False},
        "writer_waited_for_reader_or_syncer": False,
    }
    reader = {
        "status": "pass",
        "role": "proposal_reader",
        "rank": 1,
        "hostname": "node-b",
        "state_identities": state,
    }
    (result_root / "snapshot_writer.json").write_text(json.dumps(writer))
    (result_root / "proposal_reader.json").write_text(json.dumps(reader))
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("node-a\nnode-b\n")
    modules = tmp_path / "modules"
    modules.write_text("nv-hpcx/25.9\n")
    args = SimpleNamespace(
        nodefile=str(nodefile),
        modules_file=str(modules),
        result_root=str(result_root),
        repository="https://example.invalid/repository.git",
        branch="codex/S1-07-snapshot-publish",
        commit="1" * 40,
        run_id="s1-07-123.opbs",
        config_sha256="a" * 64,
        research_sha256="b" * 64,
        spec_sha256="c" * 64,
        skill_repository="https://example.invalid/skill.git",
        skill_commit="2" * 40,
        initial_hostname="miyabi-g1",
        project_root="/work/project",
        evidence_root="/work/evidence",
        job_id="123.opbs",
        qtime_utc="2026-07-15T00:00:00Z",
        queue="debug-g",
        group="xg24i002",
        output=str(tmp_path / "manifest.json"),
    )
    write_manifest(args)
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "building"

    reader["rank"] = 0
    (result_root / "proposal_reader.json").write_text(json.dumps(reader))
    with pytest.raises(SnapshotStressError, match="status rank or host"):
        write_manifest(args)
    reader["rank"] = 1
    reader["state_identities"] = {**state, "config_identity": "f" * 64}
    (result_root / "proposal_reader.json").write_text(json.dumps(reader))
    with pytest.raises(SnapshotStressError, match="state/config/run"):
        write_manifest(args)
