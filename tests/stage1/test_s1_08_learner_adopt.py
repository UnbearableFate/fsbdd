from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from fsbdd.diloco.model.fragment_map import build_fragment_map  # noqa: E402
from fsbdd.diloco.protocol.global_state import (  # noqa: E402
    BootstrapFragment,
    FragmentStateDescriptor,
    GlobalStateIdentities,
    GlobalStateStore,
)
from fsbdd.diloco.learner.runtime import (  # noqa: E402
    ConstantStepScheduler,
    LearnerProgress,
    LearnerRng,
    LearnerRuntime,
    SafeBoundaryEvent,
)
from fsbdd.diloco.learner.adoption import (  # noqa: E402
    AdoptionError,
    FragmentAdoptionCoordinator,
    LatestFragmentPoller,
    optimizer_fragment_state_sha256,
)
from fsbdd.diloco.learner.publication import (  # noqa: E402
    AdoptedFragmentBase,
    FragmentPublicationSchedule,
    FragmentSnapshotCoordinator,
    build_fragment_descriptors,
    build_fragment_parameter_groups,
    fragment_payload_sha256,
    serialize_fragment_parameters,
)
from fsbdd.diloco.model.model_registry import build_logical_layer_registry  # noqa: E402
from fsbdd.diloco.protocol.proposal import ProposalStore  # noqa: E402
from fsbdd.diloco.protocol.storage import (  # noqa: E402
    PosixStorageBackend,
    PublicationError,
    PublicationNotReady,
)


IDENTITIES = GlobalStateIdentities(
    run_identity="run-s1-08",
    config_identity="a" * 64,
    model_identity="b" * 64,
    fragment_map_identity="c" * 64,
)


def _descriptors(
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[FragmentStateDescriptor, ...]:
    return tuple(
        FragmentStateDescriptor(
            index=index,
            identity=f"fragment-{index}",
            dtype="float32",
            shape=(parameter.numel(),),
            parameter_identities=(f"parameter-{index}",),
        )
        for index, parameter in enumerate(parameters)
    )


def _bootstrap(
    root: Path,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    identities: GlobalStateIdentities = IDENTITIES,
    descriptors: tuple[FragmentStateDescriptor, ...] | None = None,
) -> tuple[GlobalStateStore, tuple]:
    descriptors = descriptors or _descriptors(parameters)
    store = GlobalStateStore(
        PosixStorageBackend(root),
        identities=identities,
        descriptors=descriptors,
        s_max=0,
    )
    result = store.bootstrap(
        tuple(
            BootstrapFragment(
                descriptor=descriptor,
                parameters=serialize_fragment_parameters(
                    (parameter,), parameter.numel() * 4
                ),
                outer_state=b"outer-v0",
            )
            for descriptor, parameter in zip(descriptors, parameters, strict=True)
        )
    )
    return store, result.snapshot.states


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


def _shift(payload: bytes, amount: float) -> bytes:
    value = np.frombuffer(payload, dtype="<f4").copy()
    value += np.float32(amount)
    return value.astype("<f4", copy=False).tobytes()


def _initialize_adam_moments(
    parameters: tuple[torch.nn.Parameter, ...], optimizer: torch.optim.Optimizer
) -> None:
    loss = sum(parameter.square().sum() for parameter in parameters)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_latest_only_target_scope_counters_and_moments(tmp_path: Path) -> None:
    parameters = tuple(
        torch.nn.Parameter(torch.arange(4, dtype=torch.float32) + index * 10)
        for index in range(3)
    )
    store, initial = _bootstrap(tmp_path / "global", parameters)
    optimizer = torch.optim.AdamW(parameters, lr=0.001)
    _initialize_adam_moments(parameters, optimizer)
    progress = LearnerProgress.initialize("learner-00", (0, 0, 0))
    for version in range(1, 4):
        previous = store.load_fragment(0)
        store.publish_successor(
            0,
            parameters=_shift(previous.parameters, 0.25),
            outer_state=f"outer-v{version}".encode(),
        )
    latest = store.load_fragment(0)
    before_parameters = [
        fragment_payload_sha256((parameter,)) for parameter in parameters
    ]
    before_moments = [
        optimizer_fragment_state_sha256(optimizer, (parameter,))
        for parameter in parameters
    ]
    coordinator = FragmentAdoptionCoordinator(
        store=store,
        progress=progress,
        initial_states=initial,
        fragment_parameters=tuple((parameter,) for parameter in parameters),
        optimizer=optimizer,
        identity_audit=True,
        autostart=False,
    )
    discoveries = coordinator.poll_once()
    assert [(row["fragment_index"], row["version"]) for row in discoveries] == [(0, 3)]
    _advance(progress)
    metrics = coordinator.on_safe_boundary(_event(progress))
    assert metrics["adoption_applied_fragments"] == [0]
    trace = metrics["adoption_traces"][0]
    assert (trace["from_version"], trace["to_version"]) == (0, 3)
    assert trace["skipped_intermediate_versions"] == 2
    assert (
        trace["target_payload_sha256"] == hashlib.sha256(latest.parameters).hexdigest()
    )
    assert trace["parameter_hashes_after"][0] == trace["target_payload_sha256"]
    assert (
        trace["optimizer_state_hashes_before"] == trace["optimizer_state_hashes_after"]
    )
    assert trace["optimizer_state_entry_counts_before"] == [1, 1, 1]
    assert trace["optimizer_state_entry_counts_after"] == [1, 1, 1]
    assert [fragment_payload_sha256((parameter,)) for parameter in parameters][
        1:
    ] == before_parameters[1:]
    assert [
        optimizer_fragment_state_sha256(optimizer, (parameter,))
        for parameter in parameters
    ] == before_moments
    assert [fragment.global_version for fragment in progress.fragments] == [3, 0, 0]
    assert [fragment.local_steps_since_adoption for fragment in progress.fragments] == [
        0,
        1,
        1,
    ]
    assert [
        fragment.processed_input_tokens_since_adoption
        for fragment in progress.fragments
    ] == [0, 10, 10]
    coordinator.close()


def test_pending_is_latest_wins_bounded_and_repeated_current_is_ignored(
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    store, initial = _bootstrap(tmp_path / "global", (parameter,))
    progress = LearnerProgress.initialize("learner-00", (0,))
    coordinator = FragmentAdoptionCoordinator(
        store=store,
        progress=progress,
        initial_states=initial,
        fragment_parameters=((parameter,),),
        autostart=False,
    )
    for version in range(1, 4):
        previous = store.load_fragment(0)
        store.publish_successor(
            0,
            parameters=_shift(previous.parameters, 1.0),
            outer_state=f"outer-v{version}".encode(),
        )
        coordinator.poll_once()
    coordinator.poll_once()
    polling = coordinator.summary()["poller"]
    assert polling["pending_count"] == 1
    assert polling["maximum_pending_per_fragment"] == [1]
    assert polling["pending_replacement_count"] == 2
    assert polling["repeated_current_count"] >= 1
    _advance(progress)
    metrics = coordinator.on_safe_boundary(_event(progress))
    assert metrics["adoption_traces"][0]["to_version"] == 3
    assert metrics["adoption_traces"][0]["skipped_intermediate_versions"] == 2
    coordinator.close()


def test_invalid_payload_and_active_gradient_fail_before_target_mutation(
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    store, initial = _bootstrap(tmp_path / "global", (parameter,))
    store.publish_successor(0, parameters=b"short", outer_state=b"outer-v1")
    progress = LearnerProgress.initialize("learner-00", (0,))
    coordinator = FragmentAdoptionCoordinator(
        store=store,
        progress=progress,
        initial_states=initial,
        fragment_parameters=((parameter,),),
        autostart=False,
    )
    coordinator.poll_once()
    before = parameter.detach().clone()
    _advance(progress)
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(AdoptionError, match="gradients remain active"):
        coordinator.on_safe_boundary(_event(progress))
    assert torch.equal(parameter, before)
    parameter.grad = None
    with pytest.raises(AdoptionError, match="payload byte count"):
        coordinator.on_safe_boundary(_event(progress))
    assert torch.equal(parameter, before)
    coordinator.close()


def test_wrong_fragment_map_identity_is_rejected_by_fixed_slot_poll(
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    store, _initial = _bootstrap(tmp_path / "global", (parameter,))
    wrong = GlobalStateStore(
        PosixStorageBackend(tmp_path / "global"),
        identities=dataclasses.replace(
            store.identities, fragment_map_identity="f" * 64
        ),
        descriptors=store.descriptors,
        s_max=0,
    )
    poller = LatestFragmentPoller(
        wrong,
        adopted_versions=(0,),
        adopted_content_identities=("0" * 64,),
        completed_step=lambda: 0,
        autostart=False,
    )
    with pytest.raises(PublicationError, match="fragment_map_identity"):
        poller.poll_once()
    poller.close()


def test_poller_rejects_same_version_visibility_record_rewrite(
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    store, initial = _bootstrap(tmp_path / "global", (parameter,))
    poller = LatestFragmentPoller(
        store,
        adopted_versions=(0,),
        adopted_content_identities=(initial[0].content_identity,),
        completed_step=lambda: 0,
        autostart=False,
    )
    poller.poll_once()
    visibility = tmp_path / "global" / "visibility" / "global-current-000000.json"
    record = json.loads(visibility.read_bytes())
    source = tmp_path / "global" / record["payload_relative_path"]
    replacement = tmp_path / "global" / "payloads" / "record-rewrite.bin"
    shutil.copyfile(source, replacement)
    record["payload_relative_path"] = "payloads/record-rewrite.bin"
    visibility.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AdoptionError, match="record changed at the same version"):
        poller.poll_once()
    poller.close()


class _OneTransientReadBackend:
    def __init__(self, backend: PosixStorageBackend) -> None:
        self.backend = backend
        self.remaining_transient_reads = 0

    @property
    def root(self) -> Path:
        return self.backend.root

    def publish(self, *args: object, **kwargs: object) -> object:
        return self.backend.publish(*args, **kwargs)

    def read(self, *args: object, **kwargs: object) -> object:
        if self.remaining_transient_reads > 0:
            self.remaining_transient_reads -= 1
            raise PublicationNotReady("injected eventually-readable payload")
        return self.backend.read(*args, **kwargs)


def test_background_poller_retries_eventually_readable_payload_and_adopts(
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    descriptor = _descriptors((parameter,))
    backend = _OneTransientReadBackend(PosixStorageBackend(tmp_path / "global"))
    store = GlobalStateStore(
        backend,
        identities=IDENTITIES,
        descriptors=descriptor,
        s_max=0,
    )
    initial = store.bootstrap(
        (
            BootstrapFragment(
                descriptor=descriptor[0],
                parameters=serialize_fragment_parameters((parameter,), 16),
                outer_state=b"outer-v0",
            ),
        )
    ).snapshot.states
    store.publish_successor(
        0,
        parameters=_shift(initial[0].parameters, 1.0),
        outer_state=b"outer-v1",
    )
    backend.remaining_transient_reads = 1
    progress = LearnerProgress.initialize("learner-00", (0,))
    coordinator = FragmentAdoptionCoordinator(
        store=store,
        progress=progress,
        initial_states=initial,
        fragment_parameters=((parameter,),),
        poll_interval_seconds=0.001,
        autostart=True,
    )
    deadline = time.monotonic() + 5
    while coordinator.summary()["poller"]["discovery_count"] == 0:
        if time.monotonic() >= deadline:
            raise AssertionError("eventually readable fragment was not discovered")
        time.sleep(0.001)
    polling = coordinator.summary()["poller"]
    assert polling["transient_read_retry_count"] == 1
    assert polling["errors"] == []
    _advance(progress)
    metrics = coordinator.on_safe_boundary(_event(progress))
    assert metrics["adoption_applied_fragments"] == [0]
    assert progress.fragments[0].global_version == 1
    coordinator.close()


def _tiny_neox() -> torch.nn.Module:
    torch.manual_seed(1808)
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


def test_real_learner_sustains_finite_mixed_version_training(tmp_path: Path) -> None:
    model = _tiny_neox()
    registry = build_logical_layer_registry(model)
    fragment_map = build_fragment_map(registry, 2)
    descriptors = build_fragment_descriptors(fragment_map)
    groups = build_fragment_parameter_groups(model, registry, fragment_map)
    identities = dataclasses.replace(
        IDENTITIES,
        model_identity=registry.digest,
        fragment_map_identity=fragment_map.digest,
    )
    store = GlobalStateStore(
        PosixStorageBackend(tmp_path / "global"),
        identities=identities,
        descriptors=descriptors,
        s_max=0,
    )
    initial = store.bootstrap(
        tuple(
            BootstrapFragment(
                descriptor=descriptor,
                parameters=serialize_fragment_parameters(
                    group, descriptor.shape[0] * 4
                ),
                outer_state=b"outer-v0",
            )
            for descriptor, group in zip(descriptors, groups, strict=True)
        )
    ).snapshot.states
    store.publish_successor(
        0,
        parameters=_shift(initial[0].parameters, 0.02),
        outer_state=b"outer-v1",
    )
    progress = LearnerProgress.initialize("learner-00", (0, 0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    coordinator = FragmentAdoptionCoordinator(
        store=store,
        progress=progress,
        initial_states=initial,
        fragment_parameters=groups,
        optimizer=optimizer,
        identity_audit=True,
        autostart=False,
    )
    coordinator.poll_once()
    runtime = LearnerRuntime(
        model=model,
        optimizer=optimizer,
        scheduler=ConstantStepScheduler(),
        progress=progress,
        rng=LearnerRng.initialize(progress.learner_id, 1808, torch.device("cpu")),
        fragment_parameters=groups,
        device=torch.device("cpu"),
        precision="fp32",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        comparison={"claim": "mixed-version-test"},
        safe_boundary_observers=(coordinator.on_safe_boundary,),
    )
    run = runtime.run(_batches(8), optimizer_steps=8)
    adoption = run.events[0].active_metrics["adoption_traces"][0]
    assert (
        adoption["optimizer_state_hashes_before"]
        == adoption["optimizer_state_hashes_after"]
    )
    mixed_window = run.events[1:]
    assert len(mixed_window) == 7
    assert all(event.fragment_global_versions == (1, 0) for event in mixed_window)
    assert all(math.isfinite(event.token_weighted_loss) for event in mixed_window)
    assert [event.local_optimizer_step for event in mixed_window] == list(range(2, 9))
    assert all(
        "fs_to_cpu_seconds" not in event.inactive_metrics for event in run.events
    )
    assert all(not event.distributed_initialized for event in run.events)
    coordinator.close()


def test_tied_parameter_has_one_fragment_owner_and_target_only_adoption(
    tmp_path: Path,
) -> None:
    torch.manual_seed(1808)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=2,
            n_head=4,
            n_positions=16,
            use_cache=False,
            tie_word_embeddings=True,
        )
    )
    assert model.transformer.wte.weight is model.lm_head.weight
    registry = build_logical_layer_registry(model)
    fragment_map = build_fragment_map(registry, 2)
    descriptors = build_fragment_descriptors(fragment_map)
    groups = build_fragment_parameter_groups(model, registry, fragment_map)
    tied_identity = registry.tied_identities[0]
    target = next(
        fragment.index
        for fragment in fragment_map.fragments
        if tied_identity in fragment.parameter_identities
    )
    assert (
        sum(
            tied_identity in fragment.parameter_identities
            for fragment in fragment_map.fragments
        )
        == 1
    )
    identities = dataclasses.replace(
        IDENTITIES,
        model_identity=registry.digest,
        fragment_map_identity=fragment_map.digest,
    )
    store = GlobalStateStore(
        PosixStorageBackend(tmp_path / "global"),
        identities=identities,
        descriptors=descriptors,
        s_max=0,
    )
    initial = store.bootstrap(
        tuple(
            BootstrapFragment(
                descriptor=descriptor,
                parameters=serialize_fragment_parameters(
                    group, descriptor.shape[0] * 4
                ),
                outer_state=b"outer-v0",
            )
            for descriptor, group in zip(descriptors, groups, strict=True)
        )
    ).snapshot.states
    store.publish_successor(
        target,
        parameters=_shift(initial[target].parameters, 0.125),
        outer_state=b"outer-v1",
    )
    progress = LearnerProgress.initialize("learner-00", (0, 0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = [fragment_payload_sha256(group) for group in groups]
    coordinator = FragmentAdoptionCoordinator(
        store=store,
        progress=progress,
        initial_states=initial,
        fragment_parameters=groups,
        optimizer=optimizer,
        identity_audit=True,
        autostart=False,
    )
    coordinator.poll_once()
    _advance(progress)
    coordinator.on_safe_boundary(_event(progress))
    after = [fragment_payload_sha256(group) for group in groups]
    assert after[target] != before[target]
    assert all(after[index] == before[index] for index in range(2) if index != target)
    assert model.transformer.wte.weight is model.lm_head.weight
    coordinator.close()


def test_inflight_old_base_proposal_survives_later_adoption(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    descriptors = _descriptors((parameter,))
    global_store, initial = _bootstrap(
        tmp_path / "global", (parameter,), descriptors=descriptors
    )
    proposal_store = ProposalStore(
        PosixStorageBackend(tmp_path / "proposals"),
        identities=IDENTITIES,
        descriptors=descriptors,
        learner_ids=("learner-00",),
        maximum_local_steps=100,
        maximum_processed_tokens=1_000_000,
    )
    progress = LearnerProgress.initialize("learner-00", (0,))
    entered = threading.Event()
    release = threading.Event()

    def gate(_proposal: object) -> None:
        entered.set()
        assert release.wait(5)

    snapshot = FragmentSnapshotCoordinator(
        identities=IDENTITIES,
        progress=progress,
        descriptors=descriptors,
        fragment_parameters=((parameter,),),
        adopted_bases=(AdoptedFragmentBase(0, initial[0].content_identity),),
        schedule=FragmentPublicationSchedule.unified((parameter.numel() * 4,), 1),
        store=proposal_store,
        before_publish=gate,
    )
    global_store.publish_successor(
        0,
        parameters=_shift(initial[0].parameters, 2.0),
        outer_state=b"outer-v1",
    )
    adoption = FragmentAdoptionCoordinator(
        store=global_store,
        progress=progress,
        initial_states=initial,
        fragment_parameters=((parameter,),),
        base_context_sink=snapshot.update_adopted_base_context,
        autostart=False,
    )
    adoption.poll_once()
    _advance(progress)
    event = _event(progress)
    old_parameters = serialize_fragment_parameters((parameter,), parameter.numel() * 4)
    snapshot.on_safe_boundary(event)
    assert entered.wait(5)
    adoption.on_safe_boundary(event)
    during = snapshot.summary()["publication"]["active_in_flight"][0]
    assert during["base_version"] == 0
    assert during["base_content_identity"] == initial[0].content_identity
    assert snapshot.summary()["adopted_bases"][0]["version"] == 1
    release.set()
    snapshot.drain()
    published = proposal_store.load_latest("learner-00", 0)
    assert published.base_version == 0
    assert published.base_content_identity == initial[0].content_identity
    assert published.parameters == old_parameters
    adoption.close()
    snapshot.close()
