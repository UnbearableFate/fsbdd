from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import socket
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fsbdd_stage0.oracle import (
    OuterSGDState,
    inverse_staleness_weights,
    outer_sgd_step,
    weighted_direct_merge,
)

from .evaluation import (
    FrozenEvaluationPlan,
    load_evaluation_snapshot,
    materialize_evaluation_snapshot,
)
from .fragment_map import build_fragment_map
from .global_commit import AtomicGlobalCommitStore
from .global_state import (
    BootstrapFragment,
    FragmentStateDescriptor,
    GlobalStateIdentities,
    GlobalStateStore,
)
from .identity import canonical_digest, file_digest
from .learner import (
    ConstantStepScheduler,
    LearnerProgress,
    LearnerRng,
    LearnerRuntime,
)
from .learner_adopt import FragmentAdoptionCoordinator
from .learner_publish import (
    AdoptedFragmentBase,
    FragmentPublicationSchedule,
    FragmentSnapshotCoordinator,
    build_fragment_descriptors,
    build_fragment_parameter_groups,
    serialize_fragment_parameters,
)
from .manifest import build_manifest
from .model_registry import build_logical_layer_registry
from .profile_a import (
    ComparisonBasis,
    FrozenStopPolicy,
    ProfileAConfig,
    ProfileAFragmentExecutor,
    ProfileAProgressTracker,
    profile_a_policy_identity,
)
from .proposal import Proposal, ProposalStore
from .storage import PosixStorageBackend
from .syncer_merge import OuterSGDPolicy


class ProfileAStressError(RuntimeError):
    pass


_CLAIM = "protocol_and_runtime_acceptance_only_no_quality_or_efficiency_comparison"


def _write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _wait_json(path: Path, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"coordination record did not appear: {path}")
            time.sleep(0.01)
            continue
        if not isinstance(value, dict) or value.get("complete") is not True:
            raise ProfileAStressError(f"coordination record is malformed: {path}")
        return value


def _wait_all(root: Path, prefix: str, count: int, timeout: float) -> list[dict[str, Any]]:
    return [
        _wait_json(root / f"{prefix}-{rank:02d}.json", timeout)
        for rank in range(count)
    ]


def _rank() -> int:
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"):
        if name in os.environ:
            return int(os.environ[name])
    raise ProfileAStressError("role command requires an MPI launcher rank")


def _size() -> int:
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"):
        if name in os.environ:
            return int(os.environ[name])
    raise ProfileAStressError("role command requires an MPI launcher size")


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _load_config(path: Path, expected_identity: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_identity:
        raise ProfileAStressError("Profile A config identity mismatch")
    value = json.loads(raw)
    expected = {
        "schema_version",
        "profile",
        "outer_optimizer",
        "numeric_oracle",
        "evaluation",
        "real_fs",
        "progress",
        "comparison",
        "coordination_timeout_seconds",
    }
    if not isinstance(value, dict) or set(value) != expected or value["schema_version"] != 1:
        raise ProfileAStressError("Profile A config schema mismatch")
    profile = value["profile"]
    outer = value["outer_optimizer"]
    real = value["real_fs"]
    if (
        profile.get("learner_count") != 4
        or profile.get("fragment_count") != 2
        or profile.get("q") != 4
        or profile.get("q_fresh") != 4
        or profile.get("s_max") != 0
        or profile.get("grace_period_seconds") != 0.0
        or profile.get("deterministic_fragment_schedule") != "round_robin"
        or outer.get("merge") != "direct_weighted_average"
        or outer.get("accumulation_dtype") != "float32"
        or value["numeric_oracle"].get("updates_per_fragment") != 50
        or real.get("nodes") != 4
        or real.get("ranks") != 4
        or real.get("learner_processes") != 4
        or real.get("application_coordination") != "shared_filesystem_only"
        or value["comparison"]
        != {
            "matched_compute": False,
            "matched_communication": False,
            "matched_tokens": False,
            "claim": _CLAIM,
        }
    ):
        raise ProfileAStressError("Profile A frozen values differ")
    ProfileAConfig(
        learner_count=profile["learner_count"],
        fragment_count=profile["fragment_count"],
        q=profile["q"],
        q_fresh=profile["q_fresh"],
        s_max=profile["s_max"],
        grace_period_ns=int(profile["grace_period_seconds"] * 1e9),
        lambda_s=profile["lambda_s"],
        schedule=profile["deterministic_fragment_schedule"],
    )
    return value


def _profile(value: Mapping[str, Any]) -> ProfileAConfig:
    item = value["profile"]
    return ProfileAConfig(
        learner_count=int(item["learner_count"]),
        fragment_count=int(item["fragment_count"]),
        q=int(item["q"]),
        q_fresh=int(item["q_fresh"]),
        s_max=int(item["s_max"]),
        grace_period_ns=int(float(item["grace_period_seconds"]) * 1e9),
        lambda_s=float(item["lambda_s"]),
        schedule=str(item["deterministic_fragment_schedule"]),
    )


def _outer_policy(value: Mapping[str, Any]) -> OuterSGDPolicy:
    item = value["outer_optimizer"]
    return OuterSGDPolicy(
        learning_rate=float(item["learning_rate"]),
        momentum=float(item["momentum"]),
        nesterov=bool(item["nesterov"]),
    )


def _comparison() -> ComparisonBasis:
    return ComparisonBasis(False, False, False, _CLAIM)


def _learner_ids(count: int = 4) -> tuple[str, ...]:
    return tuple(f"learner-{index:02d}" for index in range(count))


def _descriptor_from_dict(value: Mapping[str, Any]) -> FragmentStateDescriptor:
    return FragmentStateDescriptor(
        index=int(value["index"]),
        identity=str(value["identity"]),
        dtype=str(value["dtype"]),
        shape=tuple(int(item) for item in value["shape"]),
        parameter_identities=tuple(str(item) for item in value["parameter_identities"]),
    )


def _catalog_matches_model(
    catalog: Mapping[str, Any],
    identities: GlobalStateIdentities,
    descriptors: Sequence[FragmentStateDescriptor],
) -> bool:
    try:
        catalog_descriptors = tuple(
            _descriptor_from_dict(item) for item in catalog["descriptors"]
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        catalog.get("identities") == identities.to_dict()
        and catalog_descriptors == tuple(descriptors)
    )


def _model(config: Mapping[str, Any], device: Any) -> tuple[Any, Any, Any, Any]:
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    model_spec = config["real_fs"]["tiny_hf_model"]
    torch.manual_seed(20260715)
    torch.cuda.manual_seed_all(20260715)
    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=int(model_spec["vocab_size"]),
            hidden_size=int(model_spec["hidden_size"]),
            intermediate_size=int(model_spec["intermediate_size"]),
            num_hidden_layers=int(model_spec["num_hidden_layers"]),
            num_attention_heads=int(model_spec["num_attention_heads"]),
            max_position_embeddings=int(model_spec["max_position_embeddings"]),
            use_cache=False,
        )
    ).to(device)
    registry = build_logical_layer_registry(model)
    fragment_map = build_fragment_map(
        registry, int(config["profile"]["fragment_count"])
    )
    descriptors = build_fragment_descriptors(fragment_map)
    groups = build_fragment_parameter_groups(model, registry, fragment_map)
    return model, registry, fragment_map, groups


def _batches(
    *,
    count: int,
    rank: int,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    start_unix_ns: int | None = None,
    period_seconds: float | None = None,
) -> Iterator[dict[str, Any]]:
    import torch

    base = torch.arange(batch_size * sequence_length).reshape(
        batch_size, sequence_length
    )
    for step in range(count):
        if start_unix_ns is not None and period_seconds is not None:
            target = start_unix_ns + int(step * period_seconds * 1e9)
            while time.time_ns() < target:
                remaining = (target - time.time_ns()) / 1e9
                time.sleep(min(0.01, max(0.0, remaining)))
        input_ids = (base + rank * 17 + step * 3) % vocab_size
        yield {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }


def _runtime(
    *,
    model: Any,
    groups: Sequence[Sequence[Any]],
    progress: LearnerProgress,
    scheduler: ConstantStepScheduler,
    rng: LearnerRng,
    device: Any,
    learning_rate: float,
    optimizer: Any | None = None,
    observers: Sequence[Any] = (),
) -> tuple[LearnerRuntime, Any]:
    import torch

    owned_optimizer = optimizer or torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )
    return (
        LearnerRuntime(
            model=model,
            optimizer=owned_optimizer,
            scheduler=scheduler,
            progress=progress,
            rng=rng,
            fragment_parameters=groups,
            device=device,
            precision="fp32",
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            comparison={"claim": _CLAIM},
            safe_boundary_observers=observers,
        ),
        owned_optimizer,
    )


def _speed_trial(
    *,
    config: Mapping[str, Any],
    rank: int,
    device: Any,
    phase: Mapping[str, Any],
    injected: bool,
) -> dict[str, Any]:
    import torch

    speed = config["real_fs"]["speed_injection"]
    training = config["real_fs"]["training"]
    interval = float(speed["common_interval_seconds"])
    unaffected_period = float(speed["unaffected_period_seconds"])
    slow_period = float(speed["slow_period_seconds"])
    slow_rank = int(speed["slow_learner_index"])
    period = slow_period if injected and rank == slow_rank else unaffected_period
    control_steps = int(math.floor(interval / unaffected_period))
    steps = int(math.floor(interval / period))
    model, _, _, groups = _model(config, device)
    progress = LearnerProgress.initialize(f"speed-{rank}", (0, 0))
    scheduler = ConstantStepScheduler()
    runtime, _ = _runtime(
        model=model,
        groups=groups,
        progress=progress,
        scheduler=scheduler,
        rng=LearnerRng.initialize(f"speed-{rank}", 8100 + rank, device),
        device=device,
        learning_rate=float(training["inner_learning_rate"]),
    )
    start = int(phase["start_unix_ns"])
    end = int(phase["end_unix_ns"])
    while time.time_ns() < start:
        time.sleep(0.01)
    run = runtime.run(
        _batches(
            count=steps,
            rank=rank,
            batch_size=int(training["batch_size"]),
            sequence_length=int(training["sequence_length"]),
            vocab_size=int(config["real_fs"]["tiny_hf_model"]["vocab_size"]),
            start_unix_ns=start,
            period_seconds=period,
        ),
        optimizer_steps=steps,
    )
    completed_unix_ns = time.time_ns()
    if completed_unix_ns > end:
        raise ProfileAStressError("speed trial exceeded its frozen common interval")
    while time.time_ns() < end:
        time.sleep(0.01)
    tokens = run.processed_input_tokens
    result = {
        "injected": injected,
        "start_unix_ns": start,
        "end_unix_ns": end,
        "common_interval_seconds": interval,
        "scheduled_period_seconds": period,
        "control_step_count": control_steps,
        "completed_steps": run.optimizer_steps_completed,
        "processed_input_tokens": tokens,
        "common_interval_input_tokens_per_second": tokens / interval,
        "training_completed_unix_ns": completed_unix_ns,
        "finite_loss": math.isfinite(run.token_weighted_loss),
        "token_weighted_loss": run.token_weighted_loss,
        "distributed_initialized": run.distributed_initialized,
    }
    del runtime, model
    torch.cuda.empty_cache()
    return result


def _state_catalog(
    *,
    run_id: str,
    config_identity: str,
    registry: Any,
    fragment_map: Any,
    descriptors: Sequence[FragmentStateDescriptor],
    authorities: Sequence[Any],
) -> dict[str, Any]:
    return {
        "complete": True,
        "run_id": run_id,
        "config_identity": config_identity,
        "identities": GlobalStateIdentities(
            run_identity=run_id,
            config_identity=config_identity,
            model_identity=registry.digest,
            fragment_map_identity=fragment_map.digest,
        ).to_dict(),
        "descriptors": [item.to_dict() for item in descriptors],
        "bootstrap": [
            {
                "fragment_index": item.state.descriptor.index,
                "version": item.version,
                "content_identity": item.content_identity,
                "parameters_sha256": hashlib.sha256(item.parameters).hexdigest(),
            }
            for item in authorities
        ],
    }


def _stores(
    root: Path,
    *,
    catalog: Mapping[str, Any],
    policy: OuterSGDPolicy,
) -> tuple[AtomicGlobalCommitStore, ProposalStore]:
    identities = GlobalStateIdentities(**catalog["identities"])
    descriptors = tuple(_descriptor_from_dict(item) for item in catalog["descriptors"])
    learners = _learner_ids()
    atomic = AtomicGlobalCommitStore(
        GlobalStateStore(
            PosixStorageBackend(root / "global"),
            identities=identities,
            descriptors=descriptors,
            s_max=0,
        ),
        learner_ids=learners,
        policy_identity=profile_a_policy_identity(policy),
    )
    proposals = ProposalStore(
        PosixStorageBackend(root / "proposals"),
        identities=identities,
        descriptors=descriptors,
        learner_ids=learners,
        maximum_local_steps=1_000_000,
        maximum_processed_tokens=1_000_000_000,
    )
    return atomic, proposals


def _update_summary(update: Any) -> dict[str, Any]:
    return {
        "fragment_index": update.successor.state.descriptor.index,
        "from_version": update.previous.version,
        "to_version": update.successor.version,
        "from_content_identity": update.previous.content_identity,
        "to_content_identity": update.successor.content_identity,
        "selection_identity": update.selection.selection_identity,
        "selected_learners": list(update.selection.learner_ids),
        "weights": [item.normalized_weight for item in update.selection.weights],
        "update_identity": update.result.update_identity,
        "byte_accounting": update.result.byte_accounting.to_dict(),
        "source_metrics": dataclasses.asdict(update.source_metrics),
    }


def _publish_synthetic_round(
    atomic: AtomicGlobalCommitStore,
    proposals: ProposalStore,
    fragment_index: int,
    *,
    tag: str,
) -> None:
    authority = atomic.load_fragment(fragment_index)
    current = np.frombuffer(authority.parameters, dtype="<f4")
    for learner_index, learner_id in enumerate(_learner_ids()):
        delta = np.float32((learner_index + 1) * (authority.version + 1) * 1e-5)
        local = np.subtract(current, delta, dtype=np.float32).astype("<f4").tobytes()
        proposals.publish(
            Proposal.create(
                proposal_id=(
                    f"{tag}-{learner_id}-fragment-{fragment_index}-"
                    f"base-{authority.version}-sequence-{authority.version + 1}"
                ),
                identities=authority.state.identities,
                learner_id=learner_id,
                descriptor=authority.state.descriptor,
                sequence=authority.version + 1,
                base_version=authority.version,
                base_content_identity=authority.content_identity,
                local_steps=1,
                processed_tokens=100 + learner_index,
                snapshot_local_step=authority.version + 1,
                parameters=local,
            )
        )


def _numeric_descriptor(index: int, elements: int) -> FragmentStateDescriptor:
    return FragmentStateDescriptor(
        index=index,
        identity=_identity(f"s1-12-numeric-fragment-{index}"),
        dtype="float32",
        shape=(elements,),
        parameter_identities=(f"numeric-parameter-{index}",),
    )


def _relative_l2(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(float(np.linalg.norm(right_array)), 1e-12)
    )


def run_numeric_e2e(
    root: Path,
    *,
    run_id: str,
    config_identity: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    numeric = config["numeric_oracle"]
    elements = int(numeric["fragment_elements"])
    updates = int(numeric["updates_per_fragment"])
    policy = _outer_policy(config)
    identities = GlobalStateIdentities(
        run_identity=f"{run_id}-numeric",
        config_identity=config_identity,
        model_identity=_identity("s1-12-numeric-model"),
        fragment_map_identity=_identity("s1-12-numeric-map"),
    )
    descriptors = tuple(_numeric_descriptor(index, elements) for index in range(2))
    atomic = AtomicGlobalCommitStore(
        GlobalStateStore(
            PosixStorageBackend(root / "global"),
            identities=identities,
            descriptors=descriptors,
            s_max=0,
        ),
        learner_ids=_learner_ids(),
        policy_identity=profile_a_policy_identity(policy),
    )
    atomic.bootstrap(
        tuple(
            BootstrapFragment(
                descriptor=descriptor,
                parameters=np.linspace(
                    -0.75 + index * 0.1,
                    0.75 + index * 0.1,
                    elements,
                    dtype=np.float32,
                ).astype("<f4").tobytes(),
                outer_state=b"",
            )
            for index, descriptor in enumerate(descriptors)
        )
    )
    proposals = ProposalStore(
        PosixStorageBackend(root / "proposals"),
        identities=identities,
        descriptors=descriptors,
        learner_ids=_learner_ids(),
        maximum_local_steps=1_000_000,
        maximum_processed_tokens=1_000_000_000,
    )
    tracker = ProfileAProgressTracker(_learner_ids(), (0, 0), comparison=_comparison())
    executor = ProfileAFragmentExecutor(
        atomic_store=atomic,
        proposal_store=proposals,
        profile=_profile(config),
        outer_policy=policy,
        progress=tracker,
    )
    oracle_states = [OuterSGDState(None), OuterSGDState(None)]
    traces: list[dict[str, Any]] = []
    for cycle in range(updates):
        raw: dict[int, dict[str, Any]] = {}
        for fragment_index in range(2):
            authority = atomic.load_fragment(fragment_index)
            current = np.frombuffer(authority.parameters, dtype="<f4").astype(np.float64).tolist()
            locals_: list[list[float]] = []
            tokens: list[int] = []
            proposal_rows = []
            for learner_index, learner_id in enumerate(_learner_ids()):
                token_count = (learner_index + 1) * 17 + cycle
                tokens.append(token_count)
                gradient = [
                    float(
                        np.float32(
                            (((cycle + 1) * (learner_index + 2) * (position + 5)) % 31 - 15)
                            * 0.0002
                        )
                    )
                    for position in range(elements)
                ]
                local = [
                    float(np.float32(value - delta))
                    for value, delta in zip(current, gradient, strict=True)
                ]
                local_payload = np.asarray(local, dtype="<f4").tobytes()
                proposal = Proposal.create(
                    proposal_id=(
                        f"numeric-{learner_id}-fragment-{fragment_index}-"
                        f"base-{cycle}-sequence-{cycle + 1}"
                    ),
                    identities=identities,
                    learner_id=learner_id,
                    descriptor=authority.state.descriptor,
                    sequence=cycle + 1,
                    base_version=cycle,
                    base_content_identity=authority.content_identity,
                    local_steps=1,
                    processed_tokens=token_count,
                    snapshot_local_step=cycle + 1,
                    parameters=local_payload,
                )
                proposals.publish(proposal)
                locals_.append(local)
                proposal_rows.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "content_identity": proposal.content_identity,
                        "learner_id": learner_id,
                        "base_version": proposal.base_version,
                        "base_content_identity": proposal.base_content_identity,
                        "sequence": proposal.sequence,
                        "processed_tokens": token_count,
                        "parameters_sha256": proposal.parameters_sha256,
                        "parameters": local,
                    }
                )
            raw[fragment_index] = {
                "fragment_index": fragment_index,
                "cycle": cycle,
                "current_version": authority.version,
                "current_content_identity": authority.content_identity,
                "current_parameters": current,
                "proposals": proposal_rows,
                "tokens": tokens,
                "locals": locals_,
            }
        for expected_fragment in range(2):
            update = executor.execute_next(observed_ns=cycle * 2 + expected_fragment + 1)
            if update is None:
                raise ProfileAStressError("numeric Profile A executor was not ready")
            index = update.successor.state.descriptor.index
            if index != expected_fragment:
                raise ProfileAStressError("round-robin fragment schedule changed")
            item = raw[index]
            selected_ids = [
                proposal.proposal_id for proposal in update.selection.proposals
            ]
            raw_by_id = {
                row["proposal_id"]: row for row in item["proposals"]
            }
            ordered_rows = [raw_by_id[proposal_id] for proposal_id in selected_ids]
            ordered_tokens = [int(row["processed_tokens"]) for row in ordered_rows]
            ordered_locals = [row["parameters"] for row in ordered_rows]
            weights = inverse_staleness_weights(ordered_tokens, [0] * 4, 1.0)
            merged = weighted_direct_merge(
                current=item["current_parameters"],
                bases=[item["current_parameters"]] * 4,
                locals_=ordered_locals,
                weights=weights,
            )
            expected, oracle_state = outer_sgd_step(
                item["current_parameters"],
                merged,
                oracle_states[index],
                learning_rate=policy.f32_learning_rate,
                momentum=policy.f32_momentum,
                nesterov=policy.nesterov,
            )
            oracle_states[index] = oracle_state
            production = np.frombuffer(
                update.successor.parameters, dtype="<f4"
            ).astype(np.float64).tolist()
            production_momentum = np.frombuffer(
                update.successor.outer_optimizer_state, dtype="<f4"
            ).astype(np.float64).tolist()
            trace = {
                **item,
                "selected_proposal_ids": selected_ids,
                "selected_weights": [
                    weight.normalized_weight for weight in update.selection.weights
                ],
                "selection_identity": update.selection.selection_identity,
                "production_parameters": production,
                "production_parameters_sha256": hashlib.sha256(
                    update.successor.parameters
                ).hexdigest(),
                "production_momentum": production_momentum,
                "production_momentum_sha256": hashlib.sha256(
                    update.successor.outer_optimizer_state
                ).hexdigest(),
                "successor_version": update.successor.version,
                "successor_content_identity": update.successor.content_identity,
                "relative_l2": _relative_l2(production, expected),
                "momentum_relative_l2": _relative_l2(
                    production_momentum, oracle_state.momentum_buffer
                ),
                "byte_accounting": update.result.byte_accounting.to_dict(),
                "maximum_active_payloads": update.source_metrics.maximum_active_payloads,
            }
            traces.append(trace)
    report = tracker.report()
    stop = FrozenStopPolicy(
        target_global_cycles=int(config["progress"]["target_global_cycles"]),
        maximum_walltime_seconds=float(
            config["progress"]["maximum_walltime_seconds"]
        ),
        maximum_compute_steps=int(config["progress"]["maximum_compute_steps"]),
        maximum_accepted_tokens=int(
            config["progress"]["maximum_accepted_tokens"]
        ),
        maximum_flops=config["progress"]["maximum_flops"],
    ).evaluate(report, walltime_seconds=1.0)
    result = {
        "schema_version": 1,
        "status": "pass",
        "updates_per_fragment": updates,
        "traces": traces,
        "progress": report.to_dict(),
        "stop": stop.to_dict(),
        "final_version_vector": list(atomic.load_snapshot().version_vector),
    }
    _write_json_new(root / "numeric-trace.json", result)
    return result


def _run_real_role(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_identity: str,
    config: Mapping[str, Any],
    rank: int,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise ProfileAStressError("M=4 real FS role requires one CUDA GPU")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise ProfileAStressError("torch.distributed must remain uninitialized")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", 0)
    coordination = root / "coordination"
    timeout = float(config["coordination_timeout_seconds"])
    speed = config["real_fs"]["speed_injection"]
    interval_ns = int(float(speed["common_interval_seconds"]) * 1e9)
    if rank == 0:
        start = time.time_ns() + 3_000_000_000
        _replace_json(
            coordination / "speed-control.json",
            {"complete": True, "start_unix_ns": start, "end_unix_ns": start + interval_ns},
        )
    control_phase = _wait_json(coordination / "speed-control.json", timeout)
    control = _speed_trial(
        config=config, rank=rank, device=device, phase=control_phase, injected=False
    )
    _replace_json(
        coordination / f"speed-control-done-{rank:02d}.json",
        {"complete": True, "rank": rank},
    )
    if rank == 0:
        _wait_all(coordination, "speed-control-done", 4, timeout)
        start = time.time_ns() + 3_000_000_000
        _replace_json(
            coordination / "speed-injected.json",
            {"complete": True, "start_unix_ns": start, "end_unix_ns": start + interval_ns},
        )
    injected_phase = _wait_json(coordination / "speed-injected.json", timeout)
    injected = _speed_trial(
        config=config, rank=rank, device=device, phase=injected_phase, injected=True
    )
    _replace_json(
        coordination / f"speed-injected-done-{rank:02d}.json",
        {"complete": True, "rank": rank},
    )
    if rank == 0:
        _wait_all(coordination, "speed-injected-done", 4, timeout)

    training = config["real_fs"]["training"]
    model, registry, fragment_map, groups = _model(config, device)
    descriptors = build_fragment_descriptors(fragment_map)
    identities = GlobalStateIdentities(
        run_identity=run_id,
        config_identity=config_identity,
        model_identity=registry.digest,
        fragment_map_identity=fragment_map.digest,
    )
    policy = _outer_policy(config)
    provisional = {
        "identities": identities.to_dict(),
        "descriptors": [item.to_dict() for item in descriptors],
    }
    atomic, proposals = _stores(root / "real", catalog=provisional, policy=policy)
    if rank == 0:
        boot = atomic.bootstrap(
            tuple(
                BootstrapFragment(
                    descriptor=descriptor,
                    parameters=serialize_fragment_parameters(
                        group, int(descriptor.shape[0]) * 4
                    ),
                    outer_state=b"",
                )
                for descriptor, group in zip(descriptors, groups, strict=True)
            )
        )
        catalog = _state_catalog(
            run_id=run_id,
            config_identity=config_identity,
            registry=registry,
            fragment_map=fragment_map,
            descriptors=descriptors,
            authorities=boot.snapshot.authorities,
        )
        _replace_json(coordination / "catalog.json", catalog)
    catalog = _wait_json(coordination / "catalog.json", timeout)
    if not _catalog_matches_model(catalog, identities, descriptors):
        raise ProfileAStressError("real learner model or fragment identities differ")
    atomic, proposals = _stores(root / "real", catalog=catalog, policy=policy)
    bootstrap_authorities = atomic.load_snapshot().authorities
    if tuple(item.version for item in bootstrap_authorities) != (0, 0):
        raise ProfileAStressError("learners did not begin from bootstrap vector")
    with torch.no_grad():
        for authority, group in zip(bootstrap_authorities, groups, strict=True):
            values = np.frombuffer(authority.parameters, dtype="<f4")
            offset = 0
            for parameter in group:
                count = int(parameter.numel())
                source = torch.from_numpy(values[offset : offset + count].copy()).reshape(
                    parameter.shape
                )
                parameter.copy_(source.to(device=device, dtype=parameter.dtype))
                offset += count

    learner_id = _learner_ids()[rank]
    progress = LearnerProgress.initialize(learner_id, (0, 0))
    scheduler = ConstantStepScheduler()
    rng = LearnerRng.initialize(learner_id, 9200 + rank, device)
    runtime, optimizer = _runtime(
        model=model,
        groups=groups,
        progress=progress,
        scheduler=scheduler,
        rng=rng,
        device=device,
        learning_rate=float(training["inner_learning_rate"]),
    )
    fragment_bytes = tuple(int(item.shape[0]) * 4 for item in descriptors)
    publisher = FragmentSnapshotCoordinator(
        identities=identities,
        progress=progress,
        descriptors=descriptors,
        fragment_parameters=groups,
        adopted_bases=tuple(
            AdoptedFragmentBase(item.version, item.content_identity)
            for item in bootstrap_authorities
        ),
        schedule=FragmentPublicationSchedule(
            fragment_bytes=fragment_bytes,
            intervals=(2, 2),
            offsets=(0, 1),
            offset_algorithm="s1_12_two_step_round_robin",
        ),
        store=proposals,
    )
    runtime.safe_boundary_observers = (publisher.on_safe_boundary,)
    proposal_run = runtime.run(
        _batches(
            count=int(training["proposal_local_steps"]),
            rank=rank,
            batch_size=int(training["batch_size"]),
            sequence_length=int(training["sequence_length"]),
            vocab_size=int(config["real_fs"]["tiny_hf_model"]["vocab_size"]),
        ),
        optimizer_steps=int(training["proposal_local_steps"]),
    )
    publisher.close(timeout)
    proposal_summary = publisher.summary()
    _replace_json(
        coordination / f"proposals-ready-{rank:02d}.json",
        {
            "complete": True,
            "rank": rank,
            "hostname": socket.gethostname().split(".")[0],
            "published": proposal_summary["publication"][
                "published_snapshot_count"
            ],
        },
    )
    real_tracker = None
    executor = None
    real_updates: list[dict[str, Any]] = []
    real_cycle_progress: dict[str, Any] | None = None
    if rank == 0:
        _wait_all(coordination, "proposals-ready", 4, timeout)
        real_tracker = ProfileAProgressTracker(
            _learner_ids(), (0, 0), comparison=_comparison()
        )
        executor = ProfileAFragmentExecutor(
            atomic_store=atomic,
            proposal_store=proposals,
            profile=_profile(config),
            outer_policy=policy,
            progress=real_tracker,
        )
        first = executor.execute_next()
        if first is None or first.successor.state.descriptor.index != 0:
            raise ProfileAStressError("real syncer did not commit fragment zero first")
        real_updates.append(_update_summary(first))
        _replace_json(
            coordination / "fragment-zero-committed.json",
            {"complete": True, "version_vector": list(atomic.load_snapshot().version_vector)},
        )
    _wait_json(coordination / "fragment-zero-committed.json", timeout)

    adoption_traces: list[dict[str, Any]] = []
    adoption = FragmentAdoptionCoordinator(
        store=atomic.store,
        progress=progress,
        initial_states=tuple(item.state for item in bootstrap_authorities),
        fragment_parameters=groups,
        optimizer=optimizer,
        identity_audit=True,
        trace_sink=lambda trace: adoption_traces.append(dict(trace)),
        autostart=False,
    )
    adoption.poll_once()
    mixed_steps = int(training["mixed_version_steps"])
    mixed_runtime, _ = _runtime(
        model=model,
        groups=groups,
        progress=progress,
        scheduler=scheduler,
        rng=rng,
        device=device,
        learning_rate=float(training["inner_learning_rate"]),
        optimizer=optimizer,
        observers=(adoption.on_safe_boundary,),
    )
    mixed_run = mixed_runtime.run(
        _batches(
            count=mixed_steps + 1,
            rank=rank,
            batch_size=int(training["batch_size"]),
            sequence_length=int(training["sequence_length"]),
            vocab_size=int(config["real_fs"]["tiny_hf_model"]["vocab_size"]),
        ),
        optimizer_steps=mixed_steps + 1,
    )
    _replace_json(
        coordination / f"mixed-complete-{rank:02d}.json",
        {"complete": True, "rank": rank},
    )
    if rank == 0:
        _wait_all(coordination, "mixed-complete", 4, timeout)
        if executor is None or real_tracker is None:
            raise AssertionError("rank zero lost its syncer")
        second = executor.execute_next()
        if second is None or second.successor.state.descriptor.index != 1:
            raise ProfileAStressError("real syncer did not commit fragment one second")
        real_updates.append(_update_summary(second))
        real_cycle_progress = real_tracker.report().to_dict()
        _replace_json(
            coordination / "fragment-one-committed.json",
            {"complete": True, "version_vector": list(atomic.load_snapshot().version_vector)},
        )
    _wait_json(coordination / "fragment-one-committed.json", timeout)
    adoption.poll_once()
    final_runtime, _ = _runtime(
        model=model,
        groups=groups,
        progress=progress,
        scheduler=scheduler,
        rng=rng,
        device=device,
        learning_rate=float(training["inner_learning_rate"]),
        optimizer=optimizer,
        observers=(adoption.on_safe_boundary,),
    )
    final_run = final_runtime.run(
        _batches(
            count=1,
            rank=rank,
            batch_size=int(training["batch_size"]),
            sequence_length=int(training["sequence_length"]),
            vocab_size=int(config["real_fs"]["tiny_hf_model"]["vocab_size"]),
        ),
        optimizer_steps=1,
    )
    adoption.close(timeout)
    role = {
        "schema_version": 1,
        "status": "pass",
        "role": "learner_and_syncer" if rank == 0 else "learner",
        "rank": rank,
        "size": 4,
        "learner_id": learner_id,
        "hostname": socket.gethostname().split(".")[0],
        "pid": os.getpid(),
        "run_id": run_id,
        "config_identity": config_identity,
        "torch": {
            "version": torch.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "distributed_initialized": torch.distributed.is_initialized(),
        },
        "model": {
            "implementation": type(model).__name__,
            "registry_digest": registry.digest,
            "fragment_map_digest": fragment_map.digest,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "fragment_parameter_counts": [
                sum(parameter.numel() for parameter in group) for group in groups
            ],
        },
        "speed": {"control": control, "injected": injected},
        "proposal_training": proposal_run.to_dict(),
        "proposal_publication": proposal_summary,
        "mixed_training": mixed_run.to_dict(),
        "final_training": final_run.to_dict(),
        "adoption": adoption.summary(),
        "adoption_traces": adoption_traces,
        "final_version_vector": [
            item.global_version for item in progress.fragments
        ],
        "waited_for_peer_local_step": False,
        "waited_for_version_alignment_before_mixed_training": False,
        "application_coordination": "shared_filesystem_only",
        "mpi_usage": "launcher_only",
    }
    _write_json_new(result_root / f"learner-{rank:02d}.json", role)
    _replace_json(
        coordination / f"learner-final-{rank:02d}.json",
        {"complete": True, "rank": rank},
    )

    if rank == 0:
        _wait_all(coordination, "learner-final", 4, timeout)
        if executor is None or real_tracker is None or real_cycle_progress is None:
            raise AssertionError("rank zero lost Profile A executor")
        completed_roles = [
            json.loads((result_root / f"learner-{index:02d}.json").read_text())
            for index in range(4)
        ]
        real_cycle_progress["learners"] = [
            {
                "learner_id": item["learner_id"],
                "local_optimizer_steps": item["final_training"]["progress"][
                    "local_optimizer_steps"
                ],
                "processed_input_tokens": item["final_training"]["progress"][
                    "processed_input_tokens"
                ],
                "loss_bearing_target_tokens": item["final_training"]["progress"][
                    "loss_bearing_target_tokens"
                ],
            }
            for item in completed_roles
        ]
        plan = FrozenEvaluationPlan.capture(atomic)
        evaluation_started = threading.Event()
        writer_interval: dict[str, int] = {}

        def advance_current() -> None:
            writer_interval["start_unix_ns"] = time.time_ns()
            evaluation_started.set()
            for cycle in range(
                int(config["evaluation"]["concurrent_successor_updates"]) // 2
            ):
                _publish_synthetic_round(
                    atomic, proposals, 0, tag=f"evaluation-{cycle}"
                )
                _publish_synthetic_round(
                    atomic, proposals, 1, tag=f"evaluation-{cycle}"
                )
                first = executor.execute_next()
                second = executor.execute_next()
                if first is None or second is None:
                    raise ProfileAStressError("evaluation writer lost Profile A readiness")
            writer_interval["end_unix_ns"] = time.time_ns()

        writer = threading.Thread(target=advance_current, name="evaluation-successor-writer")
        writer.start()
        evaluation_started.wait(timeout)
        materialize_start = time.time_ns()
        manifest = materialize_evaluation_snapshot(
            plan, result_root / "evaluation"
        )
        materialize_end = time.time_ns()
        writer.join(timeout)
        if writer.is_alive() or "end_unix_ns" not in writer_interval:
            raise ProfileAStressError("evaluation successor writer failed to finish")
        loaded = load_evaluation_snapshot(result_root / "evaluation")
        evaluation_result = {
            "captured_version_vector": list(plan.version_vector),
            "captured_content_identities": list(plan.content_identities),
            "manifest_identity": manifest.snapshot_identity,
            "loaded_version_vector": list(loaded.manifest.version_vector),
            "loaded_content_identities": list(loaded.manifest.content_identities),
            "current_version_vector_after_writer": list(
                atomic.load_snapshot().version_vector
            ),
            "state_payload_bytes": loaded.state_payload_bytes,
            "parameter_payload_bytes": loaded.parameter_payload_bytes,
            "materialize_start_unix_ns": materialize_start,
            "materialize_end_unix_ns": materialize_end,
            **writer_interval,
            "latest_reads_during_restart_load": 0,
            "purpose": "evaluation",
            "steady_state": False,
        }
        numeric_result = run_numeric_e2e(
            root / "numeric",
            run_id=run_id,
            config_identity=config_identity,
            config=config,
        )
        syncer = {
            "schema_version": 1,
            "status": "pass",
            "role": "profile_a_syncer",
            "rank": 0,
            "hostname": socket.gethostname().split(".")[0],
            "run_id": run_id,
            "config_identity": config_identity,
            "real_updates": real_updates,
            "real_progress_after_cycle": real_cycle_progress,
            "evaluation": evaluation_result,
            "numeric_path": "runtime/shared-state/numeric/numeric-trace.json",
            "numeric_final_version_vector": numeric_result["final_version_vector"],
            "application_coordination": "shared_filesystem_only",
            "mpi_usage": "launcher_only",
        }
        _write_json_new(result_root / "syncer.json", syncer)
    return role


def run_role(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_path: Path,
    config_identity: str,
) -> dict[str, Any]:
    rank = _rank()
    size = _size()
    if size != 4 or rank not in range(4):
        raise ProfileAStressError("Profile A formal run requires exactly four ranks")
    config = _load_config(config_path, config_identity)
    return _run_real_role(
        root=root,
        result_root=result_root,
        run_id=run_id,
        config_identity=config_identity,
        config=config,
        rank=rank,
    )


def _analyze_numeric(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    numeric = config["numeric_oracle"]
    updates = int(numeric["updates_per_fragment"])
    traces = value.get("traces")
    if not isinstance(traces, list) or len(traces) != updates * 2:
        raise ProfileAStressError("numeric trace does not contain two times fifty updates")
    policy = _outer_policy(config)
    states = [OuterSGDState(None), OuterSGDState(None)]
    previous_successors: list[list[float] | None] = [None, None]
    errors: list[list[float]] = [[], []]
    momentum_errors: list[list[float]] = [[], []]
    for position, trace in enumerate(traces):
        if not isinstance(trace, dict):
            raise ProfileAStressError("numeric trace row is malformed")
        index = int(trace["fragment_index"])
        cycle = int(trace["cycle"])
        if index != position % 2 or cycle != position // 2:
            raise ProfileAStressError("numeric round-robin trace order differs")
        current = trace["current_parameters"]
        if previous_successors[index] is not None and _relative_l2(
            current, previous_successors[index]
        ) > 1e-12:
            raise ProfileAStressError("numeric production authority chain is discontinuous")
        proposals = trace["proposals"]
        if (
            not isinstance(proposals, list)
            or len(proposals) != 4
            or len({item["learner_id"] for item in proposals}) != 4
            or any(
                item["base_version"] != cycle
                or item["base_content_identity"] != trace["current_content_identity"]
                or hashlib.sha256(
                    np.asarray(item["parameters"], dtype="<f4").tobytes()
                ).hexdigest()
                != item["parameters_sha256"]
                for item in proposals
            )
        ):
            raise ProfileAStressError("raw numeric proposal facts differ")
        by_id = {item["proposal_id"]: item for item in proposals}
        selected_ids = trace["selected_proposal_ids"]
        if set(selected_ids) != set(by_id) or len(selected_ids) != 4:
            raise ProfileAStressError("numeric selected proposal set differs")
        ordered = [by_id[proposal_id] for proposal_id in selected_ids]
        tokens = [int(item["processed_tokens"]) for item in ordered]
        locals_ = [item["parameters"] for item in ordered]
        weights = inverse_staleness_weights(tokens, [0] * 4, 1.0)
        if _relative_l2(trace["selected_weights"], weights) > 1e-6:
            raise ProfileAStressError("production selected weights differ from oracle")
        merged = weighted_direct_merge(
            current=current,
            bases=[current] * 4,
            locals_=locals_,
            weights=weights,
        )
        expected, state = outer_sgd_step(
            current,
            merged,
            states[index],
            learning_rate=policy.f32_learning_rate,
            momentum=policy.f32_momentum,
            nesterov=policy.nesterov,
        )
        states[index] = state
        error = _relative_l2(trace["production_parameters"], expected)
        momentum_error = _relative_l2(
            trace["production_momentum"], state.momentum_buffer
        )
        if (
            not math.isclose(error, trace["relative_l2"], rel_tol=0, abs_tol=1e-15)
            or not math.isclose(
                momentum_error,
                trace["momentum_relative_l2"],
                rel_tol=0,
                abs_tol=1e-15,
            )
            or trace["successor_version"] != cycle + 1
            or trace["byte_accounting"]["full_model_operations"] != 0
            or trace["maximum_active_payloads"] != 1
        ):
            raise ProfileAStressError("numeric production trace differs from recomputation")
        errors[index].append(error)
        momentum_errors[index].append(momentum_error)
        previous_successors[index] = trace["production_parameters"]
    single_max = max(item[0] for item in errors)
    fifty_max = max(max(item) for item in errors)
    first_max = max(max(item[:10]) for item in errors)
    last_max = max(max(item[-10:]) for item in errors)
    if (
        single_max > float(numeric["single_update_relative_l2_max"])
        or fifty_max > float(numeric["fifty_update_relative_l2_max"])
        or last_max
        > max(first_max, float(numeric["late_window_amplification_floor"]))
        or value["final_version_vector"] != [updates, updates]
        or value["progress"]["global_cycle"] != updates
        or value["stop"]["stop"] is not True
        or value["stop"]["reason"] != "target_global_cycles"
    ):
        raise ProfileAStressError("numeric gate progress or stop failed")
    return {
        "updates_per_fragment": updates,
        "single_update_maximum_relative_l2": single_max,
        "fifty_update_maximum_relative_l2": fifty_max,
        "maximum_momentum_relative_l2": max(max(item) for item in momentum_errors),
        "first_ten_maximum_relative_l2": first_max,
        "last_ten_maximum_relative_l2": last_max,
        "final_version_vector": [updates, updates],
        "global_cycle": updates,
        "stop_reason": "target_global_cycles",
        "raw_checker_update": {
            "fragment_index": traces[0]["fragment_index"],
            "cycle": traces[0]["cycle"],
            "current_content_identity": traces[0]["current_content_identity"],
            "proposal_ids": [item["proposal_id"] for item in traces[0]["proposals"]],
            "successor_content_identity": traces[0]["successor_content_identity"],
            "recomputed_relative_l2": errors[0][0],
        },
    }


def analyze(
    *,
    result_root: Path,
    shared_root: Path,
    config: Mapping[str, Any],
    config_identity: str,
) -> dict[str, Any]:
    learners = [
        json.loads((result_root / f"learner-{rank:02d}.json").read_text())
        for rank in range(4)
    ]
    syncer = json.loads((result_root / "syncer.json").read_text())
    hosts = [item["hostname"] for item in learners]
    if len(set(hosts)) != 4:
        raise ProfileAStressError("M=4 learners did not use four distinct hosts")
    for rank, learner in enumerate(learners):
        if (
            learner.get("status") != "pass"
            or learner.get("rank") != rank
            or learner.get("size") != 4
            or learner.get("learner_id") != _learner_ids()[rank]
            or learner.get("config_identity") != config_identity
            or learner["torch"]["distributed_initialized"] is not False
            or learner["application_coordination"] != "shared_filesystem_only"
            or learner["mpi_usage"] != "launcher_only"
            or learner["waited_for_peer_local_step"] is not False
            or learner["waited_for_version_alignment_before_mixed_training"] is not False
            or learner["final_version_vector"] != [1, 1]
        ):
            raise ProfileAStressError("real learner identity or topology contract failed")
    speed = config["real_fs"]["speed_injection"]
    for phase_name in ("control", "injected"):
        intervals = {
            (
                item["speed"][phase_name]["start_unix_ns"],
                item["speed"][phase_name]["end_unix_ns"],
                item["speed"][phase_name]["common_interval_seconds"],
            )
            for item in learners
        }
        if len(intervals) != 1 or next(iter(intervals))[2] != float(
            speed["common_interval_seconds"]
        ):
            raise ProfileAStressError("speed trial common interval differs across ranks")
    control_rates = [
        float(item["speed"]["control"]["common_interval_input_tokens_per_second"])
        for item in learners
    ]
    injected_rates = [
        float(item["speed"]["injected"]["common_interval_input_tokens_per_second"])
        for item in learners
    ]
    changes = [
        abs(injected_rates[index] / control_rates[index] - 1.0)
        for index in range(3)
    ]
    slow_index = int(speed["slow_learner_index"])
    slow_ratio = injected_rates[slow_index] / control_rates[slow_index]
    if (
        any(value > float(speed["unaffected_throughput_change_max"]) for value in changes)
        or abs(slow_ratio - float(speed["slowdown_target"]))
        > float(speed["slowdown_absolute_tolerance"])
        or any(
            item["speed"][phase]["finite_loss"] is not True
            or item["speed"][phase]["distributed_initialized"] is not False
            for item in learners
            for phase in ("control", "injected")
        )
    ):
        raise ProfileAStressError("speed heterogeneity timing gate failed")
    minimum_mixed = int(config["real_fs"]["training"]["mixed_version_steps"])
    mixed_counts = []
    for learner in learners:
        events = learner["mixed_training"]["events"]
        mixed = [
            item
            for item in events
            if item["fragment_global_versions"] == [1, 0]
        ]
        if (
            len(mixed) < minimum_mixed
            or any(not math.isfinite(float(item["token_weighted_loss"])) for item in mixed)
            or learner["adoption"]["adoption_count"] != 2
            or len(learner["adoption_traces"]) != 2
        ):
            raise ProfileAStressError("sustained mixed-version learner gate failed")
        mixed_counts.append(len(mixed))
    if (
        syncer.get("status") != "pass"
        or syncer.get("config_identity") != config_identity
        or [item["fragment_index"] for item in syncer["real_updates"]] != [0, 1]
        or syncer["real_progress_after_cycle"]["global_cycle"] != 1
        or [
            item["learner_id"]
            for item in syncer["real_progress_after_cycle"]["learners"]
        ]
        != list(_learner_ids())
        or any(
            int(item["local_optimizer_steps"]) <= 0
            or int(item["processed_input_tokens"]) <= 0
            or int(item["loss_bearing_target_tokens"]) <= 0
            for item in syncer["real_progress_after_cycle"]["learners"]
        )
        or any(
            item["byte_accounting"]["full_model_operations"] != 0
            or item["source_metrics"]["maximum_active_payloads"] != 1
            for item in syncer["real_updates"]
        )
    ):
        raise ProfileAStressError("real Profile A syncer closure failed")
    evaluation = syncer["evaluation"]
    loaded = load_evaluation_snapshot(result_root / "evaluation")
    if (
        evaluation["captured_version_vector"] != [1, 1]
        or evaluation["loaded_version_vector"] != [1, 1]
        or evaluation["captured_content_identities"]
        != evaluation["loaded_content_identities"]
        or evaluation["current_version_vector_after_writer"] == [1, 1]
        or evaluation["latest_reads_during_restart_load"] != 0
        or evaluation["purpose"] != "evaluation"
        or evaluation["steady_state"] is not False
        or loaded.manifest.snapshot_identity != evaluation["manifest_identity"]
        or int(evaluation["start_unix_ns"])
        > int(evaluation["materialize_end_unix_ns"])
        or int(evaluation["end_unix_ns"])
        < int(evaluation["materialize_start_unix_ns"])
    ):
        raise ProfileAStressError("frozen evaluation snapshot gate failed")
    numeric_value = json.loads(
        (shared_root / "numeric" / "numeric-trace.json").read_text()
    )
    numeric = _analyze_numeric(numeric_value, config)
    return {
        "schema_version": 1,
        "status": "pass",
        "profile": "Profile A",
        "learner_count": 4,
        "hosts": hosts,
        "application_coordination": "shared_filesystem_only",
        "torch_distributed_initialized": False,
        "speed_heterogeneity": {
            "control_rates": control_rates,
            "injected_rates": injected_rates,
            "unaffected_absolute_relative_changes": changes,
            "maximum_unaffected_change": max(changes),
            "slow_learner_index": slow_index,
            "slow_ratio": slow_ratio,
        },
        "mixed_version_training": {
            "version_vector": [1, 0],
            "finite_consecutive_steps_per_learner": mixed_counts,
            "minimum_required": minimum_mixed,
        },
        "real_fs_cycle": {
            "updates": syncer["real_updates"],
            "global_cycle": 1,
            "final_learner_version_vectors": [
                item["final_version_vector"] for item in learners
            ],
        },
        "numeric": numeric,
        "evaluation": {
            **evaluation,
            "manifest_path": str(result_root / "evaluation" / "manifest.json"),
        },
        "comparison": config["comparison"],
    }


def summarize(
    *,
    result_root: Path,
    shared_root: Path,
    config_path: Path,
    config_identity: str,
    output: Path,
) -> dict[str, Any]:
    config = _load_config(config_path, config_identity)
    summary = analyze(
        result_root=result_root,
        shared_root=shared_root,
        config=config,
        config_identity=config_identity,
    )
    _write_json_new(output, summary)
    return summary


def manifest_command(args: argparse.Namespace) -> dict[str, Any]:
    learners = [
        json.loads((args.result_root / f"learner-{rank:02d}.json").read_text())
        for rank in range(4)
    ]
    declared = {
        "learner-00+logical-syncer": [learners[0]["hostname"]],
        "learner-01": [learners[1]["hostname"]],
        "learner-02": [learners[2]["hostname"]],
        "learner-03-slow-injection": [learners[3]["hostname"]],
    }
    modules = [
        line.strip()
        for line in args.modules_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identities = {
        "code": {
            "repository": args.repository,
            "branch": args.branch,
            "commit": args.commit,
            "dirty": False,
        },
        "config": {"sha256": args.config_sha256},
        "source": {
            "research_plan_sha256": args.research_sha256,
            "stage0_4_spec_sha256": args.spec_sha256,
        },
        "skill": {
            "repository": args.skill_repository,
            "commit": args.skill_commit,
        },
        "execution": {
            "identity": args.run_id,
            "initial_hostname": args.initial_hostname,
            "compute_hostname": learners[0]["hostname"],
            "workflow": "four-node-four-gpu-profile-a-filesystem-only-e2e",
        },
        "roles": {"declared": declared, "actual": declared},
        "paths": {
            "project_root": str(args.project_root.resolve()),
            "evidence_root": str(args.evidence_root.resolve()),
        },
    }
    scheduler = {
        "job_id": args.job_id,
        "qtime_utc": args.qtime_utc,
        "queue": args.queue,
        "group": args.group,
        "nodefile_sha256": file_digest(args.nodefile),
        "modules": modules,
    }
    manifest = build_manifest(
        "S1-12",
        "L2",
        args.qtime_utc,
        "pbs_qtime",
        identities,
        scheduler=scheduler,
    )
    _write_json_new(args.output, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S1-12 Profile A formal harness")
    sub = parser.add_subparsers(dest="command", required=True)
    role = sub.add_parser("role")
    role.add_argument("--root", type=Path, required=True)
    role.add_argument("--result-root", type=Path, required=True)
    role.add_argument("--run-id", required=True)
    role.add_argument("--config-path", type=Path, required=True)
    role.add_argument("--config-identity", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--result-root", type=Path, required=True)
    summary.add_argument("--shared-root", type=Path, required=True)
    summary.add_argument("--config-path", type=Path, required=True)
    summary.add_argument("--config-identity", required=True)
    summary.add_argument("--output", type=Path, required=True)
    manifest = sub.add_parser("manifest")
    for name in (
        "repository",
        "branch",
        "commit",
        "run-id",
        "config-sha256",
        "research-sha256",
        "spec-sha256",
        "skill-repository",
        "skill-commit",
        "initial-hostname",
        "job-id",
        "qtime-utc",
        "queue",
        "group",
    ):
        manifest.add_argument(f"--{name}", required=True)
    for name in (
        "project-root",
        "evidence-root",
        "nodefile",
        "modules-file",
        "result-root",
        "output",
    ):
        manifest.add_argument(f"--{name}", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "role":
        result = run_role(
            root=args.root,
            result_root=args.result_root,
            run_id=args.run_id,
            config_path=args.config_path,
            config_identity=args.config_identity,
        )
    elif args.command == "summarize":
        result = summarize(
            result_root=args.result_root,
            shared_root=args.shared_root,
            config_path=args.config_path,
            config_identity=args.config_identity,
            output=args.output,
        )
    else:
        result = manifest_command(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
