from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import FrozenEvaluationPlan, load_evaluation_snapshot, materialize_evaluation_snapshot
from .global_commit import AtomicGlobalCommitStore
from .global_state import BootstrapFragment, FragmentStateDescriptor, GlobalStateIdentities, GlobalStateStore
from .identity import canonical_digest
from .learner import ConstantStepScheduler, LearnerProgress, LearnerRng, LearnerRuntime, PackedTokenShard
from .learner_adopt import FragmentAdoptionCoordinator
from .learner_assets import load_learner_profile, validate_materialized_shards, verify_profile_assets
from .learner_publish import (
    AdoptedFragmentBase,
    FragmentPublicationSchedule,
    FragmentSnapshotCoordinator,
    build_fragment_parameter_groups,
)
from .learner_smoke import _load_real_model, _optimizer
from .logging import StructuredLogger
from .model_registry import build_logical_layer_registry
from .fragment_map import build_fragment_map
from .profile_a import (
    ComparisonBasis,
    ProfileAConfig,
    ProfileAFragmentExecutor,
    ProfileAProgressTracker,
    profile_a_policy_identity,
)
from .proposal import ProposalStore
from .storage import PosixStorageBackend
from .syncer_merge import OuterSGDPolicy


class Stage1CloseError(RuntimeError):
    pass


_CLAIM = "stage1_runtime_and_training_occurrence_only"
_PROFILE_CLAIM = "protocol_and_runtime_acceptance_only_no_quality_or_efficiency_comparison"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage1CloseError(f"JSON object required: {path}")
    return value


def _wait_json(path: Path, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            value = _read_json(path)
            if value.get("complete") is True:
                return value
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for coordination record: {path}")
        time.sleep(0.05)


def _wait_roles(root: Path, prefix: str, count: int, timeout_seconds: float) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        paths = [root / f"{prefix}-{index:02d}.json" for index in range(count)]
        try:
            if all(path.exists() for path in paths):
                return [_read_json(path) for path in paths]
        except json.JSONDecodeError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {count} {prefix} records")
        time.sleep(0.1)


def _descriptor(value: Mapping[str, Any]) -> FragmentStateDescriptor:
    return FragmentStateDescriptor(
        index=int(value["index"]),
        identity=str(value["identity"]),
        dtype=str(value["dtype"]),
        shape=tuple(int(item) for item in value["shape"]),
        parameter_identities=tuple(str(item) for item in value["parameter_identities"]),
    )


def _learner_ids(count: int) -> tuple[str, ...]:
    return tuple(f"learner-{index:02d}" for index in range(count))


def _outer_policy(config: Mapping[str, Any]) -> OuterSGDPolicy:
    value = config["protocol"]["outer_optimizer"]
    return OuterSGDPolicy(
        learning_rate=float(value["learning_rate"]),
        momentum=float(value["momentum"]),
        nesterov=bool(value["nesterov"]),
    )


def _load_contract(
    config_path: Path,
    *,
    expected_config_sha256: str,
    expected_asset_marker_sha256: str,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    if _hash_file(config_path) != expected_config_sha256:
        raise Stage1CloseError("resolved config identity mismatch")
    config = _read_json(config_path)
    selected = str(config.get("selected_workload"))
    if selected not in {"nine_node", "long_run"}:
        raise Stage1CloseError("resolved config has no valid selected workload")
    resolved = config.get("resolved_runtime_fields", {})
    if resolved.get("asset_marker_sha256") != expected_asset_marker_sha256:
        raise Stage1CloseError("resolved config asset marker identity mismatch")
    asset_root = Path(str(resolved["asset_bundle_root"]))
    marker = _read_json(asset_root / "complete.json")
    if _hash_file(asset_root / "complete.json") != expected_asset_marker_sha256:
        raise Stage1CloseError("asset completion marker checksum mismatch")
    if marker.get("manifest_sha256") != _hash_file(asset_root / "manifest.json"):
        raise Stage1CloseError("asset completion marker does not bind the manifest")
    asset_manifest = _read_json(asset_root / "manifest.json")
    bootstrap_path = asset_root / "workloads" / selected / "manifest.json"
    bootstrap = _read_json(bootstrap_path)
    if (
        asset_manifest.get("status") != "complete"
        or asset_manifest.get("kind") != "s1_13_immutable_asset_bundle"
        or asset_manifest["workloads"][selected]["manifest_sha256"]
        != _hash_file(bootstrap_path)
    ):
        raise Stage1CloseError("asset manifest does not bind the workload bootstrap")
    if bootstrap.get("bootstrap_identity") != resolved.get("workload_bootstrap_identity"):
        raise Stage1CloseError("bootstrap identity differs from resolved config")
    if [row["descriptor"] for row in bootstrap["fragments"]] != resolved["fragment_descriptors"]:
        raise Stage1CloseError("resolved descriptors differ from immutable bootstrap")
    return config, asset_root, asset_manifest, bootstrap


def _load_gate_contract(
    path: Path,
    *,
    expected_sha256: str,
    workload: str,
    non_formal_long_smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _hash_file(path) != expected_sha256:
        raise Stage1CloseError("gate contract identity mismatch")
    contract = _read_json(path)
    if contract.get("loop_id") != "S1-13" or contract.get("schema_version") != 1:
        raise Stage1CloseError("invalid S1-13 gate contract")
    if workload not in {"nine_node", "long_run"}:
        raise Stage1CloseError("gate contract workload is invalid")
    workload_contract = contract.get(workload)
    if not isinstance(workload_contract, dict):
        raise Stage1CloseError("gate contract omits the selected workload")
    if non_formal_long_smoke:
        if workload != "long_run":
            raise Stage1CloseError("the non-formal long smoke requires long_run")
        smoke = workload_contract.get("preflight_smoke")
        if not isinstance(smoke, dict) or smoke.get("formal_evidence") is not False:
            raise Stage1CloseError("gate contract omits the non-formal long smoke")
        return contract, smoke
    return contract, workload_contract


def _stores(
    shared_root: Path,
    *,
    run_id: str,
    config_sha256: str,
    bootstrap: Mapping[str, Any],
    learner_count: int,
    policy: OuterSGDPolicy,
) -> tuple[AtomicGlobalCommitStore, ProposalStore, PosixStorageBackend, PosixStorageBackend]:
    descriptors = tuple(_descriptor(row["descriptor"]) for row in bootstrap["fragments"])
    identities = GlobalStateIdentities(
        run_identity=run_id,
        config_identity=config_sha256,
        model_identity=str(bootstrap["registry_digest"]),
        fragment_map_identity=str(bootstrap["fragment_map_digest"]),
    )
    global_backend = PosixStorageBackend(shared_root / "protocol" / "global")
    proposal_backend = PosixStorageBackend(shared_root / "protocol" / "proposals")
    global_store = GlobalStateStore(
        global_backend,
        identities=identities,
        descriptors=descriptors,
        s_max=0,
    )
    learners = _learner_ids(learner_count)
    atomic = AtomicGlobalCommitStore(
        global_store,
        learner_ids=learners,
        policy_identity=profile_a_policy_identity(policy),
    )
    proposals = ProposalStore(
        proposal_backend,
        identities=identities,
        descriptors=descriptors,
        learner_ids=learners,
        maximum_local_steps=10_000_000,
        maximum_processed_tokens=2_000_000_000,
    )
    return atomic, proposals, global_backend, proposal_backend


def _bootstrap(
    atomic: AtomicGlobalCommitStore,
    asset_root: Path,
    workload: str,
    bootstrap: Mapping[str, Any],
) -> None:
    fragments = []
    for row in bootstrap["fragments"]:
        path = asset_root / "workloads" / workload / row["path"]
        payload = path.read_bytes()
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise Stage1CloseError("bootstrap fragment payload identity mismatch")
        fragments.append(
            BootstrapFragment(
                descriptor=_descriptor(row["descriptor"]),
                parameters=payload,
                outer_state=b"",
            )
        )
    result = atomic.bootstrap(tuple(fragments))
    if result.snapshot.version_vector != tuple(0 for _ in fragments):
        raise Stage1CloseError("bootstrap did not produce the version-zero authority vector")


def _gpu_identity() -> dict[str, Any]:
    import torch

    uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": uuid,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "torch_version": torch.__version__,
    }


def _role_identity(role: str, shared_root: Path) -> dict[str, Any]:
    return {
        "complete": True,
        "role": role,
        "hostname": socket.gethostname().split(".")[0],
        "pid": os.getpid(),
        "pbs_job_id": os.environ.get("PBS_JOBID", "local"),
        "pbs_qtime_utc": os.environ.get("PBS_QTIME_UTC"),
        "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX"),
        "shared_device": os.stat(shared_root).st_dev,
        "shared_root": str(shared_root.resolve()),
        "timestamp_unix_ns": time.time_ns(),
    }


def _apply_fragment_payloads(groups: Sequence[Sequence[Any]], payloads: Sequence[bytes]) -> None:
    import torch

    with torch.no_grad():
        for group, payload in zip(groups, payloads, strict=True):
            array = np.frombuffer(payload, dtype="<f4")
            offset = 0
            for parameter in group:
                count = int(parameter.numel())
                source = torch.from_numpy(array[offset : offset + count].copy()).reshape(parameter.shape)
                parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
                offset += count
            if offset != array.size:
                raise Stage1CloseError("fragment payload does not cover its model group")
    if any(parameter.device.type == "cuda" for group in groups for parameter in group):
        torch.cuda.synchronize()


def _batches(shard: PackedTokenShard, batch_size: int) -> Iterator[dict[str, Any]]:
    while True:
        yield shard.next_batch(batch_size)


def _evaluate_validation(model: Any, validation_root: Path, device: Any) -> dict[str, Any]:
    import torch

    manifest = _read_json(validation_root / "manifest.json")
    rows = sorted(manifest["shards"], key=lambda row: row["learner_index"])
    sequence_length = int(manifest["sequence_length"])
    arrays = [
        np.memmap(
            validation_root / row["path"],
            dtype="<u4",
            mode="r",
            shape=(int(row["blocks"]), sequence_length),
        )
        for row in rows
    ]
    total_blocks = int(manifest["packed_blocks"])
    batch_size = 4
    loss_numerator = 0.0
    target_tokens = 0
    model.eval()
    started = time.monotonic_ns()
    with torch.no_grad():
        for start in range(0, total_blocks, batch_size):
            blocks = []
            for block_index in range(start, min(total_blocks, start + batch_size)):
                shard_index = block_index % len(arrays)
                local_index = block_index // len(arrays)
                blocks.append(np.asarray(arrays[shard_index][local_index], dtype=np.int64))
            input_ids = torch.from_numpy(np.stack(blocks)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(input_ids=input_ids, labels=input_ids, attention_mask=torch.ones_like(input_ids))
            count = int(input_ids.shape[0]) * (sequence_length - 1)
            loss = float(output.loss.detach().float().item())
            if not math.isfinite(loss):
                raise Stage1CloseError("validation loss is nonfinite")
            loss_numerator += loss * count
            target_tokens += count
    torch.cuda.synchronize(device)
    completed = time.monotonic_ns()
    return {
        "finite": True,
        "all_blocks": True,
        "shuffle": False,
        "packed_blocks": total_blocks,
        "processed_input_tokens": total_blocks * sequence_length,
        "loss_bearing_target_tokens": target_tokens,
        "loss_numerator": loss_numerator,
        "token_weighted_loss": loss_numerator / target_tokens,
        "latency_seconds": (completed - started) / 1e9,
    }


def run_learner(
    *,
    shared_root: Path,
    result_root: Path,
    run_id: str,
    config_path: Path,
    config_sha256: str,
    asset_marker_sha256: str,
    gate_contract_path: Path,
    gate_contract_sha256: str,
    non_formal_long_smoke: bool,
    hub_cache: Path,
    learner_index: int,
    learner_count_override: int | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    import torch

    config, asset_root, _asset_manifest, bootstrap = _load_contract(
        config_path,
        expected_config_sha256=config_sha256,
        expected_asset_marker_sha256=asset_marker_sha256,
    )
    workload = str(config["selected_workload"])
    workload_config = config["workloads"][workload]
    _gate_contract, execution_contract = _load_gate_contract(
        gate_contract_path,
        expected_sha256=gate_contract_sha256,
        workload=workload,
        non_formal_long_smoke=non_formal_long_smoke,
    )
    learner_count = int(workload_config["learner_count"] if learner_count_override is None else learner_count_override)
    if not 0 <= learner_index < learner_count:
        raise Stage1CloseError("learner index is outside the role topology")
    role_id = f"learner-{learner_index:02d}"
    policy = _outer_policy(config)
    atomic, proposals, _global_backend, _proposal_backend = _stores(
        shared_root,
        run_id=run_id,
        config_sha256=config_sha256,
        bootstrap=bootstrap,
        learner_count=learner_count,
        policy=policy,
    )
    _wait_json(shared_root / "coordination" / "bootstrap.json", timeout_seconds)
    profile_path = config_path.resolve().parents[2] / workload_config["learner_profile"]
    profile = load_learner_profile(profile_path)
    inventory = verify_profile_assets(profile, hub_cache)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise Stage1CloseError("learner requires one bf16-capable CUDA GPU")
    device = torch.device("cuda", 0)
    seed = int(profile.training["seed"]) + learner_index
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _load_real_model(profile, inventory, device)
    registry = build_logical_layer_registry(model)
    fragment_map = build_fragment_map(registry, int(config["protocol"]["fragment_count"]))
    if registry.digest != bootstrap["registry_digest"] or fragment_map.digest != bootstrap["fragment_map_digest"]:
        raise Stage1CloseError("learner model registry differs from immutable bootstrap")
    groups = build_fragment_parameter_groups(model, registry, fragment_map)
    initial = atomic.load_snapshot().authorities
    _apply_fragment_payloads(groups, [item.parameters for item in initial])
    optimizer = _optimizer(profile, model)
    progress = LearnerProgress.initialize(role_id, tuple(item.version for item in initial))
    logger = StructuredLogger(result_root / "logs" / f"{role_id}.jsonl", role="learner", run_id=run_id)
    schedule_row = bootstrap["per_learner_schedules"][learner_index]
    schedule = FragmentPublicationSchedule(
        fragment_bytes=tuple(int(item) for item in schedule_row["fragment_bytes"]),
        intervals=tuple(int(item) for item in schedule_row["intervals"]),
        offsets=tuple(int(item) for item in schedule_row["offsets"]),
        offset_algorithm=str(schedule_row["offset_algorithm"]),
        learner_phase_offset=int(schedule_row["learner_phase_offset"]),
    )
    publisher = FragmentSnapshotCoordinator(
        identities=atomic.store.identities,
        progress=progress,
        descriptors=atomic.store.descriptors,
        fragment_parameters=groups,
        adopted_bases=tuple(
            AdoptedFragmentBase(item.version, item.content_identity) for item in initial
        ),
        schedule=schedule,
        store=proposals,
        logger=logger,
    )
    adoption = FragmentAdoptionCoordinator(
        store=atomic.store,
        progress=progress,
        initial_states=tuple(item.state for item in initial),
        fragment_parameters=groups,
        optimizer=optimizer,
        identity_audit=False,
        base_context_sink=publisher.update_adopted_base_context,
        logger=logger,
        poll_interval_seconds=0.01,
        autostart=True,
    )
    dataset_key = "gpt2_train" if workload == "nine_node" else "pythia_long"
    dataset_root = Path(str(_asset_manifest["datasets"][dataset_key]["root"]))
    shard = PackedTokenShard(profile, dataset_root, learner_index=learner_index)
    validation: dict[str, Any] = {}
    if workload == "nine_node" and learner_index == 0:
        loaded = load_evaluation_snapshot(shared_root / "evaluation" / "initial")
        _apply_fragment_payloads(groups, [item.parameters for item in loaded.authorities])
        validation["initial"] = {
            **_evaluate_validation(
                model,
                Path(str(_asset_manifest["datasets"]["gpt2_validation"]["root"])),
                device,
            ),
            "snapshot_identity": loaded.manifest.snapshot_identity,
            "version_vector": list(loaded.manifest.version_vector),
            "restart_load_access_audit": loaded.access_audit,
        }
    ready = {
        **_role_identity(role_id, shared_root),
        "gpu": _gpu_identity(),
        "run_id": run_id,
        "config_sha256": config_sha256,
        "asset_marker_sha256": asset_marker_sha256,
        "gate_contract_sha256": gate_contract_sha256,
        "execution_mode": (
            "non_formal_long_profile_smoke"
            if non_formal_long_smoke
            else "formal_workload"
        ),
        "model_revision": profile.model["revision"],
        "dataset_revision": profile.dataset["revision"],
    }
    _replace_json(shared_root / "coordination" / f"ready-{learner_index:02d}.json", ready)
    active = _wait_json(shared_root / "coordination" / "active.json", timeout_seconds)
    runtime = LearnerRuntime(
        model=model,
        optimizer=optimizer,
        scheduler=ConstantStepScheduler(),
        progress=progress,
        rng=LearnerRng.initialize(role_id, seed, device),
        fragment_parameters=groups,
        device=device,
        precision="bf16",
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        max_grad_norm=float(config["training"]["inner_optimizer"]["gradient_clip_norm"]),
        comparison=config["comparison"],
        logger=logger,
        # Snapshot the completed step against its pre-adoption base first.
        # Adoption then owns the counter reset and advances publisher context
        # for the next safe boundary, matching the proven S1-12 ordering.
        safe_boundary_observers=(publisher.on_safe_boundary, adoption.on_safe_boundary),
        update_norm_interval=50 if workload == "nine_node" else 1000,
        retain_events=False,
    )
    requested_steps = (
        int(execution_contract["optimizer_steps_per_learner"])
        if non_formal_long_smoke
        else (
            int(workload_config["maximum_optimizer_steps"])
            if workload == "nine_node"
            else int(workload_config["optimizer_steps_per_learner"])
        )
    )
    completion_path = shared_root / "coordination" / "complete.json"
    minimum_points = int(workload_config.get("minimum_loss_points", 1))
    run = runtime.run(
        _batches(shard, int(config["training"]["batch_size"])),
        optimizer_steps=requested_steps,
        # The frozen gate contract paces each workload so fixed-slot proposals
        # stay observable while retaining the exact optimizer-step/token budget.
        minimum_step_seconds=float(execution_contract["minimum_step_seconds"]),
        stop_requested=(
            (lambda: completion_path.exists() and progress.local_optimizer_steps >= minimum_points)
            if workload == "nine_node"
            else None
        ),
    )
    publisher.drain(timeout_seconds)
    if workload == "nine_node":
        completion = _wait_json(completion_path, timeout_seconds)
        if learner_index == 0:
            loaded = load_evaluation_snapshot(shared_root / "evaluation" / "final")
            _apply_fragment_payloads(groups, [item.parameters for item in loaded.authorities])
            validation["final"] = {
                **_evaluate_validation(
                    model,
                    Path(str(_asset_manifest["datasets"]["gpt2_validation"]["root"])),
                    device,
                ),
                "snapshot_identity": loaded.manifest.snapshot_identity,
                "version_vector": list(loaded.manifest.version_vector),
                "restart_load_access_audit": loaded.access_audit,
            }
    else:
        completion = None
    publisher.close(timeout_seconds)
    adoption.close(timeout_seconds)
    result = {
        "schema_version": 1,
        "status": "pass",
        "complete": True,
        "workload": workload,
        "run_id": run_id,
        "learner_id": role_id,
        "learner_index": learner_index,
        "learner_count": learner_count,
        "identity": ready,
        "active": active,
        "completed_unix_ns": time.time_ns(),
        "runtime": run.to_dict(),
        "update_norm_sampling": {
            "interval_optimizer_steps": 50 if workload == "nine_node" else 1000,
            "first_step_always_sampled": True,
            "parameter_update_norm_is_sampled_diagnostic": True,
        },
        "step_pacing": {
            "minimum_step_seconds": float(execution_contract["minimum_step_seconds"]),
            "scope": (
                "non_formal_long_profile_capacity_smoke"
                if non_formal_long_smoke
                else workload
            ),
            "reason": (
                "prove_the_formal_long_schedule_can_sustain_its_frozen_cycle_margin"
                if workload == "long_run"
                else "preserve_H50_fixed_slot_observability_below_the_800_step_ceiling"
            ),
        },
        "gate_contract_sha256": gate_contract_sha256,
        "non_formal_long_smoke": non_formal_long_smoke,
        "progress": progress.to_dict(),
        "data_state": shard.state_dict(),
        "publication": publisher.summary(),
        "adoption": adoption.summary(),
        "validation": validation,
        "completion": completion,
        "forbidden_runtime": {
            "torch_distributed_initialized": bool(torch.distributed.is_initialized()),
            "network_data_plane": False,
            "full_model_operations": 0,
        },
    }
    _replace_json(result_root / "roles" / f"{role_id}.json", result)
    _replace_json(shared_root / "coordination" / f"final-{learner_index:02d}.json", result)
    return result


def _update_record(update: Any, global_cycle: int, completed_ns: int) -> dict[str, Any]:
    return {
        "fragment_index": update.successor.state.descriptor.index,
        "from_version": update.previous.version,
        "to_version": update.successor.version,
        "selection_identity": update.selection.selection_identity,
        "selected_learners": list(update.selection.learner_ids),
        "proposal_ids": [item.proposal_id for item in update.selection.proposals],
        "proposal_processed_tokens": [item.processed_tokens for item in update.selection.proposals],
        "weights": [item.normalized_weight for item in update.selection.weights],
        "staleness": [item.staleness for item in update.selection.weights],
        "byte_accounting": update.result.byte_accounting.to_dict(),
        "source_metrics": dataclasses.asdict(update.source_metrics),
        "successor_authority_identity": update.successor.authority_identity,
        "successor_parameters_sha256": hashlib.sha256(update.successor.parameters).hexdigest(),
        "successor_momentum_sha256": hashlib.sha256(update.successor.outer_optimizer_state).hexdigest(),
        "global_cycle_after": global_cycle,
        "completed_unix_ns": completed_ns,
    }


def run_syncer(
    *,
    shared_root: Path,
    result_root: Path,
    run_id: str,
    config_path: Path,
    config_sha256: str,
    asset_marker_sha256: str,
    gate_contract_path: Path,
    gate_contract_sha256: str,
    non_formal_long_smoke: bool,
    learner_count_override: int | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Stage1CloseError("syncer requires CUDA_VISIBLE_DEVICES to disable accelerators")
    config, asset_root, _asset_manifest, bootstrap = _load_contract(
        config_path,
        expected_config_sha256=config_sha256,
        expected_asset_marker_sha256=asset_marker_sha256,
    )
    workload = str(config["selected_workload"])
    workload_config = config["workloads"][workload]
    _gate_contract, execution_contract = _load_gate_contract(
        gate_contract_path,
        expected_sha256=gate_contract_sha256,
        workload=workload,
        non_formal_long_smoke=non_formal_long_smoke,
    )
    learner_count = int(workload_config["learner_count"] if learner_count_override is None else learner_count_override)
    policy = _outer_policy(config)
    atomic, proposals, global_backend, proposal_backend = _stores(
        shared_root,
        run_id=run_id,
        config_sha256=config_sha256,
        bootstrap=bootstrap,
        learner_count=learner_count,
        policy=policy,
    )
    _bootstrap(atomic, asset_root, workload, bootstrap)
    initial_plan = FrozenEvaluationPlan.capture(atomic)
    if workload == "nine_node":
        materialize_evaluation_snapshot(initial_plan, shared_root / "evaluation" / "initial")
    bootstrap_record = {
        **_role_identity("syncer", shared_root),
        "run_id": run_id,
        "config_sha256": config_sha256,
        "asset_marker_sha256": asset_marker_sha256,
        "gate_contract_sha256": gate_contract_sha256,
        "execution_mode": (
            "non_formal_long_profile_smoke"
            if non_formal_long_smoke
            else "formal_workload"
        ),
        "version_vector": list(initial_plan.version_vector),
        "bootstrap_identity": bootstrap["bootstrap_identity"],
    }
    _replace_json(shared_root / "coordination" / "bootstrap.json", bootstrap_record)
    _replace_json(shared_root / "coordination" / "syncer-ready.json", bootstrap_record)
    ready = _wait_roles(shared_root / "coordination", "ready", learner_count, timeout_seconds)
    hosts = [str(item["hostname"]) for item in ready] + [bootstrap_record["hostname"]]
    active = {
        "complete": True,
        "run_id": run_id,
        "active_start_unix_ns": time.time_ns(),
        "role_hosts": hosts,
        "learner_count": learner_count,
        "application_coordination": "shared_filesystem_only",
    }
    _replace_json(shared_root / "coordination" / "active.json", active)
    tracker = ProfileAProgressTracker(
        _learner_ids(learner_count),
        tuple(0 for _ in bootstrap["fragments"]),
        comparison=ComparisonBasis(False, False, False, _PROFILE_CLAIM),
    )
    executor = ProfileAFragmentExecutor(
        atomic_store=atomic,
        proposal_store=proposals,
        profile=ProfileAConfig(
            learner_count=learner_count,
            fragment_count=len(bootstrap["fragments"]),
            q=learner_count,
            q_fresh=learner_count,
            s_max=0,
            grace_period_ns=0,
            lambda_s=float(config["protocol"]["lambda_s"]),
        ),
        outer_policy=policy,
        progress=tracker,
        merge_backend="numpy",
    )
    logger = StructuredLogger(result_root / "logs" / "syncer.jsonl", role="syncer", run_id=run_id)
    updates: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []
    last_inventory_cycle = -1
    inventory_policy = {
        "minimum_unreferenced_payload_age_seconds": float(
            config["bounded_state"]["minimum_unreferenced_payload_age_seconds"]
        ),
        "retained_recent_unreferenced_payloads": int(
            config["bounded_state"]["retained_recent_unreferenced_payloads"]
        ),
        "inventory_every_global_cycles": int(
            config["bounded_state"]["inventory_every_global_cycles"]
        ),
        "history_scan_allowed": False,
    }
    target = int(
        execution_contract["minimum_global_cycles"]
        if non_formal_long_smoke
        else (
            workload_config["target_global_cycles"]
            if workload == "nine_node"
            else workload_config["minimum_global_cycles"]
        )
    )
    active_deadline = time.monotonic() + float(
        execution_contract["runtime_budget_seconds"]
        if non_formal_long_smoke
        else workload_config["runtime_budget_seconds"]
    )
    final_records: list[dict[str, Any]] = []
    while True:
        executor.poll(observed_ns=time.monotonic_ns())
        made_progress = False
        while True:
            update_started_ns = time.monotonic_ns()
            update = executor.execute_next(observed_ns=time.monotonic_ns(), poll_store=False)
            if update is None:
                break
            made_progress = True
            report = tracker.report()
            row = _update_record(update, report.global_cycle, time.time_ns())
            row["update_latency_seconds"] = (
                time.monotonic_ns() - update_started_ns
            ) / 1_000_000_000
            updates.append(row)
            logger.emit("fragment_outer_update", **row)
        report = tracker.report()
        cycle = report.global_cycle
        interval = int(config["bounded_state"]["inventory_every_global_cycles"])
        if cycle > 0 and cycle != last_inventory_cycle and cycle % interval == 0:
            inventory = {
                "global_cycle": cycle,
                "timestamp_unix_ns": time.time_ns(),
                "policy": inventory_policy,
                "global": global_backend.reclaim_unreferenced_payloads(
                    retain_recent=int(config["bounded_state"]["retained_recent_unreferenced_payloads"]),
                    minimum_age_seconds=float(config["bounded_state"]["minimum_unreferenced_payload_age_seconds"]),
                ),
                "proposals": proposal_backend.reclaim_unreferenced_payloads(
                    retain_recent=int(config["bounded_state"]["retained_recent_unreferenced_payloads"]),
                    minimum_age_seconds=float(config["bounded_state"]["minimum_unreferenced_payload_age_seconds"]),
                ),
            }
            inventories.append(inventory)
            logger.emit("bounded_storage_inventory", **inventory)
            last_inventory_cycle = cycle
        if workload == "nine_node" and cycle >= target:
            final_plan = FrozenEvaluationPlan.capture(atomic)
            final_manifest = materialize_evaluation_snapshot(final_plan, shared_root / "evaluation" / "final")
            completion = {
                "complete": True,
                "run_id": run_id,
                "reason": "target_global_cycles",
                "global_cycle": cycle,
                "version_vector": list(final_plan.version_vector),
                "evaluation_snapshot_identity": final_manifest.snapshot_identity,
                "active_end_unix_ns": time.time_ns(),
            }
            _replace_json(shared_root / "coordination" / "complete.json", completion)
            final_records = _wait_roles(shared_root / "coordination", "final", learner_count, timeout_seconds)
            break
        if workload == "long_run":
            paths = [shared_root / "coordination" / f"final-{index:02d}.json" for index in range(learner_count)]
            if all(path.exists() for path in paths) and cycle >= target and not made_progress:
                final_records = [_read_json(path) for path in paths]
                completion = {
                    "complete": True,
                    "run_id": run_id,
                    "reason": "learners_complete_and_minimum_global_cycles",
                    "global_cycle": cycle,
                    "version_vector": list(atomic.load_snapshot().version_vector),
                    "active_end_unix_ns": time.time_ns(),
                }
                _replace_json(shared_root / "coordination" / "complete.json", completion)
                break
        if time.monotonic() >= active_deadline:
            raise TimeoutError("frozen active runtime budget expired")
        if not made_progress:
            time.sleep(0.01)
    for item in final_records:
        progress = item["progress"]
        tracker.update_learner(
            item["learner_id"],
            local_optimizer_steps=int(progress["local_optimizer_steps"]),
            processed_input_tokens=int(progress["processed_input_tokens"]),
            loss_bearing_target_tokens=int(progress["loss_bearing_target_tokens"]),
        )
    final_report = tracker.report()
    readiness = dataclasses.asdict(executor.readiness.snapshot())
    final_inventory = {
        "global_cycle": final_report.global_cycle,
        "timestamp_unix_ns": time.time_ns(),
        "policy": inventory_policy,
        "global": global_backend.inspect_inventory(),
        "proposals": proposal_backend.inspect_inventory(),
    }
    inventories.append(final_inventory)
    result = {
        "schema_version": 1,
        "status": "pass",
        "complete": True,
        "workload": workload,
        "run_id": run_id,
        "gate_contract_sha256": gate_contract_sha256,
        "non_formal_long_smoke": non_formal_long_smoke,
        "identity": {
            **bootstrap_record,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_module_imported": "torch" in sys.modules,
            "gpu_count": 0,
        },
        "active": active,
        "completion": completion,
        "completed_unix_ns": time.time_ns(),
        "progress": final_report.to_dict(),
        "update_count": len(updates),
        "updates": updates,
        "readiness": readiness,
        "inventories": inventories,
        "final_roles": [
            {
                "learner_id": item["learner_id"],
                "hostname": item["identity"]["hostname"],
                "pbs_job_id": item["identity"]["pbs_job_id"],
                "optimizer_steps": item["progress"]["local_optimizer_steps"],
                "processed_input_tokens": item["progress"]["processed_input_tokens"],
            }
            for item in final_records
        ],
        "forbidden_runtime": {
            "torch_distributed_initialized": False,
            "network_data_plane": False,
            "full_model_operations": 0,
            "history_scan_operations": 0,
        },
    }
    _replace_json(result_root / "roles" / "syncer.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the S1-13 filesystem-only closure workload")
    parser.add_argument("role", choices=("learner", "syncer", "mpi"))
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--asset-marker-sha256", required=True)
    parser.add_argument("--gate-contract", type=Path, required=True)
    parser.add_argument("--gate-contract-sha256", required=True)
    parser.add_argument("--non-formal-long-smoke", action="store_true")
    parser.add_argument("--hub-cache", type=Path)
    parser.add_argument("--learner-index", type=int)
    parser.add_argument("--learner-count-override", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.role == "mpi":
        rank_value = next(
            (os.environ[name] for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK") if name in os.environ),
            None,
        )
        size_value = next(
            (os.environ[name] for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE") if name in os.environ),
            None,
        )
        if rank_value is None or size_value is None or int(size_value) != 5:
            raise Stage1CloseError("mpi role requires exactly five launcher ranks")
        rank = int(rank_value)
        if rank < 4:
            arguments.role = "learner"
            arguments.learner_index = rank
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif rank == 4:
            arguments.role = "syncer"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            raise Stage1CloseError("mpi rank is outside the frozen 4+1 topology")
    if arguments.role == "learner":
        if arguments.learner_index is None or arguments.hub_cache is None:
            raise Stage1CloseError("learner role requires --learner-index and --hub-cache")
        result = run_learner(
            shared_root=arguments.shared_root,
            result_root=arguments.result_root,
            run_id=arguments.run_id,
            config_path=arguments.config_path,
            config_sha256=arguments.config_sha256,
            asset_marker_sha256=arguments.asset_marker_sha256,
            gate_contract_path=arguments.gate_contract,
            gate_contract_sha256=arguments.gate_contract_sha256,
            non_formal_long_smoke=arguments.non_formal_long_smoke,
            hub_cache=arguments.hub_cache,
            learner_index=arguments.learner_index,
            learner_count_override=arguments.learner_count_override,
            timeout_seconds=arguments.timeout_seconds,
        )
    else:
        result = run_syncer(
            shared_root=arguments.shared_root,
            result_root=arguments.result_root,
            run_id=arguments.run_id,
            config_path=arguments.config_path,
            config_sha256=arguments.config_sha256,
            asset_marker_sha256=arguments.asset_marker_sha256,
            gate_contract_path=arguments.gate_contract,
            gate_contract_sha256=arguments.gate_contract_sha256,
            non_formal_long_smoke=arguments.non_formal_long_smoke,
            learner_count_override=arguments.learner_count_override,
            timeout_seconds=arguments.timeout_seconds,
        )
    print(json.dumps({"status": result["status"], "workload": result["workload"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
