from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from fsbdd.diloco.model.fragment_map import build_fragment_map
from fsbdd.diloco.protocol.global_state import (
    BootstrapFragment,
    FragmentStateDescriptor,
    GlobalStateIdentities,
    GlobalStateStore,
)
from fsbdd.diloco.common.identity import file_digest
from fsbdd.diloco.learner.runtime import (
    ConstantStepScheduler,
    LearnerProgress,
    LearnerRng,
    LearnerRuntime,
)
from fsbdd.diloco.learner.adoption import FragmentAdoptionCoordinator
from fsbdd.diloco.learner.publication import (
    build_fragment_descriptors,
    build_fragment_parameter_groups,
    serialize_fragment_parameters,
)
from fsbdd.auxiliary.contracts.manifest import build_manifest
from fsbdd.diloco.model.model_registry import build_logical_layer_registry
from fsbdd.diloco.protocol.storage import PosixStorageBackend


class AdoptionStressError(RuntimeError):
    pass


def _write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _wait(path: Path, timeout_seconds: float) -> dict[str, Any]:
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
            raise AdoptionStressError(f"coordination record is malformed: {path}")
        return value


def _load_config(path: Path, expected_identity: str) -> dict[str, Any]:
    if file_digest(path) != expected_identity:
        raise AdoptionStressError("frozen adoption profile identity mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or set(value)
        != {
            "schema_version",
            "profile_id",
            "model",
            "training",
            "adoption",
        }
    ):
        raise AdoptionStressError("adoption profile schema mismatch")
    model = value["model"]
    training = value["training"]
    adoption = value["adoption"]
    if (
        model.get("family") != "gpt_neox"
        or training.get("learner_id") != "learner-00"
        or training.get("fragment_count") != 4
        or training.get("precision") != "fp32"
        or training.get("optimizer", {}).get("class") != "AdamW"
        or adoption.get("maximum_pending_per_fragment") != 1
        or adoption.get("identity_audit") is not True
        or adoption.get("expected_final_version_vector") != [3, 0, 1, 0]
        or adoption.get("updates")
        != [
            {"fragment_index": 0, "successor_count": 3, "delta_per_version": 0.01},
            {"fragment_index": 2, "successor_count": 1, "delta_per_version": 0.02},
        ]
    ):
        raise AdoptionStressError("adoption profile violates the frozen workload")
    return value


def _descriptor_from_dict(value: Mapping[str, Any]) -> FragmentStateDescriptor:
    return FragmentStateDescriptor(
        index=int(value["index"]),
        identity=str(value["identity"]),
        dtype=str(value["dtype"]),
        shape=tuple(int(item) for item in value["shape"]),
        parameter_identities=tuple(str(item) for item in value["parameter_identities"]),
    )


def _shift(payload: bytes, amount: float) -> bytes:
    array = np.frombuffer(payload, dtype="<f4").copy()
    array += np.float32(amount)
    if not bool(np.isfinite(array).all()):
        raise AdoptionStressError("writer generated a nonfinite fragment")
    return array.astype("<f4", copy=False).tobytes()


def _delayed_batches(
    *,
    count: int,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    delay_seconds: float,
    updates_ready: Path,
    coordinator: FragmentAdoptionCoordinator,
    control: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    import torch

    base = torch.arange(batch_size * sequence_length).reshape(
        batch_size, sequence_length
    )
    for step in range(count):
        if not control["poller_started"] and updates_ready.exists():
            coordinator.start()
            control["poller_started"] = True
            control["poller_started_before_batch"] = step
            control["poller_started_unix_ns"] = time.time_ns()
        time.sleep(delay_seconds)
        input_ids = (base + step) % vocab_size
        yield {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }


def _learner(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_path: Path,
    config_identity: str,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    if not torch.cuda.is_available():
        raise AdoptionStressError("formal mixed-version learner requires one CUDA GPU")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise AdoptionStressError("application distributed process group is forbidden")
    device = torch.device("cuda", 0)
    model_spec = profile["model"]
    training = profile["training"]
    adoption = profile["adoption"]
    seed = int(model_spec["initialization_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=int(model_spec["vocab_size"]),
            hidden_size=int(model_spec["hidden_size"]),
            intermediate_size=int(model_spec["intermediate_size"]),
            num_hidden_layers=int(model_spec["num_hidden_layers"]),
            num_attention_heads=int(model_spec["num_attention_heads"]),
            max_position_embeddings=int(model_spec["max_position_embeddings"]),
            use_cache=bool(model_spec["use_cache"]),
        )
    ).to(device)
    registry = build_logical_layer_registry(model)
    fragment_map = build_fragment_map(registry, int(training["fragment_count"]))
    descriptors = build_fragment_descriptors(fragment_map)
    groups = build_fragment_parameter_groups(model, registry, fragment_map)
    identities = GlobalStateIdentities(
        run_identity=run_id,
        config_identity=config_identity,
        model_identity=registry.digest,
        fragment_map_identity=fragment_map.digest,
    )
    store = GlobalStateStore(
        PosixStorageBackend(root / "backend"),
        identities=identities,
        descriptors=descriptors,
        s_max=0,
    )
    bootstrap = store.bootstrap(
        tuple(
            BootstrapFragment(
                descriptor=descriptor,
                parameters=serialize_fragment_parameters(
                    group, int(descriptor.shape[0]) * 4
                ),
                outer_state=b'{"optimizer":"none","version":0}',
            )
            for descriptor, group in zip(descriptors, groups, strict=True)
        )
    )
    initial_states = bootstrap.snapshot.states
    catalog = {
        "complete": True,
        "profile_id": profile["profile_id"],
        "config_path": str(config_path),
        "state_identities": identities.to_dict(),
        "descriptors": [descriptor.to_dict() for descriptor in descriptors],
        "base_states": [
            {
                "fragment_index": state.descriptor.index,
                "version": state.version,
                "content_identity": state.content_identity,
                "parameters_sha256": hashlib_sha256(state.parameters),
            }
            for state in initial_states
        ],
        "catalog_unix_ns": time.time_ns(),
    }
    coordination = root / "coordination"
    _replace_json(coordination / "catalog.json", catalog)

    progress = LearnerProgress.initialize(
        str(training["learner_id"]), bootstrap.snapshot.version_vector
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["optimizer"]["lr"]),
        weight_decay=float(training["optimizer"]["weight_decay"]),
    )
    adoption_traces: list[dict[str, Any]] = []
    coordinator = FragmentAdoptionCoordinator(
        store=store,
        progress=progress,
        initial_states=initial_states,
        fragment_parameters=groups,
        optimizer=optimizer,
        identity_audit=bool(adoption["identity_audit"]),
        trace_sink=lambda trace: adoption_traces.append(dict(trace)),
        poll_interval_seconds=float(adoption["poll_interval_seconds"]),
        autostart=False,
    )
    runtime = LearnerRuntime(
        model=model,
        optimizer=optimizer,
        scheduler=ConstantStepScheduler(),
        progress=progress,
        rng=LearnerRng.initialize(progress.learner_id, seed, device),
        fragment_parameters=groups,
        device=device,
        precision=str(training["precision"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_grad_norm=float(training["optimizer"]["grad_clip_norm"]),
        comparison={"claim": "S1-08 adoption protocol semantics"},
        safe_boundary_observers=(coordinator.on_safe_boundary,),
    )
    control: dict[str, Any] = {
        "poller_started": False,
        "poller_started_before_batch": None,
        "poller_started_unix_ns": None,
    }
    training_started_unix_ns = time.time_ns()
    run = runtime.run(
        _delayed_batches(
            count=int(training["optimizer_steps"]),
            batch_size=int(training["batch_size"]),
            sequence_length=int(training["sequence_length"]),
            vocab_size=int(model_spec["vocab_size"]),
            delay_seconds=float(adoption["batch_delay_seconds"]),
            updates_ready=coordination / "updates-ready.json",
            coordinator=coordinator,
            control=control,
        ),
        optimizer_steps=int(training["optimizer_steps"]),
    )
    training_returned_unix_ns = time.time_ns()
    coordinator.close(float(adoption["reader_timeout_seconds"]))
    final_adoption = coordinator.summary()
    if not control["poller_started"]:
        raise AdoptionStressError("learner completed before the update record appeared")
    learner = {
        "schema_version": 1,
        "status": "pass",
        "role": "mixed_version_learner",
        "hostname": socket.gethostname().split(".")[0],
        "rank": 0,
        "state_identities": identities.to_dict(),
        "torch": {
            "version": torch.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "distributed_initialized": False,
        },
        "model": {
            "registry_digest": registry.digest,
            "fragment_map_digest": fragment_map.digest,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "fragment_parameter_counts": [
                sum(parameter.numel() for parameter in group) for group in groups
            ],
        },
        "bootstrap": catalog["base_states"],
        "training": {
            "started_unix_ns": training_started_unix_ns,
            "returned_unix_ns": training_returned_unix_ns,
            "optimizer_steps": run.optimizer_steps_completed,
            "processed_input_tokens": run.processed_input_tokens,
            "token_weighted_loss": run.token_weighted_loss,
            "finite_loss": math.isfinite(run.token_weighted_loss),
            "events": [event.to_dict() for event in run.events],
            "distributed_initialized": run.distributed_initialized,
        },
        "worker_control": control,
        "adoption": final_adoption,
        "adoption_traces": sorted(
            adoption_traces,
            key=lambda trace: (
                int(trace["safe_boundary_local_step"]),
                int(trace["fragment_index"]),
            ),
        ),
        "learner_waited_for_writer_or_version_alignment": False,
        "application_data_plane": "shared_filesystem_only",
    }
    _write_json_new(result_root / "mixed_version_learner.json", learner)
    _replace_json(
        coordination / "learner-complete.json",
        {"complete": True, "completed_unix_ns": time.time_ns()},
    )
    return learner


def hashlib_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _writer(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_identity: str,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    adoption = profile["adoption"]
    timeout = float(adoption["reader_timeout_seconds"])
    coordination = root / "coordination"
    catalog = _wait(coordination / "catalog.json", timeout)
    identities = GlobalStateIdentities(**catalog["state_identities"])
    if (
        identities.run_identity != run_id
        or identities.config_identity != config_identity
    ):
        raise AdoptionStressError("writer catalog identity mismatch")
    descriptors = tuple(
        _descriptor_from_dict(value) for value in catalog["descriptors"]
    )
    store = GlobalStateStore(
        PosixStorageBackend(root / "backend"),
        identities=identities,
        descriptors=descriptors,
        s_max=0,
    )
    time.sleep(float(adoption["writer_start_delay_seconds"]))
    publications: list[dict[str, Any]] = []
    for update in adoption["updates"]:
        index = int(update["fragment_index"])
        for _ in range(int(update["successor_count"])):
            previous = store.load_fragment(index, timeout_seconds=0)
            started = time.time_ns()
            successor = store.publish_successor(
                index,
                parameters=_shift(
                    previous.parameters, float(update["delta_per_version"])
                ),
                outer_state=json.dumps(
                    {
                        "fragment_index": index,
                        "version": previous.version + 1,
                    },
                    sort_keys=True,
                ).encode("utf-8"),
            )
            completed = time.time_ns()
            publications.append(
                {
                    "fragment_index": index,
                    "version": successor.version,
                    "content_identity": successor.content_identity,
                    "parameters_sha256": hashlib_sha256(successor.parameters),
                    "publication_started_unix_ns": started,
                    "publication_completed_unix_ns": completed,
                    "payload_bytes": len(successor.parameters),
                }
            )
    ready_unix_ns = time.time_ns()
    _replace_json(
        coordination / "updates-ready.json",
        {
            "complete": True,
            "updates_ready_unix_ns": ready_unix_ns,
            "publication_count": len(publications),
        },
    )
    _wait(coordination / "learner-complete.json", timeout)
    writer = {
        "schema_version": 1,
        "status": "pass",
        "role": "global_fragment_writer",
        "hostname": socket.gethostname().split(".")[0],
        "rank": 1,
        "state_identities": identities.to_dict(),
        "torch_imported": "torch" in sys.modules,
        "gpu_memory_allocated_bytes": 0,
        "publications": publications,
        "updates_ready_unix_ns": ready_unix_ns,
        "final_version_vector": list(store.load_snapshot().version_vector),
        "writer_waited_for_learner_before_publication": False,
        "application_data_plane": "shared_filesystem_only",
    }
    _write_json_new(result_root / "global_fragment_writer.json", writer)
    return writer


def run_role(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_path: Path,
    config_identity: str,
) -> dict[str, Any]:
    try:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise AdoptionStressError("adoption stress requires an MPI launcher") from error
    if size != 2 or rank not in (0, 1):
        raise AdoptionStressError("adoption stress requires exactly two ranks")
    profile = _load_config(config_path, config_identity)
    if rank == 0:
        return _learner(
            root=root,
            result_root=result_root,
            run_id=run_id,
            config_path=config_path,
            config_identity=config_identity,
            profile=profile,
        )
    return _writer(
        root=root,
        result_root=result_root,
        run_id=run_id,
        config_identity=config_identity,
        profile=profile,
    )


def summarize(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_path: Path,
    config_identity: str,
    output: Path,
) -> dict[str, Any]:
    profile = _load_config(config_path, config_identity)
    learner = json.loads(
        (result_root / "mixed_version_learner.json").read_text(encoding="utf-8")
    )
    writer = json.loads(
        (result_root / "global_fragment_writer.json").read_text(encoding="utf-8")
    )
    if (
        learner.get("status") != "pass"
        or learner.get("role") != "mixed_version_learner"
        or learner.get("rank") != 0
        or writer.get("status") != "pass"
        or writer.get("role") != "global_fragment_writer"
        or writer.get("rank") != 1
    ):
        raise AdoptionStressError("formal role status rank or identity is invalid")
    if learner["hostname"] == writer["hostname"]:
        raise AdoptionStressError("formal roles did not use distinct hosts")
    if learner["state_identities"] != writer["state_identities"]:
        raise AdoptionStressError("formal role state identities differ")
    if (
        learner["state_identities"]["run_identity"] != run_id
        or learner["state_identities"]["config_identity"] != config_identity
    ):
        raise AdoptionStressError("formal run or config identity differs")
    if (
        learner["torch"]["distributed_initialized"]
        or learner["training"]["distributed_initialized"]
        or writer["torch_imported"]
        or writer["gpu_memory_allocated_bytes"] != 0
        or learner["application_data_plane"] != "shared_filesystem_only"
        or writer["application_data_plane"] != "shared_filesystem_only"
    ):
        raise AdoptionStressError(
            "formal topology or filesystem-only data plane failed"
        )
    training = learner["training"]
    events = training["events"]
    expected_steps = int(profile["training"]["optimizer_steps"])
    tokens_per_step = int(profile["training"]["batch_size"]) * int(
        profile["training"]["sequence_length"]
    )
    if (
        training["optimizer_steps"] != expected_steps
        or training["processed_input_tokens"] != expected_steps * tokens_per_step
        or not training["finite_loss"]
        or not math.isfinite(float(training["token_weighted_loss"]))
        or [int(event["local_optimizer_step"]) for event in events]
        != list(range(1, expected_steps + 1))
        or any(
            not math.isfinite(float(event["token_weighted_loss"])) for event in events
        )
    ):
        raise AdoptionStressError("formal learner step or finite-loss trace failed")
    expected_vector = list(profile["adoption"]["expected_final_version_vector"])
    adoption = learner["adoption"]
    traces = learner["adoption_traces"]
    if (
        adoption["adoption_count"] != 2
        or len(traces) != 2
        or adoption["progress"]["fragments"]
        and [
            fragment["global_version"] for fragment in adoption["progress"]["fragments"]
        ]
        != expected_vector
        or writer["final_version_vector"] != expected_vector
        or {int(trace["fragment_index"]) for trace in traces} != {0, 2}
    ):
        raise AdoptionStressError("formal independently adopted version vector failed")
    fragment_zero = [trace for trace in traces if int(trace["fragment_index"]) == 0]
    if (
        len(fragment_zero) != 1
        or fragment_zero[0]["from_version"] != 0
        or fragment_zero[0]["to_version"] != 3
        or fragment_zero[0]["skipped_intermediate_versions"] != 2
        or adoption["skipped_intermediate_versions"] != 2
    ):
        raise AdoptionStressError("formal latest-only direct jump failed")
    publications = {
        (int(row["fragment_index"]), int(row["version"])): row
        for row in writer["publications"]
    }
    if set(publications) != {(0, 1), (0, 2), (0, 3), (2, 1)}:
        raise AdoptionStressError("formal writer successor coverage failed")
    lag_rows = []
    for trace in traces:
        index = int(trace["fragment_index"])
        target = index
        publication = publications[(index, int(trace["to_version"]))]
        before_parameters = trace["parameter_hashes_before"]
        after_parameters = trace["parameter_hashes_after"]
        before_moments = trace["optimizer_state_hashes_before"]
        after_moments = trace["optimizer_state_hashes_after"]
        before_moment_entries = trace["optimizer_state_entry_counts_before"]
        after_moment_entries = trace["optimizer_state_entry_counts_after"]
        if (
            trace["content_identity"] != publication["content_identity"]
            or trace["target_payload_sha256"] != publication["parameters_sha256"]
            or after_parameters[target] != publication["parameters_sha256"]
            or any(
                before != after
                for position, (before, after) in enumerate(
                    zip(before_parameters, after_parameters, strict=True)
                )
                if position != target
            )
            or before_moments != after_moments
            or before_moment_entries != after_moment_entries
            or any(int(value) <= 0 for value in before_moment_entries)
            or len(before_parameters) != int(profile["training"]["fragment_count"])
            or len(before_moments) != int(profile["training"]["fragment_count"])
        ):
            raise AdoptionStressError(
                "formal parameter or optimizer identity audit failed"
            )
        before_counters = trace["counters_before"]
        after_counters = trace["counters_after"]
        for position, (before, after) in enumerate(
            zip(before_counters, after_counters, strict=True)
        ):
            if position == target:
                if (
                    after["global_version"] != trace["to_version"]
                    or after["local_steps"] != 0
                    or after["processed_input_tokens"] != 0
                    or after["proposal_sequence"] != before["proposal_sequence"]
                ):
                    raise AdoptionStressError("formal target counter reset failed")
            elif before != after:
                raise AdoptionStressError("formal non-target counter changed")
        lag_ns = int(trace["adoption_unix_ns"]) - int(
            publication["publication_completed_unix_ns"]
        )
        if (
            lag_ns < 0
            or int(trace["extra_local_steps_before_adoption"]) < 0
            or float(trace["fs_to_cpu_seconds"]) < 0
            or float(trace["cpu_to_gpu_seconds"]) < 0
        ):
            raise AdoptionStressError("formal adoption lag or transfer timing failed")
        lag_rows.append(
            {
                "fragment_index": index,
                "version": int(trace["to_version"]),
                "content_identity": trace["content_identity"],
                "publication_to_adoption_seconds": lag_ns / 1e9,
                "extra_local_steps_before_adoption": int(
                    trace["extra_local_steps_before_adoption"]
                ),
                "fs_to_cpu_seconds": float(trace["fs_to_cpu_seconds"]),
                "cpu_to_gpu_seconds": float(trace["cpu_to_gpu_seconds"]),
            }
        )
    first_adoption_step = min(
        int(trace["safe_boundary_local_step"]) for trace in traces
    )
    mixed_window = [
        event
        for event in events
        if int(event["local_optimizer_step"]) > first_adoption_step
        and len(set(int(value) for value in event["fragment_global_versions"])) > 1
    ]
    minimum_mixed = int(profile["training"]["minimum_mixed_version_steps"])
    if (
        len(mixed_window) < minimum_mixed
        or int(mixed_window[0]["local_optimizer_step"]) != first_adoption_step + 1
        or int(mixed_window[-1]["local_optimizer_step"]) != expected_steps
        or [int(event["local_optimizer_step"]) for event in mixed_window]
        != list(range(first_adoption_step + 1, expected_steps + 1))
        or any(
            not math.isfinite(float(event["token_weighted_loss"]))
            for event in mixed_window
        )
    ):
        raise AdoptionStressError("formal sustained mixed-version loss window failed")
    poller = adoption["poller"]
    if (
        any(value > 1 for value in poller["maximum_pending_per_fragment"])
        or poller["resident_pending_bound"]
        != int(profile["training"]["fragment_count"])
        or poller["pending_count"] != 0
        or poller["errors"]
        or poller["fixed_slot_read_count"] <= int(profile["training"]["fragment_count"])
        or poller["repeated_current_count"] < 1
        or adoption["resident_trace_records"] > adoption["resident_trace_record_bound"]
    ):
        raise AdoptionStressError("formal bounded polling state failed")
    first_publication = min(
        int(row["publication_started_unix_ns"]) for row in writer["publications"]
    )
    if (
        learner["learner_waited_for_writer_or_version_alignment"]
        or writer["writer_waited_for_learner_before_publication"]
        or int(training["started_unix_ns"]) >= first_publication
        or not learner["worker_control"]["poller_started"]
        or learner["worker_control"]["poller_started_before_batch"] is None
    ):
        raise AdoptionStressError("formal learner blocked on writer or alignment")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "run_id": run_id,
        "config_identity": config_identity,
        "hosts": [learner["hostname"], writer["hostname"]],
        "state_identities": learner["state_identities"],
        "topology": {
            "mixed_version_learner": learner["hostname"],
            "global_fragment_writer": writer["hostname"],
            "learner_gpu_count": 1,
            "writer_gpu_memory_allocated_bytes": 0,
            "application_data_plane": "shared_filesystem_only",
            "distributed_initialized": False,
        },
        "latest_only": {
            "expected_final_version_vector": expected_vector,
            "actual_final_version_vector": [
                fragment["global_version"]
                for fragment in adoption["progress"]["fragments"]
            ],
            "fragment_zero_direct_jump": [0, 3],
            "skipped_intermediate_versions": 2,
            "applied_fragments": sorted(
                int(trace["fragment_index"]) for trace in traces
            ),
        },
        "identity_audit": {
            "adoption_count": len(traces),
            "all_non_target_parameters_unchanged": True,
            "all_target_parameters_match_verified_global_payload": True,
            "all_optimizer_states_preserved": True,
            "all_target_only_counters_reset": True,
            "traces": traces,
        },
        "mixed_version_training": {
            "first_adoption_boundary_step": first_adoption_step,
            "window_steps": [
                int(event["local_optimizer_step"]) for event in mixed_window
            ],
            "window_version_vectors": [
                event["fragment_global_versions"] for event in mixed_window
            ],
            "window_losses": [
                float(event["token_weighted_loss"]) for event in mixed_window
            ],
            "minimum_required_steps": minimum_mixed,
            "all_finite": True,
        },
        "adoption_lag": lag_rows,
        "poller": poller,
        "writer_publications": writer["publications"],
        "learner_started_before_writer_publication": True,
    }
    _write_json_new(output, summary)
    shutil.rmtree(root)
    return summary


def write_manifest(args: argparse.Namespace) -> None:
    hosts = sorted(set(Path(args.nodefile).read_text(encoding="utf-8").splitlines()))
    if len(hosts) != 2:
        raise AdoptionStressError(f"manifest requires two distinct hosts, got {hosts}")
    modules = [
        line.strip()
        for line in Path(args.modules_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    learner = json.loads(
        (Path(args.result_root) / "mixed_version_learner.json").read_text(
            encoding="utf-8"
        )
    )
    writer = json.loads(
        (Path(args.result_root) / "global_fragment_writer.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        learner.get("status") != "pass"
        or learner.get("role") != "mixed_version_learner"
        or learner.get("rank") != 0
        or writer.get("status") != "pass"
        or writer.get("role") != "global_fragment_writer"
        or writer.get("rank") != 1
        or learner.get("hostname") == writer.get("hostname")
    ):
        raise AdoptionStressError("manifest role status rank or host is invalid")
    expected_run_id = f"s1-08-{args.job_id}"
    if args.run_id != expected_run_id:
        raise AdoptionStressError("manifest run identity is not bound to PBS job id")
    if (
        learner.get("state_identities") != writer.get("state_identities")
        or learner["state_identities"].get("run_identity") != args.run_id
        or learner["state_identities"].get("config_identity") != args.config_sha256
        or learner.get("torch", {}).get("distributed_initialized") is not False
        or writer.get("torch_imported") is not False
        or learner.get("learner_waited_for_writer_or_version_alignment") is not False
    ):
        raise AdoptionStressError(
            "manifest role state config or run identity is invalid"
        )
    if args.queue != "debug-g" or args.group != "xg24i002":
        raise AdoptionStressError(
            "manifest scheduler queue or group differs from contract"
        )
    role_map = {
        "mixed_version_learner": [learner["hostname"]],
        "global_fragment_writer": [writer["hostname"]],
    }
    if set(
        role_map["mixed_version_learner"] + role_map["global_fragment_writer"]
    ) != set(hosts):
        raise AdoptionStressError("actual adoption roles do not match allocated hosts")
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
            "compute_hostname": socket.gethostname().split(".")[0],
            "workflow": "two-node-gpu-mixed-version-learner-filesystem-global-writer",
        },
        "roles": {"declared": role_map, "actual": role_map},
        "paths": {
            "project_root": args.project_root,
            "evidence_root": args.evidence_root,
        },
    }
    scheduler = {
        "job_id": args.job_id,
        "qtime_utc": args.qtime_utc,
        "queue": args.queue,
        "group": args.group,
        "nodefile_sha256": file_digest(Path(args.nodefile)),
        "modules": modules,
    }
    manifest = build_manifest(
        "S1-08", "L2", args.qtime_utc, "pbs_qtime", identities, scheduler=scheduler
    )
    _write_json_new(Path(args.output), manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fsbdd.auxiliary.stress.learner_adopt_stress"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("role", "summarize"):
        current = commands.add_parser(command)
        current.add_argument("--root", required=True, type=Path)
        current.add_argument("--result-root", required=True, type=Path)
        current.add_argument("--run-id", required=True)
        current.add_argument("--config-path", required=True, type=Path)
        current.add_argument("--config-identity", required=True)
        if command == "summarize":
            current.add_argument("--output", required=True, type=Path)
    manifest = commands.add_parser("manifest")
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
        "project-root",
        "evidence-root",
        "job-id",
        "qtime-utc",
        "queue",
        "group",
        "nodefile",
        "modules-file",
        "result-root",
        "output",
    ):
        manifest.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    if args.command == "role":
        value = run_role(
            root=args.root,
            result_root=args.result_root,
            run_id=args.run_id,
            config_path=args.config_path,
            config_identity=args.config_identity,
        )
        print(
            json.dumps(
                {"status": value["status"], "role": value["role"]}, sort_keys=True
            )
        )
        return 0
    if args.command == "summarize":
        value = summarize(
            root=args.root,
            result_root=args.result_root,
            run_id=args.run_id,
            config_path=args.config_path,
            config_identity=args.config_identity,
            output=args.output,
        )
        print(json.dumps({"status": value["status"]}, sort_keys=True))
        return 0
    write_manifest(args)
    print(json.dumps({"status": "building"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
