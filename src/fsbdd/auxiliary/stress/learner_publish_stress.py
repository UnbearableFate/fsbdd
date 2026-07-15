from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import threading
import time
from collections.abc import Iterator, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

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
from fsbdd.diloco.learner.publication import (
    AdoptedFragmentBase,
    FragmentPublicationSchedule,
    FragmentSnapshotCoordinator,
    build_fragment_descriptors,
    build_fragment_parameter_groups,
    serialize_fragment_parameters,
)
from fsbdd.auxiliary.contracts.manifest import build_manifest
from fsbdd.diloco.model.model_registry import build_logical_layer_registry
from fsbdd.diloco.protocol.proposal import Proposal, ProposalStore
from fsbdd.diloco.protocol.storage import PosixStorageBackend, PublicationNotFound


class SnapshotStressError(RuntimeError):
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
            raise SnapshotStressError(f"coordination record is malformed: {path}")
        return value


def _load_config(path: Path, expected_identity: str) -> dict[str, Any]:
    if file_digest(path) != expected_identity:
        raise SnapshotStressError("frozen snapshot profile identity mismatch")
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
            "publication",
        }
    ):
        raise SnapshotStressError("snapshot profile schema mismatch")
    model = value["model"]
    training = value["training"]
    publication = value["publication"]
    if (
        model.get("family") != "gpt_neox"
        or training.get("learner_id") != "learner-00"
        or training.get("precision") != "fp32"
        or publication.get("offset_algorithm")
        != "byte_weighted_midpoint_nearest_free_v1"
        or publication.get("maximum_pending_per_fragment") != 1
        or publication.get("maximum_in_flight_per_fragment") != 1
    ):
        raise SnapshotStressError("snapshot profile violates the frozen workload")
    return value


def _descriptor_from_dict(value: Mapping[str, Any]) -> FragmentStateDescriptor:
    return FragmentStateDescriptor(
        index=int(value["index"]),
        identity=str(value["identity"]),
        dtype=str(value["dtype"]),
        shape=tuple(int(item) for item in value["shape"]),
        parameter_identities=tuple(str(item) for item in value["parameter_identities"]),
    )


def _batches(
    *, count: int, batch_size: int, sequence_length: int, vocab_size: int
) -> Iterator[dict[str, Any]]:
    import torch

    base = torch.arange(batch_size * sequence_length).reshape(
        batch_size, sequence_length
    )
    for step in range(count):
        input_ids = (base + step) % vocab_size
        yield {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }


def _writer(
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
        raise SnapshotStressError("formal snapshot writer requires one CUDA GPU")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise SnapshotStressError("application distributed process group is forbidden")
    device = torch.device("cuda", 0)
    model_spec = profile["model"]
    training = profile["training"]
    publication = profile["publication"]
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
    backend_root = root / "backend"
    backend = PosixStorageBackend(backend_root)
    global_store = GlobalStateStore(
        backend,
        identities=identities,
        descriptors=descriptors,
        s_max=0,
    )
    bootstrap_fragments = tuple(
        BootstrapFragment(
            descriptor=descriptor,
            parameters=serialize_fragment_parameters(
                group, int(descriptor.shape[0]) * 4
            ),
            outer_state=b'{"optimizer":"none","version":0}',
        )
        for descriptor, group in zip(descriptors, groups, strict=True)
    )
    bootstrap = global_store.bootstrap(bootstrap_fragments)
    adopted_bases = tuple(
        AdoptedFragmentBase(state.version, state.content_identity)
        for state in bootstrap.snapshot.states
    )
    fragment_bytes = tuple(fragment.sync_bytes for fragment in fragment_map.fragments)
    schedule = FragmentPublicationSchedule.unified(
        fragment_bytes,
        int(publication["unified_h"]),
        learner_phase_offset=int(publication["learner_phase_offset"]),
    )
    if schedule.offset_algorithm != publication["offset_algorithm"]:
        raise SnapshotStressError("runtime offset algorithm differs from profile")
    full_model_bytes = sum(parameter.numel() for parameter in model.parameters()) * 4
    catalog = {
        "complete": True,
        "profile_id": profile["profile_id"],
        "config_path": str(config_path),
        "state_identities": identities.to_dict(),
        "descriptors": [descriptor.to_dict() for descriptor in descriptors],
        "fragment_bytes": list(fragment_bytes),
        "full_model_bytes": full_model_bytes,
        "base_states": [
            {
                "fragment_index": state.descriptor.index,
                "version": state.version,
                "content_identity": state.content_identity,
                "parameters_sha256": state.base_history[-1].parameters_sha256,
            }
            for state in bootstrap.snapshot.states
        ],
        "schedule": schedule.to_dict(),
    }
    coordination = root / "coordination"
    _replace_json(coordination / "catalog.json", catalog)

    proposal_store = ProposalStore(
        backend,
        identities=identities,
        descriptors=descriptors,
        learner_ids=(str(training["learner_id"]),),
        maximum_local_steps=int(publication["maximum_local_steps"]),
        maximum_processed_tokens=int(publication["maximum_processed_tokens"]),
    )
    slow_fragment = int(publication["slow_fragment_index"])
    slow_entered = threading.Event()
    slow_release = threading.Event()
    gate_lock = threading.Lock()
    gate_claimed = False

    def before_publish(proposal: Proposal) -> None:
        nonlocal gate_claimed
        if proposal.descriptor.index != slow_fragment:
            return
        with gate_lock:
            if gate_claimed:
                return
            gate_claimed = True
        slow_entered.set()
        if not slow_release.wait(float(publication["reader_timeout_seconds"])):
            raise TimeoutError("slow publication gate was not released")

    progress = LearnerProgress.initialize(
        str(training["learner_id"]),
        tuple(base.version for base in adopted_bases),
    )
    terminal_traces: list[dict[str, Any]] = []
    coordinator = FragmentSnapshotCoordinator(
        identities=identities,
        progress=progress,
        descriptors=descriptors,
        fragment_parameters=groups,
        adopted_bases=adopted_bases,
        schedule=schedule,
        store=proposal_store,
        before_publish=before_publish,
        trace_sink=lambda trace: terminal_traces.append(dict(trace)),
    )
    runtime = LearnerRuntime(
        model=model,
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=float(training["optimizer"]["lr"]),
            weight_decay=float(training["optimizer"]["weight_decay"]),
        ),
        scheduler=ConstantStepScheduler(),
        progress=progress,
        rng=LearnerRng.initialize(progress.learner_id, seed, device),
        fragment_parameters=groups,
        device=device,
        precision=str(training["precision"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_grad_norm=float(training["optimizer"]["grad_clip_norm"]),
        comparison={"claim": "S1-07 protocol semantics only"},
        safe_boundary_observers=(coordinator.on_safe_boundary,),
    )
    training_started_ns = time.monotonic_ns()
    run = runtime.run(
        _batches(
            count=int(training["optimizer_steps"]),
            batch_size=int(training["batch_size"]),
            sequence_length=int(training["sequence_length"]),
            vocab_size=int(model_spec["vocab_size"]),
        ),
        optimizer_steps=int(training["optimizer_steps"]),
    )
    training_returned_ns = time.monotonic_ns()
    before_release = coordinator.summary()
    if not slow_entered.is_set():
        raise SnapshotStressError(
            "slow fragment publication did not begin before training returned"
        )
    if before_release["publication"]["per_fragment_in_flight"][slow_fragment] != 1:
        raise SnapshotStressError("slow fragment is not in flight at training return")
    slow_released_ns = time.monotonic_ns()
    slow_release.set()
    coordinator.drain(float(publication["reader_timeout_seconds"]))
    final = coordinator.summary()
    coordinator.close(float(publication["reader_timeout_seconds"]))
    latest = tuple(
        proposal_store.load_latest(progress.learner_id, descriptor.index)
        for descriptor in descriptors
    )
    writer = {
        "schema_version": 1,
        "status": "pass",
        "role": "snapshot_writer",
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
            "full_model_bytes": full_model_bytes,
        },
        "bootstrap": catalog["base_states"],
        "schedule": schedule.to_dict(),
        "training": {
            "started_monotonic_ns": training_started_ns,
            "returned_monotonic_ns": training_returned_ns,
            "optimizer_steps": run.optimizer_steps_completed,
            "processed_input_tokens": run.processed_input_tokens,
            "finite_loss": math.isfinite(run.token_weighted_loss),
            "token_weighted_loss": run.token_weighted_loss,
            "distributed_initialized": run.distributed_initialized,
            "safe_boundary_steps": [event.local_optimizer_step for event in run.events],
            "active_snapshot_metrics": [
                dict(event.active_metrics) for event in run.events
            ],
        },
        "slow_publication": {
            "fragment_index": slow_fragment,
            "entered_before_training_return": True,
            "training_returned_before_release": training_returned_ns
            <= slow_released_ns,
            "release_monotonic_ns": slow_released_ns,
            "state_at_training_return": before_release["publication"],
        },
        "publication": final["publication"],
        "publication_terminal_traces": sorted(
            terminal_traces,
            key=lambda trace: (
                int(trace["fragment_index"]),
                int(trace["sequence"]),
            ),
        ),
        "final_progress": progress.to_dict(),
        "final_latest": [
            {
                "fragment_index": proposal.descriptor.index,
                "sequence": proposal.sequence,
                "base_version": proposal.base_version,
                "base_content_identity": proposal.base_content_identity,
                "local_steps": proposal.local_steps,
                "processed_tokens": proposal.processed_tokens,
                "snapshot_local_step": proposal.snapshot_local_step,
                "payload_bytes": proposal.payload_bytes,
                "parameters_sha256": proposal.parameters_sha256,
                "content_identity": proposal.content_identity,
            }
            for proposal in latest
        ],
        "writer_waited_for_reader_or_syncer": False,
    }
    _write_json_new(result_root / "snapshot_writer.json", writer)
    _replace_json(coordination / "writer-complete.json", {"complete": True})
    return writer


def _reader(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_identity: str,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    publication = profile["publication"]
    timeout = float(publication["reader_timeout_seconds"])
    coordination = root / "coordination"
    catalog = _wait(coordination / "catalog.json", timeout)
    identities = GlobalStateIdentities(**catalog["state_identities"])
    if (
        identities.run_identity != run_id
        or identities.config_identity != config_identity
    ):
        raise SnapshotStressError("reader catalog identity mismatch")
    descriptors = tuple(
        _descriptor_from_dict(value) for value in catalog["descriptors"]
    )
    store = ProposalStore(
        PosixStorageBackend(root / "backend"),
        identities=identities,
        descriptors=descriptors,
        learner_ids=(str(profile["training"]["learner_id"]),),
        maximum_local_steps=int(publication["maximum_local_steps"]),
        maximum_processed_tokens=int(publication["maximum_processed_tokens"]),
    )
    bases = {int(value["fragment_index"]): value for value in catalog["base_states"]}
    observed: dict[int, list[int]] = {
        descriptor.index: [] for descriptor in descriptors
    }
    polls = 0
    deadline = time.monotonic() + timeout
    while not (coordination / "writer-complete.json").exists():
        polls += 1
        for descriptor in descriptors:
            try:
                proposal = store.load_latest(
                    str(profile["training"]["learner_id"]), descriptor.index
                )
            except PublicationNotFound:
                continue
            if (
                not observed[descriptor.index]
                or observed[descriptor.index][-1] != proposal.sequence
            ):
                if (
                    observed[descriptor.index]
                    and proposal.sequence < observed[descriptor.index][-1]
                ):
                    raise SnapshotStressError(
                        "reader observed a regressing latest sequence"
                    )
                observed[descriptor.index].append(proposal.sequence)
            if proposal.payload_bytes != descriptor.shape[0] * 4:
                raise SnapshotStressError("reader observed wrong target-fragment bytes")
            if (
                proposal.base_version != bases[descriptor.index]["version"]
                or proposal.base_content_identity
                != bases[descriptor.index]["content_identity"]
            ):
                raise SnapshotStressError("reader observed an invented base identity")
        if time.monotonic() >= deadline:
            raise TimeoutError("reader timed out before writer completion")
        time.sleep(0.005)
    _wait(coordination / "writer-complete.json", timeout)
    final = tuple(
        store.load_latest(str(profile["training"]["learner_id"]), descriptor.index)
        for descriptor in descriptors
    )
    reader = {
        "schema_version": 1,
        "status": "pass",
        "role": "proposal_reader",
        "hostname": socket.gethostname().split(".")[0],
        "rank": 1,
        "state_identities": identities.to_dict(),
        "polls": polls,
        "observed_sequences": {
            str(index): values for index, values in observed.items()
        },
        "monotonic_latest": all(
            values == sorted(set(values)) for values in observed.values()
        ),
        "partial_or_corrupt_proposals_observed": 0,
        "final_latest": [
            {
                "fragment_index": proposal.descriptor.index,
                "sequence": proposal.sequence,
                "base_version": proposal.base_version,
                "base_content_identity": proposal.base_content_identity,
                "local_steps": proposal.local_steps,
                "processed_tokens": proposal.processed_tokens,
                "snapshot_local_step": proposal.snapshot_local_step,
                "payload_bytes": proposal.payload_bytes,
                "parameters_sha256": proposal.parameters_sha256,
                "content_identity": proposal.content_identity,
            }
            for proposal in final
        ],
        "application_reads_only_shared_filesystem": True,
    }
    _write_json_new(result_root / "proposal_reader.json", reader)
    return reader


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
        raise SnapshotStressError("snapshot stress requires an MPI launcher") from error
    if size != 2 or rank not in (0, 1):
        raise SnapshotStressError("snapshot stress requires exactly two ranks")
    profile = _load_config(config_path, config_identity)
    if rank == 0:
        return _writer(
            root=root,
            result_root=result_root,
            run_id=run_id,
            config_path=config_path,
            config_identity=config_identity,
            profile=profile,
        )
    return _reader(
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
    writer = json.loads(
        (result_root / "snapshot_writer.json").read_text(encoding="utf-8")
    )
    reader = json.loads(
        (result_root / "proposal_reader.json").read_text(encoding="utf-8")
    )
    if (
        writer.get("status") != "pass"
        or writer.get("role") != "snapshot_writer"
        or writer.get("rank") != 0
        or reader.get("status") != "pass"
        or reader.get("role") != "proposal_reader"
        or reader.get("rank") != 1
    ):
        raise SnapshotStressError("formal role status rank or identity is invalid")
    if writer["hostname"] == reader["hostname"]:
        raise SnapshotStressError("formal roles did not use distinct hosts")
    if writer["state_identities"] != reader["state_identities"]:
        raise SnapshotStressError("formal role state identities differ")
    if (
        writer["state_identities"]["run_identity"] != run_id
        or writer["state_identities"]["config_identity"] != config_identity
    ):
        raise SnapshotStressError("formal run or config identity differs")
    schedule = writer["schedule"]
    publication = writer["publication"]
    traces = writer["publication_terminal_traces"]
    latest_writer = writer["final_latest"]
    latest_reader = reader["final_latest"]
    fragment_count = int(profile["training"]["fragment_count"])
    h = int(profile["publication"]["unified_h"])
    fragment_bytes = tuple(int(value) for value in schedule["fragment_bytes"])
    expected_schedule = FragmentPublicationSchedule.unified(
        fragment_bytes,
        h,
        learner_phase_offset=int(profile["publication"]["learner_phase_offset"]),
    ).to_dict()
    if schedule != expected_schedule or len(set(schedule["offsets"])) != fragment_count:
        raise SnapshotStressError("formal byte-aware offset derivation failed")
    exact_rate = sum(
        (Fraction(value, h) for value in fragment_bytes),
        start=Fraction(0, 1),
    )
    budget = schedule["frequency_budget"]
    if (
        budget["exact_numerator"] != exact_rate.numerator
        or budget["exact_denominator"] != exact_rate.denominator
        or not math.isclose(
            float(budget["bytes_per_local_step"]),
            float(exact_rate),
            rel_tol=0,
            abs_tol=1e-9,
        )
        or budget["maximum_due_bytes"] != max(fragment_bytes)
    ):
        raise SnapshotStressError("formal PROG-06 byte-rate arithmetic failed")
    expected_steps = int(profile["training"]["optimizer_steps"])
    tokens_per_step = int(profile["training"]["batch_size"]) * int(
        profile["training"]["sequence_length"]
    )
    training = writer["training"]
    if (
        training["optimizer_steps"] != expected_steps
        or training["safe_boundary_steps"] != list(range(1, expected_steps + 1))
        or training["processed_input_tokens"] != expected_steps * tokens_per_step
        or not training["finite_loss"]
        or not math.isfinite(float(training["token_weighted_loss"]))
        or training["distributed_initialized"]
    ):
        raise SnapshotStressError("formal safe-boundary training trace failed")
    if publication["snapshot_replacement_count"] < 1:
        raise SnapshotStressError("formal slow writer did not exercise replacement")
    if (
        any(value > 1 for value in publication["maximum_pending_per_fragment"])
        or any(value > 1 for value in publication["maximum_in_flight_per_fragment"])
        or max(publication["maximum_in_flight_per_fragment"]) != 1
        or publication["resident_trace_records"]
        > publication["resident_trace_record_bound"]
        or publication["resident_trace_record_bound"] != 3 * fragment_count
    ):
        raise SnapshotStressError("formal backpressure bounds failed")
    if (
        publication["pending_upload_count"] != 0
        or publication["in_flight_publication_count"] != 0
        or publication["terminal_emits_in_progress"] != 0
        or publication["errors"]
        or publication["captured_snapshot_count"] != len(traces)
        or sum(publication["terminal_outcome_counts"].values()) != len(traces)
    ):
        raise SnapshotStressError("formal terminal publication accounting failed")
    if latest_writer != latest_reader:
        raise SnapshotStressError("remote reader final latest differs from writer")
    if (
        reader["partial_or_corrupt_proposals_observed"] != 0
        or not reader["monotonic_latest"]
        or not reader["application_reads_only_shared_filesystem"]
    ):
        raise SnapshotStressError("remote reader observed partial proposal data")
    bases = {value["fragment_index"]: value for value in writer["bootstrap"]}
    if set(bases) != set(range(fragment_count)):
        raise SnapshotStressError("bootstrap base catalog is incomplete")
    allowed_outcomes = {"published", "replaced_before_publish"}
    trace_keys: set[tuple[int, int]] = set()
    for trace in traces:
        index = int(trace["fragment_index"])
        sequence = int(trace["sequence"])
        key = (index, sequence)
        if (
            index not in bases
            or key in trace_keys
            or trace["outcome"] not in allowed_outcomes
            or trace["base_version"] != bases[index]["version"]
            or trace["base_content_identity"] != bases[index]["content_identity"]
            or trace["payload_bytes"] != fragment_bytes[index]
            or trace["payload_bytes"] >= writer["model"]["full_model_bytes"]
            or trace["local_steps"] != trace["snapshot_local_step"]
            or trace["processed_tokens"]
            != trace["snapshot_local_step"] * tokens_per_step
            or (trace["snapshot_local_step"] - schedule["offsets"][index]) % h != 0
            or len(trace["parameters_sha256"]) != 64
            or len(trace["proposal_content_identity"]) != 64
            or trace["staging_started_monotonic_ns"]
            < trace["safe_boundary_monotonic_ns"]
            or trace["staging_completed_monotonic_ns"]
            < trace["staging_started_monotonic_ns"]
            or trace["gpu_to_cpu_seconds"] < 0
        ):
            raise SnapshotStressError("formal capture trace identity/byte chain failed")
        trace_keys.add(key)
        if trace["outcome"] == "published":
            if (
                trace["publication_started_monotonic_ns"] is None
                or trace["publication_completed_monotonic_ns"]
                < trace["publication_started_monotonic_ns"]
                or trace["cpu_to_fs_seconds"] is None
                or trace["cpu_to_fs_seconds"] < 0
            ):
                raise SnapshotStressError("formal published transfer timing failed")
        elif (
            trace["publication_started_monotonic_ns"] is not None
            or trace["cpu_to_fs_seconds"] is not None
        ):
            raise SnapshotStressError("replaced snapshot was incorrectly started")
    if trace_keys != {
        (index, sequence)
        for index in range(fragment_count)
        for sequence in range(1, expected_steps // h + 1)
    }:
        raise SnapshotStressError("formal due snapshot sequence coverage is incomplete")
    published = [trace for trace in traces if trace["outcome"] == "published"]
    if (
        publication["published_snapshot_count"] != len(published)
        or publication["published_payload_bytes"]
        != sum(int(trace["payload_bytes"]) for trace in published)
        or not math.isclose(
            float(publication["cpu_to_fs_seconds"]),
            sum(float(trace["cpu_to_fs_seconds"]) for trace in published),
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(publication["gpu_to_cpu_seconds"]),
            sum(float(trace["gpu_to_cpu_seconds"]) for trace in traces),
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise SnapshotStressError("formal publication totals do not reconcile")
    slow = writer["slow_publication"]
    slow_index = int(slow["fragment_index"])
    slow_published = sorted(
        (trace for trace in published if int(trace["fragment_index"]) == slow_index),
        key=lambda trace: int(trace["sequence"]),
    )
    if not slow_published:
        raise SnapshotStressError("slow fragment produced no publication")
    first_slow = slow_published[0]
    if (
        not slow["entered_before_training_return"]
        or not slow["training_returned_before_release"]
        or writer["writer_waited_for_reader_or_syncer"]
        or first_slow["publication_started_monotonic_ns"]
        >= training["returned_monotonic_ns"]
        or training["returned_monotonic_ns"] > slow["release_monotonic_ns"]
        or slow["release_monotonic_ns"]
        > first_slow["publication_completed_monotonic_ns"]
        or slow["state_at_training_return"]["per_fragment_in_flight"][slow_index] != 1
    ):
        raise SnapshotStressError("formal learner waited for publication or reader")
    latest_by_fragment = {value["fragment_index"]: value for value in latest_reader}
    if set(latest_by_fragment) != set(range(fragment_count)):
        raise SnapshotStressError("remote final latest set is incomplete")
    trace_chains: list[dict[str, Any]] = []
    for index in range(fragment_count):
        final = latest_by_fragment[index]
        candidates = [
            trace
            for trace in published
            if trace["fragment_index"] == index
            and trace["sequence"] == final["sequence"]
        ]
        if len(candidates) != 1:
            raise SnapshotStressError("remote latest lacks one capture trace")
        trace = candidates[0]
        expected_fields = {
            "sequence": trace["sequence"],
            "base_version": trace["base_version"],
            "base_content_identity": trace["base_content_identity"],
            "local_steps": trace["local_steps"],
            "processed_tokens": trace["processed_tokens"],
            "snapshot_local_step": trace["snapshot_local_step"],
            "payload_bytes": trace["payload_bytes"],
            "parameters_sha256": trace["parameters_sha256"],
            "content_identity": trace["proposal_content_identity"],
        }
        if any(final[field] != value for field, value in expected_fields.items()):
            raise SnapshotStressError("capture writer reader final chain differs")
        if final["sequence"] != max(
            item["sequence"] for item in published if item["fragment_index"] == index
        ):
            raise SnapshotStressError(
                "remote final is not the newest published sequence"
            )
        trace_chains.append(
            {
                "fragment_index": index,
                "fragment_identity": trace["fragment_identity"],
                **expected_fields,
                "bootstrap_base_matches": True,
                "remote_reader_matches": True,
            }
        )
    all_published_steps_due = all(
        (trace["snapshot_local_step"] - schedule["offsets"][trace["fragment_index"]])
        % h
        == 0
        for trace in published
    )
    all_payloads_target_only = all(
        trace["payload_bytes"] == fragment_bytes[trace["fragment_index"]]
        and trace["payload_bytes"] < writer["model"]["full_model_bytes"]
        for trace in traces
    )
    if not all_published_steps_due or not all_payloads_target_only:
        raise SnapshotStressError("formal due-step or target-only predicate failed")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "run_id": run_id,
        "config_identity": config_identity,
        "hosts": [writer["hostname"], reader["hostname"]],
        "state_identities": writer["state_identities"],
        "schedule": schedule,
        "safe_boundary": {
            "optimizer_steps": writer["training"]["optimizer_steps"],
            "steps": writer["training"]["safe_boundary_steps"],
            "all_published_steps_due": all_published_steps_due,
        },
        "backpressure": {
            "pending_bound": 1,
            "in_flight_bound": 1,
            "maximum_pending_per_fragment": publication["maximum_pending_per_fragment"],
            "maximum_in_flight_per_fragment": publication[
                "maximum_in_flight_per_fragment"
            ],
            "replacement_count": publication["snapshot_replacement_count"],
            "skip_count": publication["snapshot_skip_count"],
            "training_returned_before_slow_release": slow[
                "training_returned_before_release"
            ],
        },
        "byte_accounting": {
            "fragment_bytes": fragment_bytes,
            "full_model_bytes": writer["model"]["full_model_bytes"],
            "all_payloads_target_only": all_payloads_target_only,
            "frequency_budget": schedule["frequency_budget"],
        },
        "publication": publication,
        "publication_terminal_traces": traces,
        "remote_reader": {
            "hostname": reader["hostname"],
            "polls": reader["polls"],
            "observed_sequences": reader["observed_sequences"],
            "monotonic_latest": reader["monotonic_latest"],
            "partial_or_corrupt_proposals_observed": 0,
        },
        "trace_chains": trace_chains,
        "latest": latest_reader,
    }
    _write_json_new(output, summary)
    shutil.rmtree(root)
    return summary


def write_manifest(args: argparse.Namespace) -> None:
    hosts = sorted(set(Path(args.nodefile).read_text(encoding="utf-8").splitlines()))
    if len(hosts) != 2:
        raise SnapshotStressError(f"manifest requires two distinct hosts, got {hosts}")
    modules = [
        line.strip()
        for line in Path(args.modules_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    writer = json.loads(
        (Path(args.result_root) / "snapshot_writer.json").read_text(encoding="utf-8")
    )
    reader = json.loads(
        (Path(args.result_root) / "proposal_reader.json").read_text(encoding="utf-8")
    )
    if (
        writer.get("status") != "pass"
        or writer.get("role") != "snapshot_writer"
        or writer.get("rank") != 0
        or reader.get("status") != "pass"
        or reader.get("role") != "proposal_reader"
        or reader.get("rank") != 1
        or writer.get("hostname") == reader.get("hostname")
    ):
        raise SnapshotStressError("manifest role status rank or host is invalid")
    expected_run_id = f"s1-07-{args.job_id}"
    if args.run_id != expected_run_id:
        raise SnapshotStressError("manifest run identity is not bound to PBS job id")
    expected_state_prefix = {
        "run_identity": args.run_id,
        "config_identity": args.config_sha256,
    }
    if (
        writer.get("state_identities") != reader.get("state_identities")
        or any(
            writer["state_identities"].get(field) != value
            for field, value in expected_state_prefix.items()
        )
        or writer.get("torch", {}).get("distributed_initialized") is not False
        or writer.get("writer_waited_for_reader_or_syncer") is not False
    ):
        raise SnapshotStressError("manifest role state/config/run identity is invalid")
    if args.queue != "debug-g" or args.group != "xg24i002":
        raise SnapshotStressError(
            "manifest scheduler queue or group differs from contract"
        )
    role_map = {
        "snapshot_writer": [writer["hostname"]],
        "proposal_reader": [reader["hostname"]],
    }
    if set(role_map["snapshot_writer"] + role_map["proposal_reader"]) != set(hosts):
        raise SnapshotStressError("actual snapshot roles do not match allocated hosts")
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
            "workflow": "two-node-gpu-snapshot-writer-filesystem-reader",
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
        "S1-07",
        "L2",
        args.qtime_utc,
        "pbs_qtime",
        identities,
        scheduler=scheduler,
    )
    _write_json_new(Path(args.output), manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fsbdd.auxiliary.stress.learner_publish_stress"
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
    print(json.dumps({"status": "building", "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
