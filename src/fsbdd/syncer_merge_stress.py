from __future__ import annotations

import argparse
import contextlib
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

from fsbdd_stage0.oracle import (
    OuterSGDState,
    inverse_staleness_weights,
    outer_sgd_step,
    weighted_direct_merge,
)

from .global_state import FragmentStateDescriptor
from .identity import canonical_digest, file_digest
from .manifest import build_manifest
from .syncer_merge import (
    ContributionFact,
    FragmentMergeRequest,
    FragmentOuterState,
    ImmutableFileBaseSource,
    ImmutableFileContributionSource,
    OuterSGDPolicy,
    PayloadLocation,
    execute_streaming_fragment_update,
    linux_process_memory_bytes,
)


class MergeStressError(RuntimeError):
    pass


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
            raise MergeStressError(f"coordination record is malformed: {path}")
        return value


def _load_config(path: Path, expected_identity: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_identity:
        raise MergeStressError("merge config identity mismatch")
    value = json.loads(raw)
    expected = {
        "schema_version",
        "accumulation_dtype",
        "merge_policy",
        "outer_optimizer",
        "direct_average_control",
        "formal_workload",
        "numeric_gates",
        "memory_gates",
        "coordination_timeout_seconds",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise MergeStressError("merge config schema mismatch")
    if (
        value["schema_version"] != 1
        or value["accumulation_dtype"] != "float32"
        or value["merge_policy"] != "direct_weighted_average"
    ):
        raise MergeStressError("merge config freezes schema one fp32 direct averaging")
    if set(value["outer_optimizer"]) != {"learning_rate", "momentum", "nesterov"}:
        raise MergeStressError("outer optimizer config schema mismatch")
    if set(value["direct_average_control"]) != {
        "learning_rate",
        "momentum",
        "nesterov",
    }:
        raise MergeStressError("direct control config schema mismatch")
    if set(value["formal_workload"]) != {
        "learner_count",
        "fragment_count",
        "fragment_elements",
        "reference_updates",
        "order_permutations",
    }:
        raise MergeStressError("formal workload config schema mismatch")
    if set(value["numeric_gates"]) != {
        "single_update_relative_l2_max",
        "fifty_update_relative_l2_max",
        "order_relative_l2_max",
        "late_window_amplification_floor",
    }:
        raise MergeStressError("numeric gate config schema mismatch")
    if set(value["memory_gates"]) != {
        "maximum_live_local_payloads",
        "maximum_tensor_fragment_multiples",
        "maximum_m8_minus_m1_peak_rss_bytes",
    }:
        raise MergeStressError("memory gate config schema mismatch")
    workload = value["formal_workload"]
    if (
        workload["learner_count"] != 8
        or workload["fragment_count"] != 4
        or workload["reference_updates"] != 50
        or workload["order_permutations"] != 16
        or workload["fragment_elements"] <= 0
    ):
        raise MergeStressError("formal workload differs from the frozen profile")
    return value


def _payload(values: Sequence[float]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _values(payload: bytes) -> list[float]:
    return np.frombuffer(payload, dtype="<f4").astype(np.float64).tolist()


def _relative_l2(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(float(np.linalg.norm(right_array)), 1e-12)
    )


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _descriptor(elements: int) -> FragmentStateDescriptor:
    return FragmentStateDescriptor(
        index=0,
        identity=_identity("s1-10-formal-fragment-0"),
        dtype="float32",
        shape=(elements,),
        parameter_identities=("formal-fragment-flat-fp32",),
    )


def _fact(
    *,
    proposal_id: str,
    learner_id: str,
    local_payload: bytes,
    base_version: int,
    base_identity: str,
    current_version: int,
    tokens: int,
    weight: float,
) -> dict[str, Any]:
    return dataclasses.asdict(
        ContributionFact(
            proposal_id=proposal_id,
            content_identity=_identity(f"content:{proposal_id}"),
            learner_id=learner_id,
            base_version=base_version,
            base_content_identity=base_identity,
            processed_tokens=tokens,
            staleness=current_version - base_version,
            normalized_weight=weight,
            parameters_sha256=hashlib.sha256(local_payload).hexdigest(),
            payload_bytes=len(local_payload),
        )
    )


def _write_payload_new(root: Path, relative_path: str, payload: bytes) -> None:
    path = root / relative_path
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable workload: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_workload(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    learner_count = int(config["formal_workload"]["learner_count"])
    update_count = int(config["formal_workload"]["reference_updates"])
    small_elements = 64
    tokens = list(range(1, learner_count + 1))
    weights = inverse_staleness_weights(tokens, [0] * learner_count, 1.0)
    policy = config["outer_optimizer"]
    oracle_parameters = [
        struct_value
        for struct_value in np.linspace(-0.75, 0.75, small_elements, dtype=np.float32).tolist()
    ]
    oracle_state = OuterSGDState(None)
    numeric_updates: list[dict[str, Any]] = []
    for update in range(update_count):
        current = list(oracle_parameters)
        current_identity = _identity(f"numeric-current:{update}")
        bases: list[list[float]] = []
        locals_: list[list[float]] = []
        contributions: list[dict[str, Any]] = []
        for learner in range(learner_count):
            gradient = [
                float(
                    np.float32(
                        (((update + 1) * (learner + 1) * ((index % 7) - 3)) % 29 - 14)
                        * 0.0002
                    )
                )
                for index in range(small_elements)
            ]
            local = [
                float(np.float32(base - delta))
                for base, delta in zip(current, gradient, strict=True)
            ]
            local_payload = _payload(local)
            proposal_id = f"numeric-u{update:03d}-learner-{learner:02d}"
            relative_path = f"payloads/numeric/u{update:03d}/learner-{learner:02d}.f32"
            _write_payload_new(root, relative_path, local_payload)
            contributions.append(
                {
                    "fact": _fact(
                        proposal_id=proposal_id,
                        learner_id=f"learner-{learner:02d}",
                        local_payload=local_payload,
                        base_version=update,
                        base_identity=current_identity,
                        current_version=update,
                        tokens=tokens[learner],
                        weight=weights[learner],
                    ),
                    "relative_path": relative_path,
                    "local": local,
                }
            )
            bases.append(current)
            locals_.append(local)
        merged = weighted_direct_merge(
            current=current, bases=bases, locals_=locals_, weights=weights
        )
        oracle_parameters, oracle_state = outer_sgd_step(
            current,
            merged,
            oracle_state,
            learning_rate=policy["learning_rate"],
            momentum=policy["momentum"],
            nesterov=policy["nesterov"],
        )
        numeric_updates.append(
            {
                "update": update,
                "current": current,
                "current_identity": current_identity,
                "selection_identity": _identity(f"numeric-selection:{update}"),
                "weights": weights,
                "contributions": contributions,
                "expected_merged_gradient": merged,
                "expected_parameters": oracle_parameters,
                "expected_momentum_buffer": oracle_state.momentum_buffer,
            }
        )

    old_base = [3.0, 5.0]
    mixed_current = [12.0, -2.0]
    mixed_locals = [[1.0, 1.0], [8.0, -4.0]]
    mixed_weights = [0.5, 0.5]
    mixed_contributions = []
    mixed_current_identity = _identity("mixed-current")
    mixed_old_identity = _identity("mixed-old")
    for index, local in enumerate(mixed_locals):
        raw = _payload(local)
        proposal_id = f"mixed-{index}"
        relative_path = f"payloads/mixed/local-{index}.f32"
        _write_payload_new(root, relative_path, raw)
        is_old = index == 0
        mixed_contributions.append(
            {
                "fact": _fact(
                    proposal_id=proposal_id,
                    learner_id=f"mixed-learner-{index}",
                    local_payload=raw,
                    base_version=0 if is_old else 1,
                    base_identity=mixed_old_identity if is_old else mixed_current_identity,
                    current_version=1,
                    tokens=1,
                    weight=mixed_weights[index],
                ),
                "relative_path": relative_path,
                "local": local,
            }
        )
    old_base_path = "payloads/mixed/old-base.f32"
    old_base_payload = _payload(old_base)
    _write_payload_new(root, old_base_path, old_base_payload)

    memory_elements = int(config["formal_workload"]["fragment_elements"])
    memory_current_array = np.linspace(-1.0, 1.0, memory_elements, dtype=np.float32)
    memory_current = memory_current_array.astype("<f4", copy=False).tobytes()
    memory_current_path = "payloads/memory/current.f32"
    _write_payload_new(root, memory_current_path, memory_current)
    memory_contributions = []
    for learner in range(learner_count):
        delta = np.float32((learner + 1) * 0.0001)
        local_array = np.subtract(memory_current_array, delta, dtype=np.float32)
        raw = local_array.astype("<f4", copy=False).tobytes()
        proposal_id = f"memory-{learner:02d}"
        relative_path = f"payloads/memory/learner-{learner:02d}.f32"
        _write_payload_new(root, relative_path, raw)
        memory_contributions.append(
            {
                "proposal_id": proposal_id,
                "learner_id": f"learner-{learner:02d}",
                "relative_path": relative_path,
                "parameters_sha256": hashlib.sha256(raw).hexdigest(),
                "payload_bytes": len(raw),
            }
        )

    workload = {
        "schema_version": 1,
        "fragment_map_identity": _identity("s1-10-formal-fragment-map"),
        "small_descriptor": _descriptor(small_elements).to_dict(),
        "memory_descriptor": _descriptor(memory_elements).to_dict(),
        "numeric": {
            "initial_parameters": numeric_updates[0]["current"],
            "updates": numeric_updates,
        },
        "mixed_base": {
            "current": mixed_current,
            "current_identity": mixed_current_identity,
            "old_base": old_base,
            "old_base_identity": mixed_old_identity,
            "old_base_relative_path": old_base_path,
            "old_base_parameters_sha256": hashlib.sha256(old_base_payload).hexdigest(),
            "selection_identity": _identity("mixed-selection"),
            "contributions": mixed_contributions,
        },
        "memory": {
            "current_relative_path": memory_current_path,
            "current_sha256": hashlib.sha256(memory_current).hexdigest(),
            "current_identity": _identity("memory-current"),
            "selection_identity": _identity("memory-selection"),
            "fragment_elements": memory_elements,
            "fragment_bytes": len(memory_current),
            "contributions": memory_contributions,
        },
    }
    return {**workload, "workload_identity": canonical_digest(workload)}


def _descriptor_from_dict(value: Mapping[str, Any]) -> FragmentStateDescriptor:
    return FragmentStateDescriptor(
        index=value["index"],
        identity=value["identity"],
        dtype=value["dtype"],
        shape=tuple(value["shape"]),
        parameter_identities=tuple(value["parameter_identities"]),
    )


def _result_values(raw: bytes) -> list[float]:
    return np.frombuffer(raw, dtype="<f4").astype(np.float64).tolist()


def execute_numeric_workload(root: Path, workload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = _descriptor_from_dict(workload["small_descriptor"])
    policy = OuterSGDPolicy(**config["outer_optimizer"])
    current = _payload(workload["numeric"]["initial_parameters"])
    outer_state = FragmentOuterState(0)
    traces = []
    update_identities: set[str] = set()
    maximum_source_active = 0
    maximum_tensor_multiple = 0.0
    total_local_bytes = 0
    for item in workload["numeric"]["updates"]:
        facts = tuple(ContributionFact(**entry["fact"]) for entry in item["contributions"])
        locations = tuple(
            PayloadLocation(entry["fact"]["proposal_id"], entry["relative_path"])
            for entry in item["contributions"]
        )
        source = ImmutableFileContributionSource(root, locations)
        request = FragmentMergeRequest(
            descriptor=descriptor,
            fragment_map_identity=workload["fragment_map_identity"],
            current_version=item["update"],
            current_content_identity=item["current_identity"],
            current_parameters=current,
            outer_state=outer_state,
            selection_identity=item["selection_identity"],
            contributions=facts,
        )
        result = execute_streaming_fragment_update(request, source, policy)
        production_parameters = _result_values(result.parameters)
        production_buffer = (
            None
            if result.outer_state.momentum_buffer is None
            else _result_values(result.outer_state.momentum_buffer)
        )
        error = _relative_l2(production_parameters, item["expected_parameters"])
        buffer_error = _relative_l2(
            production_buffer or [0.0] * len(production_parameters),
            item["expected_momentum_buffer"] or [0.0] * len(production_parameters),
        )
        metrics = source.metrics()
        maximum_source_active = max(maximum_source_active, metrics.maximum_active_payloads)
        maximum_tensor_multiple = max(
            maximum_tensor_multiple,
            result.memory_accounting.maximum_tensor_fragment_multiples,
        )
        total_local_bytes += result.byte_accounting.local_payload_bytes
        if result.update_identity in update_identities:
            raise MergeStressError("distinct updates produced a duplicate update identity")
        update_identities.add(result.update_identity)
        traces.append(
            {
                "update": item["update"],
                "production_parameters": production_parameters,
                "production_momentum_buffer": production_buffer,
                "production_merged_gradient": _result_values(result.merged_gradient),
                "relative_l2": error,
                "buffer_relative_l2": buffer_error,
                "update_identity": result.update_identity,
                "update_facts": result.update_facts,
                "byte_accounting": result.byte_accounting.to_dict(),
                "memory_accounting": result.memory_accounting.to_dict(),
                "source_metrics": dataclasses.asdict(metrics),
            }
        )
        current = result.parameters
        outer_state = result.outer_state
    errors = [item["relative_l2"] for item in traces]
    return {
        "update_count": len(traces),
        "traces": traces,
        "maximum_relative_l2": max(errors),
        "first_ten_maximum_relative_l2": max(errors[:10]),
        "last_ten_maximum_relative_l2": max(errors[-10:]),
        "maximum_buffer_relative_l2": max(item["buffer_relative_l2"] for item in traces),
        "distinct_update_identities": len(update_identities),
        "maximum_source_active_payloads": maximum_source_active,
        "maximum_tensor_fragment_multiples": maximum_tensor_multiple,
        "total_local_payload_bytes": total_local_bytes,
    }


def execute_order_workload(root: Path, workload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    first = workload["numeric"]["updates"][0]
    descriptor = _descriptor_from_dict(workload["small_descriptor"])
    count = int(config["formal_workload"]["order_permutations"])
    entries = first["contributions"]
    orders = []
    base_order = list(range(len(entries)))
    for offset in range(len(entries)):
        orders.append(base_order[offset:] + base_order[:offset])
        orders.append(list(reversed(base_order[offset:] + base_order[:offset])))
    outputs: list[list[float]] = []
    for order_index, order in enumerate(orders[:count]):
        selected = [entries[index] for index in order]
        facts = tuple(ContributionFact(**entry["fact"]) for entry in selected)
        source = ImmutableFileContributionSource(
            root,
            tuple(
                PayloadLocation(entry["fact"]["proposal_id"], entry["relative_path"])
                for entry in selected
            ),
        )
        request = FragmentMergeRequest(
            descriptor=descriptor,
            fragment_map_identity=workload["fragment_map_identity"],
            current_version=0,
            current_content_identity=first["current_identity"],
            current_parameters=_payload(first["current"]),
            outer_state=FragmentOuterState(0),
            selection_identity=_identity(f"order-selection:{order_index}"),
            contributions=facts,
        )
        result = execute_streaming_fragment_update(
            request, source, OuterSGDPolicy(**config["direct_average_control"])
        )
        outputs.append(_result_values(result.parameters))
    canonical = outputs[0]
    errors = [_relative_l2(value, canonical) for value in outputs]
    return {
        "permutation_count": len(outputs),
        "orders": orders[:count],
        "outputs": outputs,
        "maximum_relative_l2": max(errors),
        "relative_l2": errors,
        "output_sha256": [hashlib.sha256(_payload(value)).hexdigest() for value in outputs],
    }


def execute_mixed_base_workload(root: Path, workload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    mixed = workload["mixed_base"]
    facts = tuple(ContributionFact(**entry["fact"]) for entry in mixed["contributions"])
    source = ImmutableFileContributionSource(
        root,
        tuple(
            PayloadLocation(entry["fact"]["proposal_id"], entry["relative_path"])
            for entry in mixed["contributions"]
        ),
    )
    base_source = ImmutableFileBaseSource(
        root,
        (PayloadLocation(mixed["old_base_identity"], mixed["old_base_relative_path"]),),
        {
            mixed["old_base_identity"]: (
                0,
                mixed["old_base_identity"],
                mixed["old_base_parameters_sha256"],
            )
        },
    )
    request = FragmentMergeRequest(
        descriptor=_descriptor_from_dict(workload["small_descriptor"]),
        fragment_map_identity=workload["fragment_map_identity"],
        current_version=1,
        current_content_identity=mixed["current_identity"],
        current_parameters=_payload(mixed["current"]),
        outer_state=FragmentOuterState(1),
        selection_identity=mixed["selection_identity"],
        contributions=facts,
    )
    result = execute_streaming_fragment_update(
        request,
        source,
        OuterSGDPolicy(**config["direct_average_control"]),
        base_source=base_source,
    )
    expected_gradient = weighted_direct_merge(
        current=mixed["current"],
        bases=[mixed["old_base"], mixed["current"]],
        locals_=[entry["local"] for entry in mixed["contributions"]],
        weights=[entry["fact"]["normalized_weight"] for entry in mixed["contributions"]],
    )
    expected_parameters, _ = outer_sgd_step(
        mixed["current"],
        expected_gradient,
        OuterSGDState(None),
        learning_rate=1.0,
        momentum=0.0,
        nesterov=False,
    )
    wrong_gradient = weighted_direct_merge(
        current=mixed["current"],
        bases=[mixed["current"], mixed["current"]],
        locals_=[entry["local"] for entry in mixed["contributions"]],
        weights=[entry["fact"]["normalized_weight"] for entry in mixed["contributions"]],
    )
    wrong_parameters, _ = outer_sgd_step(
        mixed["current"],
        wrong_gradient,
        OuterSGDState(None),
        learning_rate=1.0,
        momentum=0.0,
        nesterov=False,
    )
    production = _result_values(result.parameters)
    return {
        "production_parameters": production,
        "expected_parameters": expected_parameters,
        "wrong_current_relative_parameters": wrong_parameters,
        "relative_l2": _relative_l2(production, expected_parameters),
        "wrong_relative_l2": _relative_l2(wrong_parameters, expected_parameters),
        "retained_base_reads": result.byte_accounting.retained_base_reads,
        "retained_base_bytes": result.byte_accounting.retained_base_bytes,
        "maximum_active_base_payloads": result.memory_accounting.maximum_active_base_payloads,
        "update_identity": result.update_identity,
        "update_facts": result.update_facts,
    }


class _MemorySource:
    def __init__(self, payloads: Mapping[str, bytes]) -> None:
        self.payloads = dict(payloads)

    @contextlib.contextmanager
    def open_payload(self, contribution: ContributionFact) -> Iterator[bytes]:
        yield self.payloads[contribution.proposal_id]


def _warm_memory_runtime() -> None:
    raw = _payload([1.0, 2.0])
    local = _payload([0.0, 1.0])
    fact = ContributionFact(
        proposal_id="warmup",
        content_identity=_identity("warmup-content"),
        learner_id="warmup-learner",
        base_version=0,
        base_content_identity=_identity("warmup-current"),
        processed_tokens=1,
        staleness=0,
        normalized_weight=1.0,
        parameters_sha256=hashlib.sha256(local).hexdigest(),
        payload_bytes=len(local),
    )
    request = FragmentMergeRequest(
        descriptor=_descriptor(2),
        fragment_map_identity=_identity("warmup-map"),
        current_version=0,
        current_content_identity=fact.base_content_identity,
        current_parameters=raw,
        outer_state=FragmentOuterState(0),
        selection_identity=_identity("warmup-selection"),
        contributions=(fact,),
    )
    execute_streaming_fragment_update(
        request, _MemorySource({"warmup": local}), OuterSGDPolicy(1.0, 0.0, False)
    )


def run_memory_child(root: Path, workload_path: Path, contributor_count: int) -> dict[str, Any]:
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    memory = workload["memory"]
    if contributor_count not in (1, 8):
        raise MergeStressError("memory child contributor count must be one or eight")
    current = (root / memory["current_relative_path"]).read_bytes()
    if hashlib.sha256(current).hexdigest() != memory["current_sha256"]:
        raise MergeStressError("memory current checksum mismatch")
    selected = memory["contributions"][:contributor_count]
    weight = float(np.float32(1.0 / contributor_count))
    facts = tuple(
        ContributionFact(
            proposal_id=item["proposal_id"],
            content_identity=_identity(f"memory-content:{item['proposal_id']}"),
            learner_id=item["learner_id"],
            base_version=0,
            base_content_identity=memory["current_identity"],
            processed_tokens=1,
            staleness=0,
            normalized_weight=weight,
            parameters_sha256=item["parameters_sha256"],
            payload_bytes=item["payload_bytes"],
        )
        for item in selected
    )
    source = ImmutableFileContributionSource(
        root,
        tuple(PayloadLocation(item["proposal_id"], item["relative_path"]) for item in selected),
    )
    request = FragmentMergeRequest(
        descriptor=_descriptor_from_dict(workload["memory_descriptor"]),
        fragment_map_identity=workload["fragment_map_identity"],
        current_version=0,
        current_content_identity=memory["current_identity"],
        current_parameters=current,
        outer_state=FragmentOuterState(0),
        selection_identity=_identity(f"memory-selection:{contributor_count}"),
        contributions=facts,
    )
    _warm_memory_runtime()
    before_rss, before_hwm = linux_process_memory_bytes()
    result = execute_streaming_fragment_update(
        request,
        source,
        OuterSGDPolicy(1.0, 0.0, False),
        measure_process_rss=True,
    )
    after_rss, after_hwm = linux_process_memory_bytes()
    metrics = source.metrics()
    measured_peak = result.memory_accounting.peak_process_rss_bytes
    if measured_peak is None:
        raise MergeStressError("memory child did not measure target-update RSS")
    return {
        "contributor_count": contributor_count,
        "before_rss_bytes": before_rss,
        "before_hwm_bytes": before_hwm,
        "after_rss_bytes": after_rss,
        "after_hwm_bytes": after_hwm,
        "target_update_peak_rss_bytes": measured_peak,
        "source_metrics": dataclasses.asdict(metrics),
        "memory_accounting": result.memory_accounting.to_dict(),
        "byte_accounting": result.byte_accounting.to_dict(),
        "parameters_sha256": hashlib.sha256(result.parameters).hexdigest(),
    }


def execute_memory_workload(root: Path, workload_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    runs = {}
    for count in (1, 8):
        output = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "fsbdd.syncer_merge_stress",
                "memory-child",
                "--root",
                str(root),
                "--workload-path",
                str(workload_path),
                "--contributors",
                str(count),
            ],
            text=True,
        )
        runs[str(count)] = json.loads(output)
    difference = runs["8"]["target_update_peak_rss_bytes"] - runs["1"][
        "target_update_peak_rss_bytes"
    ]
    return {
        "runs": runs,
        "m8_minus_m1_peak_rss_bytes": difference,
        "rss_gate_bytes": config["memory_gates"]["maximum_m8_minus_m1_peak_rss_bytes"],
    }


def _gpu_memory_for_current_process() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    total = 0
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            total += int(fields[1])
    return total


def run_role(
    *, root: Path, result_root: Path, run_id: str, config_path: Path, config_identity: str
) -> dict[str, Any]:
    try:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise MergeStressError("merge role requires an MPI launcher environment") from error
    if size != 2 or rank not in (0, 1):
        raise MergeStressError("merge stress requires exactly two ranks")
    config = _load_config(config_path, config_identity)
    timeout = float(config["coordination_timeout_seconds"])
    hostname = socket.gethostname().split(".")[0]
    gpu_before = _gpu_memory_for_current_process()
    workload_path = root / "coordination/workload.json"
    ready_path = root / "coordination/ready.json"
    if rank == 1:
        torch_before = "torch" in sys.modules
        workload = build_workload(root, config)
        _write_json_new(workload_path, workload)
        workload_sha = file_digest(workload_path)
        _replace_json(
            ready_path,
            {"complete": True, "workload_sha256": workload_sha, "workload_identity": workload["workload_identity"]},
        )
        role = {
            "schema_version": 1,
            "complete": True,
            "role": "proposal_writer",
            "rank": rank,
            "size": size,
            "hostname": hostname,
            "pid": os.getpid(),
            "run_id": run_id,
            "config_identity": config_identity,
            "workload_sha256": workload_sha,
            "workload_identity": workload["workload_identity"],
            "workload": workload,
            "torch_imported_before": torch_before,
            "torch_imported_after": "torch" in sys.modules,
            "gpu_memory_before": gpu_before,
            "gpu_memory_after": _gpu_memory_for_current_process(),
        }
    else:
        ready = _wait_json(ready_path, timeout)
        if file_digest(workload_path) != ready["workload_sha256"]:
            raise MergeStressError("published workload checksum mismatch")
        workload = json.loads(workload_path.read_text(encoding="utf-8"))
        if workload["workload_identity"] != ready["workload_identity"]:
            raise MergeStressError("published workload identity mismatch")
        torch_before = "torch" in sys.modules
        numeric = execute_numeric_workload(root, workload, config)
        order = execute_order_workload(root, workload, config)
        mixed = execute_mixed_base_workload(root, workload, config)
        memory = execute_memory_workload(root, workload_path, config)
        role = {
            "schema_version": 1,
            "complete": True,
            "role": "outer_syncer",
            "rank": rank,
            "size": size,
            "hostname": hostname,
            "pid": os.getpid(),
            "run_id": run_id,
            "config_identity": config_identity,
            "workload_sha256": ready["workload_sha256"],
            "workload_identity": workload["workload_identity"],
            "numeric": numeric,
            "order": order,
            "mixed_base": mixed,
            "memory": memory,
            "application_coordination": "filesystem_only",
            "mpi_usage": "launcher_only",
            "torch_imported_before": torch_before,
            "torch_imported_after": "torch" in sys.modules,
            "gpu_memory_before": gpu_before,
            "gpu_memory_after": _gpu_memory_for_current_process(),
        }
    _write_json_new(result_root / f"{role['role']}.json", role)
    return role


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise MergeStressError(f"{name} schema mismatch")


def analyze_roles(
    writer: Mapping[str, Any],
    syncer: Mapping[str, Any],
    config: Mapping[str, Any],
    expected_config_identity: str,
) -> dict[str, Any]:
    _require_exact_keys(
        writer,
        {
            "schema_version",
            "complete",
            "role",
            "rank",
            "size",
            "hostname",
            "pid",
            "run_id",
            "config_identity",
            "workload_sha256",
            "workload_identity",
            "workload",
            "torch_imported_before",
            "torch_imported_after",
            "gpu_memory_before",
            "gpu_memory_after",
        },
        "proposal writer role",
    )
    _require_exact_keys(
        syncer,
        {
            "schema_version",
            "complete",
            "role",
            "rank",
            "size",
            "hostname",
            "pid",
            "run_id",
            "config_identity",
            "workload_sha256",
            "workload_identity",
            "numeric",
            "order",
            "mixed_base",
            "memory",
            "application_coordination",
            "mpi_usage",
            "torch_imported_before",
            "torch_imported_after",
            "gpu_memory_before",
            "gpu_memory_after",
        },
        "outer syncer role",
    )
    if (
        writer["schema_version"] != 1
        or syncer["schema_version"] != 1
        or writer["complete"] is not True
        or syncer["complete"] is not True
        or writer["role"] != "proposal_writer"
        or syncer["role"] != "outer_syncer"
        or writer["rank"] != 1
        or syncer["rank"] != 0
        or writer["size"] != syncer["size"]
        or writer["size"] != 2
    ):
        raise MergeStressError("formal role identity mismatch")
    for field in ("run_id", "config_identity", "workload_sha256", "workload_identity"):
        if writer[field] != syncer[field]:
            raise MergeStressError(f"formal roles disagree on {field}")
    if writer["config_identity"] != expected_config_identity:
        raise MergeStressError("formal role config identity differs from frozen config")
    if writer["hostname"] == syncer["hostname"]:
        raise MergeStressError("formal roles must run on distinct hosts")
    if writer["torch_imported_before"] or writer["torch_imported_after"]:
        raise MergeStressError("proposal writer imported Torch")
    if not syncer["torch_imported_after"]:
        raise MergeStressError("outer syncer did not execute Torch tensor math")
    if any(
        value != 0
        for value in (
            writer["gpu_memory_before"],
            writer["gpu_memory_after"],
            syncer["gpu_memory_before"],
            syncer["gpu_memory_after"],
        )
    ):
        raise MergeStressError("application role used GPU memory")
    if (
        syncer["application_coordination"] != "filesystem_only"
        or syncer["mpi_usage"] != "launcher_only"
    ):
        raise MergeStressError("application data plane is not filesystem-only")
    workload = writer["workload"]
    semantic = {key: value for key, value in workload.items() if key != "workload_identity"}
    if canonical_digest(semantic) != workload["workload_identity"]:
        raise MergeStressError("packaged workload identity mismatch")

    numeric = syncer["numeric"]
    updates = workload["numeric"]["updates"]
    traces = numeric["traces"]
    expected_updates = config["formal_workload"]["reference_updates"]
    if len(updates) != len(traces) or len(traces) != expected_updates:
        raise MergeStressError("numeric trace does not contain exactly 50 updates")
    state = OuterSGDState(None)
    expected_current = workload["numeric"]["initial_parameters"]
    production_current = workload["numeric"]["initial_parameters"]
    production_buffer: list[float] | None = None
    recomputed_errors = []
    update_ids: set[str] = set()
    for position, (source, trace) in enumerate(zip(updates, traces, strict=True)):
        if source["update"] != trace["update"] or source["update"] != position:
            raise MergeStressError("numeric update ordering mismatch")
        if _relative_l2(source["current"], expected_current) > 1e-12:
            raise MergeStressError("writer numeric current diverges from independent oracle")
        weights = [entry["fact"]["normalized_weight"] for entry in source["contributions"]]
        bases = [source["current"] for _ in source["contributions"]]
        locals_ = [entry["local"] for entry in source["contributions"]]
        merged = weighted_direct_merge(
            current=source["current"], bases=bases, locals_=locals_, weights=weights
        )
        expected_current, state = outer_sgd_step(
            source["current"],
            merged,
            state,
            learning_rate=config["outer_optimizer"]["learning_rate"],
            momentum=config["outer_optimizer"]["momentum"],
            nesterov=config["outer_optimizer"]["nesterov"],
        )
        error = _relative_l2(trace["production_parameters"], expected_current)
        buffer_error = _relative_l2(
            trace["production_momentum_buffer"], state.momentum_buffer
        )
        if not math.isclose(error, trace["relative_l2"], rel_tol=0.0, abs_tol=1e-15):
            raise MergeStressError("reported parameter error is not independently reproducible")
        if not math.isclose(
            buffer_error, trace["buffer_relative_l2"], rel_tol=0.0, abs_tol=1e-15
        ):
            raise MergeStressError("reported buffer error is not independently reproducible")
        if _relative_l2(trace["production_merged_gradient"], merged) > 1e-6:
            raise MergeStressError("production merged gradient differs from direct oracle")
        facts = trace["update_facts"]
        _require_exact_keys(
            facts,
            {
                "schema_version",
                "current_version",
                "current_content_identity",
                "current_parameters_sha256",
                "current_outer_state",
                "selection_identity",
                "ordered_proposal_identities",
                "ordered_proposal_content_identities",
                "ordered_base_versions",
                "ordered_base_content_identities",
                "ordered_processed_tokens",
                "ordered_staleness",
                "ordered_float32_weights",
                "ordered_local_parameters_sha256",
                "merge_policy",
                "outer_optimizer_policy",
                "outer_hyperparameters",
                "accumulation_dtype",
                "fragment_map_identity",
                "fragment_identity",
                "fragment_index",
            },
            "update identity facts",
        )
        _require_exact_keys(
            facts["current_outer_state"],
            {"update_count", "momentum_buffer_sha256"},
            "current outer state identity",
        )
        _require_exact_keys(
            facts["merge_policy"], {"name", "formula"}, "merge policy identity"
        )
        _require_exact_keys(
            facts["outer_hyperparameters"],
            {"learning_rate", "momentum", "nesterov"},
            "outer hyperparameters identity",
        )
        if canonical_digest(facts) != trace["update_identity"]:
            raise MergeStressError("update identity does not bind its packaged facts")
        source_facts = [entry["fact"] for entry in source["contributions"]]
        descriptor = workload["small_descriptor"]
        expected_outer_buffer_sha = (
            None
            if production_buffer is None
            else hashlib.sha256(_payload(production_buffer)).hexdigest()
        )
        if (
            facts["schema_version"] != 1
            or facts["current_version"] != position
            or facts["current_content_identity"] != source["current_identity"]
            or facts["current_parameters_sha256"]
            != hashlib.sha256(_payload(production_current)).hexdigest()
            or facts["current_outer_state"]
            != {
                "update_count": position,
                "momentum_buffer_sha256": expected_outer_buffer_sha,
            }
            or facts["selection_identity"] != source["selection_identity"]
            or facts["ordered_proposal_identities"]
            != [item["proposal_id"] for item in source_facts]
            or facts["ordered_proposal_content_identities"]
            != [item["content_identity"] for item in source_facts]
            or facts["ordered_base_versions"]
            != [item["base_version"] for item in source_facts]
            or facts["ordered_base_content_identities"]
            != [item["base_content_identity"] for item in source_facts]
            or facts["ordered_processed_tokens"]
            != [item["processed_tokens"] for item in source_facts]
            or facts["ordered_staleness"]
            != [item["staleness"] for item in source_facts]
            or facts["ordered_float32_weights"]
            != [item["normalized_weight"] for item in source_facts]
            or facts["ordered_local_parameters_sha256"]
            != [item["parameters_sha256"] for item in source_facts]
            or facts["merge_policy"]
            != {
                "name": "direct_weighted_average",
                "formula": "sum(weight * (declared_base - local))",
            }
            or facts["outer_optimizer_policy"] != "sgd"
            or facts["outer_hyperparameters"]
            != {
                "learning_rate": float(
                    np.float32(config["outer_optimizer"]["learning_rate"])
                ),
                "momentum": float(np.float32(config["outer_optimizer"]["momentum"])),
                "nesterov": config["outer_optimizer"]["nesterov"],
            }
            or facts["accumulation_dtype"] != "float32"
            or facts["fragment_map_identity"] != workload["fragment_map_identity"]
            or facts["fragment_identity"] != descriptor["identity"]
            or facts["fragment_index"] != descriptor["index"]
        ):
            raise MergeStressError("update identity facts differ from selected transition")
        contribution_count = len(source["contributions"])
        if (
            len(facts["ordered_proposal_identities"]) != contribution_count
            or len(facts["ordered_float32_weights"]) != contribution_count
            or facts["ordered_float32_weights"] != weights
        ):
            raise MergeStressError("update identity omits or changes selected contribution facts")
        if trace["update_identity"] in update_ids:
            raise MergeStressError("numeric update identities are not distinct")
        update_ids.add(trace["update_identity"])
        if trace["source_metrics"]["maximum_active_payloads"] != 1:
            raise MergeStressError("numeric source retained more than one payload")
        if trace["byte_accounting"]["full_model_operations"] != 0:
            raise MergeStressError("numeric update performed a full-model operation")
        recomputed_errors.append(error)
        production_current = trace["production_parameters"]
        production_buffer = trace["production_momentum_buffer"]
    gates = config["numeric_gates"]
    if max(recomputed_errors) > gates["fifty_update_relative_l2_max"]:
        raise MergeStressError("50-update relative L2 gate failed")
    first_max = max(recomputed_errors[:10])
    last_max = max(recomputed_errors[-10:])
    if last_max > max(first_max, gates["late_window_amplification_floor"]):
        raise MergeStressError("50-update error amplifies in the late window")
    if numeric["distinct_update_identities"] != expected_updates:
        raise MergeStressError("numeric update identity cardinality mismatch")

    order = syncer["order"]
    permutation_count = config["formal_workload"]["order_permutations"]
    if any(
        len(order[field]) != permutation_count
        for field in ("orders", "outputs", "relative_l2", "output_sha256")
    ) or order["permutation_count"] != permutation_count:
        raise MergeStressError("contribution-order trace cardinality mismatch")
    first_update = updates[0]
    independently_expected_order_outputs = []
    for indices, packaged_output, packaged_sha in zip(
        order["orders"], order["outputs"], order["output_sha256"], strict=True
    ):
        if sorted(indices) != list(range(len(first_update["contributions"]))):
            raise MergeStressError("contribution-order trace is not a permutation")
        selected = [first_update["contributions"][index] for index in indices]
        expected_gradient = weighted_direct_merge(
            current=first_update["current"],
            bases=[first_update["current"] for _ in selected],
            locals_=[entry["local"] for entry in selected],
            weights=[entry["fact"]["normalized_weight"] for entry in selected],
        )
        expected_output, _ = outer_sgd_step(
            first_update["current"],
            expected_gradient,
            OuterSGDState(None),
            learning_rate=config["direct_average_control"]["learning_rate"],
            momentum=config["direct_average_control"]["momentum"],
            nesterov=config["direct_average_control"]["nesterov"],
        )
        if _relative_l2(packaged_output, expected_output) > gates["order_relative_l2_max"]:
            raise MergeStressError("permuted production output differs from independent oracle")
        if hashlib.sha256(_payload(packaged_output)).hexdigest() != packaged_sha:
            raise MergeStressError("permuted output checksum mismatch")
        independently_expected_order_outputs.append(packaged_output)
    canonical_order_output = independently_expected_order_outputs[0]
    recomputed_order_errors = [
        _relative_l2(value, canonical_order_output)
        for value in independently_expected_order_outputs
    ]
    if (
        any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)
            for left, right in zip(
                recomputed_order_errors, order["relative_l2"], strict=True
            )
        )
        or not math.isclose(
            max(recomputed_order_errors),
            order["maximum_relative_l2"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or order["maximum_relative_l2"] > gates["order_relative_l2_max"]
    ):
        raise MergeStressError("contribution-order tolerance gate failed")

    mixed = syncer["mixed_base"]
    mixed_source = workload["mixed_base"]
    mixed_gradient = weighted_direct_merge(
        current=mixed_source["current"],
        bases=[mixed_source["old_base"], mixed_source["current"]],
        locals_=[entry["local"] for entry in mixed_source["contributions"]],
        weights=[entry["fact"]["normalized_weight"] for entry in mixed_source["contributions"]],
    )
    mixed_expected, _ = outer_sgd_step(
        mixed_source["current"],
        mixed_gradient,
        OuterSGDState(None),
        learning_rate=1.0,
        momentum=0.0,
        nesterov=False,
    )
    wrong_gradient = weighted_direct_merge(
        current=mixed_source["current"],
        bases=[mixed_source["current"], mixed_source["current"]],
        locals_=[entry["local"] for entry in mixed_source["contributions"]],
        weights=[entry["fact"]["normalized_weight"] for entry in mixed_source["contributions"]],
    )
    wrong_expected, _ = outer_sgd_step(
        mixed_source["current"],
        wrong_gradient,
        OuterSGDState(None),
        learning_rate=1.0,
        momentum=0.0,
        nesterov=False,
    )
    recomputed_wrong_error = _relative_l2(wrong_expected, mixed_expected)
    if (
        _relative_l2(mixed["production_parameters"], mixed_expected)
        > gates["single_update_relative_l2_max"]
        or _relative_l2(mixed["wrong_current_relative_parameters"], wrong_expected)
        > 1e-12
        or not math.isclose(
            mixed["wrong_relative_l2"],
            recomputed_wrong_error,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or recomputed_wrong_error <= gates["single_update_relative_l2_max"]
        or mixed["retained_base_reads"] != 1
        or mixed["maximum_active_base_payloads"] != 1
        or canonical_digest(mixed["update_facts"]) != mixed["update_identity"]
    ):
        raise MergeStressError("declared-base current-application counterexample failed")

    memory = syncer["memory"]
    memory_gates = config["memory_gates"]
    if set(memory["runs"]) != {"1", "8"}:
        raise MergeStressError("memory profile requires M=1 and M=8 child runs")
    for contributor_count in (1, 8):
        run = memory["runs"][str(contributor_count)]
        process_memory = run["memory_accounting"]
        if (
            run["contributor_count"] != contributor_count
            or run["source_metrics"]["maximum_active_payloads"]
            != memory_gates["maximum_live_local_payloads"]
            or run["source_metrics"]["active_payloads"] != 0
            or run["source_metrics"]["opens"] != contributor_count
            or run["memory_accounting"]["maximum_active_local_payloads"] != 1
            or run["memory_accounting"]["maximum_tensor_fragment_multiples"]
            > memory_gates["maximum_tensor_fragment_multiples"]
            or process_memory["process_rss_at_entry_bytes"] is None
            or process_memory["maximum_observed_process_rss_bytes"] is None
            or process_memory["peak_process_rss_bytes"] is None
            or process_memory["peak_process_rss_bytes"] < 0
            or process_memory["maximum_observed_process_rss_bytes"]
            - process_memory["process_rss_at_entry_bytes"]
            != process_memory["peak_process_rss_bytes"]
            or run["target_update_peak_rss_bytes"]
            != process_memory["peak_process_rss_bytes"]
            or run["byte_accounting"]["local_payload_reads"] != contributor_count
            or run["byte_accounting"]["local_payload_bytes"]
            != contributor_count * workload["memory"]["fragment_bytes"]
            or run["byte_accounting"]["full_model_operations"] != 0
        ):
            raise MergeStressError(f"M={contributor_count} streaming memory evidence failed")
    recomputed_difference = (
        memory["runs"]["8"]["target_update_peak_rss_bytes"]
        - memory["runs"]["1"]["target_update_peak_rss_bytes"]
    )
    if (
        recomputed_difference != memory["m8_minus_m1_peak_rss_bytes"]
        or memory["rss_gate_bytes"]
        != memory_gates["maximum_m8_minus_m1_peak_rss_bytes"]
        or recomputed_difference > memory["rss_gate_bytes"]
    ):
        raise MergeStressError("M=8 versus M=1 RSS gate failed")

    return {
        "schema_version": 1,
        "status": "pass",
        "run_id": writer["run_id"],
        "config_identity": writer["config_identity"],
        "workload_identity": writer["workload_identity"],
        "roles": {
            "proposal_writer": writer["hostname"],
            "outer_syncer": syncer["hostname"],
        },
        "application_coordination": "filesystem_only",
        "mpi_usage": "launcher_only",
        "numeric": {
            "updates": expected_updates,
            "maximum_relative_l2": max(recomputed_errors),
            "first_ten_maximum_relative_l2": first_max,
            "last_ten_maximum_relative_l2": last_max,
            "distinct_update_identities": len(update_ids),
        },
        "order": order,
        "mixed_base": {
            "production_parameters": mixed["production_parameters"],
            "expected_parameters": mixed_expected,
            "wrong_current_relative_parameters": mixed["wrong_current_relative_parameters"],
            "relative_l2": mixed["relative_l2"],
            "wrong_relative_l2": mixed["wrong_relative_l2"],
        },
        "memory": memory,
        "byte_accounting": {
            "fragment_count": config["formal_workload"]["fragment_count"],
            "fragment_bytes": workload["memory"]["fragment_bytes"],
            "full_model_bytes": config["formal_workload"]["fragment_count"]
            * workload["memory"]["fragment_bytes"],
            "full_model_operations": 0,
            "maximum_live_local_payloads": max(
                memory["runs"]["1"]["source_metrics"]["maximum_active_payloads"],
                memory["runs"]["8"]["source_metrics"]["maximum_active_payloads"],
            ),
        },
        "gpu_memory_bytes": {
            "proposal_writer_before": writer["gpu_memory_before"],
            "proposal_writer_after": writer["gpu_memory_after"],
            "outer_syncer_before": syncer["gpu_memory_before"],
            "outer_syncer_after": syncer["gpu_memory_after"],
        },
    }


def summarize(
    *, result_root: Path, config_path: Path, config_identity: str, output: Path
) -> dict[str, Any]:
    config = _load_config(config_path, config_identity)
    writer = json.loads((result_root / "proposal_writer.json").read_text(encoding="utf-8"))
    syncer = json.loads((result_root / "outer_syncer.json").read_text(encoding="utf-8"))
    summary = analyze_roles(writer, syncer, config, config_identity)
    _write_json_new(output, summary)
    return summary


def manifest_command(args: argparse.Namespace) -> dict[str, Any]:
    writer = json.loads((args.result_root / "proposal_writer.json").read_text(encoding="utf-8"))
    syncer = json.loads((args.result_root / "outer_syncer.json").read_text(encoding="utf-8"))
    declared = {
        "proposal_writer": [writer["hostname"]],
        "outer_syncer": [syncer["hostname"]],
    }
    actual = {key: list(value) for key, value in declared.items()}
    modules = args.modules_file.read_text(encoding="utf-8").splitlines()
    manifest = build_manifest(
        loop_id="S1-10",
        resource_level="L2",
        repository=args.repository,
        branch=args.branch,
        commit=args.commit,
        dirty=False,
        run_id=args.run_id,
        config_sha256=args.config_sha256,
        research_sha256=args.research_sha256,
        spec_sha256=args.spec_sha256,
        skill_repository=args.skill_repository,
        skill_commit=args.skill_commit,
        initial_hostname=args.initial_hostname,
        project_root=str(args.project_root),
        evidence_root=str(args.evidence_root),
        job_id=args.job_id,
        qtime_utc=args.qtime_utc,
        queue=args.queue,
        group=args.group,
        modules=modules,
        nodefile_sha256=file_digest(args.nodefile),
        declared_roles=declared,
        actual_roles=actual,
        workflow="two-node-filesystem-only-streaming-fragment-merge",
    )
    _write_json_new(args.output, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S1-10 streaming merge formal stress")
    sub = parser.add_subparsers(dest="command", required=True)
    role = sub.add_parser("role")
    role.add_argument("--root", type=Path, required=True)
    role.add_argument("--result-root", type=Path, required=True)
    role.add_argument("--run-id", required=True)
    role.add_argument("--config-path", type=Path, required=True)
    role.add_argument("--config-identity", required=True)
    memory = sub.add_parser("memory-child")
    memory.add_argument("--root", type=Path, required=True)
    memory.add_argument("--workload-path", type=Path, required=True)
    memory.add_argument("--contributors", type=int, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--result-root", type=Path, required=True)
    summary.add_argument("--config-path", type=Path, required=True)
    summary.add_argument("--config-identity", required=True)
    summary.add_argument("--output", type=Path, required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--repository", required=True)
    manifest.add_argument("--branch", required=True)
    manifest.add_argument("--commit", required=True)
    manifest.add_argument("--run-id", required=True)
    manifest.add_argument("--config-sha256", required=True)
    manifest.add_argument("--research-sha256", required=True)
    manifest.add_argument("--spec-sha256", required=True)
    manifest.add_argument("--skill-repository", required=True)
    manifest.add_argument("--skill-commit", required=True)
    manifest.add_argument("--initial-hostname", required=True)
    manifest.add_argument("--project-root", type=Path, required=True)
    manifest.add_argument("--evidence-root", type=Path, required=True)
    manifest.add_argument("--job-id", required=True)
    manifest.add_argument("--qtime-utc", required=True)
    manifest.add_argument("--queue", required=True)
    manifest.add_argument("--group", required=True)
    manifest.add_argument("--nodefile", type=Path, required=True)
    manifest.add_argument("--modules-file", type=Path, required=True)
    manifest.add_argument("--result-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
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
    elif args.command == "memory-child":
        result = run_memory_child(args.root, args.workload_path, args.contributors)
    elif args.command == "summarize":
        result = summarize(
            result_root=args.result_root,
            config_path=args.config_path,
            config_identity=args.config_identity,
            output=args.output,
        )
    elif args.command == "manifest":
        result = manifest_command(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
