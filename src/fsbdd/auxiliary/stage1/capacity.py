from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from fsbdd.diloco.common.identity import canonical_digest
from fsbdd.diloco.protocol.global_commit import (
    AtomicCommitRequest,
    AtomicGlobalCommitStore,
)
from fsbdd.diloco.protocol.global_state import (
    BootstrapFragment,
    FragmentStateDescriptor,
    GlobalStateIdentities,
    GlobalStateStore,
)
from fsbdd.diloco.protocol.proposal import Proposal, compute_candidate_weights
from fsbdd.diloco.protocol.storage import PosixStorageBackend
from fsbdd.diloco.syncer.merge import (
    ContributionFact,
    FragmentMergeRequest,
    FragmentOuterState,
    OuterSGDPolicy,
    execute_numpy_streaming_fragment_update,
)
from fsbdd.diloco.syncer.profile_a import profile_a_policy_identity
from fsbdd.diloco.syncer.readiness import FrozenSelection


class CapacityPreflightError(RuntimeError):
    """The S1-13 exact-size publication preflight is not admissible."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityPreflightError(f"cannot read JSON contract: {path}") from error
    if not isinstance(value, dict):
        raise CapacityPreflightError("capacity contract must be a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with path.open("xb") as stream:
            if stream.write(content) != len(content):
                raise CapacityPreflightError("capacity result write was incomplete")
    except OSError as error:
        raise CapacityPreflightError(f"cannot write capacity result: {path}") from error


def _require_positive_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CapacityPreflightError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_float(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise CapacityPreflightError(f"{field} must be nonnegative")
    return float(value)


def _validated_contract(path: Path) -> tuple[dict[str, Any], str]:
    actual_sha256 = _hash_file(path)
    contract = _read_json(path)
    if (
        contract.get("schema_version") != 2
        or contract.get("kind")
        != "s1_13_exact_size_numpy_merge_commit_capacity_preflight"
    ):
        raise CapacityPreflightError("capacity contract identity is invalid")
    sizes = contract.get("payload_bytes")
    if not isinstance(sizes, list) or not sizes:
        raise CapacityPreflightError("capacity contract omits payload sizes")
    for index, size in enumerate(sizes):
        _require_positive_integer(size, f"payload_bytes[{index}]")
    parameter_sizes = contract.get("fragment_parameter_bytes")
    parameter_identity_counts = contract.get("parameter_identity_counts")
    if (
        not isinstance(parameter_sizes, list)
        or not isinstance(parameter_identity_counts, list)
        or len(parameter_sizes) != len(sizes)
        or len(parameter_identity_counts) != len(sizes)
    ):
        raise CapacityPreflightError(
            "fragment sizes and parameter identity counts must match payload sizes"
        )
    for index, size in enumerate(parameter_sizes):
        size = _require_positive_integer(size, f"fragment_parameter_bytes[{index}]")
        if size % 4:
            raise CapacityPreflightError(
                "fragment parameter bytes must be float32 aligned"
            )
    for index, count in enumerate(parameter_identity_counts):
        _require_positive_integer(count, f"parameter_identity_counts[{index}]")
    _require_positive_integer(
        contract.get("numpy_update_repetitions"), "numpy_update_repetitions"
    )
    thresholds = contract.get("thresholds")
    if not isinstance(thresholds, dict):
        raise CapacityPreflightError("capacity contract omits thresholds")
    for field in ("maximum_numpy_merge_commit_seconds",):
        _require_nonnegative_float(thresholds.get(field), f"thresholds.{field}")
    baseline = contract.get("failed_smoke_baseline")
    if not isinstance(baseline, dict):
        raise CapacityPreflightError("capacity contract omits the failed baseline")
    _require_nonnegative_float(
        baseline.get("mean_update_latency_seconds"),
        "failed_smoke_baseline.mean_update_latency_seconds",
    )
    return contract, actual_sha256


class _PrevalidatedContributionSource:
    """Mirror the in-memory source used after proposal records are validated."""

    prevalidated_payload_integrity = True

    def __init__(self, proposals: tuple[Proposal, ...]) -> None:
        self._payloads = {item.proposal_id: item.parameters for item in proposals}

    @contextlib.contextmanager
    def open_payload(self, contribution: ContributionFact) -> Iterator[bytes]:
        try:
            yield self._payloads[contribution.proposal_id]
        except KeyError as error:
            raise CapacityPreflightError(
                "NumPy update proposal payload is absent"
            ) from error


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _frozen_selection(
    authority: Any, proposals: tuple[Proposal, ...]
) -> FrozenSelection:
    weights = compute_candidate_weights(
        proposals, current_version=authority.version, lambda_s=1.0
    )
    semantic = {
        "schema_version": 1,
        "logical_syncer_id": "capacity-preflight-syncer",
        "fragment_index": authority.state.descriptor.index,
        "current_version": authority.version,
        "authority_identity": authority.authority_identity,
        "proposal_content_identities": [item.content_identity for item in proposals],
        "weights": [item.to_dict() for item in weights],
    }
    return FrozenSelection(
        logical_syncer_id="capacity-preflight-syncer",
        fragment_index=authority.state.descriptor.index,
        current_version=authority.version,
        authority_identity=authority.authority_identity,
        generation=1,
        grace_started_ns=0,
        frozen_observed_ns=0,
        proposals=proposals,
        weights=weights,
        selection_identity=canonical_digest(semantic),
    )


def _run_numpy_update_trial(
    *,
    root: Path,
    run_id: str,
    parameter_bytes: int,
    target_global_payload_bytes: int,
    parameter_identity_count: int,
    repetition: int,
) -> dict[str, Any]:
    """Measure the exact production NumPy merge and atomic commit regions."""

    if parameter_bytes <= 0 or parameter_bytes % 4:
        raise CapacityPreflightError("NumPy trial parameter bytes are invalid")
    learners = tuple(f"learner-{index:02d}" for index in range(4))
    identities = GlobalStateIdentities(
        run_identity=f"{run_id}-numpy-{parameter_bytes}-{repetition}",
        config_identity=_identity(f"{run_id}-capacity-config"),
        model_identity=_identity(f"{run_id}-capacity-model"),
        fragment_map_identity=_identity(f"{run_id}-capacity-map-{parameter_bytes}"),
    )
    descriptor = FragmentStateDescriptor(
        index=0,
        identity=_identity(f"{run_id}-capacity-fragment-{parameter_bytes}"),
        dtype="float32",
        shape=(parameter_bytes // 4,),
        parameter_identities=tuple(
            _identity(f"{run_id}-parameter-{parameter_bytes}-{index}")
            for index in range(parameter_identity_count)
        ),
    )
    policy = OuterSGDPolicy(learning_rate=0.1, momentum=0.9, nesterov=True)
    backend = PosixStorageBackend(root, verify_payload_readback=False)
    global_store = GlobalStateStore(
        backend,
        identities=identities,
        descriptors=(descriptor,),
        s_max=0,
    )
    atomic = AtomicGlobalCommitStore(
        global_store,
        learner_ids=learners,
        policy_identity=profile_a_policy_identity(policy),
    )
    current_payload = bytes(parameter_bytes)
    atomic.bootstrap(
        (
            BootstrapFragment(
                descriptor=descriptor,
                parameters=current_payload,
                outer_state=b"",
            ),
        )
    )
    authority = atomic.load_fragment(0)
    local_payload = bytes(parameter_bytes)
    proposals = tuple(
        Proposal.create(
            proposal_id=f"capacity-{learner_id}-{parameter_bytes}-{repetition}",
            identities=identities,
            learner_id=learner_id,
            descriptor=descriptor,
            sequence=1,
            base_version=authority.version,
            base_content_identity=authority.content_identity,
            local_steps=50,
            processed_tokens=2048 * (index + 1),
            snapshot_local_step=50,
            parameters=local_payload,
        )
        for index, learner_id in enumerate(learners)
    )
    selection = _frozen_selection(authority, proposals)
    contributions = tuple(
        ContributionFact(
            proposal_id=proposal.proposal_id,
            content_identity=proposal.content_identity,
            learner_id=proposal.learner_id,
            base_version=proposal.base_version,
            base_content_identity=proposal.base_content_identity,
            processed_tokens=proposal.processed_tokens,
            staleness=weight.staleness,
            normalized_weight=weight.normalized_weight,
            parameters_sha256=proposal.parameters_sha256,
            payload_bytes=proposal.payload_bytes,
        )
        for proposal, weight in zip(proposals, selection.weights, strict=True)
    )
    merge_started_ns = time.monotonic_ns()
    result = execute_numpy_streaming_fragment_update(
        FragmentMergeRequest(
            descriptor=descriptor,
            fragment_map_identity=identities.fragment_map_identity,
            current_version=authority.version,
            current_content_identity=authority.content_identity,
            current_parameters=authority.parameters,
            outer_state=FragmentOuterState(
                update_count=authority.version,
                momentum_buffer=None,
            ),
            selection_identity=selection.selection_identity,
            contributions=contributions,
        ),
        _PrevalidatedContributionSource(proposals),
        policy,
    )
    merge_seconds = (time.monotonic_ns() - merge_started_ns) / 1_000_000_000
    commit_started_ns = time.monotonic_ns()
    committed = atomic.commit(
        AtomicCommitRequest(
            fragment_index=0,
            expected_current_version=authority.version,
            expected_current_content_identity=authority.content_identity,
            next_version=authority.version + 1,
            parameters=result.parameters,
            outer_optimizer_state=(
                b""
                if result.outer_state.momentum_buffer is None
                else result.outer_state.momentum_buffer
            ),
            selection=selection,
            policy_identity=atomic.policy_identity,
            update_identity=result.update_identity,
        )
    )
    commit_seconds = (time.monotonic_ns() - commit_started_ns) / 1_000_000_000
    visible_record = global_store.peek_fragment_record(0, timeout_seconds=0)
    if (
        not committed.published
        or committed.authority.version != 1
        or committed.authority.parameters != current_payload
        or result.byte_accounting.fragment_bytes != parameter_bytes
    ):
        raise CapacityPreflightError("NumPy merge/commit correctness check failed")
    return {
        "repetition": repetition,
        "fragment_parameter_bytes": parameter_bytes,
        "target_global_payload_bytes": target_global_payload_bytes,
        "committed_global_payload_bytes": visible_record.payload_bytes,
        "committed_payload_delta_bytes": abs(
            visible_record.payload_bytes - target_global_payload_bytes
        ),
        "parameter_identity_count": parameter_identity_count,
        "merge_seconds": merge_seconds,
        "commit_seconds": commit_seconds,
        "merge_commit_seconds": merge_seconds + commit_seconds,
        "successor_version": committed.authority.version,
        "successor_parameters_sha256": hashlib.sha256(
            committed.authority.parameters
        ).hexdigest(),
        "verification_mode": backend.publication_verification_mode,
        "torch_not_imported": "torch" not in sys.modules,
    }


def run_preflight(
    *,
    shared_root: Path,
    output: Path,
    run_id: str,
    contract_path: Path,
) -> dict[str, Any]:
    contract, actual_contract_sha256 = _validated_contract(contract_path)
    if shared_root.exists():
        raise CapacityPreflightError("capacity shared root already exists")
    shared_root.mkdir(parents=True)
    sizes = tuple(int(value) for value in contract["payload_bytes"])
    numpy_trials: list[dict[str, Any]] = []
    numpy_repetitions = int(contract["numpy_update_repetitions"])
    for size_index, size in enumerate(sizes):
        for repetition in range(numpy_repetitions):
            numpy_trials.append(
                _run_numpy_update_trial(
                    root=(
                        shared_root
                        / "numpy-merge-commit"
                        / f"payload-{size}"
                        / f"repetition-{repetition}"
                    ),
                    run_id=run_id,
                    parameter_bytes=int(
                        contract["fragment_parameter_bytes"][size_index]
                    ),
                    target_global_payload_bytes=size,
                    parameter_identity_count=int(
                        contract["parameter_identity_counts"][size_index]
                    ),
                    repetition=repetition,
                )
            )
            gc.collect()

    baseline_mean = float(
        contract["failed_smoke_baseline"]["mean_update_latency_seconds"]
    )
    thresholds = contract["thresholds"]
    maximum_numpy_merge_commit = max(
        float(row["merge_commit_seconds"]) for row in numpy_trials
    )
    checks = {
        "exact_fragment_parameter_sizes_exercised": sorted(
            int(row["fragment_parameter_bytes"]) for row in numpy_trials
        )
        == sorted(
            [
                int(value)
                for value in contract["fragment_parameter_bytes"]
                for _ in range(numpy_repetitions)
            ]
        ),
        "numpy_merge_commit_within_budget": maximum_numpy_merge_commit
        <= float(thresholds["maximum_numpy_merge_commit_seconds"]),
        "all_numpy_updates_correct": all(
            row["successor_version"] == 1
            and row["verification_mode"] == "source_sha256_complete_write"
            and row["torch_not_imported"]
            for row in numpy_trials
        ),
        "torch_not_imported": "torch" not in sys.modules,
    }
    result = {
        "schema_version": 2,
        "kind": "s1_13_exact_size_numpy_merge_commit_capacity_preflight_result",
        "status": "pass" if all(checks.values()) else "fail",
        "complete": True,
        "run_id": run_id,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "hostname": os.uname().nodename,
        "code_commit": os.environ.get("FSBDD_CODE_COMMIT"),
        "contract_path": str(contract_path),
        "contract_sha256": actual_contract_sha256,
        "publication_mode": "source_sha256_complete_write",
        "numpy_merge_commit_trials": numpy_trials,
        "aggregate": {
            "failed_smoke_mean_update_latency_seconds": baseline_mean,
            "maximum_numpy_merge_commit_seconds": maximum_numpy_merge_commit,
        },
        "thresholds": thresholds,
        "checks": checks,
        "forbidden_runtime": {
            "torch_module_imported": "torch" in sys.modules,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    _write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = run_preflight(
            shared_root=arguments.shared_root,
            output=arguments.output,
            run_id=arguments.run_id,
            contract_path=arguments.contract,
        )
    except Exception as error:
        if not arguments.output.exists():
            _write_json(
                arguments.output,
                {
                    "schema_version": 2,
                    "kind": "s1_13_exact_size_numpy_merge_commit_capacity_preflight_result",
                    "status": "fail",
                    "complete": False,
                    "run_id": arguments.run_id,
                    "pbs_job_id": os.environ.get("PBS_JOBID"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        return 1
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
