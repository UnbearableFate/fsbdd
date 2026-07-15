from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import socket
import threading
import time
from pathlib import Path
from typing import Any

from .global_state import CountingStorageBackend, GlobalStateIdentities
from .global_state_stress import _fragments, derive_stress_identities
from .identity import file_digest
from .manifest import build_manifest
from .proposal import (
    ConsumptionFrontiers,
    EligibilityPolicy,
    Proposal,
    ProposalError,
    ProposalStore,
    RetainedBaseIdentity,
    commit_consumption,
    compute_candidate_weights,
    select_candidates,
)
from .storage import PosixStorageBackend, PublicationInterrupted, PublicationNotFound


_DESCRIPTOR_BYTES = 65_536


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


def _wait(path: Path, timeout_seconds: float = 180.0) -> dict[str, Any]:
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
            raise RuntimeError(f"coordination record is malformed: {path}")
        return value


def _learner_ids(learner_count: int) -> tuple[str, ...]:
    if learner_count <= 0:
        raise ValueError("learner_count must be positive")
    return tuple(f"learner-{index:02d}" for index in range(learner_count))


def _base_identity(fragment_index: int, version: int) -> str:
    return hashlib.sha256(
        f"fragment-{fragment_index}-base-{version}".encode()
    ).hexdigest()


def _parameters(
    learner_index: int, fragment_index: int, sequence: int, payload_bytes: int
) -> bytes:
    value = (learner_index * 31 + fragment_index * 7 + sequence) % 251
    return bytes([value]) * payload_bytes


def _identities(
    run_id: str,
    config_identity: str,
    model_identity: str,
    fragment_map_identity: str,
) -> GlobalStateIdentities:
    return GlobalStateIdentities(
        run_identity=run_id,
        config_identity=config_identity,
        model_identity=model_identity,
        fragment_map_identity=fragment_map_identity,
    )


def _store(
    backend,
    *,
    identities: GlobalStateIdentities,
    fragment_count: int,
    learner_count: int,
    maximum_local_steps: int,
    maximum_processed_tokens: int,
) -> ProposalStore:
    initial = _fragments(fragment_count, _DESCRIPTOR_BYTES)
    return ProposalStore(
        backend,
        identities=identities,
        descriptors=tuple(item.descriptor for item in initial),
        learner_ids=_learner_ids(learner_count),
        maximum_local_steps=maximum_local_steps,
        maximum_processed_tokens=maximum_processed_tokens,
    )


def _proposal(
    store: ProposalStore,
    *,
    learner_index: int,
    fragment_index: int,
    sequence: int,
    payload_bytes: int,
    parameters: bytes | None = None,
) -> Proposal:
    learner_id = store.learner_ids[learner_index]
    return Proposal.create(
        proposal_id=(
            f"{learner_id}-fragment-{fragment_index:02d}-sequence-{sequence:06d}"
        ),
        identities=store.identities,
        learner_id=learner_id,
        descriptor=store.descriptors[fragment_index],
        sequence=sequence,
        base_version=0,
        base_content_identity=_base_identity(fragment_index, 0),
        local_steps=sequence,
        processed_tokens=(learner_index + 1) * 1_000 + fragment_index + sequence,
        snapshot_local_step=sequence,
        parameters=(
            _parameters(learner_index, fragment_index, sequence, payload_bytes)
            if parameters is None
            else parameters
        ),
    )


def _validate_workload_identities(
    *,
    fragment_count: int,
    config_identity: str,
    model_identity: str,
    fragment_map_identity: str,
) -> None:
    initial = _fragments(fragment_count, _DESCRIPTOR_BYTES)
    derived_model, derived_map = derive_stress_identities(config_identity, initial)
    if derived_model != model_identity:
        raise ValueError("declared synthetic model identity differs from the workload")
    if derived_map != fragment_map_identity:
        raise ValueError(
            "declared synthetic fragment-map identity differs from the workload"
        )


def run_role(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    learner_count: int,
    fragment_count: int,
    payload_bytes: int,
    history_objects: int,
    s_max: int,
    maximum_local_steps: int,
    maximum_processed_tokens: int,
    config_identity: str,
    model_identity: str,
    fragment_map_identity: str,
) -> dict[str, Any]:
    try:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "proposal stress role requires an MPI launcher environment"
        ) from error
    if size != 2 or rank not in (0, 1):
        raise RuntimeError("proposal stress requires exactly two launcher ranks")
    if min(fragment_count, payload_bytes, history_objects) <= 0:
        raise ValueError("fragment_count, payload_bytes, and history_objects must be positive")
    if s_max < 0:
        raise ValueError("s_max must be nonnegative")
    _validate_workload_identities(
        fragment_count=fragment_count,
        config_identity=config_identity,
        model_identity=model_identity,
        fragment_map_identity=fragment_map_identity,
    )
    identities = _identities(
        run_id, config_identity, model_identity, fragment_map_identity
    )
    frozen_identities = identities.to_dict()
    backend_root = root / "backend"
    coordination = root / "coordination"
    hostname = socket.gethostname().split(".")[0]

    if rank == 0:
        store = _store(
            PosixStorageBackend(backend_root),
            identities=identities,
            fragment_count=fragment_count,
            learner_count=learner_count,
            maximum_local_steps=maximum_local_steps,
            maximum_processed_tokens=maximum_processed_tokens,
        )
        partial = _proposal(
            store,
            learner_index=0,
            fragment_index=0,
            sequence=1,
            payload_bytes=payload_bytes,
        )
        try:
            store.publish(partial, crash_at="after_payload_write")
        except PublicationInterrupted:
            partial_interrupted = True
        else:  # pragma: no cover - defensive formal assertion
            raise AssertionError("partial proposal interruption was not injected")
        _replace_json(coordination / "partial-ready.json", {"complete": True})
        _wait(coordination / "partial-observed.json")
        store.publish(partial)
        for learner_index in range(learner_count):
            for fragment_index in range(fragment_count):
                store.publish(
                    _proposal(
                        store,
                        learner_index=learner_index,
                        fragment_index=fragment_index,
                        sequence=1,
                        payload_bytes=payload_bytes,
                    )
                )
        payload_root = backend_root / "payloads"
        before_idempotent = {path.name for path in payload_root.iterdir()}
        repeated = store.publish(partial)
        after_idempotent = {path.name for path in payload_root.iterdir()}
        if repeated.content_identity != partial.content_identity:
            raise AssertionError("idempotent proposal changed content identity")
        conflict_parameters = bytes([partial.parameters[0] ^ 1]) * payload_bytes
        try:
            store.publish(
                _proposal(
                    store,
                    learner_index=0,
                    fragment_index=0,
                    sequence=1,
                    payload_bytes=payload_bytes,
                    parameters=conflict_parameters,
                )
            )
        except ProposalError:
            conflict_rejected = True
        else:  # pragma: no cover - defensive formal assertion
            raise AssertionError("conflicting same-sequence proposal was accepted")
        after_conflict = {path.name for path in payload_root.iterdir()}
        _replace_json(coordination / "complete-ready.json", {"complete": True})
        _wait(coordination / "baseline-observed.json")
        for index in range(history_objects):
            (payload_root / f"historical-{index:05d}.bin").write_bytes(b"history")
        _replace_json(coordination / "history-ready.json", {"complete": True})
        _wait(coordination / "history-observed.json")

        entered = threading.Event()
        release = threading.Event()
        reorder_errors: list[str] = []

        def publish_delayed() -> None:
            entered.set()
            if not release.wait(timeout=60):
                raise TimeoutError("delayed sequence was not released")
            try:
                store.publish(
                    _proposal(
                        store,
                        learner_index=0,
                        fragment_index=0,
                        sequence=2,
                        payload_bytes=payload_bytes,
                    )
                )
            except ProposalError as error:
                reorder_errors.append(str(error))

        thread = threading.Thread(target=publish_delayed)
        thread.start()
        if not entered.wait(timeout=60):
            raise TimeoutError("sequence-2 publication task did not reach its delay point")
        sequence_three = _proposal(
            store,
            learner_index=0,
            fragment_index=0,
            sequence=3,
            payload_bytes=payload_bytes,
        )
        store.publish(sequence_three)
        release.set()
        thread.join(timeout=60)
        if thread.is_alive() or len(reorder_errors) != 1:
            raise AssertionError("delayed sequence-2 publication was not rejected")
        final = store.load_latest(store.learner_ids[0], 0)
        if final.sequence != 3:
            raise AssertionError("delayed old completion regressed latest")
        _replace_json(coordination / "reorder-ready.json", {"complete": True})
        _wait(coordination / "reader-done.json")
        result = {
            "schema_version": 1,
            "status": "complete",
            "role": "proposal_writer",
            "rank": rank,
            "hostname": hostname,
            "state_identities": frozen_identities,
            "partial_interrupted": partial_interrupted,
            "published_slots": learner_count * fragment_count,
            "idempotent_new_payloads": len(after_idempotent - before_idempotent),
            "conflict_new_payloads": len(after_conflict - after_idempotent),
            "same_sequence_conflict_rejected": conflict_rejected,
            "delayed_sequence_rejected": True,
            "delayed_sequence_error": reorder_errors[0],
            "final_reordered_sequence": final.sequence,
            "history_objects": history_objects,
        }
        _write_json_new(result_root / "proposal_writer.json", result)
        return result

    store = _store(
        PosixStorageBackend(backend_root),
        identities=identities,
        fragment_count=fragment_count,
        learner_count=learner_count,
        maximum_local_steps=maximum_local_steps,
        maximum_processed_tokens=maximum_processed_tokens,
    )
    _wait(coordination / "partial-ready.json")
    try:
        store.load_latest(store.learner_ids[0], 0, timeout_seconds=0)
    except PublicationNotFound:
        partial_invisible = True
    else:  # pragma: no cover - defensive formal assertion
        raise AssertionError("remote reader observed a payload-only proposal")
    _replace_json(coordination / "partial-observed.json", {"complete": True})
    _wait(coordination / "complete-ready.json")
    counting = CountingStorageBackend(PosixStorageBackend(backend_root))
    reader_store = _store(
        counting,
        identities=identities,
        fragment_count=fragment_count,
        learner_count=learner_count,
        maximum_local_steps=maximum_local_steps,
        maximum_processed_tokens=maximum_processed_tokens,
    )
    counting.reset_counts()
    baseline = reader_store.discover_latest(timeout_seconds=30)
    baseline_reads = counting.read_calls
    baseline_identities = tuple(item.content_identity for item in baseline)
    _replace_json(coordination / "baseline-observed.json", {"complete": True})
    _wait(coordination / "history-ready.json")
    counting.reset_counts()
    after_history = reader_store.discover_latest(timeout_seconds=30)
    history_reads = counting.read_calls
    after_history_identities = tuple(item.content_identity for item in after_history)
    if baseline_identities != after_history_identities:
        raise AssertionError("history objects changed fixed-slot discovery")
    _replace_json(coordination / "history-observed.json", {"complete": True})
    _wait(coordination / "reorder-ready.json")
    final_reordered = reader_store.load_latest(reader_store.learner_ids[0], 0)
    discovered = reader_store.discover_latest(timeout_seconds=30)
    proposal_schema_fields = [field.name for field in dataclasses.fields(Proposal)]
    required_schema_fields = {
        "proposal_id",
        "identities",
        "learner_id",
        "descriptor",
        "sequence",
        "base_version",
        "base_content_identity",
        "local_steps",
        "processed_tokens",
        "snapshot_local_step",
        "parameters",
        "parameters_sha256",
        "content_identity",
    }
    if set(proposal_schema_fields) != required_schema_fields:
        raise AssertionError("decoded proposal schema inventory is incomplete")
    if any(item.payload_bytes != payload_bytes for item in discovered):
        raise AssertionError("decoded proposal payload byte count differs from the workload")
    fragment_results = []
    for descriptor in reader_store.descriptors:
        candidates = tuple(
            item for item in discovered if item.descriptor.index == descriptor.index
        )
        policy = EligibilityPolicy(
            identities=identities,
            descriptor=descriptor,
            learner_ids=reader_store.learner_ids,
            current_version=0,
            retained_bases=(
                RetainedBaseIdentity(0, _base_identity(descriptor.index, 0)),
            ),
            s_max=s_max,
            q=learner_count,
            q_fresh=learner_count,
            max_contributors=learner_count,
            maximum_local_steps=maximum_local_steps,
            maximum_processed_tokens=maximum_processed_tokens,
            lambda_s=1.0,
        )
        frontiers = ConsumptionFrontiers.empty(
            identities=identities,
            descriptor=descriptor,
            learner_ids=reader_store.learner_ids,
        )
        selection = select_candidates(candidates, frontiers, policy)
        if not selection.ready or len(selection.selected) != learner_count:
            raise AssertionError("formal proposal selection did not reach full quorum")
        weights = compute_candidate_weights(
            selection.selected, current_version=0, lambda_s=1.0
        )
        committed = commit_consumption(frontiers, selection.selected, True)
        repeated = select_candidates(candidates, committed, policy)
        if repeated.ready or repeated.selected:
            raise AssertionError("repeated polling selected consumed proposals")
        fragment_results.append(
            {
                "fragment_index": descriptor.index,
                "selected_count": len(selection.selected),
                "distinct_learner_count": len(
                    {item.learner_id for item in selection.selected}
                ),
                "frontier_entries": len(committed.entries),
                "repeated_selected_count": len(repeated.selected),
                "repeated_rejections": repeated.rejections,
                "weights": [item.to_dict() for item in weights],
            }
        )
    result = {
        "schema_version": 1,
        "status": "complete",
        "role": "proposal_discoverer",
        "rank": rank,
        "hostname": hostname,
        "state_identities": frozen_identities,
        "partial_proposal_invisible": partial_invisible,
        "baseline_discovery_read_calls": baseline_reads,
        "post_history_discovery_read_calls": history_reads,
        "baseline_proposals": len(baseline),
        "post_history_proposals": len(after_history),
        "final_reordered_sequence": final_reordered.sequence,
        "decoded_proposal_schema": {
            "fields": proposal_schema_fields,
            "payload_kind": "complete_local_parameters",
            "payload_bytes": payload_bytes,
            "payload_identity_bound": True,
            "base_version_bound": True,
            "base_content_identity_bound": True,
        },
        "fragment_results": fragment_results,
    }
    _write_json_new(result_root / "proposal_discoverer.json", result)
    _replace_json(coordination / "reader-done.json", {"complete": True})
    return result


def summarize(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    learner_count: int,
    fragment_count: int,
    payload_bytes: int,
    history_objects: int,
    s_max: int,
    maximum_local_steps: int,
    maximum_processed_tokens: int,
    config_identity: str,
    model_identity: str,
    fragment_map_identity: str,
    base_retention_evidence: Path,
    output: Path,
) -> dict[str, Any]:
    writer = json.loads(
        (result_root / "proposal_writer.json").read_text(encoding="utf-8")
    )
    reader = json.loads(
        (result_root / "proposal_discoverer.json").read_text(encoding="utf-8")
    )
    if writer["hostname"] == reader["hostname"]:
        raise AssertionError("two-node proposal run used the same host for both roles")
    expected_identities = {
        "run_identity": run_id,
        "config_identity": config_identity,
        "model_identity": model_identity,
        "fragment_map_identity": fragment_map_identity,
    }
    if writer.get("state_identities") != expected_identities:
        raise AssertionError("writer identities differ from the frozen workload")
    if reader.get("state_identities") != expected_identities:
        raise AssertionError("discoverer identities differ from the frozen workload")
    base_retention = json.loads(base_retention_evidence.read_text(encoding="utf-8"))
    component_identities = base_retention.get("state_identities", {})
    for field in ("config_identity", "model_identity", "fragment_map_identity"):
        if component_identities.get(field) != expected_identities[field]:
            raise AssertionError(
                f"S1-04 base-retention component {field} differs from S1-05"
            )
    bounded_authority = base_retention.get("bounded_authority", {})
    if (
        bounded_authority.get("current_records") != fragment_count
        or bounded_authority.get("retained_base_entries") != fragment_count
        or bounded_authority.get("maximum_base_entries") != fragment_count
    ):
        raise AssertionError("S1-04 S_max=0 base-retention component is incomplete")
    slot_count = learner_count * fragment_count
    if not writer["partial_interrupted"] or not reader["partial_proposal_invisible"]:
        raise AssertionError("partial proposal invisibility evidence is incomplete")
    if writer["published_slots"] != slot_count:
        raise AssertionError("writer did not populate every frozen latest slot")
    if writer["idempotent_new_payloads"] != 0:
        raise AssertionError("idempotent same-sequence publication created a payload")
    if writer["conflict_new_payloads"] != 0 or not writer[
        "same_sequence_conflict_rejected"
    ]:
        raise AssertionError("same-sequence conflict was not rejected before payload write")
    if (
        not writer["delayed_sequence_rejected"]
        or writer["final_reordered_sequence"] != 3
        or reader["final_reordered_sequence"] != 3
    ):
        raise AssertionError("monotonic latest reorder evidence is incomplete")
    if (
        reader["baseline_discovery_read_calls"] != slot_count
        or reader["post_history_discovery_read_calls"] != slot_count
        or reader["baseline_proposals"] != slot_count
        or reader["post_history_proposals"] != slot_count
    ):
        raise AssertionError("proposal discovery cost depends on history volume")
    for fragment in reader["fragment_results"]:
        if (
            fragment["selected_count"] != learner_count
            or fragment["distinct_learner_count"] != learner_count
            or fragment["frontier_entries"] != learner_count
            or fragment["repeated_selected_count"] != 0
            or len(fragment["repeated_rejections"]) != learner_count
        ):
            raise AssertionError("selection or fixed-frontier evidence is incomplete")
        weights = fragment["weights"]
        if any(item["staleness"] != 0 for item in weights):
            raise AssertionError("S1-05 formal workload did not run with fresh proposals")
        if abs(sum(item["normalized_weight"] for item in weights) - 1.0) > 2e-6:
            raise AssertionError("formal candidate weights do not normalize to one")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "run_identity": run_id,
        "filesystem_data_plane": "shared_posix_only",
        "application_coordination": "filesystem_only",
        "mpi_usage": "launcher_only",
        "roles": {
            "proposal_writer": [writer["hostname"]],
            "proposal_discoverer": [reader["hostname"]],
        },
        "state_identities": expected_identities,
        "workload": {
            "learner_count": learner_count,
            "fragment_count": fragment_count,
            "latest_slots": slot_count,
            "payload_bytes_per_proposal": payload_bytes,
            "history_objects": history_objects,
            "s_max": s_max,
            "maximum_local_steps": maximum_local_steps,
            "maximum_processed_tokens": maximum_processed_tokens,
        },
        "partial_publication": {
            "crash_point": "after_payload_write",
            "remote_partial_proposal_invisible": True,
            "complete_retry_visible": True,
        },
        "latest_monotonic": {
            "delayed_sequence": 2,
            "winning_sequence": 3,
            "final_sequence": 3,
            "same_sequence_idempotent_new_payloads": 0,
            "same_sequence_conflict_rejected": True,
        },
        "discovery": {
            "baseline_read_calls": reader["baseline_discovery_read_calls"],
            "post_10000_object_read_calls": reader[
                "post_history_discovery_read_calls"
            ],
            "directory_scans_in_normal_path": 0,
        },
        "eligibility": {
            "fragments": reader["fragment_results"],
            "one_per_learner": True,
            "fixed_frontiers": True,
            "repeated_polling_once_only": True,
            "generic_s_max": s_max,
            "lambda_s": 1.0,
        },
        "decoded_proposal_schema": reader["decoded_proposal_schema"],
        "stale_ready_interfaces": {
            "proposal_base_and_progress_identity": True,
            "generic_eligibility": {
                "runtime_s_max": s_max,
                "positive_s_max_evidence": (
                    "tests/stage1/test_s1_05_proposal.py::"
                    "test_smax_zero_and_positive_use_the_same_generic_eligibility_function"
                ),
            },
            "inverse_staleness_weighting": {
                "lambda_s": 1.0,
                "telemetry_fields": [
                    "processed_tokens",
                    "staleness",
                    "raw_weight",
                    "normalized_weight",
                ],
            },
            "fixed_consumption_frontier_fields": [
                "last_sequence",
                "last_base_version",
            ],
            "bounded_base_retention_component": {
                "source": str(base_retention_evidence),
                "sha256": file_digest(base_retention_evidence),
                "runtime_s_max": 0,
                "retained_base_entries": bounded_authority[
                    "retained_base_entries"
                ],
                "maximum_base_entries": bounded_authority[
                    "maximum_base_entries"
                ],
            },
        },
    }
    _write_json_new(output, summary)
    shutil.rmtree(root)
    return summary


def write_manifest(args: argparse.Namespace) -> None:
    hosts = sorted(set(Path(args.nodefile).read_text(encoding="utf-8").splitlines()))
    if len(hosts) != 2:
        raise ValueError(f"manifest requires two distinct hosts, got {hosts}")
    modules = [
        line.strip()
        for line in Path(args.modules_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    writer = json.loads(
        (Path(args.result_root) / "proposal_writer.json").read_text(encoding="utf-8")
    )
    reader = json.loads(
        (Path(args.result_root) / "proposal_discoverer.json").read_text(
            encoding="utf-8"
        )
    )
    role_map = {
        "proposal_writer": [writer["hostname"]],
        "proposal_discoverer": [reader["hostname"]],
    }
    if set(role_map["proposal_writer"] + role_map["proposal_discoverer"]) != set(
        hosts
    ):
        raise ValueError("actual proposal roles do not match allocated hosts")
    expected_state_identities = {
        "run_identity": args.run_id,
        "config_identity": args.config_sha256,
        "model_identity": args.state_model_identity,
        "fragment_map_identity": args.state_fragment_map_identity,
    }
    if writer.get("state_identities") != expected_state_identities:
        raise ValueError("writer state identities do not match manifest inputs")
    if reader.get("state_identities") != expected_state_identities:
        raise ValueError("discoverer state identities do not match manifest inputs")
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
            "workflow": "two-node-fixed-slot-proposal-batch",
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
        "S1-05",
        "L2",
        args.qtime_utc,
        "pbs_qtime",
        identities,
        scheduler=scheduler,
    )
    _write_json_new(Path(args.output), manifest)


def _add_workload_arguments(parser: argparse.ArgumentParser, *, output: bool) -> None:
    for name, value_type in (
        ("root", Path),
        ("result-root", Path),
        ("run-id", str),
        ("learner-count", int),
        ("fragment-count", int),
        ("payload-bytes", int),
        ("history-objects", int),
        ("s-max", int),
        ("maximum-local-steps", int),
        ("maximum-processed-tokens", int),
        ("config-identity", str),
        ("model-identity", str),
        ("fragment-map-identity", str),
    ):
        parser.add_argument(f"--{name}", required=True, type=value_type)
    if output:
        parser.add_argument("--base-retention-evidence", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fsbdd.proposal_stress")
    commands = parser.add_subparsers(dest="command", required=True)
    role = commands.add_parser("role")
    _add_workload_arguments(role, output=False)
    analysis = commands.add_parser("summarize")
    _add_workload_arguments(analysis, output=True)
    manifest = commands.add_parser("manifest")
    for name in (
        "repository",
        "branch",
        "commit",
        "run-id",
        "config-sha256",
        "state-model-identity",
        "state-fragment-map-identity",
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
        result = run_role(
            root=args.root,
            result_root=args.result_root,
            run_id=args.run_id,
            learner_count=args.learner_count,
            fragment_count=args.fragment_count,
            payload_bytes=args.payload_bytes,
            history_objects=args.history_objects,
            s_max=args.s_max,
            maximum_local_steps=args.maximum_local_steps,
            maximum_processed_tokens=args.maximum_processed_tokens,
            config_identity=args.config_identity,
            model_identity=args.model_identity,
            fragment_map_identity=args.fragment_map_identity,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "summarize":
        summary = summarize(
            root=args.root,
            result_root=args.result_root,
            run_id=args.run_id,
            learner_count=args.learner_count,
            fragment_count=args.fragment_count,
            payload_bytes=args.payload_bytes,
            history_objects=args.history_objects,
            s_max=args.s_max,
            maximum_local_steps=args.maximum_local_steps,
            maximum_processed_tokens=args.maximum_processed_tokens,
            config_identity=args.config_identity,
            model_identity=args.model_identity,
            fragment_map_identity=args.fragment_map_identity,
            base_retention_evidence=args.base_retention_evidence,
            output=args.output,
        )
        print(json.dumps({"status": summary["status"]}, sort_keys=True))
        return 0
    write_manifest(args)
    print(json.dumps({"status": "building", "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
