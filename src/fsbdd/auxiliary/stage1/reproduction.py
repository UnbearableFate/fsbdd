from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ReproductionError(RuntimeError):
    """The corrected two-node reproduction evidence is inadmissible."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReproductionError(f"JSON evidence is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ReproductionError(f"JSON object required: {path}")
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for ordinal, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ReproductionError(
                        f"JSONL object required at {path}:{ordinal}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ReproductionError(f"JSONL evidence is unreadable: {path}") from error
    return rows


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise ReproductionError(f"refusing to replace evidence: {path}") from error


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReproductionError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReproductionError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReproductionError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ReproductionError(
            f"{name} must be finite" + (" and positive" if positive else "")
        )
    return result


def _event(rows: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("event") == name]


def _identity_pass(
    learner: Mapping[str, Any],
    syncer: Mapping[str, Any],
    *,
    run_id: str,
    config_sha256: str,
    asset_marker_sha256: str,
    gate_contract_sha256: str,
) -> bool:
    identities = [learner.get("identity"), syncer.get("identity")]
    if not all(isinstance(value, dict) for value in identities):
        return False
    return all(
        identity.get("run_id") == run_id
        and identity.get("config_sha256") == config_sha256
        and identity.get("asset_marker_sha256") == asset_marker_sha256
        and identity.get("gate_contract_sha256") == gate_contract_sha256
        and identity.get("execution_mode") == "reduced_two_node_reproduction"
        for identity in identities
        if isinstance(identity, dict)
    )


def _artifact_inventory(result_root: Path) -> list[dict[str, Any]]:
    excluded = {
        "checksums.sha256",
        "package-manifest.json",
        "analysis/package-validator.json",
    }
    rows = []
    for path in sorted(item for item in result_root.rglob("*") if item.is_file()):
        relative = path.relative_to(result_root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
        )
    return rows


def analyze_reproduction(
    *,
    result_root: Path,
    shared_root: Path,
    config_path: Path,
    asset_root: Path,
    gate_contract_path: Path,
    reproduction_contract_path: Path,
    expected_commit: str,
    run_id: str,
    job_id: str,
) -> dict[str, Any]:
    config = _read_object(config_path)
    contract = _read_object(reproduction_contract_path)
    identities = contract.get("identities")
    topology = contract.get("topology")
    runtime = contract.get("runtime")
    required_evidence = contract.get("required_evidence")
    if (
        contract.get("schema_version") != 1
        or contract.get("loop_id") != "S1-13"
        or contract.get("kind") != "corrected_two_node_filesystem_reproduction"
        or not isinstance(identities, dict)
        or not isinstance(topology, dict)
        or not isinstance(runtime, dict)
        or not isinstance(required_evidence, list)
        or not all(isinstance(item, str) and item for item in required_evidence)
    ):
        raise ReproductionError("invalid S1-13 reproduction contract")

    config_sha256 = _sha256(
        identities.get("resolved_config_sha256"), "resolved config identity"
    )
    asset_marker_sha256 = _sha256(
        identities.get("asset_marker_sha256"), "asset marker identity"
    )
    gate_contract_sha256 = _sha256(
        identities.get("gate_contract_sha256"), "gate contract identity"
    )
    resolved = config.get("resolved_runtime_fields")
    if not isinstance(resolved, dict):
        raise ReproductionError("resolved config omits runtime fields")
    fragment_bytes_value = resolved.get("fragment_bytes")
    if not isinstance(fragment_bytes_value, list):
        raise ReproductionError("resolved config omits fragment byte sizes")
    fragment_bytes = [
        _integer(value, f"fragment_bytes[{index}]", minimum=1)
        for index, value in enumerate(fragment_bytes_value)
    ]
    if len(fragment_bytes) != 4:
        raise ReproductionError("reproduction requires four fragment byte sizes")

    learner_path = result_root / "roles" / "learner-00.json"
    syncer_path = result_root / "roles" / "syncer.json"
    learner_log_path = result_root / "logs" / "learner-00.jsonl"
    syncer_log_path = result_root / "logs" / "syncer.jsonl"
    learner = _read_object(learner_path)
    syncer = _read_object(syncer_path)
    learner_rows = _records(learner_log_path)
    syncer_rows = _records(syncer_log_path)
    staged = _event(learner_rows, "fragment_snapshot_staged")
    terminal = _event(learner_rows, "proposal_publication_terminal")
    published = [row for row in terminal if row.get("outcome") == "published"]
    updates = _event(syncer_rows, "fragment_outer_update")

    expected_updates = _integer(
        runtime.get("expected_fragment_updates"),
        "expected_fragment_updates",
        minimum=1,
    )
    target_cycles = _integer(
        runtime.get("target_global_cycles"), "target_global_cycles", minimum=1
    )
    learner_identity = learner.get("identity")
    syncer_identity = syncer.get("identity")
    if not isinstance(learner_identity, dict) or not isinstance(syncer_identity, dict):
        raise ReproductionError("role identity objects are required")
    learner_host = learner_identity.get("hostname")
    syncer_host = syncer_identity.get("hostname")
    learner_gpu = learner_identity.get("gpu")
    if not isinstance(learner_gpu, dict):
        raise ReproductionError("learner GPU identity is required")
    shared_devices = {
        learner_identity.get("shared_device"),
        syncer_identity.get("shared_device"),
    }
    shared_device_ids = sorted(
        value
        for value in shared_devices
        if isinstance(value, int) and not isinstance(value, bool)
    )
    topology_pass = (
        topology.get("learners") == 1
        and topology.get("syncers") == 1
        and topology.get("distinct_compute_hosts") == 2
        and topology.get("learner_gpus") == 1
        and topology.get("syncer_gpu_count") == 0
        and topology.get("syncer_torch_module_imported") is False
        and topology.get("application_data_plane") == "shared_filesystem_only"
        and topology.get("launcher_ranks") == 2
        and learner_host != syncer_host
        and all(
            isinstance(value, str) and value for value in (learner_host, syncer_host)
        )
        and learner_gpu.get("cuda_available") is True
        and isinstance(learner_gpu.get("gpu_uuid"), str)
        and learner_identity.get("role") == "learner-00"
        and syncer_identity.get("role") == "syncer"
        and learner_identity.get("pbs_job_id")
        == syncer_identity.get("pbs_job_id")
        == job_id
        and all(
            isinstance(identity.get("pbs_qtime_utc"), str)
            and str(identity.get("pbs_qtime_utc")).endswith("Z")
            for identity in (learner_identity, syncer_identity)
        )
        and syncer_identity.get("gpu_count") == 0
        and syncer_identity.get("torch_module_imported") is False
        and syncer_identity.get("cuda_visible_devices") in {"", "-1"}
        and len(shared_devices) == 1
        and None not in shared_devices
        and syncer.get("active", {}).get("application_coordination")
        == "shared_filesystem_only"
    )

    identity_pass = (
        config.get("selected_workload") == "nine_node"
        and _hash_file(config_path) == config_sha256
        and _hash_file(asset_root / "complete.json") == asset_marker_sha256
        and _hash_file(gate_contract_path) == gate_contract_sha256
        and _hash_file(result_root / "contracts" / "resolved-config.json")
        == config_sha256
        and _hash_file(result_root / "contracts" / "gate-contract.json")
        == gate_contract_sha256
        and _hash_file(result_root / "contracts" / "reproduction-contract.json")
        == _hash_file(reproduction_contract_path)
        and resolved.get("asset_bundle_root") == str(asset_root.resolve())
        and resolved.get("asset_marker_sha256") == asset_marker_sha256
        and learner_identity.get("shared_root") == str(shared_root.resolve())
        and syncer_identity.get("shared_root") == str(shared_root.resolve())
        and (result_root / "env" / "code-commit.txt")
        .read_text(encoding="utf-8")
        .strip()
        == expected_commit
        and not (result_root / "env" / "git-status.txt")
        .read_text(encoding="utf-8")
        .strip()
        and learner.get("status") == syncer.get("status") == "pass"
        and learner.get("run_id") == syncer.get("run_id") == run_id
        and _identity_pass(
            learner,
            syncer,
            run_id=run_id,
            config_sha256=config_sha256,
            asset_marker_sha256=asset_marker_sha256,
            gate_contract_sha256=gate_contract_sha256,
        )
    )

    expected_by_fragment = dict(enumerate(fragment_bytes))

    def matches_fragment_payload(row: Mapping[str, Any]) -> bool:
        index = row.get("fragment_index")
        return (
            isinstance(index, int)
            and not isinstance(index, bool)
            and expected_by_fragment.get(index) == row.get("payload_bytes")
        )

    staged_payload_pass = len(staged) == expected_updates and all(
        matches_fragment_payload(row)
        and row.get("identity_materialization") == "background_publisher"
        and row.get("parameters_sha256") is None
        and row.get("proposal_content_identity") is None
        for row in staged
    )
    proposal_ids = [row.get("proposal_id") for row in published]
    materialization_seconds = []
    materialization_pass = (
        len(published) == expected_updates
        and len(set(proposal_ids)) == expected_updates
    )
    for row in published:
        index = row.get("fragment_index")
        elapsed = row.get("proposal_materialization_seconds")
        started = row.get("proposal_materialization_started_monotonic_ns")
        completed = row.get("proposal_materialization_completed_monotonic_ns")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or expected_by_fragment.get(index) != row.get("payload_bytes")
            or not isinstance(started, int)
            or not isinstance(completed, int)
            or completed < started
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
            or not isinstance(row.get("parameters_sha256"), str)
            or len(str(row.get("parameters_sha256"))) != 64
            or not isinstance(row.get("proposal_content_identity"), str)
            or len(str(row.get("proposal_content_identity"))) != 64
        ):
            materialization_pass = False
            continue
        materialization_seconds.append(float(elapsed))

    publication_value = learner.get("publication")
    publication = (
        publication_value.get("publication")
        if isinstance(publication_value, dict)
        else None
    )
    if not isinstance(publication, dict):
        raise ReproductionError("learner publication summary is required")
    summary_materialization_seconds = _number(
        publication.get("proposal_materialization_seconds"),
        "proposal_materialization_seconds",
    )
    publication_pass = (
        publication.get("published_snapshot_count") == expected_updates
        and publication.get("captured_snapshot_count") == expected_updates
        and publication.get("snapshot_replacement_count") == 0
        and publication.get("snapshot_skip_count") == 0
        and publication.get("pending_upload_count") == 0
        and publication.get("in_flight_publication_count") == 0
        and publication.get("terminal_emits_in_progress") == 0
        and publication.get("errors") == []
        and publication.get("terminal_outcome_counts")
        == {"published": expected_updates}
        and math.isclose(
            summary_materialization_seconds,
            sum(materialization_seconds),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )

    byte_accounting_pass = len(updates) == expected_updates
    update_latencies = []
    completion_times = []
    for row in updates:
        index = row.get("fragment_index")
        accounting = row.get("byte_accounting")
        if not isinstance(index, int) or not isinstance(accounting, dict):
            byte_accounting_pass = False
            continue
        expected = expected_by_fragment.get(index)
        if (
            expected is None
            or accounting.get("fragment_bytes") != expected
            or accounting.get("current_input_bytes") != expected
            or accounting.get("local_payload_bytes") != expected
            or accounting.get("successor_output_bytes") != expected
            or accounting.get("retained_base_bytes") != 0
            or accounting.get("retained_base_reads") != 0
            or accounting.get("full_model_operations") != 0
        ):
            byte_accounting_pass = False
        update_latencies.append(
            _number(row.get("update_latency_seconds"), "update latency", positive=True)
        )
        completion_times.append(
            _integer(
                row.get("completed_unix_ns"), "update completed_unix_ns", minimum=1
            )
        )

    readiness = syncer.get("readiness")
    if not isinstance(readiness, dict):
        raise ReproductionError("syncer readiness summary is required")
    misses = _integer(readiness.get("proposal_payload_cache_misses"), "cache misses")
    hits = _integer(readiness.get("proposal_payload_cache_hits"), "cache hits")
    cache_pass = (
        misses == expected_updates
        and hits > 0
        and readiness.get("resident_latest_proposals") == len(fragment_bytes)
        and _integer(readiness.get("fixed_slot_reads"), "fixed slot reads")
        >= misses + hits
    )

    completion = syncer.get("completion")
    active = syncer.get("active")
    if not isinstance(completion, dict) or not isinstance(active, dict):
        raise ReproductionError("syncer active and completion records are required")
    active_seconds = (
        _integer(completion.get("active_end_unix_ns"), "active end", minimum=1)
        - _integer(active.get("active_start_unix_ns"), "active start", minimum=1)
    ) / 1e9
    mean_cycle_seconds = active_seconds / target_cycles
    target_pass = (
        completion.get("global_cycle") == target_cycles
        and completion.get("version_vector") == [target_cycles] * len(fragment_bytes)
        and syncer.get("update_count") == expected_updates
        and active_seconds > 0
        and active_seconds
        <= _number(
            runtime.get("maximum_active_runtime_seconds"),
            "maximum active runtime",
            positive=True,
        )
        and mean_cycle_seconds
        <= _number(
            runtime.get("maximum_mean_cycle_seconds"),
            "maximum mean cycle seconds",
            positive=True,
        )
        and update_latencies
        and statistics.median(update_latencies)
        <= _number(
            runtime.get("maximum_median_fragment_update_seconds"),
            "maximum median fragment update seconds",
            positive=True,
        )
    )

    strictly_increasing = all(
        right > left for left, right in zip(completion_times, completion_times[1:])
    )
    per_fragment_times: dict[int, list[int]] = {}
    for row in updates:
        index = row.get("fragment_index")
        completed = row.get("completed_unix_ns")
        if isinstance(index, int) and isinstance(completed, int):
            per_fragment_times.setdefault(index, []).append(completed)
    same_fragment_intervals = [
        (right - left) / 1e9
        for times in per_fragment_times.values()
        for left, right in zip(times, times[1:])
    ]
    consecutive_gaps = [
        (right - left) / 1e9
        for left, right in zip(completion_times, completion_times[1:])
    ]
    normal_interval = (
        statistics.median(same_fragment_intervals)
        if same_fragment_intervals
        else math.nan
    )
    maximum_gap = max(consecutive_gaps) if consecutive_gaps else math.inf
    gap_multiple = _number(
        runtime.get("maximum_unexpected_heartbeat_gap_multiple"),
        "heartbeat gap multiple",
        positive=True,
    )
    cadence_pass = (
        strictly_increasing
        and math.isfinite(normal_interval)
        and normal_interval > 0
        and maximum_gap <= gap_multiple * normal_interval
    )

    stream_pass = (
        syncer.get("update_stream")
        == {
            "path": "logs/syncer.jsonl",
            "event": "fragment_outer_update",
            "count": expected_updates,
            "retained_in_memory": 0,
        }
        and isinstance(syncer.get("inventory_stream"), dict)
        and syncer["inventory_stream"].get("path") == "logs/syncer.jsonl"
        and syncer["inventory_stream"].get("event") == "bounded_storage_inventory"
        and syncer["inventory_stream"].get("retained_in_memory") == 0
        and syncer.get("forbidden_runtime", {}).get("in_memory_update_history") == 0
        and syncer.get("forbidden_runtime", {}).get("in_memory_inventory_history") == 0
    )
    stderr_files = sorted((result_root / "stderr").glob("*"))
    stderr_pass = bool(stderr_files) and all(
        path.stat().st_size == 0 for path in stderr_files
    )

    checks = {
        "identity": identity_pass,
        "topology": topology_pass,
        "staged_payloads": staged_payload_pass,
        "background_materialization": materialization_pass and publication_pass,
        "single_current_parameter_accounting": byte_accounting_pass,
        "readiness_cache_reconciliation": cache_pass,
        "target_and_throughput": target_pass,
        "cycle_cadence": cadence_pass,
        "bounded_live_state": stream_pass,
        "empty_stderr": stderr_pass,
    }
    passed = all(checks.values())
    gate = {
        "schema_version": 1,
        "loop_id": "S1-13",
        "gate": "corrected_two_node_filesystem_reproduction",
        "status": "pass" if passed else "fail",
        "run_id": run_id,
        "pbs_job_id": job_id,
        "code_commit": expected_commit,
        "identities": {
            "resolved_config_sha256": config_sha256,
            "asset_marker_sha256": asset_marker_sha256,
            "gate_contract_sha256": gate_contract_sha256,
            "reproduction_contract_sha256": _hash_file(reproduction_contract_path),
        },
        "role_map": {
            "learner-00": {
                "hostname": learner_host,
                "gpu_uuid": learner_gpu.get("gpu_uuid"),
                "cuda": True,
            },
            "syncer": {
                "hostname": syncer_host,
                "gpu_count": syncer_identity.get("gpu_count"),
                "torch_module_imported": syncer_identity.get("torch_module_imported"),
                "cuda": False,
            },
            "shared_filesystem_device_ids": shared_device_ids,
        },
        "fragment_payload_bytes": fragment_bytes,
        "publication": {
            "staged": len(staged),
            "published": len(published),
            "distinct_proposal_ids": len(set(proposal_ids)),
            "materialization_trace_count": len(materialization_seconds),
            "materialization_seconds": summary_materialization_seconds,
        },
        "readiness_cache": {
            "payload_cache_misses": misses,
            "payload_cache_hits": hits,
            "fixed_slot_reads": readiness.get("fixed_slot_reads"),
            "resident_latest_proposals": readiness.get("resident_latest_proposals"),
        },
        "runtime": {
            "global_cycle": completion.get("global_cycle"),
            "fragment_updates": len(updates),
            "active_runtime_seconds": active_seconds,
            "mean_cycle_seconds": mean_cycle_seconds,
            "median_fragment_update_seconds": (
                statistics.median(update_latencies) if update_latencies else None
            ),
            "normal_same_fragment_interval_seconds": normal_interval,
            "maximum_consecutive_update_gap_seconds": maximum_gap,
            "maximum_allowed_gap_multiple": gap_multiple,
        },
        "checks": checks,
    }
    gate_path = result_root / "analysis" / "correction-reproduction-gate.json"
    _write_new_json(gate_path, gate)
    role_map_path = result_root / "env" / "role-map.json"
    _write_new_json(role_map_path, gate["role_map"])
    if not passed:
        failed = ", ".join(name for name, value in checks.items() if not value)
        raise ReproductionError(f"correction reproduction gates failed: {failed}")

    required_before_validation = {
        str(item)
        for item in required_evidence
        if item != "analysis/package-validator.json"
    }
    missing = sorted(
        relative
        for relative in required_before_validation
        if not (result_root / relative).is_file()
    )
    if missing:
        raise ReproductionError(
            "reproduction package omits required evidence: " + ", ".join(missing)
        )
    artifacts = _artifact_inventory(result_root)
    manifest = {
        "schema_version": 1,
        "loop_id": "S1-13",
        "kind": "corrected_two_node_filesystem_reproduction_package",
        "status": "ready_for_validation",
        "run_id": run_id,
        "pbs_job_id": job_id,
        "code_commit": expected_commit,
        "shared_root": str(shared_root.resolve()),
        "result_root": str(result_root.resolve()),
        "contract_sha256": _hash_file(reproduction_contract_path),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_new_json(result_root / "package-manifest.json", manifest)
    return gate


def validate_package(
    *, result_root: Path, reproduction_contract_path: Path
) -> dict[str, Any]:
    contract = _read_object(reproduction_contract_path)
    manifest = _read_object(result_root / "package-manifest.json")
    gate = _read_object(result_root / "analysis" / "correction-reproduction-gate.json")
    required = contract.get("required_evidence")
    artifacts = manifest.get("artifacts")
    if (
        contract.get("schema_version") != 1
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "ready_for_validation"
        or manifest.get("contract_sha256") != _hash_file(reproduction_contract_path)
        or gate.get("status") != "pass"
        or not isinstance(required, list)
        or not isinstance(artifacts, list)
    ):
        raise ReproductionError("reproduction package authority is invalid")
    current = _artifact_inventory(result_root)
    if artifacts != current or manifest.get("artifact_count") != len(current):
        raise ReproductionError("reproduction artifact inventory changed")
    for row in current:
        if not isinstance(row, dict):
            raise ReproductionError("artifact inventory row is invalid")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative:
            raise ReproductionError("artifact path is invalid")
        path = result_root / relative
        if row.get("bytes") != path.stat().st_size or row.get("sha256") != _hash_file(
            path
        ):
            raise ReproductionError(f"artifact checksum mismatch: {relative}")
    missing = sorted(
        str(relative)
        for relative in required
        if relative != "analysis/package-validator.json"
        and not (result_root / str(relative)).is_file()
    )
    if missing:
        raise ReproductionError(
            "validated package omits required evidence: " + ", ".join(missing)
        )
    result = {
        "schema_version": 1,
        "loop_id": "S1-13",
        "validator": "corrected_two_node_filesystem_reproduction",
        "status": "admissible",
        "run_id": manifest.get("run_id"),
        "pbs_job_id": manifest.get("pbs_job_id"),
        "code_commit": manifest.get("code_commit"),
        "contract_sha256": manifest.get("contract_sha256"),
        "artifact_count": len(current),
        "artifact_inventory_recomputed": True,
        "semantic_gate_status": gate.get("status"),
    }
    _write_new_json(result_root / "analysis" / "package-validator.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the corrected S1-13 two-node reproduction"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--result-root", type=Path, required=True)
    analyze.add_argument("--shared-root", type=Path, required=True)
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--asset-root", type=Path, required=True)
    analyze.add_argument("--gate-contract", type=Path, required=True)
    analyze.add_argument("--reproduction-contract", type=Path, required=True)
    analyze.add_argument("--expected-commit", required=True)
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--job-id", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--result-root", type=Path, required=True)
    validate.add_argument("--reproduction-contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "analyze":
        result = analyze_reproduction(
            result_root=arguments.result_root,
            shared_root=arguments.shared_root,
            config_path=arguments.config,
            asset_root=arguments.asset_root,
            gate_contract_path=arguments.gate_contract,
            reproduction_contract_path=arguments.reproduction_contract,
            expected_commit=arguments.expected_commit,
            run_id=arguments.run_id,
            job_id=arguments.job_id,
        )
    else:
        result = validate_package(
            result_root=arguments.result_root,
            reproduction_contract_path=arguments.reproduction_contract,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
