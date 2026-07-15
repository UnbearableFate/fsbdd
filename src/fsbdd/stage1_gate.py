from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class Stage1GateError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage1GateError(f"JSON object required: {path}")
    return value


def _write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise Stage1GateError(f"refusing to overwrite gate report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _records(path: Path, event: str) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise Stage1GateError(f"invalid JSONL at {path}:{line_number}") from error
            if value.get("event") == event:
                result.append(value)
    return result


def _rolling_medians(values: Sequence[float], window: int) -> list[float]:
    return [statistics.median(values[index - window : index]) for index in range(window, len(values) + 1)]


def _fraction_median(values: Sequence[float], fraction: float, *, final: bool) -> float:
    count = max(1, math.ceil(len(values) * fraction))
    return float(statistics.median(values[-count:] if final else values[:count]))


def _robust_lag_slope(values: Sequence[float], minimum_lag: int) -> float:
    lag = max(minimum_lag, len(values) // 10)
    if len(values) <= lag:
        raise Stage1GateError("too few points for robust lag slope")
    slopes = [(values[index + lag] - values[index]) / lag for index in range(len(values) - lag)]
    return float(statistics.median(slopes))


def _loss_gate(
    result_root: Path,
    roles: Sequence[Mapping[str, Any]],
    *,
    workload: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if workload == "nine_node":
        contract = config["loss_gate"]
        warmup = int(contract["warmup_points"])
        window = int(contract["rolling_median_window"])
        initial_fraction = float(contract["initial_fraction"])
        final_fraction = float(contract["final_fraction"])
        maximum_ratio = float(contract["maximum_final_to_initial_median"])
        maximum_learner_ratio = float(contract["maximum_per_learner_final_to_initial"])
        minimum_points = int(config["workloads"][workload]["minimum_loss_points"])
    else:
        warmup = 100
        window = 20
        initial_fraction = 0.05
        final_fraction = 0.05
        maximum_ratio = 0.99
        maximum_learner_ratio: float | None = None
        minimum_points = int(config["workloads"][workload]["optimizer_steps_per_learner"])
    learner_rows = []
    series = []
    for role in sorted(roles, key=lambda item: int(item["learner_index"])):
        learner_id = str(role["learner_id"])
        events = _records(result_root / "logs" / f"{learner_id}.jsonl", "safe_boundary")
        steps = [int(item["local_optimizer_step"]) for item in events]
        losses = [float(item["token_weighted_loss"]) for item in events]
        target_tokens = [int(item["loss_bearing_target_tokens_step"]) for item in events]
        finite = all(math.isfinite(item) for item in losses)
        contiguous = steps == list(range(1, len(steps) + 1))
        if len(losses) < minimum_points or not finite or not contiguous:
            raise Stage1GateError(f"loss stream failed basic checks for {learner_id}")
        smoothed = _rolling_medians(losses[warmup:], window)
        initial = _fraction_median(smoothed, initial_fraction, final=False)
        final = _fraction_median(smoothed, final_fraction, final=True)
        ratio = final / initial
        learner_rows.append(
            {
                "learner_id": learner_id,
                "points": len(losses),
                "finite": finite,
                "contiguous_local_steps": contiguous,
                "initial_rolling_median": initial,
                "final_rolling_median": final,
                "final_to_initial": ratio,
                "maximum_final_to_initial": maximum_learner_ratio,
                "pass": maximum_learner_ratio is None or ratio <= maximum_learner_ratio,
            }
        )
        series.append((losses, target_tokens))
    common = min(len(item[0]) for item in series)
    aggregate = []
    for index in range(common):
        numerator = sum(losses[index] * tokens[index] for losses, tokens in series)
        denominator = sum(tokens[index] for _losses, tokens in series)
        aggregate.append(numerator / denominator)
    aggregate_smoothed = _rolling_medians(aggregate[warmup:], window)
    initial = _fraction_median(aggregate_smoothed, initial_fraction, final=False)
    final = _fraction_median(aggregate_smoothed, final_fraction, final=True)
    ratio = final / initial
    slope = _robust_lag_slope(aggregate_smoothed, window)
    validation = None
    validation_pass = True
    if workload == "nine_node":
        leader = next(item for item in roles if int(item["learner_index"]) == 0)
        validation = leader["validation"]
        validation_pass = (
            set(validation) == {"initial", "final"}
            and all(bool(validation[key]["finite"]) for key in ("initial", "final"))
            and all(int(validation[key]["packed_blocks"]) == 487 for key in ("initial", "final"))
            and all(bool(validation[key]["all_blocks"]) and not bool(validation[key]["shuffle"]) for key in ("initial", "final"))
        )
    passed = (
        all(item["pass"] for item in learner_rows)
        and ratio <= maximum_ratio
        and slope < 0
        and validation_pass
    )
    return {
        "schema_version": 1,
        "gate": "loss",
        "workload": workload,
        "status": "pass" if passed else "fail",
        "aggregation": "target-token-weighted loss at common local-step index",
        "smoothing": {"warmup_points": warmup, "rolling_median_window": window},
        "robust_slope": {
            "method": "median of fixed-lag finite differences",
            "lag": max(window, len(aggregate_smoothed) // 10),
            "value": slope,
            "must_be_negative": True,
        },
        "aggregate": {
            "common_points": common,
            "initial_median": initial,
            "final_median": final,
            "final_to_initial": ratio,
            "maximum_final_to_initial": maximum_ratio,
        },
        "learners": learner_rows,
        "validation": validation,
    }


def _topology_gate(
    roles: Sequence[Mapping[str, Any]],
    syncer: Mapping[str, Any],
    *,
    workload: str,
    expected_learners: int,
) -> dict[str, Any]:
    learner_hosts = [str(item["identity"]["hostname"]) for item in roles]
    syncer_host = str(syncer["identity"]["hostname"])
    gpu_uuids = [str(item["identity"]["gpu"]["gpu_uuid"]) for item in roles]
    job_ids = [str(item["identity"]["pbs_job_id"]) for item in roles] + [str(syncer["identity"]["pbs_job_id"])]
    qtimes = [item["identity"].get("pbs_qtime_utc") for item in roles] + [
        syncer["identity"].get("pbs_qtime_utc")
    ]
    hosts = learner_hosts + [syncer_host]
    require_independent = workload == "nine_node"
    run_ids = {str(item["run_id"]) for item in roles} | {str(syncer["run_id"])}
    config_identities = {str(item["identity"]["config_sha256"]) for item in roles} | {
        str(syncer["identity"]["config_sha256"])
    }
    asset_identities = {str(item["identity"]["asset_marker_sha256"]) for item in roles} | {
        str(syncer["identity"]["asset_marker_sha256"])
    }
    passed = (
        len(roles) == expected_learners
        and len(set(learner_hosts)) == expected_learners
        and len(set(gpu_uuids)) == expected_learners
        and syncer_host not in learner_hosts
        and syncer["identity"]["gpu_count"] == 0
        and syncer["identity"]["torch_module_imported"] is False
        and syncer["identity"]["cuda_visible_devices"] in {"", "-1"}
        and len(run_ids) == len(config_identities) == len(asset_identities) == 1
        and (not require_independent or len(set(job_ids)) == expected_learners + 1)
        and all(isinstance(value, str) and value.endswith("Z") for value in qtimes)
        and len({int(item["identity"]["shared_device"]) for item in roles} | {int(syncer["identity"]["shared_device"])}) == 1
    )
    return {
        "schema_version": 1,
        "gate": "topology",
        "workload": workload,
        "status": "pass" if passed else "fail",
        "expected_learners": expected_learners,
        "learner_hosts": learner_hosts,
        "syncer_host": syncer_host,
        "distinct_hosts": len(set(hosts)),
        "gpu_uuids": gpu_uuids,
        "distinct_gpu_uuids": len(set(gpu_uuids)),
        "pbs_job_ids": job_ids,
        "pbs_qtime_utc": qtimes,
        "run_ids": sorted(run_ids),
        "config_sha256": sorted(config_identities),
        "asset_marker_sha256": sorted(asset_identities),
        "independent_job_ids_required": require_independent,
        "shared_filesystem_device_ids": sorted({int(item["identity"]["shared_device"]) for item in roles} | {int(syncer["identity"]["shared_device"])}),
        "syncer_cpu_only": (
            syncer["identity"]["gpu_count"] == 0
            and syncer["identity"]["torch_module_imported"] is False
        ),
    }


def _runtime_gate(
    roles: Sequence[Mapping[str, Any]],
    syncer: Mapping[str, Any],
    *,
    workload: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    start = int(syncer["active"]["active_start_unix_ns"])
    end = int(syncer["completion"]["active_end_unix_ns"])
    active_seconds = (end - start) / 1e9
    budget = float(config["workloads"][workload]["runtime_budget_seconds"])
    step_latencies = []
    for role in roles:
        events = _records(Path(role["_result_root"]) / "logs" / f"{role['learner_id']}.jsonl", "safe_boundary")
        step_latencies.extend(float(item["step_latency_seconds"]) for item in events)
    passed = 0 < active_seconds <= budget and all(math.isfinite(item) and item > 0 for item in step_latencies)
    return {
        "schema_version": 1,
        "gate": "runtime",
        "workload": workload,
        "status": "pass" if passed else "fail",
        "active_start_unix_ns": start,
        "active_end_unix_ns": end,
        "active_runtime_seconds": active_seconds,
        "budget_seconds": budget,
        "budget_fraction": active_seconds / budget,
        "learner_step_latency_seconds": {
            "points": len(step_latencies),
            "median": statistics.median(step_latencies),
            "maximum": max(step_latencies),
        },
        "queue_time_excluded": True,
    }


def _protocol_gate(
    roles: Sequence[Mapping[str, Any]],
    syncer: Mapping[str, Any],
    *,
    workload: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    learner_count = len(roles)
    updates = syncer["updates"]
    per_fragment: dict[int, list[int]] = {}
    selections_pass = True
    byte_pass = True
    for item in updates:
        index = int(item["fragment_index"])
        per_fragment.setdefault(index, []).append(int(item["to_version"]))
        selections_pass &= (
            len(item["selected_learners"]) == learner_count
            and len(set(item["selected_learners"])) == learner_count
            and all(int(value) == 0 for value in item["staleness"])
            and math.isclose(sum(float(value) for value in item["weights"]), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        )
        accounting = item["byte_accounting"]
        byte_pass &= (
            int(accounting["local_payload_reads"]) == learner_count
            and int(item["source_metrics"]["maximum_active_payloads"]) == 1
            and int(accounting["full_model_operations"]) == 0
        )
    contiguous = all(values == list(range(1, len(values) + 1)) for values in per_fragment.values())
    global_cycle = int(syncer["progress"]["global_cycle"])
    target = int(
        config["workloads"][workload]["target_global_cycles"]
        if workload == "nine_node"
        else config["workloads"][workload]["minimum_global_cycles"]
    )
    tokens = sum(int(item["progress"]["processed_input_tokens"]) for item in roles)
    expected_tokens = (
        None
        if workload == "nine_node"
        else int(config["workloads"][workload]["expected_aggregate_processed_input_tokens"])
    )
    inventory = syncer["inventories"][-1]
    inventory_pass = (
        int(inventory["global"]["visibility_records"]) == 4
        and int(inventory["proposals"]["visibility_records"]) == learner_count * 4
        and int(inventory["global"]["record_temp_files"]) == 0
        and int(inventory["proposals"]["record_temp_files"]) == 0
    )
    forbidden_pass = (
        all(not bool(item["forbidden_runtime"]["torch_distributed_initialized"]) for item in roles)
        and not bool(syncer["forbidden_runtime"]["torch_distributed_initialized"])
        and int(syncer["forbidden_runtime"]["history_scan_operations"]) == 0
        and int(syncer["forbidden_runtime"]["full_model_operations"]) == 0
    )
    frozen_asset_identity_pass = all(
        item["identity"]["model_revision"]
        == config["workloads"][workload]["model_revision"]
        and item["identity"]["dataset_revision"]
        == config["workloads"][workload]["dataset_revision"]
        for item in roles
    )
    passed = (
        len(per_fragment) == 4
        and contiguous
        and global_cycle >= target
        and selections_pass
        and byte_pass
        and inventory_pass
        and forbidden_pass
        and frozen_asset_identity_pass
        and (expected_tokens is None or tokens == expected_tokens)
    )
    return {
        "schema_version": 1,
        "gate": "protocol",
        "workload": workload,
        "status": "pass" if passed else "fail",
        "profile": {"Q": learner_count, "Q_fresh": learner_count, "S_max": 0, "H": 50},
        "fragment_version_sequences": {str(key): value for key, value in sorted(per_fragment.items())},
        "contiguous_fragment_versions": contiguous,
        "global_cycle": global_cycle,
        "minimum_global_cycle": target,
        "selections_pass": selections_pass,
        "fragment_only_byte_accounting_pass": byte_pass,
        "aggregate_processed_input_tokens": tokens,
        "expected_aggregate_processed_input_tokens": expected_tokens,
        "bounded_storage_inventory": inventory,
        "bounded_inventory_pass": inventory_pass,
        "forbidden_runtime_pass": forbidden_pass,
        "frozen_model_dataset_identity_pass": frozen_asset_identity_pass,
        "readiness": syncer["readiness"],
    }


def analyze(result_root: Path, config_path: Path, output_root: Path) -> dict[str, Any]:
    config = _read(config_path)
    workload = str(config["selected_workload"])
    count = int(config["workloads"][workload]["learner_count"])
    roles = [_read(result_root / "roles" / f"learner-{index:02d}.json") for index in range(count)]
    syncer = _read(result_root / "roles" / "syncer.json")
    for role in roles:
        role["_result_root"] = str(result_root)
    topology = _topology_gate(roles, syncer, workload=workload, expected_learners=count)
    loss = _loss_gate(result_root, roles, workload=workload, config=config)
    runtime = _runtime_gate(roles, syncer, workload=workload, config=config)
    protocol = _protocol_gate(roles, syncer, workload=workload, config=config)
    for name, value in (("topology", topology), ("loss", loss), ("runtime", runtime), ("protocol", protocol)):
        _write_new(output_root / f"{name}_gate.json", value)
    passed = all(item["status"] == "pass" for item in (topology, loss, runtime, protocol))
    summary = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "workload": workload,
        "gates": {name: value["status"] for name, value in (("topology", topology), ("loss", loss), ("runtime", runtime), ("protocol", protocol))},
    }
    _write_new(output_root / "gate_summary.json", summary)
    if not passed:
        raise Stage1GateError(f"one or more {workload} gates failed")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze S1-13 topology/loss/runtime/protocol gates")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(analyze(arguments.result_root, arguments.config, arguments.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
