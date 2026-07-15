from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from fsbdd.auxiliary.stage1.reproduction import (
    ReproductionError,
    analyze_reproduction,
    validate_package,
)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(
    root: Path, mutation: Callable[[dict[str, Any]], None] | None = None
) -> dict[str, Any]:
    result_root = root / "result"
    shared_root = root / "shared"
    asset_root = root / "assets"
    shared_root.mkdir(parents=True)
    _dump(asset_root / "complete.json", {"status": "complete"})
    gate_path = root / "gate.json"
    _dump(gate_path, {"schema_version": 1, "loop_id": "S1-13"})
    fragment_bytes = [40, 44, 48, 52]
    config_path = root / "config.json"
    _dump(
        config_path,
        {
            "selected_workload": "nine_node",
            "resolved_runtime_fields": {
                "asset_bundle_root": str(asset_root.resolve()),
                "asset_marker_sha256": _digest(asset_root / "complete.json"),
                "fragment_bytes": fragment_bytes,
            },
        },
    )
    contract_path = root / "contract.json"
    required = [
        "analysis/correction-reproduction-gate.json",
        "analysis/package-validator.json",
        "contracts/gate-contract.json",
        "contracts/reproduction-contract.json",
        "contracts/resolved-config.json",
        "env/code-commit.txt",
        "env/git-status.txt",
        "env/role-map.json",
        "logs/learner-00.jsonl",
        "logs/syncer.jsonl",
        "roles/learner-00.json",
        "roles/syncer.json",
        "stderr/mpirun.log",
    ]
    contract = {
        "schema_version": 1,
        "loop_id": "S1-13",
        "kind": "corrected_two_node_filesystem_reproduction",
        "identities": {
            "resolved_config_sha256": _digest(config_path),
            "asset_marker_sha256": _digest(asset_root / "complete.json"),
            "gate_contract_sha256": _digest(gate_path),
        },
        "topology": {},
        "runtime": {
            "target_global_cycles": 2,
            "expected_fragment_updates": 8,
            "maximum_active_runtime_seconds": 30,
            "maximum_mean_cycle_seconds": 15,
            "maximum_median_fragment_update_seconds": 2,
            "maximum_unexpected_heartbeat_gap_multiple": 2,
        },
        "required_evidence": required,
    }
    _dump(contract_path, contract)
    for source, destination in (
        (config_path, result_root / "contracts" / "resolved-config.json"),
        (gate_path, result_root / "contracts" / "gate-contract.json"),
        (contract_path, result_root / "contracts" / "reproduction-contract.json"),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (result_root / "env").mkdir(parents=True, exist_ok=True)
    (result_root / "env" / "code-commit.txt").write_text(
        "a" * 40 + "\n", encoding="utf-8"
    )
    (result_root / "env" / "git-status.txt").write_text("", encoding="utf-8")
    (result_root / "stderr").mkdir(parents=True, exist_ok=True)
    (result_root / "stderr" / "mpirun.log").write_text("", encoding="utf-8")

    common_identity = {
        "run_id": "reproduction-test",
        "config_sha256": _digest(config_path),
        "asset_marker_sha256": _digest(asset_root / "complete.json"),
        "gate_contract_sha256": _digest(gate_path),
        "execution_mode": "reduced_two_node_reproduction",
        "shared_device": 17,
        "shared_root": str(shared_root.resolve()),
        "pbs_job_id": "test.opbs",
        "pbs_qtime_utc": "2026-07-15T00:00:00Z",
    }
    staged = []
    terminal = []
    updates = []
    for sequence in range(2):
        for index, size in enumerate(fragment_bytes):
            proposal_id = f"p-{index}-{sequence}"
            staged.append(
                {
                    "event": "fragment_snapshot_staged",
                    "fragment_index": index,
                    "payload_bytes": size,
                    "identity_materialization": "background_publisher",
                    "parameters_sha256": None,
                    "proposal_content_identity": None,
                }
            )
            terminal.append(
                {
                    "event": "proposal_publication_terminal",
                    "fragment_index": index,
                    "payload_bytes": size,
                    "proposal_id": proposal_id,
                    "outcome": "published",
                    "proposal_materialization_started_monotonic_ns": 10,
                    "proposal_materialization_completed_monotonic_ns": 20,
                    "proposal_materialization_seconds": 0.01,
                    "parameters_sha256": f"{index + 1:064x}",
                    "proposal_content_identity": f"{index + 9:064x}",
                }
            )
            ordinal = sequence * 4 + index + 1
            updates.append(
                {
                    "event": "fragment_outer_update",
                    "fragment_index": index,
                    "global_cycle_after": sequence + int(index == 3),
                    "completed_unix_ns": (ordinal + 1) * 1_000_000_000,
                    "update_latency_seconds": 1.0,
                    "byte_accounting": {
                        "fragment_bytes": size,
                        "current_input_bytes": size,
                        "local_payload_bytes": size,
                        "successor_output_bytes": size,
                        "retained_base_bytes": 0,
                        "retained_base_reads": 0,
                        "full_model_operations": 0,
                    },
                }
            )
    publication = {
        "published_snapshot_count": 8,
        "captured_snapshot_count": 8,
        "snapshot_replacement_count": 0,
        "snapshot_skip_count": 0,
        "pending_upload_count": 0,
        "in_flight_publication_count": 0,
        "terminal_emits_in_progress": 0,
        "errors": [],
        "terminal_outcome_counts": {"published": 8},
        "proposal_materialization_seconds": 0.08,
    }
    learner = {
        "status": "pass",
        "run_id": "reproduction-test",
        "identity": {
            **common_identity,
            "role": "learner-00",
            "hostname": "learner-host",
            "gpu": {
                "cuda_available": True,
                "gpu_uuid": "GPU-test",
            },
        },
        "publication": {"publication": publication},
    }
    syncer = {
        "status": "pass",
        "run_id": "reproduction-test",
        "identity": {
            **common_identity,
            "role": "syncer",
            "hostname": "syncer-host",
            "gpu_count": 0,
            "torch_module_imported": False,
            "cuda_visible_devices": "",
        },
        "active": {
            "active_start_unix_ns": 1_000_000_000,
            "application_coordination": "shared_filesystem_only",
        },
        "completion": {
            "active_end_unix_ns": 21_000_000_000,
            "global_cycle": 2,
            "version_vector": [2, 2, 2, 2],
        },
        "update_count": 8,
        "update_stream": {
            "path": "logs/syncer.jsonl",
            "event": "fragment_outer_update",
            "count": 8,
            "retained_in_memory": 0,
        },
        "inventory_stream": {
            "path": "logs/syncer.jsonl",
            "event": "bounded_storage_inventory",
            "count": 1,
            "retained_in_memory": 0,
        },
        "readiness": {
            "proposal_payload_cache_misses": 8,
            "proposal_payload_cache_hits": 24,
            "resident_latest_proposals": 4,
            "fixed_slot_reads": 32,
        },
        "forbidden_runtime": {
            "in_memory_update_history": 0,
            "in_memory_inventory_history": 0,
        },
    }
    values = {
        "learner": learner,
        "syncer": syncer,
        "staged": staged,
        "terminal": terminal,
        "updates": updates,
    }
    if mutation is not None:
        mutation(values)
    _dump(result_root / "roles" / "learner-00.json", learner)
    _dump(result_root / "roles" / "syncer.json", syncer)
    _jsonl(result_root / "logs" / "learner-00.jsonl", staged + terminal)
    _jsonl(result_root / "logs" / "syncer.jsonl", updates)
    return {
        "result_root": result_root,
        "shared_root": shared_root,
        "config_path": config_path,
        "asset_root": asset_root,
        "gate_contract_path": gate_path,
        "reproduction_contract_path": contract_path,
        "expected_commit": "a" * 40,
        "run_id": "reproduction-test",
        "job_id": "test.opbs",
    }


def test_reproduction_analyzer_and_package_validator_accept_complete_fixture(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    gate = analyze_reproduction(**arguments)
    assert gate["status"] == "pass"
    validator = validate_package(
        result_root=arguments["result_root"],
        reproduction_contract_path=arguments["reproduction_contract_path"],
    )
    assert validator["status"] == "admissible"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["updates"][0]["byte_accounting"].__setitem__(
            "retained_base_bytes", 40
        ),
        lambda value: value["syncer"]["readiness"].__setitem__(
            "proposal_payload_cache_misses", 7
        ),
        lambda value: value["terminal"][0].__setitem__(
            "proposal_materialization_seconds", None
        ),
        lambda value: value["syncer"]["identity"].__setitem__(
            "execution_mode", "formal_workload"
        ),
    ],
)
def test_reproduction_analyzer_fails_closed_on_correction_mutations(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], None]
) -> None:
    arguments = _fixture(tmp_path, mutation)
    with pytest.raises(ReproductionError):
        analyze_reproduction(**arguments)
    gate = json.loads(
        (
            arguments["result_root"] / "analysis" / "correction-reproduction-gate.json"
        ).read_text(encoding="utf-8")
    )
    assert gate["status"] == "fail"
