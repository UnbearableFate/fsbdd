from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest

from fsbdd.auxiliary.stage1 import close as stage1_close
from fsbdd.auxiliary.stage1.reproduction import (
    ReproductionError,
    analyze_reproduction,
    validate_preflight,
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


def _write_preliminary_inventory(result_root: Path) -> dict[str, str]:
    excluded = {"checksums.sha256", "analysis/package-validator.json"}
    entries = {
        path.relative_to(result_root).as_posix(): _digest(path)
        for path in result_root.rglob("*")
        if path.is_file()
        and path.relative_to(result_root).as_posix() not in excluded
    }
    (result_root / "checksums.sha256").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in sorted(entries.items())),
        encoding="utf-8",
    )
    return entries


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
        "env/binding-hostnames.txt",
        "env/code-commit.txt",
        "env/git-status.txt",
        "env/mpi-bindings.txt",
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
        "topology": {
            "learners": 1,
            "syncers": 1,
            "distinct_compute_hosts": 2,
            "learner_gpus": 1,
            "syncer_gpu_count": 0,
            "syncer_torch_module_imported": False,
            "application_data_plane": "shared_filesystem_only",
            "launcher_ranks": 2,
            "mpi_binding_policy": "none_with_report_bindings_evidence",
        },
        "runtime": {
            "target_global_cycles": 2,
            "expected_fragment_updates": 8,
            "maximum_active_runtime_seconds": 30,
            "maximum_mean_cycle_seconds": 15,
            "maximum_median_fragment_update_seconds": 2,
            "maximum_unexpected_heartbeat_gap_multiple": 2,
        },
        "required_evidence": required,
        "finalization": {
            "checksum_inventory": "checksums.sha256",
            "checksums_cover_every_retained_file_except_inventory_itself": True,
            "validator_runs_after_preliminary_checksums": True,
            "validator_appended_before_final_checksum_check": True,
        },
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
    (result_root / "env" / "binding-hostnames.txt").write_text(
        "learner-host\nsyncer-host\n", encoding="utf-8"
    )
    (result_root / "env" / "mpi-bindings.txt").write_text(
        "MCW rank 0 is not bound\nMCW rank 1 is not bound\n", encoding="utf-8"
    )
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
    preliminary = _write_preliminary_inventory(arguments["result_root"])
    assert "package-manifest.json" in preliminary
    validator = validate_package(
        result_root=arguments["result_root"],
        reproduction_contract_path=arguments["reproduction_contract_path"],
    )
    assert validator["status"] == "admissible"
    final_entries = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (arguments["result_root"] / "checksums.sha256")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    retained = {
        path.relative_to(arguments["result_root"]).as_posix()
        for path in arguments["result_root"].rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    assert set(final_entries) == retained
    assert "analysis/package-validator.json" in final_entries


def test_package_validator_rejects_post_manifest_artifact_mutation(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    analyze_reproduction(**arguments)
    _write_preliminary_inventory(arguments["result_root"])
    with (arguments["result_root"] / "logs" / "syncer.jsonl").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write('{}\n')
    with pytest.raises(ReproductionError, match="artifact inventory changed"):
        validate_package(
            result_root=arguments["result_root"],
            reproduction_contract_path=arguments["reproduction_contract_path"],
        )


def test_frozen_preflight_accepts_exact_clean_package(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    asset_root = tmp_path / "assets"
    config_path = project_root / "config.json"
    gate_path = project_root / "gate.json"
    contract_path = project_root / "contract.json"
    script_path = project_root / "pbs" / "stage1_s1_13_smoke_2n.pbs"
    preflight_script_path = (
        project_root / "pbs" / "stage1_s1_13_reproduction_preflight.pbs"
    )
    analyzer_source = (
        project_root
        / "src"
        / "fsbdd"
        / "auxiliary"
        / "stage1"
        / "reproduction.py"
    )
    dynamic_test = project_root / "tests" / "stage1" / "test_s1_13_reproduction.py"
    static_test = project_root / "tests" / "stage1" / "test_s1_13_static.py"
    _dump(asset_root / "complete.json", {"status": "complete"})
    _dump(gate_path, {"gate": "frozen"})
    _dump(
        config_path,
        {
            "resolved_runtime_fields": {
                "asset_bundle_root": str(asset_root.resolve()),
                "asset_marker_sha256": _digest(asset_root / "complete.json"),
            }
        },
    )
    _dump(
        contract_path,
        {
            "schema_version": 1,
            "loop_id": "S1-13",
            "kind": "corrected_two_node_filesystem_reproduction",
            "formal_evidence": False,
            "identities": {
                "resolved_config_sha256": _digest(config_path),
                "asset_marker_sha256": _digest(asset_root / "complete.json"),
                "gate_contract_sha256": _digest(gate_path),
            },
            "topology": {
                "learners": 1,
                "syncers": 1,
                "distinct_compute_hosts": 2,
                "launcher_ranks": 2,
                "mpi_binding_policy": "none_with_report_bindings_evidence",
            },
            "runtime": {
                "internal_role_timeout_seconds": 720,
                "supervisor_term_seconds": 900,
                "supervisor_kill_after_seconds": 30,
                "pbs_walltime_seconds": 1800,
            },
            "finalization": {
                "checksums_cover_every_retained_file_except_inventory_itself": True,
                "validator_runs_after_preliminary_checksums": True,
                "validator_appended_before_final_checksum_check": True,
            },
            "required_evidence": ["analysis/package-validator.json"],
        },
    )
    script_path.parent.mkdir(parents=True)
    script_path.write_text(
        "\n".join(
            (
                "EXPECTED_COMMIT REPRODUCTION_CONTRACT_SHA256",
                'if ! mkdir "$SHARED_ROOT"',
                'if ! mkdir "$RESULT_ROOT"',
                'FSBDD_FAILURE_OUTPUT_ROOT="$RESULT_ROOT"',
                "mpirun --bind-to none --report-bindings",
                '"$RESULT_ROOT/env/binding-hostnames.txt"',
                '"$RESULT_ROOT/env/mpi-bindings.txt"',
                "timeout --signal=TERM --kill-after=30s 900s",
                "mpirun -np 2 --map-by ppr:1:node --bind-to none",
                "--learner-count-override 1 --timeout-seconds 720",
                "fsbdd.auxiliary.stage1.reproduction analyze",
                "xargs -0 sha256sum > checksums.sha256",
                "! -path './analysis/package-validator.json'",
                "fsbdd.auxiliary.stage1.reproduction validate",
                "sha256sum -c checksums.sha256",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    preflight_script_path.write_text("# frozen preflight\n", encoding="utf-8")
    analyzer_source.parent.mkdir(parents=True)
    analyzer_source.write_text("# frozen analyzer\n", encoding="utf-8")
    dynamic_test.parent.mkdir(parents=True)
    dynamic_test.write_text("# frozen dynamic test\n", encoding="utf-8")
    static_test.write_text("# frozen static test\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(project_root)], check=True)
    subprocess.run(["git", "-C", str(project_root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "-c",
            "user.name=FSBDD Test",
            "-c",
            "user.email=fsbdd-test@example.invalid",
            "commit",
            "-qm",
            "frozen package",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = validate_preflight(
        project_root=project_root,
        config_path=config_path,
        asset_root=asset_root,
        gate_contract_path=gate_path,
        reproduction_contract_path=contract_path,
        expected_commit=commit,
        expected_reproduction_contract_sha256=_digest(contract_path),
        output_path=tmp_path / "preflight.json",
    )
    assert result["status"] == "admissible"
    analyzer_source.unlink()
    with pytest.raises(ReproductionError, match="frozen reproduction preflight failed"):
        validate_preflight(
            project_root=project_root,
            config_path=config_path,
            asset_root=asset_root,
            gate_contract_path=gate_path,
            reproduction_contract_path=contract_path,
            expected_commit=commit,
            expected_reproduction_contract_sha256=_digest(contract_path),
            output_path=tmp_path / "blocked-preflight.json",
        )
    blocked = json.loads(
        (tmp_path / "blocked-preflight.json").read_text(encoding="utf-8")
    )
    assert blocked["status"] == "blocked"
    assert blocked["checks"]["frozen_package_surface"] is False
    assert blocked["identities"]["analyzer_sha256"] is None


@pytest.mark.parametrize(
    ("rank", "expected_role", "expected_cuda"),
    [(0, "learner", "0"), (1, "syncer", "")],
)
def test_explicit_one_plus_one_mpi_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    expected_role: str,
    expected_cuda: str,
) -> None:
    calls: list[tuple[str, dict[str, Any], str | None]] = []

    def fake_learner(**kwargs: Any) -> dict[str, Any]:
        calls.append(("learner", kwargs, stage1_close.os.environ.get("CUDA_VISIBLE_DEVICES")))
        return {"status": "pass", "workload": "nine_node"}

    def fake_syncer(**kwargs: Any) -> dict[str, Any]:
        calls.append(("syncer", kwargs, stage1_close.os.environ.get("CUDA_VISIBLE_DEVICES")))
        return {"status": "pass", "workload": "nine_node"}

    monkeypatch.setattr(stage1_close, "run_learner", fake_learner)
    monkeypatch.setattr(stage1_close, "run_syncer", fake_syncer)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", str(rank))
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    result = stage1_close.main(
        [
            "mpi",
            "--shared-root",
            str(tmp_path / "shared"),
            "--result-root",
            str(tmp_path / "result"),
            "--run-id",
            "test-run",
            "--config-path",
            str(tmp_path / "config.json"),
            "--config-sha256",
            "a" * 64,
            "--asset-marker-sha256",
            "b" * 64,
            "--gate-contract",
            str(tmp_path / "gate.json"),
            "--gate-contract-sha256",
            "c" * 64,
            "--hub-cache",
            str(tmp_path / "hub"),
            "--learner-count-override",
            "1",
        ]
    )
    assert result == 0
    assert len(calls) == 1
    role, kwargs, cuda_visible_devices = calls[0]
    assert role == expected_role
    assert cuda_visible_devices == expected_cuda
    assert kwargs["learner_count_override"] == 1
    if expected_role == "learner":
        assert kwargs["learner_index"] == 0


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
