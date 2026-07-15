from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_closure_runtime_has_no_application_network_data_plane() -> None:
    source = (ROOT / "src/fsbdd/auxiliary/stage1/close.py").read_text(encoding="utf-8")
    for token in (
        "init_process_group(",
        "DistributedDataParallel(",
        "init_rpc(",
        "torchrun",
        "nccl",
    ):
        assert token not in source
    assert '"network_data_plane": False' in source
    assert '"application_coordination": "shared_filesystem_only"' in source
    assert (
        'minimum_step_seconds=float(execution_contract["minimum_step_seconds"])'
        in source
    )
    assert 'merge_backend="numpy"' in source


def test_formal_topologies_freeze_independent_9n_and_coallocated_4_plus_1() -> None:
    nine = (ROOT / "pbs/stage1_s1_13_formal_9n.pbs").read_text(encoding="utf-8")
    long = (ROOT / "pbs/stage1_s1_13_formal_long.pbs").read_text(encoding="utf-8")
    assert "#PBS -J 0-8" in nine
    assert "#PBS -l select=1" in nine
    assert "PBS_ARRAY_INDEX" in nine
    assert "mpirun" not in nine
    assert "fsbdd.auxiliary.stage1.submit validate-nine" in nine
    assert "SUBMISSION_MARKER_SHA256" in nine
    assert "#PBS -l select=5" in long
    assert "mpirun -np 5 --map-by ppr:1:node" in long
    assert "CUDA_VISIBLE_DEVICES" in long or "fsbdd.auxiliary.stage1.close mpi" in long
    assert (ROOT / "pbs/stage1_s1_13_numpy_correction.pbs").is_file()
    smoke = (ROOT / "pbs/stage1_s1_13_long_smoke.pbs").read_text(encoding="utf-8")
    assert "#PBS -l select=5" in smoke
    assert "--non-formal-long-smoke" in smoke
    assert "fsbdd.auxiliary.stage1.gate" in smoke
    targeted = (ROOT / "pbs/stage1_s1_13_targeted.pbs").read_text(encoding="utf-8")
    assert "EXPECTED_COMMIT" in targeted
    assert 'if [[ -e "$OUTPUT_ROOT" ]]' in targeted
    assert "compileall -q src tests" in targeted
    assert "uvx --offline ruff check src/fsbdd tests" in targeted
    assert 'uvx --offline pyright --pythonpath "$PYTHON" src/fsbdd' in targeted
    assert 'bash -n "$script"' in targeted


def test_formal_package_retains_current_protocol_samples_and_rejects_placeholders() -> (
    None
):
    source = (ROOT / "src/fsbdd/auxiliary/stage1/package.py").read_text(
        encoding="utf-8"
    )
    assert "_reject_placeholders(resolved_config)" in source
    assert (
        "_capture_current_protocol_samples(arguments.shared_root, evidence)" in source
    )
    assert "raw-metadata/classified-current-inventory.json" in source
    assert "nine-node package requires a submission marker" in source
    assert '"gate_contract_sha256"' in source
    assert '"reports/stage1/S1-13-experiment-failures.json"' in source


def test_unresolved_contract_freezes_both_non_substitutable_workloads() -> None:
    config = json.loads(
        (ROOT / "configs/stage1/s1_13_stage1_close.json").read_text(encoding="utf-8")
    )
    nine = config["workloads"]["nine_node"]
    long = config["workloads"]["long_run"]
    assert (
        nine["learner_count"],
        nine["syncer_count"],
        nine["target_global_cycles"],
    ) == (8, 1, 10)
    assert nine["independent_pbs_roles"] is True
    assert (long["learner_count"], long["syncer_count"]) == (4, 1)
    assert long["minimum_aggregate_processed_input_tokens"] >= 1_000_000_000
    assert long["expected_aggregate_processed_input_tokens"] == 1_000_005_632
    assert long["minimum_global_cycles"] == 2400
    assert config["protocol"]["H"] == 50
    assert config["protocol"]["torch_distributed"] is False


def test_frozen_gate_contract_covers_heartbeat_stalls_and_long_capacity_smoke() -> None:
    contract = json.loads(
        (ROOT / "reports/stage1/S1-13-gate-contract.json").read_text(encoding="utf-8")
    )
    runtime = contract["runtime"]
    smoke = contract["long_run"]["preflight_smoke"]
    assert runtime["maximum_unexpected_heartbeat_gap_multiple"] == 2.0
    assert (
        runtime["pending_stall_rejection"]["learner_publication_pending_upload_count"]
        == 0
    )
    assert smoke["optimizer_steps_per_learner"] == 3000
    assert smoke["publication_opportunities_per_fragment_per_learner"] == 60
    assert smoke["minimum_global_cycles"] == 59
    assert smoke["minimum_cycle_to_publication_opportunity_ratio"] >= 59 / 60
    assert contract["long_run"]["minimum_step_seconds"] == 0.16
    overlay = contract["protocol"]["long_run_learner_phase_overlay"]
    assert overlay["algorithm"] == "aligned_zero_for_q_equals_m_capacity_v1"
    assert overlay["learner_phase_offsets"] == [0, 0, 0, 0]
    assert overlay["learner_waits_for_syncer"] is False


def test_resolved_configs_bind_the_complete_immutable_asset_bundle() -> None:
    expected_marker = "6d13e621bb912495ba9e983a305991bf716f9307d24fe241d34a39e5f0b62c7f"
    asset_root = "/work/xg24i002/x10041/fsbdd/runtime_runs/S1-13/assets-2389469.opbs"
    for workload, learners in (("nine_node", 8), ("long_run", 4)):
        path = ROOT / "configs" / "stage1" / f"s1_13_{workload}_resolved.json"
        text = path.read_text(encoding="utf-8")
        config = json.loads(text)
        resolved = config["resolved_runtime_fields"]
        assert "asset_stage" not in text
        assert config["selected_workload"] == workload
        assert resolved["asset_bundle_root"] == asset_root
        assert resolved["asset_marker_sha256"] == expected_marker
        assert len(resolved["fragment_descriptors"]) == 4
        assert len(resolved["fragment_bytes"]) == 4
        assert len(resolved["per_learner_offsets"]) == learners
        assert all(len(offsets) == 4 for offsets in resolved["per_learner_offsets"])
    long = json.loads(
        (ROOT / "configs/stage1/s1_13_long_run_resolved.json").read_text(
            encoding="utf-8"
        )
    )["workloads"]["long_run"]
    assert long["optimizer_steps_per_learner"] == 122071
    assert long["expected_aggregate_processed_input_tokens"] == 1_000_005_632
    resolved = json.loads(
        (ROOT / "configs/stage1/s1_13_long_run_resolved.json").read_text(
            encoding="utf-8"
        )
    )["resolved_runtime_fields"]
    assert resolved["per_learner_offsets"] == [resolved["fragment_offsets"]] * 4
