from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_closure_runtime_has_no_application_network_data_plane() -> None:
    source = (ROOT / "src/fsbdd/stage1_close.py").read_text(encoding="utf-8")
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
    assert 'minimum_step_seconds=0.5 if workload == "nine_node" else None' in source
    assert 'merge_backend="numpy"' in source


def test_formal_topologies_freeze_independent_9n_and_coallocated_4_plus_1() -> None:
    nine = (ROOT / "pbs/stage1_s1_13_formal_9n.pbs").read_text(encoding="utf-8")
    long = (ROOT / "pbs/stage1_s1_13_formal_long.pbs").read_text(encoding="utf-8")
    assert "#PBS -J 0-8" in nine
    assert "#PBS -l select=1" in nine
    assert "PBS_ARRAY_INDEX" in nine
    assert "mpirun" not in nine
    assert "#PBS -l select=5" in long
    assert "mpirun -np 5 --map-by ppr:1:node" in long
    assert "CUDA_VISIBLE_DEVICES" in long or "stage1_close mpi" in long
    assert (ROOT / "pbs/stage1_s1_13_numpy_correction.pbs").is_file()


def test_formal_package_retains_current_protocol_samples_and_rejects_placeholders() -> None:
    source = (ROOT / "src/fsbdd/stage1_package.py").read_text(encoding="utf-8")
    assert "_reject_placeholders(resolved_config)" in source
    assert "_capture_current_protocol_samples(arguments.shared_root, evidence)" in source
    assert "raw-metadata/classified-current-inventory.json" in source


def test_unresolved_contract_freezes_both_non_substitutable_workloads() -> None:
    config = json.loads(
        (ROOT / "configs/stage1/s1_13_stage1_close.json").read_text(encoding="utf-8")
    )
    nine = config["workloads"]["nine_node"]
    long = config["workloads"]["long_run"]
    assert (nine["learner_count"], nine["syncer_count"], nine["target_global_cycles"]) == (8, 1, 10)
    assert nine["independent_pbs_roles"] is True
    assert (long["learner_count"], long["syncer_count"]) == (4, 1)
    assert long["minimum_aggregate_processed_input_tokens"] >= 1_000_000_000
    assert long["expected_aggregate_processed_input_tokens"] == 1_000_005_632
    assert long["minimum_global_cycles"] == 2400
    assert config["protocol"]["H"] == 50
    assert config["protocol"]["torch_distributed"] is False


def test_resolved_configs_bind_the_complete_immutable_asset_bundle() -> None:
    expected_marker = "6d13e621bb912495ba9e983a305991bf716f9307d24fe241d34a39e5f0b62c7f"
    asset_root = str(
        ROOT / "runtime_runs" / "S1-13" / "assets-2389469.opbs"
    )
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
