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
