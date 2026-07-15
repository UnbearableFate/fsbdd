from __future__ import annotations

import hashlib
import json
from pathlib import Path


CONFIG = Path("configs/stage1/s1_09_syncer_readiness.json")
CONTRACT = Path("reports/stage1/S1-09-evidence-contract.json")
MATRIX = Path("reports/stage1/S1-09-requirement-matrix.csv")
FORMAL_PBS = Path("pbs/stage1_s1_09_syncer_readiness.pbs")
TARGETED_PBS = Path("pbs/stage1_s1_09_targeted.pbs")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_readiness_profile_and_contract_are_exact() -> None:
    assert _sha256(CONFIG) == "8ff3fe364fca10bb48fe80ebf0bcede63163731e47cc26fc87512b298e3c225d"
    assert _sha256(CONTRACT) == "d9003b8400a50e55af660267ed3d5716d55f9f06be04c15f28758e4b9b8ca283"
    assert _sha256(MATRIX) == "6e9a45c76c77ce9780908f97813e82af763fa2a40af31da5532b4de495c8f25f"
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    profile = config["profile_a"]
    grace = config["decoupled_grace"]
    assert config["s_max"] == 0 and config["history_objects"] == 10_000
    assert (
        profile["learner_count"],
        profile["fragment_count"],
        profile["q"],
        profile["q_fresh"],
        profile["grace_period_ns"],
    ) == (8, 4, 8, 8, 0)
    assert (
        grace["q"],
        grace["q_fresh"],
        grace["grace_period_ns"],
        grace["initial_learners"],
        grace["pre_freeze_late_learners"],
        grace["post_freeze_late_learners"],
    ) == (4, 4, 500_000_000, 4, 1, 1)


def test_formal_pbs_freezes_two_node_filesystem_only_contract() -> None:
    source = FORMAL_PBS.read_text(encoding="utf-8")
    assert "#PBS -q debug-g" in source
    assert "#PBS -l select=2" in source
    assert "mpirun -np 2 --map-by ppr:1:node" in source
    assert "fsbdd.syncer_readiness_stress role" in source
    assert "EXPECTED_COMMIT ==" not in source
    assert "status --porcelain" in source
    assert "evidence finalize" in source and "evidence validate" in source
    assert "--config-identity \"$CONFIG_SHA256\"" in source
    assert "tests/stage1" in source
    for forbidden in ("torchrun", "nccl", "init_process_group", "curl ", "wget "):
        assert forbidden not in source.lower()


def test_targeted_pbs_is_one_node_and_commit_bound() -> None:
    source = TARGETED_PBS.read_text(encoding="utf-8")
    assert "#PBS -l select=1" in source
    assert "EXPECTED_COMMIT" in source
    assert "tests/stage1" in source
    assert "mpirun" not in source


def test_formal_roles_do_not_import_torch_or_application_networking() -> None:
    source = Path("src/fsbdd/syncer_readiness_stress.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch.distributed" not in source
    assert "mpi4py" not in source
    assert "requests" not in source
    assert "urllib" not in source
    assert '"logical_syncer_count": 1' in source
