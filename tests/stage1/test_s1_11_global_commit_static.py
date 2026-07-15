from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOOP_CARD = ROOT / "plans/FS_DILOCO_CODEX_EXECUTION_PLAN/loops/stage1/S1-11.md"
MATRIX = ROOT / "reports/stage1/S1-11-requirement-matrix.csv"
CONTRACT = ROOT / "reports/stage1/S1-11-evidence-contract.json"
INJECTIONS = ROOT / "reports/stage1/S1-11-publication-injection-points.json"
CONFIG = ROOT / "configs/stage1/s1_11_global_commit.json"
PRODUCTION = ROOT / "src/fsbdd/global_commit.py"
GLOBAL_STATE = ROOT / "src/fsbdd/global_state.py"
STRESS = ROOT / "src/fsbdd/global_commit_stress.py"
FORMAL_PBS = ROOT / "pbs/stage1_s1_11_global_commit.pbs"
TARGETED_PBS = ROOT / "pbs/stage1_s1_11_targeted.pbs"


def _card_ids() -> set[str]:
    text = LOOP_CARD.read_text(encoding="utf-8")
    requirements = re.search(r"^\| 规范条款 \| ([^|]+)\|$", text, flags=re.MULTILINE)
    acceptance = re.search(r"^\| Acceptance \| ([^|]+)\|$", text, flags=re.MULTILINE)
    assert requirements is not None and acceptance is not None
    return {
        *(item.strip() for item in requirements.group(1).split(",")),
        *(item.strip() for item in acceptance.group(1).split(",")),
    }


def test_requirement_matrix_is_derived_complete_and_structural() -> None:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["id"] for row in rows} == _card_ids()
    assert len(rows) == len(_card_ids()) == 10
    for row in rows:
        assert all(row[field].strip() for field in row)
        assert row["evidence_path"].startswith("runtime_runs/S1-11/formal-l2/")


def test_config_contract_and_injections_freeze_one_success_boundary() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    injections = json.loads(INJECTIONS.read_text(encoding="utf-8"))
    assert config["learner_count"] == 4
    assert config["fragment_count"] == 2
    assert config["s_max"] == 0
    assert config["selection"] == {
        "q": 4,
        "q_fresh": 4,
        "max_contributors": 4,
        "lambda_s": 1.0,
    }
    assert config["formal_workload"] == {
        "successful_updates": 20,
        "fault_replays_per_point": 4,
        "concurrent_reader_threads": 8,
        "minimum_reader_observations": 100,
    }
    assert [item["name"] for item in injections["fault_points"]] == config[
        "publication_fault_points"
    ]
    assert [item["publication_succeeded"] for item in injections["fault_points"]] == [
        False,
        False,
        False,
        True,
    ]
    assert contract["atomic_authority_contract"]["success_boundary"].startswith(
        "replacement of the current visibility record"
    )
    assert contract["topology"]["nodes"] == contract["topology"]["ranks"] == 2
    assert contract["topology"]["application_coordination"] == "filesystem_only"


def test_production_uses_one_compound_envelope_and_fixed_current_record() -> None:
    production = PRODUCTION.read_text(encoding="utf-8")
    global_state = GLOBAL_STATE.read_text(encoding="utf-8")
    assert "outer_optimizer_state" in production
    assert "frontiers" in production
    assert "selected" in production
    assert "policy_identity" in production
    assert "update_identity" in production
    assert "encode_commit_envelope(envelope)" in production
    assert "commit_consumption(" in production
    assert "publication_succeeded=False" not in production
    assert "before_visibility" in global_state
    assert "crash_at=crash_at" in global_state
    assert not re.search(r"latest_(parameters|outer|frontier)|global_head", production)


def test_stress_roles_do_not_import_torch_or_use_mpi_as_data_plane() -> None:
    stress = STRESS.read_text(encoding="utf-8")
    assert not re.search(r"^import torch$|^from torch", stress, flags=re.MULTILINE)
    assert "PosixStorageBackend" in stress
    assert '"application_coordination": "filesystem_only"' in stress
    assert '"mpi_usage": "launcher_only"' in stress
    assert "concurrent_reader_threads" in stress
    assert "publication_fault_points" in stress


def test_pbs_scripts_preserve_compute_routing_and_formal_identity_guards() -> None:
    formal = FORMAL_PBS.read_text(encoding="utf-8")
    targeted = TARGETED_PBS.read_text(encoding="utf-8")
    assert "#PBS -l select=2" in formal
    assert "#PBS -l select=1" in targeted
    assert "EXPECTED_COMMIT" in formal and "EXPECTED_COMMIT" in targeted
    assert "git status --porcelain" in formal
    assert "evidence validate" in formal
    assert "mpirun -np 2 --map-by ppr:1:node" in formal
    assert "fsbdd.global_commit_stress role" in formal
    assert "tests/stage1" in formal
