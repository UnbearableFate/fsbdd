from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOOP_CARD = ROOT / "plans/FS_DILOCO_CODEX_EXECUTION_PLAN/loops/stage1/S1-10.md"
MATRIX = ROOT / "reports/stage1/S1-10-requirement-matrix.csv"
CONTRACT = ROOT / "reports/stage1/S1-10-evidence-contract.json"
CONFIG = ROOT / "configs/stage1/s1_10_streaming_merge.json"
PRODUCTION = ROOT / "src/fsbdd/diloco/syncer/merge.py"
STRESS = ROOT / "src/fsbdd/auxiliary/stress/syncer_merge_stress.py"
FORMAL_PBS = ROOT / "pbs/stage1_s1_10_streaming_merge.pbs"
TARGETED_PBS = ROOT / "pbs/stage1_s1_10_targeted.pbs"


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
    assert len(rows) == len(_card_ids()) == 9
    for row in rows:
        assert all(row[field].strip() for field in row)
        assert row["evidence_path"].startswith("runtime_runs/S1-10/formal-l2-retry2/")


def test_config_and_contract_freeze_numeric_memory_and_topology_semantics() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert config["formal_workload"] == {
        "learner_count": 8,
        "fragment_count": 4,
        "fragment_elements": 1048576,
        "reference_updates": 50,
        "order_permutations": 16,
    }
    assert config["accumulation_dtype"] == "float32"
    assert config["merge_policy"] == "direct_weighted_average"
    assert contract["topology"]["nodes"] == contract["topology"]["ranks"] == 2
    assert contract["topology"]["application_coordination"] == "filesystem_only"
    assert contract["numeric_contract"]["application"].startswith("apply the merged")
    assert contract["memory_contract"]["active_payload_gate"].startswith(
        "maximum simultaneous"
    )
    assert len(contract["update_identity"]["required_fields"]) == 16


def test_syncer_merge_imports_torch_only_inside_compute_paths() -> None:
    production = PRODUCTION.read_text(encoding="utf-8")
    stress = STRESS.read_text(encoding="utf-8")
    assert not re.search(r"^import torch$|^from torch", production, flags=re.MULTILINE)
    assert not re.search(r"^import torch$|^from torch", stress, flags=re.MULTILINE)
    assert "import torch" in production
    assert "load_full_model" not in production
    assert "torch.stack" not in production
    assert "torch.cat" not in production


def test_pbs_scripts_preserve_compute_routing_and_formal_identity_guards() -> None:
    formal = FORMAL_PBS.read_text(encoding="utf-8")
    targeted = TARGETED_PBS.read_text(encoding="utf-8")
    assert "#PBS -l select=2" in formal
    assert "#PBS -l select=1" in targeted
    assert "EXPECTED_COMMIT" in formal and "EXPECTED_COMMIT" in targeted
    assert "git status --porcelain" in formal
    assert "evidence validate" in formal
    assert "mpirun -np 2 --map-by ppr:1:node" in formal
    assert "fsbdd.auxiliary.stress.syncer_merge_stress role" in formal
    assert "tests/stage1" in formal
