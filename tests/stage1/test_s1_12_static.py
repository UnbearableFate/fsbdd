from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

from fsbdd.global_state import FragmentStateDescriptor, GlobalStateIdentities
from fsbdd.profile_a_stress import _catalog_matches_model


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_profile_a_config_and_four_node_topology() -> None:
    value = json.loads(
        (ROOT / "configs/stage1/s1_12_profile_a_e2e.json").read_text()
    )
    assert value["profile"] == {
        "name": "profile-a-fresh-reference",
        "learner_count": 4,
        "fragment_count": 2,
        "q": 4,
        "q_fresh": 4,
        "s_max": 0,
        "grace_period_seconds": 0.0,
        "lambda_s": 1.0,
        "deterministic_fragment_schedule": "round_robin",
    }
    assert value["numeric_oracle"]["updates_per_fragment"] == 50
    assert value["numeric_oracle"]["single_update_relative_l2_max"] == 1e-6
    assert value["real_fs"]["nodes"] == value["real_fs"]["ranks"] == 4
    assert value["real_fs"]["learner_processes"] == 4
    assert value["real_fs"]["application_coordination"] == "shared_filesystem_only"
    assert value["comparison"]["matched_compute"] is False
    assert value["comparison"]["matched_communication"] is False
    assert value["comparison"]["matched_tokens"] is False


def test_application_source_uses_mpi_only_as_launcher() -> None:
    source = (ROOT / "src/fsbdd/profile_a_stress.py").read_text()
    forbidden = (
        "init_process_group(",
        "all_reduce(",
        "all_gather(",
        "broadcast(",
        "torch.distributed.barrier(",
        "DistributedDataParallel(",
        "rpc.init_rpc(",
    )
    assert not any(token in source for token in forbidden)
    assert '"application_coordination": "shared_filesystem_only"' in source
    assert '"mpi_usage": "launcher_only"' in source


def test_real_role_reuses_its_runtime_owned_rng_across_training_phases() -> None:
    tree = ast.parse((ROOT / "src/fsbdd/profile_a_stress.py").read_text())
    role = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_real_role"
    )
    runtime_factories = [
        node
        for node in ast.walk(role)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_runtime"
    ]
    runtime_runs = [
        node
        for node in ast.walk(role)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "runtime"
        and node.func.attr == "run"
    ]
    assert len(runtime_factories) == 1
    assert len(runtime_runs) == 3


def test_requirement_matrix_has_exact_loop_ids() -> None:
    with (ROOT / "reports/stage1/S1-12-requirement-matrix.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert {row["id"] for row in rows} == {
        "GLOBAL-05",
        "PROG-01",
        "PROG-02",
        "PROG-03",
        "PROG-04",
        "REPORT-02",
        "A-ALG-01",
        "A-EVAL-01",
        "A-LEARN-01",
        "A-LEARN-03",
    }
    assert len(rows) == 10


def test_evaluation_loader_has_no_store_or_latest_argument() -> None:
    source = (ROOT / "src/fsbdd/evaluation.py").read_text()
    start = source.index("def load_evaluation_snapshot(")
    body = source[start:]
    assert "AtomicGlobalCommitStore" not in body.split("def ", 2)[1]
    assert ".load_fragment(" not in body
    assert ".load_snapshot(" not in body


def test_json_round_trip_catalog_matches_typed_model_identity() -> None:
    identities = GlobalStateIdentities(
        run_identity="catalog-test",
        config_identity=hashlib.sha256(b"config").hexdigest(),
        model_identity=hashlib.sha256(b"model").hexdigest(),
        fragment_map_identity=hashlib.sha256(b"map").hexdigest(),
    )
    descriptors = (
        FragmentStateDescriptor(
            index=0,
            identity=hashlib.sha256(b"fragment").hexdigest(),
            dtype="float32",
            shape=(8,),
            parameter_identities=("parameter",),
        ),
    )
    catalog = json.loads(
        json.dumps(
            {
                "identities": identities.to_dict(),
                "descriptors": [item.to_dict() for item in descriptors],
            }
        )
    )
    assert _catalog_matches_model(catalog, identities, descriptors)
