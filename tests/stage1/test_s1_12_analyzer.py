from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fsbdd.diloco.model.evaluation import (
    FrozenEvaluationPlan,
    load_evaluation_snapshot,
    materialize_evaluation_snapshot,
)
from fsbdd.auxiliary.stress.profile_a_stress import (
    ProfileAStressError,
    analyze,
    run_numeric_e2e,
)

from test_s1_12_profile_a import make_system, publish_round


CONFIG = Path("configs/stage1/s1_12_profile_a_e2e.json")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _advance(atomic, proposals, executor, observed: int) -> None:
    publish_round(atomic, proposals, 0)
    publish_round(atomic, proposals, 1)
    assert executor.execute_next(observed_ns=observed) is not None
    assert executor.execute_next(observed_ns=observed + 1) is not None


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, dict, str]:
    config = json.loads(CONFIG.read_text())
    identity = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    result_root = tmp_path / "roles"
    shared_root = tmp_path / "shared"
    run_numeric_e2e(
        shared_root / "numeric",
        run_id="analyzer-fixture",
        config_identity=identity,
        config=config,
    )
    atomic, proposals, executor = make_system(tmp_path / "evaluation-system")
    _advance(atomic, proposals, executor, 1)
    plan = FrozenEvaluationPlan.capture(atomic, clock_ns=lambda: 10)
    materialize_evaluation_snapshot(plan, result_root / "evaluation")
    _advance(atomic, proposals, executor, 3)
    loaded = load_evaluation_snapshot(result_root / "evaluation")
    evaluation = {
        "captured_version_vector": [1, 1],
        "captured_content_identities": list(plan.content_identities),
        "manifest_identity": loaded.manifest.snapshot_identity,
        "loaded_version_vector": [1, 1],
        "loaded_content_identities": list(loaded.manifest.content_identities),
        "current_version_vector_after_writer": [2, 2],
        "state_payload_bytes": loaded.state_payload_bytes,
        "parameter_payload_bytes": loaded.parameter_payload_bytes,
        "materialize_start_unix_ns": 20,
        "materialize_end_unix_ns": 40,
        "start_unix_ns": 10,
        "end_unix_ns": 50,
        "restart_load_access_audit": loaded.access_audit,
        "purpose": "evaluation",
        "steady_state": False,
    }
    for rank in range(4):
        control_rate = 1280.0
        injected_rate = 640.0 if rank == 3 else control_rate
        control_steps = int(control_rate * 10 / 64)
        injected_steps = int(injected_rate * 10 / 64)
        integrated_syncer = {
            "complete": True,
            "status": "pass",
            "updates": [
                {"fragment_index": 0, "completed_unix_ns": 1000},
                {"fragment_index": 1, "completed_unix_ns": 12_000_000_000},
            ],
            "phase_update_counts": {"control": 1, "injected": 1},
            "errors": [],
            "application_coordination": "shared_filesystem_only",
        }
        role = {
            "schema_version": 1,
            "status": "pass",
            "rank": rank,
            "size": 4,
            "learner_id": f"learner-{rank:02d}",
            "hostname": f"host-{rank}",
            "config_identity": identity,
            "torch": {"distributed_initialized": False},
            "speed": {
                "control": {
                    "start_unix_ns": 100,
                    "end_unix_ns": 10_000_000_100,
                    "common_interval_seconds": 10.0,
                    "common_interval_input_tokens_per_second": control_rate,
                    "scheduled_period_seconds": 0.5,
                    "local_optimizer_steps_before": 0,
                    "local_optimizer_steps_after": control_steps,
                    "completed_steps": control_steps,
                    "processed_input_tokens_before": 0,
                    "processed_input_tokens_after": control_steps * 64,
                    "processed_input_tokens": control_steps * 64,
                    "step_completion_unix_ns": list(range(101, 101 + control_steps)),
                    "publication_count_before": 0,
                    "publication_count_after": 1,
                    "publication_count_delta": 1,
                    "adoption_count_before": 0,
                    "adoption_count_after": 1,
                    "path": "learner_runtime_snapshot_publisher_adoption_and_async_syncer",
                    "measurement": "raw_progress_counter_delta_over_scheduler_defined_common_interval",
                    "finite_loss": True,
                    "distributed_initialized": False,
                },
                "injected": {
                    "start_unix_ns": 11_000_000_100,
                    "end_unix_ns": 21_000_000_100,
                    "common_interval_seconds": 10.0,
                    "common_interval_input_tokens_per_second": injected_rate,
                    "scheduled_period_seconds": 1.0 if rank == 3 else 0.5,
                    "local_optimizer_steps_before": control_steps,
                    "local_optimizer_steps_after": control_steps + injected_steps,
                    "completed_steps": injected_steps,
                    "processed_input_tokens_before": control_steps * 64,
                    "processed_input_tokens_after": (control_steps + injected_steps)
                    * 64,
                    "processed_input_tokens": injected_steps * 64,
                    "step_completion_unix_ns": list(
                        range(11_000_000_101, 11_000_000_101 + injected_steps)
                    ),
                    "publication_count_before": 1,
                    "publication_count_after": 2,
                    "publication_count_delta": 1,
                    "adoption_count_before": 1,
                    "adoption_count_after": 2,
                    "path": "learner_runtime_snapshot_publisher_adoption_and_async_syncer",
                    "measurement": "raw_progress_counter_delta_over_scheduler_defined_common_interval",
                    "finite_loss": True,
                    "distributed_initialized": False,
                },
                "integrated_syncer": integrated_syncer,
            },
            "mixed_training": {
                "events": [
                    {
                        "fragment_global_versions": [1, 0],
                        "token_weighted_loss": 4.0 - step * 0.01,
                    }
                    for step in range(5)
                ]
            },
            "adoption": {"adoption_count": 2},
            "adoption_traces": [{"fragment_index": 0}, {"fragment_index": 1}],
            "final_version_vector": [1, 1],
            "application_coordination": "shared_filesystem_only",
            "mpi_usage": "launcher_only",
        }
        _write(result_root / f"learner-{rank:02d}.json", role)
    syncer = {
        "status": "pass",
        "config_identity": identity,
        "real_updates": [
            {
                "fragment_index": index,
                "byte_accounting": {"full_model_operations": 0},
                "source_metrics": {"maximum_active_payloads": 1},
            }
            for index in range(2)
        ],
        "real_progress_after_cycle": {
            "global_cycle": 1,
            "learners": [
                {
                    "learner_id": f"learner-{rank:02d}",
                    "local_optimizer_steps": 9,
                    "processed_input_tokens": 576,
                    "loss_bearing_target_tokens": 540,
                }
                for rank in range(4)
            ],
        },
        "evaluation": evaluation,
    }
    _write(result_root / "syncer.json", syncer)
    return result_root, shared_root, config, identity


def test_analyzer_independently_recomputes_full_fixture(tmp_path: Path) -> None:
    result_root, shared_root, config, identity = _build_fixture(tmp_path)
    summary = analyze(
        result_root=result_root,
        shared_root=shared_root,
        config=config,
        config_identity=identity,
    )
    assert summary["status"] == "pass"
    assert summary["numeric"]["updates_per_fragment"] == 50
    assert summary["numeric"]["fifty_update_maximum_relative_l2"] <= 1e-6
    assert summary["speed_heterogeneity"]["maximum_unaffected_change"] == 0
    assert summary["speed_heterogeneity"]["slow_ratio"] == 0.5
    assert summary["mixed_version_training"][
        "finite_consecutive_steps_per_learner"
    ] == [
        5,
        5,
        5,
        5,
    ]

    numeric_path = shared_root / "numeric" / "numeric-trace.json"
    numeric = json.loads(numeric_path.read_text())
    numeric["traces"][0]["production_parameters"][0] += 1.0
    _write(numeric_path, numeric)
    with pytest.raises(ProfileAStressError, match="recomputation"):
        analyze(
            result_root=result_root,
            shared_root=shared_root,
            config=config,
            config_identity=identity,
        )


def test_analyzer_rejects_peer_throttling_and_mutable_evaluation(
    tmp_path: Path,
) -> None:
    result_root, shared_root, config, identity = _build_fixture(tmp_path)
    learner = json.loads((result_root / "learner-01.json").read_text())
    learner["speed"]["injected"]["common_interval_input_tokens_per_second"] *= 0.9
    _write(result_root / "learner-01.json", learner)
    with pytest.raises(ProfileAStressError, match="speed heterogeneity"):
        analyze(
            result_root=result_root,
            shared_root=shared_root,
            config=config,
            config_identity=identity,
        )

    learner["speed"]["injected"]["common_interval_input_tokens_per_second"] /= 0.9
    _write(result_root / "learner-01.json", learner)
    syncer = json.loads((result_root / "syncer.json").read_text())
    syncer["evaluation"]["loaded_version_vector"] = [2, 2]
    _write(result_root / "syncer.json", syncer)
    with pytest.raises(ProfileAStressError, match="evaluation"):
        analyze(
            result_root=result_root,
            shared_root=shared_root,
            config=config,
            config_identity=identity,
        )


def test_analyzer_rejects_corrupted_authoritative_outer_successor(
    tmp_path: Path,
) -> None:
    result_root, shared_root, config, identity = _build_fixture(tmp_path)
    successor = (
        shared_root
        / "numeric"
        / "authoritative"
        / "fragment-000001"
        / "cycle-000049"
        / "successor-state.bin"
    )
    payload = successor.read_bytes()
    successor.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(ProfileAStressError, match="integrity"):
        analyze(
            result_root=result_root,
            shared_root=shared_root,
            config=config,
            config_identity=identity,
        )
