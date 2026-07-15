from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsbdd.diloco.model.evaluation import (
    EvaluationAccessAudit,
    EvaluationSnapshotError,
    FrozenEvaluationPlan,
    load_evaluation_snapshot,
    materialize_evaluation_snapshot,
)

from test_s1_12_profile_a import make_system, publish_round


def _advance_both(atomic, proposals, executor, observed: int) -> None:
    publish_round(atomic, proposals, 0)
    publish_round(atomic, proposals, 1)
    first = executor.execute_next(observed_ns=observed)
    second = executor.execute_next(observed_ns=observed + 1)
    assert first is not None and second is not None


def test_frozen_evaluation_load_ignores_newer_current_authorities(
    tmp_path: Path,
) -> None:
    atomic, proposals, executor = make_system(tmp_path / "system")
    _advance_both(atomic, proposals, executor, 1)
    publish_round(atomic, proposals, 0)
    update = executor.execute_next(observed_ns=3)
    assert update is not None and update.successor.state.descriptor.index == 0
    plan = FrozenEvaluationPlan.capture(atomic, clock_ns=lambda: 123456)
    captured_vector = plan.version_vector
    captured_content = plan.content_identities
    captured_parameters = tuple(item.parameters for item in plan.authorities)

    publish_round(atomic, proposals, 1)
    update = executor.execute_next(observed_ns=4)
    assert update is not None and update.successor.state.descriptor.index == 1
    publish_round(atomic, proposals, 0)
    publish_round(atomic, proposals, 1)
    executor.execute_next(observed_ns=5)
    executor.execute_next(observed_ns=6)
    assert atomic.load_snapshot().version_vector != captured_vector

    destination = tmp_path / "evaluation"
    manifest = materialize_evaluation_snapshot(plan, destination)
    loaded = load_evaluation_snapshot(destination)
    assert manifest.version_vector == captured_vector
    assert manifest.content_identities == captured_content
    assert loaded.manifest.snapshot_identity == manifest.snapshot_identity
    assert tuple(item.parameters for item in loaded.authorities) == captured_parameters
    assert tuple(item.version for item in loaded.authorities) == captured_vector
    assert loaded.parameter_payload_bytes == sum(map(len, captured_parameters))
    assert loaded.manifest.to_dict()["steady_state"] is False
    assert loaded.manifest.to_dict()["purpose"] == "evaluation"
    assert loaded.access_audit == {
        "schema_version": 1,
        "instrumentation": "all_restart_loader_reads_through_path_audit",
        "operations": [
            {"purpose": "manifest", "relative_path": "manifest.json"},
            {
                "purpose": "frozen_state",
                "relative_path": "fragments/fragment-000000.state",
            },
            {
                "purpose": "frozen_state",
                "relative_path": "fragments/fragment-000001.state",
            },
        ],
        "manifest_reads": 1,
        "frozen_payload_reads": 2,
        "current_authority_reads": 0,
        "latest_resolution_reads": 0,
        "unauthorized_reads": 0,
    }


def test_restart_load_uses_only_manifest_named_payloads(tmp_path: Path) -> None:
    atomic, _, _ = make_system(tmp_path / "system")
    destination = tmp_path / "evaluation"
    materialize_evaluation_snapshot(FrozenEvaluationPlan.capture(atomic), destination)
    loaded = load_evaluation_snapshot(destination)
    assert loaded.manifest.version_vector == (0, 0)
    assert sorted(path.name for path in (destination / "fragments").iterdir()) == [
        "fragment-000000.state",
        "fragment-000001.state",
    ]


def test_restart_access_audit_rejects_current_or_latest_resolution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluation"
    (root / "visibility").mkdir(parents=True)
    (root / "visibility" / "global-current-000000.json").write_bytes(b"{}")
    audit = EvaluationAccessAudit(root)
    with pytest.raises(EvaluationSnapshotError, match="current-authority"):
        audit.read_bytes(
            root / "visibility" / "global-current-000000.json",
            purpose="frozen_state",
        )
    assert audit.current_authority_reads == 1


def test_manifest_or_state_tampering_fails_closed(tmp_path: Path) -> None:
    atomic, _, _ = make_system(tmp_path / "system-a")
    destination = tmp_path / "evaluation-a"
    materialize_evaluation_snapshot(FrozenEvaluationPlan.capture(atomic), destination)
    path = destination / "fragments" / "fragment-000000.state"
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(EvaluationSnapshotError, match="integrity"):
        load_evaluation_snapshot(destination)

    atomic, _, _ = make_system(tmp_path / "system-b")
    destination = tmp_path / "evaluation-b"
    materialize_evaluation_snapshot(FrozenEvaluationPlan.capture(atomic), destination)
    manifest_path = destination / "manifest.json"
    value = json.loads(manifest_path.read_text())
    value["version_vector"][0] = 99
    manifest_path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(EvaluationSnapshotError):
        load_evaluation_snapshot(destination)


def test_materialization_is_fail_if_exists(tmp_path: Path) -> None:
    atomic, _, _ = make_system(tmp_path / "system")
    destination = tmp_path / "evaluation"
    destination.mkdir()
    with pytest.raises(EvaluationSnapshotError, match="must be new"):
        materialize_evaluation_snapshot(
            FrozenEvaluationPlan.capture(atomic), destination
        )
