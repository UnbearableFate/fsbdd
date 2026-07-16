from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fsbdd.auxiliary.stage1.assets import (
    Stage1AssetError,
    _producer_commit,
    _sha256_identity,
)
from fsbdd.auxiliary.stage1.close import (
    Stage1CloseError,
    _require_long_run_terminal_target,
    _require_bootstrap_model_identities,
)
from fsbdd.diloco.protocol.global_state import (
    BootstrapFragment,
    FragmentStateDescriptor,
    GlobalStateIdentities,
    GlobalStateStore,
)
from fsbdd.diloco.protocol.proposal import ConsumptionFrontiers, Proposal, ProposalStore
from fsbdd.diloco.protocol.storage import (
    PosixStorageBackend,
    PublicationSpec,
    ReadExpectation,
)
from fsbdd.diloco.syncer.readiness import (
    FragmentReadinessAuthority,
    ReadinessConfig,
    SyncerReadinessMachine,
)
from fsbdd.auxiliary.stage1.gate import (
    Stage1GateError,
    _load_syncer_streams,
    _runtime_gate,
    _topology_gate,
)
from fsbdd.auxiliary.stage1.package import (
    Stage1PackageError,
    _capture_current_protocol_samples,
    _reject_placeholders,
)
from fsbdd.auxiliary.stage1.submit import (
    Stage1SubmissionError,
    prepare_nine_node_roots,
    validate_nine_node_roots,
)
from fsbdd.auxiliary.contracts.evidence import EvidencePackage


IDENTITIES = GlobalStateIdentities(
    run_identity="s1-13-unit",
    config_identity="a" * 64,
    model_identity="b" * 64,
    fragment_map_identity="c" * 64,
)
DESCRIPTOR = FragmentStateDescriptor(
    index=0,
    identity="fragment-0",
    dtype="float32",
    shape=(4,),
    parameter_identities=("parameter-0",),
)


def test_bootstrap_model_identity_mismatch_reports_exact_expected_and_actual() -> (
    None
):
    bootstrap = {
        "registry_digest": "a" * 64,
        "fragment_map_digest": "b" * 64,
    }
    with pytest.raises(Stage1CloseError) as captured:
        _require_bootstrap_model_identities(
            actual_registry_digest="c" * 64,
            actual_fragment_map_digest="d" * 64,
            bootstrap=bootstrap,
        )
    payload = json.loads(str(captured.value).split(": ", 1)[1])
    assert payload == {
        "fragment_map_digest": {"actual": "d" * 64, "expected": "b" * 64},
        "registry_digest": {"actual": "c" * 64, "expected": "a" * 64},
    }
    _require_bootstrap_model_identities(
        actual_registry_digest="a" * 64,
        actual_fragment_map_digest="b" * 64,
        bootstrap=bootstrap,
    )


def test_asset_identity_validators_reject_malformed_values() -> None:
    assert _producer_commit("a" * 40) == "a" * 40
    assert _sha256_identity("b" * 64, "test") == "b" * 64
    with pytest.raises(Stage1AssetError, match="producer commit"):
        _producer_commit("A" * 40)
    with pytest.raises(Stage1AssetError, match="SHA-256"):
        _sha256_identity("f" * 63, "test")


def test_completed_long_run_fails_fast_when_terminal_cycle_is_below_target() -> None:
    final_records = [
        {
            "learner_id": f"learner-{learner_index:02d}",
            "publication": {
                "progress": {
                    "fragments": [
                        {"global_version": 58, "proposal_sequence": 60}
                        for _ in range(4)
                    ]
                },
                "publication": {
                    "latest_terminal_per_fragment": [
                        {
                            "fragment_index": fragment_index,
                            "base_version": (
                                58
                                if fragment_index == 3 and learner_index in (0, 2)
                                else 57
                            ),
                        }
                        for fragment_index in range(4)
                    ]
                },
            },
        }
        for learner_index in range(4)
    ]
    _require_long_run_terminal_target(
        cycle=58,
        target=58,
        version_vector=(58, 58, 58, 58),
        final_records=final_records,
    )
    with pytest.raises(Stage1CloseError) as captured:
        _require_long_run_terminal_target(
            cycle=58,
            target=59,
            version_vector=(58, 58, 58, 58),
            final_records=final_records,
        )
    detail = json.loads(str(captured.value).split(": ", 1)[1])
    assert detail["global_cycle"] == 58
    assert detail["target_global_cycle"] == 59
    assert detail["version_vector"] == [58, 58, 58, 58]
    assert detail["all_learners_final"] is True
    assert detail["eligible_update_after_final_poll"] is False
    assert [
        item["fragments"][3]["latest_proposal_base_version"]
        for item in detail["learners"]
    ] == [58, 57, 58, 57]


def _spec(sequence: int) -> PublicationSpec:
    return PublicationSpec(
        run_identity="s1-13-unit",
        fragment_map_identity="c" * 64,
        fragment_identity="fragment-0",
        version=sequence,
        sequence=sequence,
        dtype="float32",
        shape=(4,),
        base_content_identity="d" * 64,
    )


def test_metadata_only_read_and_bounded_payload_reclamation(
    tmp_path: Path, monkeypatch
) -> None:
    backend = PosixStorageBackend(tmp_path)
    records = [
        backend.publish("current", bytes([index]) * 16, _spec(index))
        for index in range(5)
    ]
    assert backend.read_bound_record(records[0]).payload == bytes([0]) * 16
    for path in (tmp_path / "payloads").iterdir():
        os.utime(path, ns=(1, 1))
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path.parent.name == "payloads":
            raise AssertionError("metadata-only read touched a tensor payload")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    record = backend.read_record(
        "current",
        ReadExpectation(
            run_identity="s1-13-unit",
            fragment_map_identity="c" * 64,
            fragment_identity="fragment-0",
        ),
        timeout_seconds=0,
    )
    assert record == records[-1]
    monkeypatch.setattr(Path, "read_bytes", original)
    reclaimed = backend.reclaim_unreferenced_payloads(
        retain_recent=1, minimum_age_seconds=0
    )
    assert reclaimed["reclaimed_payload_files"] == 3
    assert reclaimed["physical_payloads"] == 2
    assert reclaimed["referenced_payloads"] == 1
    assert reclaimed["orphan_payloads"] == 1


def test_reclamation_grace_starts_when_visibility_is_replaced(tmp_path: Path) -> None:
    backend = PosixStorageBackend(tmp_path)
    previous = backend.publish("current", b"old" * 4, _spec(0))
    old_payload = tmp_path / previous.payload_relative_path
    os.utime(old_payload, ns=(1, 1))
    backend.publish("current", b"new" * 4, _spec(1))
    retained = backend.reclaim_unreferenced_payloads(
        retain_recent=0,
        minimum_age_seconds=60,
    )
    assert retained["reclaimed_payload_files"] == 0
    marker = tmp_path / ".retired" / f"{old_payload.name}.json"
    value = json.loads(marker.read_text(encoding="utf-8"))
    value["retired_unix_ns"] = 0
    marker.write_text(json.dumps(value), encoding="utf-8")
    reclaimed = backend.reclaim_unreferenced_payloads(
        retain_recent=0,
        minimum_age_seconds=60,
    )
    assert reclaimed["reclaimed_payload_files"] == 1


def test_readiness_reuses_unchanged_latest_payloads(tmp_path: Path) -> None:
    global_store = GlobalStateStore(
        PosixStorageBackend(tmp_path / "global"),
        identities=IDENTITIES,
        descriptors=(DESCRIPTOR,),
        s_max=0,
    )
    state = global_store.bootstrap(
        (BootstrapFragment(DESCRIPTOR, b"\0" * 16, b""),)
    ).snapshot.states[0]
    learners = ("learner-00", "learner-01")
    proposal_store = ProposalStore(
        PosixStorageBackend(tmp_path / "proposals"),
        identities=IDENTITIES,
        descriptors=(DESCRIPTOR,),
        learner_ids=learners,
        maximum_local_steps=100,
        maximum_processed_tokens=10000,
    )
    for index, learner_id in enumerate(learners):
        proposal_store.publish(
            Proposal.create(
                proposal_id=f"proposal-{index}",
                identities=IDENTITIES,
                learner_id=learner_id,
                descriptor=DESCRIPTOR,
                sequence=1,
                base_version=0,
                base_content_identity=state.content_identity,
                local_steps=1,
                processed_tokens=100 + index,
                snapshot_local_step=1,
                parameters=bytes([index + 1]) * 16,
            )
        )
    machine = SyncerReadinessMachine(
        proposal_store,
        authorities=(
            FragmentReadinessAuthority(
                state,
                ConsumptionFrontiers.empty(
                    identities=IDENTITIES,
                    descriptor=DESCRIPTOR,
                    learner_ids=learners,
                ),
            ),
        ),
        config=ReadinessConfig("syncer", 2, 2, 2, 0),
    )
    machine.poll_store(observed_ns=time.monotonic_ns())
    first = machine.snapshot()
    machine.poll_store(observed_ns=time.monotonic_ns())
    second = machine.snapshot()
    assert first.proposal_payload_cache_misses == 2
    assert first.proposal_payload_cache_hits == 0
    assert second.proposal_payload_cache_misses == 2
    assert second.proposal_payload_cache_hits == 2
    assert second.resident_latest_proposals == 2


def test_nine_node_topology_gate_rejects_duplicate_host_and_gpu() -> None:
    roles = [
        {
            "run_id": "run",
            "identity": {
                "hostname": f"node-{index}" if index else "node-duplicate",
                "pbs_job_id": f"job-{index}",
                "pbs_qtime_utc": "2026-07-15T00:00:00Z",
                "shared_device": 7,
                "gpu": {"gpu_uuid": f"GPU-{index}" if index else "GPU-duplicate"},
                "config_sha256": "a" * 64,
                "asset_marker_sha256": "b" * 64,
            },
        }
        for index in range(8)
    ]
    syncer = {
        "run_id": "run",
        "identity": {
            "hostname": "syncer-node",
            "pbs_job_id": "job-8",
            "pbs_qtime_utc": "2026-07-15T00:00:00Z",
            "shared_device": 7,
            "gpu_count": 0,
            "torch_module_imported": False,
            "config_sha256": "a" * 64,
            "asset_marker_sha256": "b" * 64,
            "cuda_visible_devices": "",
        },
    }
    assert (
        _topology_gate(roles, syncer, workload="nine_node", expected_learners=8)[
            "status"
        ]
        == "pass"
    )
    roles[1]["identity"]["hostname"] = "node-duplicate"
    roles[1]["identity"]["gpu"]["gpu_uuid"] = "GPU-duplicate"
    assert (
        _topology_gate(roles, syncer, workload="nine_node", expected_learners=8)[
            "status"
        ]
        == "fail"
    )


def test_protocol_sample_capture_binds_current_metadata_and_edges(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    for backend_name in ("global", "proposals"):
        backend = PosixStorageBackend(shared / "protocol" / backend_name)
        backend.publish("slot-0", bytes(range(256)) * 40, _spec(0))
    evidence = EvidencePackage.create(tmp_path / "evidence")
    summary = _capture_current_protocol_samples(shared, evidence)
    assert summary["sample_bytes_per_edge"] == 4096
    for backend_name in ("global", "proposals"):
        assert summary["backends"][backend_name]["current_visibility_records"] == 1
        sample = json.loads(
            (
                evidence.root
                / "raw-metadata"
                / backend_name
                / "samples"
                / "slot-0.json"
            ).read_text(encoding="utf-8")
        )
        assert len(bytes.fromhex(sample["prefix_hex"])) == 4096
        assert len(bytes.fromhex(sample["suffix_hex"])) == 4096
        assert sample["suffix_offset"] == 10240 - 4096


def test_formal_package_rejects_unresolved_asset_placeholders() -> None:
    with pytest.raises(Stage1PackageError, match="resolved placeholder"):
        _reject_placeholders(
            {"resolved_runtime_fields": {"asset_bundle_root": "asset_stage"}}
        )


def test_numpy_syncer_probe_imports_no_torch() -> None:
    probe = (
        Path(__file__).resolve().parents[1] / "probes" / "s1_13_numpy_syncer_probe.py"
    )
    completed = subprocess.run(
        [sys.executable, str(probe)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "torch_module_imported=false" in completed.stdout


def test_nine_node_submission_roots_are_exclusive_and_identity_bound(
    tmp_path: Path,
) -> None:
    values = {
        "shared_root": tmp_path / "shared",
        "result_root": tmp_path / "result",
        "evidence_root": tmp_path / "evidence",
        "run_id": "s1-13-nine-test",
        "submission_utc": "2026-07-15T00:00:00Z",
        "code_commit": "1" * 40,
        "config_sha256": "2" * 64,
        "asset_marker_sha256": "3" * 64,
        "gate_contract_sha256": "4" * 64,
    }
    prepared = prepare_nine_node_roots(**values)
    assert not values["evidence_root"].exists()
    assert (values["shared_root"] / "submission-root.json").read_bytes() == (
        values["result_root"] / "submission-root.json"
    ).read_bytes()
    validated = validate_nine_node_roots(
        **values,
        submission_marker_sha256=prepared["submission_marker_sha256"],
    )
    assert validated == prepared
    with pytest.raises(Stage1SubmissionError, match="refusing to reuse"):
        prepare_nine_node_roots(**values)
    with pytest.raises(Stage1SubmissionError, match="checksum mismatch"):
        validate_nine_node_roots(
            **values,
            submission_marker_sha256="5" * 64,
        )


def test_nine_node_submission_rejects_invalid_timestamp_and_overlapping_roots(
    tmp_path: Path,
) -> None:
    values = {
        "shared_root": tmp_path / "shared",
        "result_root": tmp_path / "result",
        "evidence_root": tmp_path / "evidence",
        "run_id": "s1-13-nine-test",
        "submission_utc": "invalidZ",
        "code_commit": "1" * 40,
        "config_sha256": "2" * 64,
        "asset_marker_sha256": "3" * 64,
        "gate_contract_sha256": "4" * 64,
    }
    with pytest.raises(Stage1SubmissionError, match="invalid submission_utc"):
        prepare_nine_node_roots(**values)
    values["submission_utc"] = "2026-07-15T00:00:00Z"
    values["code_commit"] = "1" * 64
    with pytest.raises(Stage1SubmissionError, match="40-hex Git identity"):
        prepare_nine_node_roots(**values)
    values["code_commit"] = "1" * 40
    values["result_root"] = values["shared_root"]
    with pytest.raises(Stage1SubmissionError, match="must differ"):
        prepare_nine_node_roots(**values)


def test_stage1_pbs_failures_install_err_and_exit_capture() -> None:
    root = Path(__file__).resolve().parents[2]
    helper = (root / "pbs/lib/record_experiment_failure.sh").read_text(
        encoding="utf-8"
    )
    assert "trap - ERR EXIT" in helper
    assert "record_experiment_exit" in helper
    assert "record_experiment_signal" in helper
    assert "scheduler signal" in helper
    assert "TERM 143" in helper
    assert "FSBDD_FAILURE_OUTPUT_ROOT" in helper
    assert "runtime_runs/S1-13/rejected-submissions" in helper
    for path in sorted(root.glob("pbs/stage1_s1_13_*.pbs")):
        source = path.read_text(encoding="utf-8")
        assert "record_experiment_failure.sh" in source
        assert "install_experiment_failure_traps" in source
        authorization_anchor = (
            "fsbdd.auxiliary.stage1.submit validate-nine"
            if path.name == "stage1_s1_13_formal_9n.pbs"
            else "refusing to reuse"
        )
        assert source.index(authorization_anchor) < source.index(
            "FSBDD_FAILURE_OUTPUT_ROOT="
        )
    targeted = (root / "pbs/stage1_s1_13_targeted.pbs").read_text(encoding="utf-8")
    assert "timeout --signal=TERM --kill-after=30s 1200s" in targeted
    assert "faulthandler_timeout=120" in targeted


def test_syncer_role_references_durable_bounded_evidence_stream(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "syncer.jsonl"
    log.parent.mkdir()
    rows = [
        {"event": "fragment_outer_update", "fragment_index": 0},
        {"event": "bounded_storage_inventory", "global_cycle": 1},
    ]
    log.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    syncer = {
        "update_count": 1,
        "update_stream": {
            "path": "logs/syncer.jsonl",
            "event": "fragment_outer_update",
            "count": 1,
            "retained_in_memory": 0,
        },
        "inventory_count": 1,
        "inventory_stream": {
            "path": "logs/syncer.jsonl",
            "event": "bounded_storage_inventory",
            "count": 1,
            "retained_in_memory": 0,
        },
    }
    _load_syncer_streams(tmp_path, syncer)
    assert syncer["_updates"] == [rows[0]]
    assert syncer["_inventories"] == [rows[1]]
    syncer["update_count"] = 2
    with pytest.raises(Stage1GateError, match="update count mismatch"):
        _load_syncer_streams(tmp_path, syncer)


def _runtime_fixture(
    tmp_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    with (log_root / "learner-00.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(1, 6):
            stream.write(
                json.dumps(
                    {
                        "event": "safe_boundary",
                        "local_optimizer_step": step,
                        "step_latency_seconds": 0.1,
                    }
                )
                + "\n"
            )
    role: dict[str, object] = {
        "_result_root": str(tmp_path),
        "learner_id": "learner-00",
        "logging": {
            "jsonl_fsync_every_events": 400,
            "final_fsync_complete": True,
            "role_json_is_written_after_final_fsync": True,
        },
        "publication": {
            "publication": {
                "pending_upload_count": 0,
                "in_flight_publication_count": 0,
                "terminal_emits_in_progress": 0,
                "errors": [],
                "maximum_pending_per_fragment": [1, 1, 1, 1],
                "maximum_in_flight_per_fragment": [1, 1, 1, 1],
            }
        },
        "adoption": {
            "poller": {
                "pending_count": 0,
                "errors": [],
                "closed": True,
                "maximum_pending_per_fragment": [1, 1, 1, 1],
            }
        },
    }
    updates = [
        {
            "fragment_index": index % 4,
            "global_cycle_after": (index + 1) // 4,
            "completed_unix_ns": (index + 1) * 1_000_000_000,
            "update_latency_seconds": 0.2,
        }
        for index in range(8)
    ]
    syncer: dict[str, object] = {
        "active": {"active_start_unix_ns": 1},
        "completion": {"active_end_unix_ns": 9_000_000_000},
        "updates": updates,
        "logging": {
            "jsonl_fsync_every_events": 400,
            "final_fsync_complete": True,
            "role_json_is_written_after_final_fsync": True,
        },
        "readiness": {
            "active_fragment": None,
            "resident_selected_proposals": 0,
            "resident_latest_proposals": 4,
            "fragments": [
                {
                    "phase": "waiting",
                    "selected_count": 0,
                    "selection_identity": None,
                }
                for _ in range(4)
            ],
        },
    }
    return [role], syncer


def test_runtime_gate_rejects_heartbeat_gap_and_pending_stall(tmp_path: Path) -> None:
    roles, syncer = _runtime_fixture(tmp_path)
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "reports/stage1/S1-13-gate-contract.json"
        ).read_text(encoding="utf-8")
    )
    execution = {
        "runtime_budget_seconds": 100,
        "pbs_walltime_seconds": 200,
        "minimum_global_cycles": 2,
    }
    config = {"bounded_state": {"inventory_every_global_cycles": 10}}
    passed = _runtime_gate(
        roles,
        syncer,
        workload="long_run",
        config=config,
        gate_contract=contract,
        execution_contract=execution,
    )
    assert passed["status"] == "pass"
    syncer["updates"][7]["completed_unix_ns"] = 100_000_000_000
    heartbeat_failure = _runtime_gate(
        roles,
        syncer,
        workload="long_run",
        config=config,
        gate_contract=contract,
        execution_contract=execution,
    )
    assert heartbeat_failure["status"] == "fail"
    assert heartbeat_failure["progress_heartbeat"]["pass"] is False
    syncer["updates"][7]["completed_unix_ns"] = 8_000_000_000
    roles[0]["publication"]["publication"]["pending_upload_count"] = 1
    stall_failure = _runtime_gate(
        roles,
        syncer,
        workload="long_run",
        config=config,
        gate_contract=contract,
        execution_contract=execution,
    )
    assert stall_failure["status"] == "fail"
    assert stall_failure["pending_stall"]["pass"] is False
    roles[0]["publication"]["publication"]["pending_upload_count"] = 0
    roles[0]["logging"]["final_fsync_complete"] = False
    logging_failure = _runtime_gate(
        roles,
        syncer,
        workload="long_run",
        config=config,
        gate_contract=contract,
        execution_contract=execution,
    )
    assert logging_failure["status"] == "fail"
    assert logging_failure["structured_logging_durability"]["pass"] is False
