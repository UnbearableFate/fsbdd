from __future__ import annotations

import os
import json
import time
from pathlib import Path

from fsbdd.global_state import BootstrapFragment, FragmentStateDescriptor, GlobalStateIdentities, GlobalStateStore
from fsbdd.proposal import ConsumptionFrontiers, Proposal, ProposalStore
from fsbdd.storage import PosixStorageBackend, PublicationSpec, ReadExpectation
from fsbdd.syncer_readiness import FragmentReadinessAuthority, ReadinessConfig, SyncerReadinessMachine
from fsbdd.stage1_gate import _topology_gate


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


def test_metadata_only_read_and_bounded_payload_reclamation(tmp_path: Path, monkeypatch) -> None:
    backend = PosixStorageBackend(tmp_path)
    records = [backend.publish("current", bytes([index]) * 16, _spec(index)) for index in range(5)]
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
    reclaimed = backend.reclaim_unreferenced_payloads(retain_recent=1, minimum_age_seconds=0)
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
            }
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
        }
    }
    assert _topology_gate(roles, syncer, workload="nine_node", expected_learners=8)["status"] == "pass"
    roles[1]["identity"]["hostname"] = "node-duplicate"
    roles[1]["identity"]["gpu"]["gpu_uuid"] = "GPU-duplicate"
    assert _topology_gate(roles, syncer, workload="nine_node", expected_learners=8)["status"] == "fail"
