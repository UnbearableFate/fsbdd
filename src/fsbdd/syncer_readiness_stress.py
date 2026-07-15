from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .global_state import CountingStorageBackend, GlobalStateIdentities, GlobalStateStore
from .global_state_stress import _fragments, derive_stress_identities
from .identity import file_digest
from .manifest import build_manifest
from .proposal import (
    ConsumptionFrontiers,
    Proposal,
    ProposalStore,
    commit_consumption,
)
from .storage import PosixStorageBackend
from .syncer_readiness import (
    FragmentReadinessAuthority,
    ReadinessConfig,
    ReadinessPhase,
    SyncerReadinessMachine,
)


class ReadinessStressError(RuntimeError):
    pass


def _write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _wait(path: Path, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"coordination record did not appear: {path}")
            time.sleep(0.005)
            continue
        if not isinstance(value, dict) or value.get("complete") is not True:
            raise ReadinessStressError(
                f"coordination record is malformed: {path}"
            )
        return value


def _load_config(path: Path, expected_identity: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_identity:
        raise ReadinessStressError("readiness config identity mismatch")
    value = json.loads(raw)
    expected = {
        "schema_version",
        "s_max",
        "maximum_local_steps",
        "maximum_processed_tokens",
        "payload_bytes",
        "profile_a",
        "decoupled_grace",
        "poll_interval_seconds",
        "coordination_timeout_seconds",
        "history_objects",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ReadinessStressError("readiness config schema mismatch")
    if value["schema_version"] != 1 or value["s_max"] != 0:
        raise ReadinessStressError("readiness config requires schema one and s_max zero")
    profile_fields = {
        "logical_syncer_id",
        "learner_count",
        "fragment_count",
        "q",
        "q_fresh",
        "max_contributors",
        "grace_period_ns",
    }
    grace_fields = profile_fields | {
        "initial_learners",
        "pre_freeze_late_learners",
        "post_freeze_late_learners",
    }
    if set(value["profile_a"]) != profile_fields:
        raise ReadinessStressError("Profile A config schema mismatch")
    if set(value["decoupled_grace"]) != grace_fields:
        raise ReadinessStressError("grace config schema mismatch")
    return value


def _learner_ids(count: int) -> tuple[str, ...]:
    if count <= 0:
        raise ReadinessStressError("learner count must be positive")
    return tuple(f"learner-{index:02d}" for index in range(count))


def _scenario(
    *,
    root: Path,
    run_identity: str,
    config_identity: str,
    profile: dict[str, Any],
    payload_bytes: int,
    maximum_local_steps: int,
    maximum_processed_tokens: int,
    counting: bool,
):
    initial = _fragments(profile["fragment_count"], payload_bytes)
    model_identity, fragment_map_identity = derive_stress_identities(
        config_identity, initial
    )
    identities = GlobalStateIdentities(
        run_identity=run_identity,
        config_identity=config_identity,
        model_identity=model_identity,
        fragment_map_identity=fragment_map_identity,
    )
    backend = PosixStorageBackend(root / "backend")
    global_store = GlobalStateStore(
        backend,
        identities=identities,
        descriptors=tuple(item.descriptor for item in initial),
        s_max=0,
    )
    proposal_backend = CountingStorageBackend(backend) if counting else backend
    proposal_store = ProposalStore(
        proposal_backend,
        identities=identities,
        descriptors=tuple(item.descriptor for item in initial),
        learner_ids=_learner_ids(profile["learner_count"]),
        maximum_local_steps=maximum_local_steps,
        maximum_processed_tokens=maximum_processed_tokens,
    )
    return initial, global_store, proposal_store, proposal_backend, identities


def _proposal(
    store: ProposalStore,
    state,
    *,
    learner_index: int,
    fragment_index: int,
    sequence: int = 1,
    payload_bytes: int,
) -> Proposal:
    learner_id = store.learner_ids[learner_index]
    return Proposal.create(
        proposal_id=(
            f"{learner_id}-fragment-{fragment_index:02d}-sequence-{sequence:06d}"
        ),
        identities=store.identities,
        learner_id=learner_id,
        descriptor=store.descriptors[fragment_index],
        sequence=sequence,
        base_version=state.version,
        base_content_identity=state.content_identity,
        local_steps=sequence,
        processed_tokens=(learner_index + 1) * 1000 + fragment_index + sequence,
        snapshot_local_step=sequence,
        parameters=bytes(
            [(learner_index * 17 + fragment_index * 5 + sequence) % 251]
        )
        * payload_bytes,
    )


def _authorities(states, learner_ids: tuple[str, ...]):
    return tuple(
        FragmentReadinessAuthority(
            state,
            ConsumptionFrontiers.empty(
                identities=state.identities,
                descriptor=state.descriptor,
                learner_ids=learner_ids,
            ),
        )
        for state in states
    )


def _readiness_config(profile: dict[str, Any]) -> ReadinessConfig:
    return ReadinessConfig(
        logical_syncer_id=profile["logical_syncer_id"],
        q=profile["q"],
        q_fresh=profile["q_fresh"],
        max_contributors=profile["max_contributors"],
        grace_period_ns=profile["grace_period_ns"],
        s_max=0,
    )


def _gpu_memory_for_current_process() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    total = 0
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            total += int(fields[1])
    return total


def _identity_dict(value: GlobalStateIdentities) -> dict[str, str]:
    return value.to_dict()


def run_role(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_path: Path,
    config_identity: str,
) -> dict[str, Any]:
    try:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise ReadinessStressError(
            "readiness role requires an MPI launcher environment"
        ) from error
    if size != 2 or rank not in (0, 1):
        raise ReadinessStressError("readiness stress requires exactly two ranks")
    config = _load_config(config_path, config_identity)
    timeout = float(config["coordination_timeout_seconds"])
    coordination = root / "coordination"
    hostname = socket.gethostname().split(".")[0]
    gpu_before = _gpu_memory_for_current_process()
    profile = config["profile_a"]
    grace = config["decoupled_grace"]
    profile_values = _scenario(
        root=root / "profile-a",
        run_identity=f"{run_id}-profile-a",
        config_identity=config_identity,
        profile=profile,
        payload_bytes=config["payload_bytes"],
        maximum_local_steps=config["maximum_local_steps"],
        maximum_processed_tokens=config["maximum_processed_tokens"],
        counting=rank == 0,
    )
    grace_values = _scenario(
        root=root / "grace",
        run_identity=f"{run_id}-grace",
        config_identity=config_identity,
        profile=grace,
        payload_bytes=config["payload_bytes"],
        maximum_local_steps=config["maximum_local_steps"],
        maximum_processed_tokens=config["maximum_processed_tokens"],
        counting=rank == 0,
    )
    (
        profile_initial,
        profile_global,
        profile_proposals,
        profile_backend,
        profile_identities,
    ) = profile_values
    (
        grace_initial,
        grace_global,
        grace_proposals,
        grace_backend,
        grace_identities,
    ) = grace_values
    frozen_identities = {
        "profile_a": _identity_dict(profile_identities),
        "decoupled_grace": _identity_dict(grace_identities),
    }

    if rank == 1:
        _wait(coordination / "authority-ready.json", timeout)
        profile_states = profile_global.load_snapshot().states
        for learner_index in range(profile["learner_count"]):
            for fragment_index in range(profile["fragment_count"]):
                profile_proposals.publish(
                    _proposal(
                        profile_proposals,
                        profile_states[fragment_index],
                        learner_index=learner_index,
                        fragment_index=fragment_index,
                        payload_bytes=config["payload_bytes"],
                    )
                )
        _replace_json(coordination / "profile-ready.json", {"complete": True})
        _wait(coordination / "profile-baseline-observed.json", timeout)
        payload_root = root / "profile-a" / "backend" / "payloads"
        for index in range(config["history_objects"]):
            (payload_root / f"unrelated-{index:05d}.bin").write_bytes(b"history")
        _replace_json(coordination / "history-ready.json", {"complete": True})
        _wait(coordination / "history-observed.json", timeout)

        grace_states = grace_global.load_snapshot().states
        for learner_index in range(grace["initial_learners"]):
            grace_proposals.publish(
                _proposal(
                    grace_proposals,
                    grace_states[0],
                    learner_index=learner_index,
                    fragment_index=0,
                    payload_bytes=config["payload_bytes"],
                )
            )
        _replace_json(coordination / "grace-initial-ready.json", {"complete": True})
        _wait(coordination / "grace-started.json", timeout)
        pre_index = grace["initial_learners"]
        grace_proposals.publish(
            _proposal(
                grace_proposals,
                grace_states[0],
                learner_index=pre_index,
                fragment_index=0,
                payload_bytes=config["payload_bytes"],
            )
        )
        _replace_json(coordination / "grace-pre-freeze-ready.json", {"complete": True})
        _wait(coordination / "grace-frozen.json", timeout)
        post_index = pre_index + grace["pre_freeze_late_learners"]
        grace_proposals.publish(
            _proposal(
                grace_proposals,
                grace_states[0],
                learner_index=post_index,
                fragment_index=0,
                payload_bytes=config["payload_bytes"],
            )
        )
        _replace_json(coordination / "grace-post-freeze-ready.json", {"complete": True})
        _wait(coordination / "syncer-done.json", timeout)
        result = {
            "schema_version": 1,
            "status": "pass",
            "role": "proposal_writer",
            "rank": rank,
            "hostname": hostname,
            "state_identities": frozen_identities,
            "torch_imported": "torch" in sys.modules,
            "gpu_memory_before_bytes": gpu_before,
            "gpu_memory_after_bytes": _gpu_memory_for_current_process(),
            "profile_a_published": profile["learner_count"]
            * profile["fragment_count"],
            "history_objects": config["history_objects"],
            "grace_initial_published": grace["initial_learners"],
            "grace_pre_freeze_published": grace["pre_freeze_late_learners"],
            "grace_post_freeze_published": grace["post_freeze_late_learners"],
        }
        _write_json_new(result_root / "proposal_writer.json", result)
        return result

    profile_states = profile_global.bootstrap(profile_initial).snapshot.states
    grace_states = grace_global.bootstrap(grace_initial).snapshot.states
    _replace_json(coordination / "authority-ready.json", {"complete": True})
    _wait(coordination / "profile-ready.json", timeout)
    profile_authorities = _authorities(
        profile_states, profile_proposals.learner_ids
    )
    machine = SyncerReadinessMachine(
        profile_proposals,
        authorities=profile_authorities,
        config=_readiness_config(profile),
    )
    profile_backend.reset_counts()
    baseline = machine.poll_store(observed_ns=0)
    baseline_reads = profile_backend.read_calls
    baseline_selections = tuple(
        machine.selection_for(index) for index in range(profile["fragment_count"])
    )
    _replace_json(
        coordination / "profile-baseline-observed.json", {"complete": True}
    )
    _wait(coordination / "history-ready.json", timeout)
    profile_backend.reset_counts()
    after_history = machine.poll_store(observed_ns=1)
    history_reads = profile_backend.read_calls
    history_selection_ids = [
        machine.selection_for(index).selection_identity  # type: ignore[union-attr]
        for index in range(profile["fragment_count"])
    ]
    _replace_json(coordination / "history-observed.json", {"complete": True})

    restarted = SyncerReadinessMachine(
        profile_proposals,
        authorities=profile_authorities,
        config=_readiness_config(profile),
    )
    profile_backend.reset_counts()
    restart_report = restarted.poll_store(observed_ns=10**12)
    restart_reads = profile_backend.read_calls
    restart_selection_ids = [
        restarted.selection_for(index).selection_identity  # type: ignore[union-attr]
        for index in range(profile["fragment_count"])
    ]

    claim_order: list[int] = []
    concurrent_claim_rejections = 0
    for expected_index in range(profile["fragment_count"]):
        lease = machine.claim_next()
        if lease is None:
            raise ReadinessStressError("Profile A fragment was not claimable")
        claim_order.append(lease.selection.fragment_index)
        if machine.claim_next() is None:
            concurrent_claim_rejections += 1
        consumed = commit_consumption(
            profile_authorities[expected_index].frontiers,
            lease.selection.proposals,
            True,
        )
        advanced = FragmentReadinessAuthority(
            profile_authorities[expected_index].state, consumed
        )
        machine.complete(lease, advanced)
        profile_authorities = tuple(
            advanced if index == expected_index else authority
            for index, authority in enumerate(profile_authorities)
        )
    profile_backend.reset_counts()
    repeated = machine.poll_store(observed_ns=2)
    repeated_reads = profile_backend.read_calls
    repeated_selections = [
        machine.selection_for(index) for index in range(profile["fragment_count"])
    ]

    _wait(coordination / "grace-initial-ready.json", timeout)
    grace_authorities = _authorities(grace_states, grace_proposals.learner_ids)
    grace_machine = SyncerReadinessMachine(
        grace_proposals,
        authorities=grace_authorities,
        config=_readiness_config(grace),
    )
    grace_backend.reset_counts()
    grace_started = grace_machine.poll_store(observed_ns=0)
    grace_start_reads = grace_backend.read_calls
    _replace_json(coordination / "grace-started.json", {"complete": True})
    _wait(coordination / "grace-pre-freeze-ready.json", timeout)
    grace_backend.reset_counts()
    grace_before = grace_machine.poll_store(
        observed_ns=grace["grace_period_ns"] - 1
    )
    grace_before_reads = grace_backend.read_calls
    grace_backend.reset_counts()
    grace_frozen = grace_machine.poll_store(
        observed_ns=grace["grace_period_ns"]
    )
    grace_freeze_reads = grace_backend.read_calls
    frozen_selection = grace_machine.selection_for(0)
    if frozen_selection is None:
        raise ReadinessStressError("grace selection did not freeze")
    frozen_dict = frozen_selection.to_dict()
    _replace_json(coordination / "grace-frozen.json", {"complete": True})
    _wait(coordination / "grace-post-freeze-ready.json", timeout)
    grace_backend.reset_counts()
    grace_after = grace_machine.poll_store(
        observed_ns=grace["grace_period_ns"] + 1
    )
    grace_after_reads = grace_backend.read_calls
    after_selection = grace_machine.selection_for(0)
    if after_selection is None:
        raise ReadinessStressError("frozen grace selection disappeared")
    grace_lease = grace_machine.claim_next()
    if grace_lease is None:
        raise ReadinessStressError("frozen grace selection was not claimable")
    grace_machine.release(grace_lease)
    _replace_json(coordination / "syncer-done.json", {"complete": True})

    profile_selection_dicts = [
        item.to_dict() if item is not None else None
        for item in baseline_selections
    ]
    result = {
        "schema_version": 1,
        "status": "pass",
        "role": "syncer_scheduler",
        "rank": rank,
        "hostname": hostname,
        "state_identities": frozen_identities,
        "torch_imported": "torch" in sys.modules,
        "gpu_memory_before_bytes": gpu_before,
        "gpu_memory_after_bytes": _gpu_memory_for_current_process(),
        "logical_syncer_count": 1,
        "profile_a": {
            "baseline_fixed_reads": baseline_reads,
            "history_fixed_reads": history_reads,
            "restart_fixed_reads": restart_reads,
            "repeated_fixed_reads": repeated_reads,
            "baseline_missing_slots": baseline.missing_slots,
            "history_missing_slots": after_history.missing_slots,
            "restart_missing_slots": restart_report.missing_slots,
            "selections": profile_selection_dicts,
            "history_selection_ids": history_selection_ids,
            "restart_selection_ids": restart_selection_ids,
            "claim_order": claim_order,
            "concurrent_claim_rejections": concurrent_claim_rejections,
            "repeated_phases": [item.phase.value for item in repeated.fragments],
            "repeated_eligible": [
                item.eligible_distinct for item in repeated.fragments
            ],
            "repeated_rejected": [item.rejected for item in repeated.fragments],
            "repeated_selected": [
                None if item is None else item.selection_identity
                for item in repeated_selections
            ],
            "frontier_entries": [
                len(authority.frontiers.entries)
                for authority in profile_authorities
            ],
            "resident_selected_proposals": machine.snapshot().resident_selected_proposals,
            "poll_cycles": machine.snapshot().poll_cycles,
        },
        "grace": {
            "start_fixed_reads": grace_start_reads,
            "before_fixed_reads": grace_before_reads,
            "freeze_fixed_reads": grace_freeze_reads,
            "after_fixed_reads": grace_after_reads,
            "start_phase": grace_started.fragments[0].phase.value,
            "start_ns": grace_started.fragments[0].grace_started_ns,
            "deadline_ns": grace_started.fragments[0].grace_deadline_ns,
            "before_phase": grace_before.fragments[0].phase.value,
            "freeze_phase": grace_frozen.fragments[0].phase.value,
            "after_phase": grace_after.fragments[0].phase.value,
            "frozen_selection": frozen_dict,
            "after_selection_identity": after_selection.selection_identity,
            "after_selected_learners": list(after_selection.learner_ids),
            "after_weights": [item.to_dict() for item in after_selection.weights],
        },
    }
    _write_json_new(result_root / "syncer_scheduler.json", result)
    return result


def _analyze_results(
    *,
    writer: dict[str, Any],
    syncer: dict[str, Any],
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    if (
        writer.get("status") != "pass"
        or syncer.get("status") != "pass"
        or writer.get("role") != "proposal_writer"
        or syncer.get("role") != "syncer_scheduler"
        or writer.get("rank") != 1
        or syncer.get("rank") != 0
        or writer.get("hostname") == syncer.get("hostname")
    ):
        raise ReadinessStressError("formal roles statuses ranks or hosts are invalid")
    expected_identities: dict[str, dict[str, str]] = {}
    for name, identity_suffix in (
        ("profile_a", "profile-a"),
        ("decoupled_grace", "grace"),
    ):
        initial = _fragments(config[name]["fragment_count"], config["payload_bytes"])
        model_identity, fragment_map_identity = derive_stress_identities(
            config["config_identity"], initial
        )
        expected_identities[name] = {
            "run_identity": f"{run_id}-{identity_suffix}",
            "config_identity": config["config_identity"],
            "model_identity": model_identity,
            "fragment_map_identity": fragment_map_identity,
        }
    if (
        writer.get("state_identities") != syncer.get("state_identities")
        or syncer.get("state_identities") != expected_identities
    ):
        raise ReadinessStressError("formal role identities differ")
    for role in (writer, syncer):
        if role.get("torch_imported") is not False:
            raise ReadinessStressError("formal readiness role imported Torch")
        if role.get("gpu_memory_before_bytes") != 0:
            raise ReadinessStressError("formal readiness role began with GPU memory")
        if role.get("gpu_memory_after_bytes") != 0:
            raise ReadinessStressError("formal readiness role allocated GPU memory")
    profile = config["profile_a"]
    grace = config["decoupled_grace"]
    slot_count = profile["learner_count"] * profile["fragment_count"]
    profile_result = syncer["profile_a"]
    if syncer.get("logical_syncer_count") != 1:
        raise ReadinessStressError("formal topology did not use one logical syncer")
    if writer.get("profile_a_published") != slot_count:
        raise ReadinessStressError("writer did not populate every Profile A slot")
    for field in (
        "baseline_fixed_reads",
        "history_fixed_reads",
        "restart_fixed_reads",
        "repeated_fixed_reads",
    ):
        if profile_result.get(field) != slot_count:
            raise ReadinessStressError(f"Profile A {field} is not exactly M times F")
    if writer.get("history_objects") != config["history_objects"]:
        raise ReadinessStressError("history-object workload differs from config")
    if any(
        profile_result.get(field) != 0
        for field in (
            "baseline_missing_slots",
            "history_missing_slots",
            "restart_missing_slots",
        )
    ):
        raise ReadinessStressError("Profile A had a missing frozen slot")
    selections = profile_result["selections"]
    if len(selections) != profile["fragment_count"] or any(
        item is None for item in selections
    ):
        raise ReadinessStressError("Profile A selection inventory is incomplete")
    baseline_ids = []
    for index, selection in enumerate(selections):
        if (
            selection["fragment_index"] != index
            or len(selection["proposal_ids"]) != profile["learner_count"]
            or len(set(selection["learner_ids"])) != profile["learner_count"]
            or any(item["staleness"] != 0 for item in selection["weights"])
            or abs(
                sum(item["normalized_weight"] for item in selection["weights"])
                - 1.0
            )
            > 2e-6
        ):
            raise ReadinessStressError("Profile A distinct fresh selection is invalid")
        baseline_ids.append(selection["selection_identity"])
    if (
        profile_result["history_selection_ids"] != baseline_ids
        or profile_result["restart_selection_ids"] != baseline_ids
    ):
        raise ReadinessStressError("selection depends on history order time or restart")
    if profile_result["claim_order"] != list(range(profile["fragment_count"])):
        raise ReadinessStressError("round-robin Profile A claim order is invalid")
    if profile_result["concurrent_claim_rejections"] != profile["fragment_count"]:
        raise ReadinessStressError("single active-lease invariant is incomplete")
    if (
        profile_result["repeated_phases"]
        != [ReadinessPhase.WAITING.value] * profile["fragment_count"]
        or profile_result["repeated_eligible"] != [0] * profile["fragment_count"]
        or profile_result["repeated_rejected"]
        != [profile["learner_count"]] * profile["fragment_count"]
        or any(item is not None for item in profile_result["repeated_selected"])
        or profile_result["frontier_entries"]
        != [profile["learner_count"]] * profile["fragment_count"]
        or profile_result["resident_selected_proposals"] != 0
    ):
        raise ReadinessStressError("repeated polling consumed-frontier evidence is invalid")
    grace_result = syncer["grace"]
    for field in (
        "start_fixed_reads",
        "before_fixed_reads",
        "freeze_fixed_reads",
        "after_fixed_reads",
    ):
        if grace_result[field] != grace["learner_count"]:
            raise ReadinessStressError(f"grace {field} is not exactly M")
    frozen = grace_result["frozen_selection"]
    expected_selected = grace["initial_learners"] + grace["pre_freeze_late_learners"]
    expected_learners = [
        f"learner-{index:02d}" for index in reversed(range(expected_selected))
    ]
    post_learner = f"learner-{expected_selected:02d}"
    if (
        grace_result["start_phase"] != ReadinessPhase.GRACE.value
        or grace_result["start_ns"] != 0
        or grace_result["deadline_ns"] != grace["grace_period_ns"]
        or grace_result["before_phase"] != ReadinessPhase.GRACE.value
        or grace_result["freeze_phase"] != ReadinessPhase.FROZEN.value
        or grace_result["after_phase"] != ReadinessPhase.FROZEN.value
        or frozen["learner_ids"] != expected_learners
        or grace_result["after_selected_learners"] != expected_learners
        or post_learner in grace_result["after_selected_learners"]
        or grace_result["after_selection_identity"] != frozen["selection_identity"]
        or grace_result["after_weights"] != frozen["weights"]
        or len(frozen["weights"]) != expected_selected
        or abs(
            sum(item["normalized_weight"] for item in frozen["weights"]) - 1.0
        )
        > 2e-6
    ):
        raise ReadinessStressError("fixed grace or late-arrival evidence is invalid")
    if (
        writer["grace_initial_published"] != grace["initial_learners"]
        or writer["grace_pre_freeze_published"]
        != grace["pre_freeze_late_learners"]
        or writer["grace_post_freeze_published"]
        != grace["post_freeze_late_learners"]
    ):
        raise ReadinessStressError("grace writer counts differ from config")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "run_identity": run_id,
        "filesystem_data_plane": "shared_posix_only",
        "application_coordination": "filesystem_only",
        "mpi_usage": "launcher_only",
        "logical_syncers": syncer["logical_syncer_count"],
        "roles": {
            "proposal_writer": [writer["hostname"]],
            "syncer_scheduler": [syncer["hostname"]],
        },
        "state_identities": syncer["state_identities"],
        "gpu_memory_bytes": {
            "proposal_writer_before": writer["gpu_memory_before_bytes"],
            "proposal_writer_after": writer["gpu_memory_after_bytes"],
            "syncer_scheduler_before": syncer["gpu_memory_before_bytes"],
            "syncer_scheduler_after": syncer["gpu_memory_after_bytes"],
        },
        "profile_a": {
            "learner_count": profile["learner_count"],
            "fragment_count": profile["fragment_count"],
            "q": profile["q"],
            "q_fresh": profile["q_fresh"],
            "grace_period_ns": profile["grace_period_ns"],
            "fixed_reads_each_poll": slot_count,
            "history_objects": config["history_objects"],
            "selections": selections,
            "claim_order": profile_result["claim_order"],
            "maximum_continuously_ready_service_window": profile["fragment_count"],
            "single_active_lease": True,
            "history_independent": True,
            "restart_rebuilt_from_authority": True,
            "repeated_polling_selected_count": 0,
            "fixed_frontier_entries_per_fragment": profile_result["frontier_entries"],
        },
        "decoupled_grace": {
            "q": grace["q"],
            "q_fresh": grace["q_fresh"],
            "grace_period_ns": grace["grace_period_ns"],
            "selected_learners": frozen["learner_ids"],
            "pre_freeze_arrival_included": True,
            "post_freeze_arrival_excluded": True,
            "selection_and_weights_stable": True,
        },
        "bounded_resident_state": {
            "maximum_frozen_proposals": profile["learner_count"]
            * profile["fragment_count"],
            "final_resident_selected_proposals": profile_result[
                "resident_selected_proposals"
            ],
            "unbounded_production_history": False,
        },
    }
    return summary


def analyze_results(
    *,
    writer: dict[str, Any],
    syncer: dict[str, Any],
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    try:
        return _analyze_results(
            writer=writer,
            syncer=syncer,
            config=config,
            run_id=run_id,
        )
    except ReadinessStressError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ReadinessStressError(
            "formal readiness result schema or value is malformed"
        ) from error


def summarize(
    *,
    root: Path,
    result_root: Path,
    run_id: str,
    config_path: Path,
    config_identity: str,
    output: Path,
) -> dict[str, Any]:
    config = _load_config(config_path, config_identity)
    config = {**config, "config_identity": config_identity}
    writer = json.loads(
        (result_root / "proposal_writer.json").read_text(encoding="utf-8")
    )
    syncer = json.loads(
        (result_root / "syncer_scheduler.json").read_text(encoding="utf-8")
    )
    summary = analyze_results(
        writer=writer,
        syncer=syncer,
        config=config,
        run_id=run_id,
    )
    _write_json_new(output, summary)
    shutil.rmtree(root)
    return summary


def write_manifest(args: argparse.Namespace) -> None:
    hosts = sorted(set(Path(args.nodefile).read_text(encoding="utf-8").splitlines()))
    if len(hosts) != 2:
        raise ReadinessStressError(
            f"readiness manifest requires two distinct hosts: {hosts}"
        )
    writer = json.loads(
        (Path(args.result_root) / "proposal_writer.json").read_text(encoding="utf-8")
    )
    syncer = json.loads(
        (Path(args.result_root) / "syncer_scheduler.json").read_text(encoding="utf-8")
    )
    if (
        writer.get("status") != "pass"
        or syncer.get("status") != "pass"
        or writer.get("hostname") == syncer.get("hostname")
        or writer.get("torch_imported") is not False
        or syncer.get("torch_imported") is not False
    ):
        raise ReadinessStressError("manifest role result validation failed")
    if writer.get("state_identities") != syncer.get("state_identities"):
        raise ReadinessStressError("manifest role state identities differ")
    state_identities = syncer.get("state_identities", {})
    expected_runs = {
        "profile_a": f"{args.run_id}-profile-a",
        "decoupled_grace": f"{args.run_id}-grace",
    }
    if not isinstance(state_identities, dict) or any(
        not isinstance(state_identities.get(name), dict)
        or state_identities[name].get("run_identity") != expected_run
        or state_identities[name].get("config_identity") != args.config_sha256
        for name, expected_run in expected_runs.items()
    ):
        raise ReadinessStressError("manifest state identity is not run/config bound")
    if args.run_id != f"s1-09-{args.job_id}":
        raise ReadinessStressError("manifest run identity is not bound to PBS job id")
    if args.queue != "debug-g" or args.group != "xg24i002":
        raise ReadinessStressError("manifest scheduler identity differs from contract")
    role_map = {
        "proposal_writer": [writer["hostname"]],
        "syncer_scheduler": [syncer["hostname"]],
    }
    if set(role_map["proposal_writer"] + role_map["syncer_scheduler"]) != set(hosts):
        raise ReadinessStressError("actual role map differs from allocated hosts")
    modules = [
        line.strip()
        for line in Path(args.modules_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identities = {
        "code": {
            "repository": args.repository,
            "branch": args.branch,
            "commit": args.commit,
            "dirty": False,
        },
        "config": {"sha256": args.config_sha256},
        "source": {
            "research_plan_sha256": args.research_sha256,
            "stage0_4_spec_sha256": args.spec_sha256,
        },
        "skill": {
            "repository": args.skill_repository,
            "commit": args.skill_commit,
        },
        "execution": {
            "identity": args.run_id,
            "initial_hostname": args.initial_hostname,
            "compute_hostname": socket.gethostname().split(".")[0],
            "workflow": "two-node-filesystem-only-readiness-and-fair-scheduler",
        },
        "roles": {"declared": role_map, "actual": role_map},
        "paths": {
            "project_root": args.project_root,
            "evidence_root": args.evidence_root,
        },
    }
    scheduler = {
        "job_id": args.job_id,
        "qtime_utc": args.qtime_utc,
        "queue": args.queue,
        "group": args.group,
        "nodefile_sha256": file_digest(Path(args.nodefile)),
        "modules": modules,
    }
    manifest = build_manifest(
        "S1-09", "L2", args.qtime_utc, "pbs_qtime", identities, scheduler=scheduler
    )
    _write_json_new(Path(args.output), manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fsbdd.syncer_readiness_stress")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("role", "summarize"):
        current = commands.add_parser(command)
        current.add_argument("--root", required=True, type=Path)
        current.add_argument("--result-root", required=True, type=Path)
        current.add_argument("--run-id", required=True)
        current.add_argument("--config-path", required=True, type=Path)
        current.add_argument("--config-identity", required=True)
        if command == "summarize":
            current.add_argument("--output", required=True, type=Path)
    manifest = commands.add_parser("manifest")
    for name in (
        "repository",
        "branch",
        "commit",
        "run-id",
        "config-sha256",
        "research-sha256",
        "spec-sha256",
        "skill-repository",
        "skill-commit",
        "initial-hostname",
        "project-root",
        "evidence-root",
        "job-id",
        "qtime-utc",
        "queue",
        "group",
        "nodefile",
        "modules-file",
        "result-root",
        "output",
    ):
        manifest.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    if args.command == "role":
        value = run_role(
            root=args.root,
            result_root=args.result_root,
            run_id=args.run_id,
            config_path=args.config_path,
            config_identity=args.config_identity,
        )
        print(
            json.dumps(
                {"status": value["status"], "role": value["role"]},
                sort_keys=True,
            )
        )
        return 0
    if args.command == "summarize":
        value = summarize(
            root=args.root,
            result_root=args.result_root,
            run_id=args.run_id,
            config_path=args.config_path,
            config_identity=args.config_identity,
            output=args.output,
        )
        print(json.dumps({"status": value["status"]}, sort_keys=True))
        return 0
    write_manifest(args)
    print(json.dumps({"status": "building"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
