from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fsbdd.global_state_stress import _fragments, derive_stress_identities
from fsbdd.syncer_readiness_stress import ReadinessStressError, analyze_results


CONFIG_IDENTITY = "8ff3fe364fca10bb48fe80ebf0bcede63163731e47cc26fc87512b298e3c225d"
RUN_ID = "s1-09-formal-fixture"


def _config() -> dict:
    value = json.loads(
        Path("configs/stage1/s1_09_syncer_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    value["config_identity"] = CONFIG_IDENTITY
    return value


def _state_identities(config: dict) -> dict:
    values = {}
    for name, suffix in (("profile_a", "profile-a"), ("decoupled_grace", "grace")):
        initial = _fragments(
            config[name]["fragment_count"], config["payload_bytes"]
        )
        model, fragment_map = derive_stress_identities(CONFIG_IDENTITY, initial)
        values[name] = {
            "run_identity": f"{RUN_ID}-{suffix}",
            "config_identity": CONFIG_IDENTITY,
            "model_identity": model,
            "fragment_map_identity": fragment_map,
        }
    return values


def _weight(learner: int, count: int) -> dict:
    return {
        "proposal_id": f"proposal-{learner}",
        "learner_id": f"learner-{learner:02d}",
        "processed_tokens": learner + 1,
        "staleness": 0,
        "raw_weight": float(learner + 1),
        "normalized_weight": 1.0 / count,
    }


def _selection(fragment: int, learners: int) -> dict:
    return {
        "fragment_index": fragment,
        "current_version": 0,
        "authority_identity": f"authority-{fragment}",
        "generation": 1,
        "grace_started_ns": 0,
        "frozen_observed_ns": 0,
        "proposal_ids": [f"f{fragment}-proposal-{index}" for index in range(learners)],
        "proposal_content_identities": [f"content-{index}" for index in range(learners)],
        "learner_ids": [f"learner-{index:02d}" for index in range(learners)],
        "weights": [_weight(index, learners) for index in range(learners)],
        "selection_identity": f"selection-{fragment}",
    }


def _valid_results() -> tuple[dict, dict, dict]:
    config = _config()
    identities = _state_identities(config)
    profile = config["profile_a"]
    grace = config["decoupled_grace"]
    slot_count = profile["learner_count"] * profile["fragment_count"]
    selections = [
        _selection(index, profile["learner_count"])
        for index in range(profile["fragment_count"])
    ]
    selection_ids = [item["selection_identity"] for item in selections]
    frozen = _selection(0, grace["initial_learners"] + grace["pre_freeze_late_learners"])
    for field in (
        "proposal_ids",
        "proposal_content_identities",
        "learner_ids",
        "weights",
    ):
        frozen[field] = list(reversed(frozen[field]))
    writer = {
        "status": "pass",
        "role": "proposal_writer",
        "rank": 1,
        "hostname": "host-b",
        "state_identities": identities,
        "torch_imported": False,
        "gpu_memory_before_bytes": 0,
        "gpu_memory_after_bytes": 0,
        "profile_a_published": slot_count,
        "history_objects": config["history_objects"],
        "grace_initial_published": grace["initial_learners"],
        "grace_pre_freeze_published": grace["pre_freeze_late_learners"],
        "grace_post_freeze_published": grace["post_freeze_late_learners"],
    }
    syncer = {
        "status": "pass",
        "role": "syncer_scheduler",
        "rank": 0,
        "hostname": "host-a",
        "state_identities": identities,
        "torch_imported": False,
        "gpu_memory_before_bytes": 0,
        "gpu_memory_after_bytes": 0,
        "logical_syncer_count": 1,
        "profile_a": {
            "baseline_fixed_reads": slot_count,
            "history_fixed_reads": slot_count,
            "restart_fixed_reads": slot_count,
            "repeated_fixed_reads": slot_count,
            "baseline_missing_slots": 0,
            "history_missing_slots": 0,
            "restart_missing_slots": 0,
            "selections": selections,
            "history_selection_ids": selection_ids,
            "restart_selection_ids": selection_ids,
            "claim_order": list(range(profile["fragment_count"])),
            "concurrent_claim_rejections": profile["fragment_count"],
            "repeated_phases": ["waiting"] * profile["fragment_count"],
            "repeated_eligible": [0] * profile["fragment_count"],
            "repeated_rejected": [profile["learner_count"]] * profile["fragment_count"],
            "repeated_selected": [None] * profile["fragment_count"],
            "frontier_entries": [profile["learner_count"]] * profile["fragment_count"],
            "resident_selected_proposals": 0,
            "poll_cycles": 3,
        },
        "grace": {
            "start_fixed_reads": grace["learner_count"],
            "before_fixed_reads": grace["learner_count"],
            "freeze_fixed_reads": grace["learner_count"],
            "after_fixed_reads": grace["learner_count"],
            "start_phase": "grace",
            "start_ns": 0,
            "deadline_ns": grace["grace_period_ns"],
            "before_phase": "grace",
            "freeze_phase": "frozen",
            "after_phase": "frozen",
            "frozen_selection": frozen,
            "after_selection_identity": frozen["selection_identity"],
            "after_selected_learners": frozen["learner_ids"],
            "after_weights": frozen["weights"],
        },
    }
    return config, writer, syncer


def test_analyzer_accepts_complete_frozen_fixture() -> None:
    config, writer, syncer = _valid_results()
    summary = analyze_results(
        writer=writer, syncer=syncer, config=config, run_id=RUN_ID
    )
    assert summary["status"] == "pass"
    assert summary["logical_syncers"] == 1
    assert summary["profile_a"]["repeated_polling_selected_count"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda writer, syncer: syncer.__setitem__("logical_syncer_count", 2),
        lambda writer, syncer: syncer["profile_a"].__setitem__("baseline_fixed_reads", 31),
        lambda writer, syncer: syncer["profile_a"].__setitem__("baseline_missing_slots", 1),
        lambda writer, syncer: syncer["profile_a"].__setitem__("claim_order", [0, 0, 1, 2]),
        lambda writer, syncer: syncer["profile_a"]["repeated_selected"].__setitem__(0, "again"),
        lambda writer, syncer: syncer["profile_a"]["selections"][0]["learner_ids"].__setitem__(1, "learner-00"),
        lambda writer, syncer: syncer["grace"].__setitem__("after_selection_identity", "changed"),
        lambda writer, syncer: syncer["grace"]["after_weights"].__setitem__(0, {"changed": True}),
        lambda writer, syncer: writer.__setitem__("gpu_memory_after_bytes", 1),
        lambda writer, syncer: writer.__setitem__("profile_a_published", 31),
        lambda writer, syncer: writer["state_identities"]["profile_a"].__setitem__("run_identity", "wrong"),
    ],
)
def test_analyzer_fails_closed_on_mutated_authoritative_fields(mutation) -> None:
    config, writer, syncer = _valid_results()
    writer = copy.deepcopy(writer)
    syncer = copy.deepcopy(syncer)
    mutation(writer, syncer)
    with pytest.raises(ReadinessStressError):
        analyze_results(writer=writer, syncer=syncer, config=config, run_id=RUN_ID)
