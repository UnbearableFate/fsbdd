from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fsbdd.identity import file_digest
from fsbdd.learner_adopt_stress import (
    AdoptionStressError,
    summarize,
    write_manifest,
)


def _state(run_id: str, config_identity: str) -> dict[str, str]:
    return {
        "run_identity": run_id,
        "config_identity": config_identity,
        "model_identity": "b" * 64,
        "fragment_map_identity": "c" * 64,
    }


def test_manifest_rejects_writer_torch_import_and_role_forgery(tmp_path: Path) -> None:
    result_root = tmp_path / "roles"
    result_root.mkdir()
    run_id = "s1-08-123.opbs"
    state = _state(run_id, "a" * 64)
    learner = {
        "status": "pass",
        "role": "mixed_version_learner",
        "rank": 0,
        "hostname": "node-a",
        "state_identities": state,
        "torch": {"distributed_initialized": False},
        "learner_waited_for_writer_or_version_alignment": False,
    }
    writer = {
        "status": "pass",
        "role": "global_fragment_writer",
        "rank": 1,
        "hostname": "node-b",
        "state_identities": state,
        "torch_imported": False,
    }
    (result_root / "mixed_version_learner.json").write_text(json.dumps(learner))
    (result_root / "global_fragment_writer.json").write_text(json.dumps(writer))
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("node-a\nnode-b\n")
    modules = tmp_path / "modules"
    modules.write_text("nv-hpcx/25.9\n")
    args = SimpleNamespace(
        nodefile=str(nodefile),
        modules_file=str(modules),
        result_root=str(result_root),
        repository="https://example.invalid/repository.git",
        branch="codex/S1-08-latest-adoption",
        commit="1" * 40,
        run_id=run_id,
        config_sha256="a" * 64,
        research_sha256="d" * 64,
        spec_sha256="e" * 64,
        skill_repository="https://example.invalid/skill.git",
        skill_commit="2" * 40,
        initial_hostname="miyabi-g1",
        project_root="/work/project",
        evidence_root="/work/evidence",
        job_id="123.opbs",
        qtime_utc="2026-07-15T00:00:00Z",
        queue="debug-g",
        group="xg24i002",
        output=str(tmp_path / "manifest.json"),
    )
    write_manifest(args)
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "building"

    writer["torch_imported"] = True
    (result_root / "global_fragment_writer.json").write_text(json.dumps(writer))
    args.output = str(tmp_path / "forged-manifest.json")
    with pytest.raises(AdoptionStressError, match="state config or run"):
        write_manifest(args)


def _counter(version: int, steps: int) -> dict[str, int]:
    return {
        "global_version": version,
        "local_steps": steps,
        "processed_input_tokens": steps * 32,
        "proposal_sequence": 0,
    }


def _trace(index: int, version: int, payload_sha: str) -> dict[str, object]:
    before_versions = [0, 0, 0, 0]
    after_versions = list(before_versions)
    after_versions[index] = version
    before = [_counter(value, 1) for value in before_versions]
    after = [dict(value) for value in before]
    after[index] = _counter(version, 0)
    parameter_before = [f"{position + 1}" * 64 for position in range(4)]
    parameter_after = list(parameter_before)
    parameter_after[index] = payload_sha
    moments = [f"{position + 5}" * 64 for position in range(4)]
    return {
        "fragment_index": index,
        "content_identity": "d" * 64 if index == 0 else "e" * 64,
        "from_version": 0,
        "to_version": version,
        "skipped_intermediate_versions": max(0, version - 1),
        "safe_boundary_local_step": 1,
        "adoption_unix_ns": 2_000_000_000,
        "extra_local_steps_before_adoption": 0,
        "fs_to_cpu_seconds": 0.001,
        "cpu_to_gpu_seconds": 0.001,
        "target_payload_sha256": payload_sha,
        "parameter_hashes_before": parameter_before,
        "parameter_hashes_after": parameter_after,
        "optimizer_state_hashes_before": moments,
        "optimizer_state_hashes_after": list(moments),
        "optimizer_state_entry_counts_before": [1, 1, 1, 1],
        "optimizer_state_entry_counts_after": [1, 1, 1, 1],
        "counters_before": before,
        "counters_after": after,
        "version_vector_after": after_versions,
    }


def test_analyzer_fails_closed_on_optimizer_moment_rewrite(tmp_path: Path) -> None:
    config = Path("configs/stage1/s1_08_latest_adoption.json").resolve()
    config_identity = file_digest(config)
    run_id = "s1-08-fixture"
    state = _state(run_id, config_identity)
    result_root = tmp_path / "roles"
    result_root.mkdir()
    sha0 = "1" * 64
    sha2 = "2" * 64
    traces = [_trace(0, 3, sha0), _trace(2, 1, sha2)]
    traces[0]["optimizer_state_hashes_after"][0] = "f" * 64  # type: ignore[index]
    events = [
        {
            "local_optimizer_step": step,
            "token_weighted_loss": 4.0 - step / 100,
            "fragment_global_versions": [0, 0, 0, 0]
            if step == 1
            else [3, 0, 1, 0],
        }
        for step in range(1, 21)
    ]
    progress_fragments = [
        {
            "global_version": version,
            "local_steps_since_adoption": 19,
            "processed_input_tokens_since_adoption": 608,
            "proposal_sequence": 0,
        }
        for version in (3, 0, 1, 0)
    ]
    learner = {
        "status": "pass",
        "role": "mixed_version_learner",
        "rank": 0,
        "hostname": "node-a",
        "state_identities": state,
        "torch": {"distributed_initialized": False},
        "training": {
            "optimizer_steps": 20,
            "processed_input_tokens": 640,
            "finite_loss": True,
            "token_weighted_loss": 3.9,
            "distributed_initialized": False,
            "events": events,
            "started_unix_ns": 100,
        },
        "adoption": {
            "adoption_count": 2,
            "skipped_intermediate_versions": 2,
            "progress": {"fragments": progress_fragments},
            "resident_trace_records": 2,
            "resident_trace_record_bound": 4,
            "poller": {
                "maximum_pending_per_fragment": [1, 0, 1, 0],
                "resident_pending_bound": 4,
                "pending_count": 0,
                "errors": [],
                "fixed_slot_read_count": 8,
                "repeated_current_count": 1,
            },
        },
        "adoption_traces": traces,
        "worker_control": {"poller_started": True, "poller_started_before_batch": 1},
        "learner_waited_for_writer_or_version_alignment": False,
        "application_data_plane": "shared_filesystem_only",
    }
    publications = [
        {
            "fragment_index": 0,
            "version": version,
            "content_identity": "d" * 64 if version == 3 else f"{version + 5}" * 64,
            "parameters_sha256": sha0,
            "publication_started_unix_ns": 1_000,
            "publication_completed_unix_ns": 2_000,
        }
        for version in (1, 2, 3)
    ] + [
        {
            "fragment_index": 2,
            "version": 1,
            "content_identity": "e" * 64,
            "parameters_sha256": sha2,
            "publication_started_unix_ns": 1_000,
            "publication_completed_unix_ns": 2_000,
        }
    ]
    writer = {
        "status": "pass",
        "role": "global_fragment_writer",
        "rank": 1,
        "hostname": "node-b",
        "state_identities": state,
        "torch_imported": False,
        "gpu_memory_allocated_bytes": 0,
        "final_version_vector": [3, 0, 1, 0],
        "publications": publications,
        "writer_waited_for_learner_before_publication": False,
        "application_data_plane": "shared_filesystem_only",
    }
    (result_root / "mixed_version_learner.json").write_text(json.dumps(learner))
    (result_root / "global_fragment_writer.json").write_text(json.dumps(writer))
    with pytest.raises(AdoptionStressError, match="parameter or optimizer identity"):
        summarize(
            root=tmp_path / "shared",
            result_root=result_root,
            run_id=run_id,
            config_path=config,
            config_identity=config_identity,
            output=tmp_path / "summary.json",
        )
