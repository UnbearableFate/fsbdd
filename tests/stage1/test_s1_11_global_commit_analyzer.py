from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest

from fsbdd.global_commit_stress import (
    GlobalCommitStressError,
    analyze_roles,
    build_analyzer_fixture,
    manifest_command,
)
from fsbdd.identity import canonical_digest


CONFIG = Path("configs/stage1/s1_11_global_commit.json")


@pytest.fixture(scope="module")
def analyzer_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("s1-11-analyzer")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_identity = hashlib.sha256(b"s1-11-analyzer-config").hexdigest()
    writer, committer = build_analyzer_fixture(root, config, config_identity)
    return writer, committer, config, config_identity


def test_analyzer_recomputes_atomic_version_consumption_and_reader_contracts(
    analyzer_case,
) -> None:
    writer, committer, config, config_identity = copy.deepcopy(analyzer_case)
    summary = analyze_roles(writer, committer, config, config_identity)
    assert summary["status"] == "pass"
    assert summary["successful_commits"] == 40
    assert summary["final_versions"] == [20, 20]
    assert summary["fault_replays"] == 16
    assert summary["fault_counts"] == {
        "before_payload_write": 4,
        "after_payload_write": 4,
        "before_record_replace": 4,
        "after_record_replace": 4,
    }
    assert summary["reader_threads"] == 8
    assert summary["reader_observations"] >= 800
    assert summary["visibility_records"] == [
        "global-current-000000.json",
        "global-current-000001.json",
    ]


def _mutate_successor_parameter(writer, committer, config):
    committer["commit_traces"][3]["after"]["parameters_sha256"] = "0" * 64


def _mutate_version_jump(writer, committer, config):
    committer["commit_traces"][4]["after"]["version"] += 1


def _mutate_frontier(writer, committer, config):
    trace = committer["commit_traces"][0]
    trace["after"]["frontiers"][0]["last_sequence"] = 99


def _mutate_weight(writer, committer, config):
    trace = committer["commit_traces"][0]
    trace["request"]["selected"][0]["normalized_weight"] = 0.5


def _mutate_fault_classification(writer, committer, config):
    committer["fault_traces"][0]["selected_eligible_after_interruption"] = False


def _mutate_duplicate_retry(writer, committer, config):
    committer["duplicate_retry"]["published"] = True


def _mutate_late_selection(writer, committer, config):
    committer["late_arrival"]["refreshed_selected_in_second"] = False


def _mutate_reader_authority(writer, committer, config):
    writer["observations"][0]["authority_fingerprint"] = "f" * 64


def _mutate_role_host(writer, committer, config):
    writer["hostname"] = committer["hostname"]


def _mutate_writer_torch(writer, committer, config):
    writer["torch_imported_after"] = True


def _mutate_visibility(writer, committer, config):
    committer["global_visibility_records"].append("global-head.json")


def _refingerprint(authority):
    authority["authority_fingerprint"] = canonical_digest(
        {
            key: value
            for key, value in authority.items()
            if key != "authority_fingerprint"
        }
    )


def _mutate_fault_final_with_consistent_fingerprint(writer, committer, config):
    final = committer["fault_traces"][0]["final"]
    final["parameters_sha256"] = hashlib.sha256(b"forged-final").hexdigest()
    _refingerprint(final)


def _mutate_fault_frontier_with_consistent_fingerprint(writer, committer, config):
    final = committer["fault_traces"][1]["final"]
    final["frontiers"][3]["last_base_version"] = -1
    _refingerprint(final)


def _mutate_reader_transition_coverage(writer, committer, config):
    initial = committer["initial_authorities"][0]
    for observation in writer["observations"]:
        if observation["reader_index"] == 0 and observation["fragment_index"] == 0:
            observation["version"] = 0
            observation["authority_fingerprint"] = initial["authority_fingerprint"]


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_successor_parameter,
        _mutate_version_jump,
        _mutate_frontier,
        _mutate_weight,
        _mutate_fault_classification,
        _mutate_duplicate_retry,
        _mutate_late_selection,
        _mutate_reader_authority,
        _mutate_role_host,
        _mutate_writer_torch,
        _mutate_visibility,
        _mutate_fault_final_with_consistent_fingerprint,
        _mutate_fault_frontier_with_consistent_fingerprint,
        _mutate_reader_transition_coverage,
    ],
)
def test_analyzer_rejects_semantic_mutations(analyzer_case, mutation) -> None:
    writer, committer, config, config_identity = copy.deepcopy(analyzer_case)
    mutation(writer, committer, config)
    with pytest.raises((GlobalCommitStressError, KeyError, TypeError)):
        analyze_roles(writer, committer, config, config_identity)


def test_analyzer_rejects_forged_common_config_identity(analyzer_case) -> None:
    writer, committer, config, config_identity = copy.deepcopy(analyzer_case)
    forged = hashlib.sha256(b"forged-config").hexdigest()
    writer["config_identity"] = forged
    committer["config_identity"] = forged
    with pytest.raises(GlobalCommitStressError, match="identity"):
        analyze_roles(writer, committer, config, config_identity)


def test_manifest_command_uses_two_distinct_formal_roles(tmp_path: Path) -> None:
    result_root = tmp_path / "roles"
    result_root.mkdir()
    (result_root / "proposal_writer_reader.json").write_text(
        json.dumps({"hostname": "writer-host"}), encoding="utf-8"
    )
    (result_root / "atomic_committer.json").write_text(
        json.dumps({"hostname": "committer-host"}), encoding="utf-8"
    )
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("committer-host\nwriter-host\n", encoding="utf-8")
    modules = tmp_path / "modules"
    modules.write_text("nv-hpcx/25.9\n", encoding="utf-8")
    output = tmp_path / "manifest.json"
    digest = "a" * 64
    manifest = manifest_command(
        argparse.Namespace(
            result_root=result_root,
            repository="https://example.invalid/fsbdd.git",
            branch="codex/S1-11-global-commit",
            commit="b" * 40,
            run_id="s1-11-123.opbs",
            config_sha256=digest,
            research_sha256="c" * 64,
            spec_sha256="d" * 64,
            skill_repository="https://example.invalid/miyabi-development.git",
            skill_commit="e" * 40,
            initial_hostname="miyabi-g1",
            project_root=tmp_path,
            evidence_root=tmp_path / "evidence",
            job_id="123.opbs",
            qtime_utc="2026-07-15T00:00:00Z",
            queue="debug-g",
            group="xg24i002",
            nodefile=nodefile,
            modules_file=modules,
            output=output,
        )
    )
    assert manifest["loop_id"] == "S1-11"
    roles = manifest["identities"]["roles"]
    assert roles["declared"] == roles["actual"] == {
        "proposal_writer_reader": ["writer-host"],
        "atomic_committer": ["committer-host"],
    }
