from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_closure_runtime_has_no_application_network_data_plane() -> None:
    source = (ROOT / "src/fsbdd/auxiliary/stage1/close.py").read_text(encoding="utf-8")
    for token in (
        "init_process_group(",
        "DistributedDataParallel(",
        "init_rpc(",
        "torchrun",
        "nccl",
    ):
        assert token not in source
    assert '"network_data_plane": False' in source
    assert '"application_coordination": "shared_filesystem_only"' in source
    assert (
        'minimum_step_seconds=float(execution_contract["minimum_step_seconds"])'
        in source
    )
    assert 'merge_backend="numpy"' in source


def test_formal_topologies_freeze_independent_9n_and_coallocated_4_plus_1() -> None:
    nine = (ROOT / "pbs/stage1_s1_13_formal_9n.pbs").read_text(encoding="utf-8")
    long = (ROOT / "pbs/stage1_s1_13_formal_long.pbs").read_text(encoding="utf-8")
    assert "#PBS -J 0-8" in nine
    assert "#PBS -l select=1" in nine
    assert "PBS_ARRAY_INDEX" in nine
    assert "mpirun" not in nine
    assert "fsbdd.auxiliary.stage1.submit validate-nine" in nine
    assert "SUBMISSION_MARKER_SHA256" in nine
    assert "#PBS -l select=5" in long
    assert "mpirun -np 5 --map-by ppr:1:node" in long
    assert "CUDA_VISIBLE_DEVICES" in long or "fsbdd.auxiliary.stage1.close mpi" in long
    assert (ROOT / "pbs/stage1_s1_13_numpy_correction.pbs").is_file()
    smoke = (ROOT / "pbs/stage1_s1_13_long_smoke.pbs").read_text(encoding="utf-8")
    assert "#PBS -l select=5" in smoke
    assert "--non-formal-long-smoke" in smoke
    assert "fsbdd.auxiliary.stage1.gate" in smoke
    targeted = (ROOT / "pbs/stage1_s1_13_targeted.pbs").read_text(encoding="utf-8")
    assert "EXPECTED_COMMIT" in targeted
    assert 'if [[ -e "$OUTPUT_ROOT" ]]' in targeted
    assert "compileall -q src tests" in targeted
    assert "uvx --offline ruff check src/fsbdd tests" in targeted
    assert 'uvx --offline pyright --pythonpath "$PYTHON" src/fsbdd' in targeted
    assert 'bash -n "$script"' in targeted
    focus = (ROOT / "pbs/stage1_s1_13_targeted_focus.pbs").read_text(
        encoding="utf-8"
    )
    assert "#PBS -l select=1" in focus
    assert "#PBS -l walltime=00:10:00" in focus
    assert "EXPECTED_COMMIT" in focus
    assert 'if [[ -e "$OUTPUT_ROOT" ]]' in focus
    assert "timeout --signal=TERM --kill-after=30s 420s" in focus
    assert "test_analyzer_accepts_complete_frozen_fixture" in focus
    assert "test_analyzer_fails_closed_on_mutated_authoritative_fields" in focus
    assert "test_concurrent_readers_observe_only_complete_committed_authorities" in focus


def test_syncer_keeps_update_history_only_in_durable_jsonl() -> None:
    close_source = (ROOT / "src/fsbdd/auxiliary/stage1/close.py").read_text(
        encoding="utf-8"
    )
    gate_source = (ROOT / "src/fsbdd/auxiliary/stage1/gate.py").read_text(
        encoding="utf-8"
    )
    assert "updates.append(" not in close_source
    assert "inventories.append(" not in close_source
    assert '"retained_in_memory": 0' in close_source
    assert 'logger.emit("fragment_outer_update"' in close_source
    assert 'logger.emit("bounded_storage_inventory"' in close_source
    assert "_load_syncer_streams(result_root, syncer)" in gate_source


def test_two_node_correction_reproduction_is_identity_bound_and_bounded() -> None:
    script = (ROOT / "pbs/stage1_s1_13_smoke_2n.pbs").read_text(
        encoding="utf-8"
    )
    for token in (
        "EXPECTED_COMMIT",
        "REPRODUCTION_CONTRACT_SHA256",
        'if ! mkdir "$SHARED_ROOT"',
        'if ! mkdir "$RESULT_ROOT"',
        'FSBDD_FAILURE_OUTPUT_ROOT="$RESULT_ROOT"',
        "timeout --signal=TERM --kill-after=30s 900s",
        "mpirun -np 2 --map-by ppr:1:node --bind-to none --report-bindings",
        "bind-policy none cpus_allowed_list",
        '"$RESULT_ROOT/env/mpi-report-bindings.txt"',
        "mpirun -np 2 --map-by ppr:1:node --bind-to none",
        "--learner-count-override 1 --timeout-seconds 720",
        "fsbdd.auxiliary.stage1.reproduction analyze",
        "fsbdd.auxiliary.stage1.reproduction validate",
        "sha256sum -c checksums.sha256",
    ):
        assert token in script
    assert script.index('if ! mkdir "$RESULT_ROOT"') < script.index(
        'FSBDD_FAILURE_OUTPUT_ROOT="$RESULT_ROOT"'
    )
    assert script.index("fsbdd.auxiliary.stage1.reproduction analyze") < script.index(
        "xargs -0 sha256sum > checksums.sha256"
    ) < script.index("fsbdd.auxiliary.stage1.reproduction validate") < script.index(
        "sha256sum -c checksums.sha256"
    )
    contract = json.loads(
        (ROOT / "reports/stage1/S1-13-reproduction-contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["kind"] == "corrected_two_node_filesystem_reproduction"
    assert contract["topology"]["distinct_compute_hosts"] == 2
    assert contract["runtime"]["target_global_cycles"] == 10
    assert contract["runtime"]["expected_fragment_updates"] == 40
    assert contract["runtime"]["supervisor_term_seconds"] == 900
    assert "env/binding-hostnames.txt" in contract["required_evidence"]
    assert "env/mpi-bindings.txt" in contract["required_evidence"]
    assert "env/mpi-report-bindings.txt" in contract["required_evidence"]
    assert (
        contract["topology"]["mpi_binding_policy"]
        == "none_with_report_bindings_and_rank_affinity_evidence"
    )
    assert contract["finalization"]["validator_runs_after_preliminary_checksums"]
    assert contract["finalization"][
        "validator_appended_before_final_checksum_check"
    ]
    assertions = contract["correction_assertions"]
    assert assertions["retained_current_base_bytes"] == 0
    assert assertions["proposal_identity_materialization"] == "background_publisher"
    assert assertions["proposal_payload_cache_misses_equal_published_proposals"]


def test_reproduction_preflight_is_single_node_identity_bound_and_focused() -> None:
    script = (ROOT / "pbs/stage1_s1_13_reproduction_preflight.pbs").read_text(
        encoding="utf-8"
    )
    for token in (
        "#PBS -l select=1",
        "#PBS -l walltime=00:10:00",
        "EXPECTED_COMMIT",
        "REPRODUCTION_CONTRACT_SHA256",
        'if [[ -e "$OUTPUT_ROOT" ]]',
        'FSBDD_FAILURE_OUTPUT_ROOT="$OUTPUT_ROOT"',
        "timeout --signal=TERM --kill-after=30s 420s",
        "tests/stage1/test_s1_13_reproduction.py",
        "test_two_node_correction_reproduction_is_identity_bound_and_bounded",
        "fsbdd.auxiliary.stage1.reproduction preflight",
        "--expected-reproduction-contract-sha256",
        "sha256sum -c checksums.sha256",
    ):
        assert token in script


def test_asset_rebuild_is_identity_bound_offline_and_reuses_frozen_datasets() -> (
    None
):
    script = (ROOT / "pbs/stage1_s1_13_assets.pbs").read_text(encoding="utf-8")
    for token in (
        "EXPECTED_COMMIT",
        "OLD_RESOLVED_CONFIG",
        "OLD_CONFIG_SHA256",
        "OLD_ASSET_ROOT",
        "OLD_ASSET_MARKER_SHA256",
        '[[ -n "$(git status --porcelain)" ]]',
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1",
        "validate-compatibility",
        '--project-root "$PROJECT_ROOT"',
        '--producer-commit "$EXPECTED_COMMIT"',
        '--reuse-gpt-train "$OLD_ASSET_ROOT/datasets/gpt2-train"',
        '--reuse-gpt-validation "$OLD_ASSET_ROOT/datasets/gpt2-validation"',
        '--reuse-pythia-smoke "$OLD_ASSET_ROOT/datasets/pythia-smoke"',
        '--reuse-pythia-long "$OLD_ASSET_ROOT/datasets/pythia-long"',
        "new-${WORKLOAD}-compatibility.json",
        "asset-authorities.sha256",
        "checksums.sha256",
    ):
        assert token in script
    assert "hf download" not in script


def test_formal_package_retains_current_protocol_samples_and_rejects_placeholders() -> (
    None
):
    source = (ROOT / "src/fsbdd/auxiliary/stage1/package.py").read_text(
        encoding="utf-8"
    )
    assert "_reject_placeholders(resolved_config)" in source
    assert "_capture_current_protocol_samples(" in source
    assert "arguments.shared_root, evidence" in source
    assert "raw-metadata/classified-current-inventory.json" in source
    assert "nine-node package requires a submission marker" in source
    assert '"gate_contract_sha256"' in source
    assert '"reports/stage1/S1-13-experiment-failures.json"' in source


def test_unresolved_contract_freezes_both_non_substitutable_workloads() -> None:
    config = json.loads(
        (ROOT / "configs/stage1/s1_13_stage1_close.json").read_text(encoding="utf-8")
    )
    nine = config["workloads"]["nine_node"]
    long = config["workloads"]["long_run"]
    assert (
        nine["learner_count"],
        nine["syncer_count"],
        nine["target_global_cycles"],
    ) == (8, 1, 10)
    assert nine["independent_pbs_roles"] is True
    assert (long["learner_count"], long["syncer_count"]) == (4, 1)
    assert long["minimum_aggregate_processed_input_tokens"] >= 1_000_000_000
    assert long["expected_aggregate_processed_input_tokens"] == 1_000_005_632
    assert long["minimum_global_cycles"] == 2400
    assert config["protocol"]["H"] == 50
    assert config["protocol"]["torch_distributed"] is False


def test_frozen_gate_contract_covers_heartbeat_stalls_and_long_capacity_smoke() -> None:
    contract = json.loads(
        (ROOT / "reports/stage1/S1-13-gate-contract.json").read_text(encoding="utf-8")
    )
    runtime = contract["runtime"]
    smoke = contract["long_run"]["preflight_smoke"]
    assert runtime["maximum_unexpected_heartbeat_gap_multiple"] == 2.0
    assert (
        runtime["pending_stall_rejection"]["learner_publication_pending_upload_count"]
        == 0
    )
    assert smoke["optimizer_steps_per_learner"] == 3000
    assert smoke["publication_opportunities_per_fragment_per_learner"] == 60
    assert smoke["minimum_global_cycles"] == 59
    assert smoke["minimum_cycle_to_publication_opportunity_ratio"] >= 59 / 60
    assert contract["long_run"]["minimum_step_seconds"] == 0.16
    overlay = contract["protocol"]["long_run_learner_phase_overlay"]
    assert overlay["algorithm"] == "aligned_zero_for_q_equals_m_capacity_v1"
    assert overlay["learner_phase_offsets"] == [0, 0, 0, 0]
    assert overlay["learner_waits_for_syncer"] is False


def test_resolved_configs_bind_the_complete_immutable_asset_bundle() -> None:
    expected_marker = "6d13e621bb912495ba9e983a305991bf716f9307d24fe241d34a39e5f0b62c7f"
    asset_root = "/work/xg24i002/x10041/fsbdd/runtime_runs/S1-13/assets-2389469.opbs"
    for workload, learners in (("nine_node", 8), ("long_run", 4)):
        path = ROOT / "configs" / "stage1" / f"s1_13_{workload}_resolved.json"
        text = path.read_text(encoding="utf-8")
        config = json.loads(text)
        resolved = config["resolved_runtime_fields"]
        assert "asset_stage" not in text
        assert config["selected_workload"] == workload
        assert resolved["asset_bundle_root"] == asset_root
        assert resolved["asset_marker_sha256"] == expected_marker
        assert len(resolved["fragment_descriptors"]) == 4
        assert len(resolved["fragment_bytes"]) == 4
        assert len(resolved["per_learner_offsets"]) == learners
        assert all(len(offsets) == 4 for offsets in resolved["per_learner_offsets"])
    long = json.loads(
        (ROOT / "configs/stage1/s1_13_long_run_resolved.json").read_text(
            encoding="utf-8"
        )
    )["workloads"]["long_run"]
    assert long["optimizer_steps_per_learner"] == 122071
    assert long["expected_aggregate_processed_input_tokens"] == 1_000_005_632
    resolved = json.loads(
        (ROOT / "configs/stage1/s1_13_long_run_resolved.json").read_text(
            encoding="utf-8"
        )
    )["resolved_runtime_fields"]
    assert resolved["per_learner_offsets"] == [resolved["fragment_offsets"]] * 4
