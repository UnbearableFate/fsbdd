from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from fsbdd.identity import canonical_digest
from fsbdd.syncer_merge_stress import (
    MergeStressError,
    analyze_roles,
    build_workload,
    execute_mixed_base_workload,
    execute_numeric_workload,
    execute_order_workload,
)


CONFIG = Path("configs/stage1/s1_10_streaming_merge.json")


@pytest.fixture(scope="module")
def analyzer_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("s1-10-analyzer")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["formal_workload"]["fragment_elements"] = 64
    workload = build_workload(root, config)
    numeric = execute_numeric_workload(root, workload, config)
    order = execute_order_workload(root, workload, config)
    mixed = execute_mixed_base_workload(root, workload, config)
    fragment_bytes = workload["memory"]["fragment_bytes"]

    def memory_run(count: int):
        return {
            "contributor_count": count,
            "baseline_rss_bytes": 100_000_000,
            "baseline_hwm_bytes": 100_000_000,
            "after_rss_bytes": 100_000_000 + 4 * fragment_bytes,
            "after_hwm_bytes": 100_000_000 + 4 * fragment_bytes,
            "peak_over_baseline_rss_bytes": 4 * fragment_bytes,
            "source_metrics": {
                "opens": count,
                "bytes_read": count * fragment_bytes,
                "active_payloads": 0,
                "maximum_active_payloads": 1,
            },
            "memory_accounting": {
                "maximum_active_local_payloads": 1,
                "maximum_active_base_payloads": 0,
                "maximum_live_tensor_bytes": 3 * fragment_bytes,
                "fragment_bytes": fragment_bytes,
                "maximum_tensor_fragment_multiples": 3.0,
            },
            "byte_accounting": {
                "fragment_index": 0,
                "fragment_bytes": fragment_bytes,
                "current_input_bytes": fragment_bytes,
                "local_payload_reads": count,
                "local_payload_bytes": count * fragment_bytes,
                "retained_base_reads": 0,
                "retained_base_bytes": 0,
                "successor_output_bytes": fragment_bytes,
                "full_model_operations": 0,
            },
            "parameters_sha256": hashlib.sha256(f"memory-{count}".encode()).hexdigest(),
        }

    memory = {
        "runs": {"1": memory_run(1), "8": memory_run(8)},
        "m8_minus_m1_peak_rss_bytes": 0,
        "rss_gate_bytes": config["memory_gates"]["maximum_m8_minus_m1_peak_rss_bytes"],
    }
    config_identity = hashlib.sha256(b"analyzer-config").hexdigest()
    common = {
        "schema_version": 1,
        "complete": True,
        "size": 2,
        "run_id": "s1-10-analyzer-fixture",
        "config_identity": config_identity,
        "workload_sha256": hashlib.sha256(b"workload-file").hexdigest(),
        "workload_identity": workload["workload_identity"],
        "gpu_memory_before": 0,
        "gpu_memory_after": 0,
    }
    writer = {
        **common,
        "role": "proposal_writer",
        "rank": 1,
        "hostname": "writer-host",
        "pid": 101,
        "workload": workload,
        "torch_imported_before": False,
        "torch_imported_after": False,
    }
    syncer = {
        **common,
        "role": "outer_syncer",
        "rank": 0,
        "hostname": "syncer-host",
        "pid": 100,
        "numeric": numeric,
        "order": order,
        "mixed_base": mixed,
        "memory": memory,
        "application_coordination": "filesystem_only",
        "mpi_usage": "launcher_only",
        "torch_imported_before": False,
        "torch_imported_after": True,
    }
    return writer, syncer, config


def test_analyzer_independently_recomputes_all_formal_transitions(analyzer_case) -> None:
    writer, syncer, config = analyzer_case
    summary = analyze_roles(copy.deepcopy(writer), copy.deepcopy(syncer), copy.deepcopy(config))
    assert summary["status"] == "pass"
    assert summary["numeric"]["updates"] == 50
    assert summary["numeric"]["distinct_update_identities"] == 50
    assert summary["order"]["permutation_count"] == 16
    assert summary["byte_accounting"]["maximum_live_local_payloads"] == 1


def _mutate_missing_update_fact(writer, syncer, config):
    trace = syncer["numeric"]["traces"][0]
    del trace["update_facts"]["ordered_processed_tokens"]
    trace["update_identity"] = canonical_digest(trace["update_facts"])


def _mutate_numeric_output(writer, syncer, config):
    syncer["numeric"]["traces"][12]["production_parameters"][3] += 0.1


def _mutate_weight_correspondence(writer, syncer, config):
    trace = syncer["numeric"]["traces"][0]
    trace["update_facts"]["ordered_float32_weights"][0] = 0.5
    trace["update_identity"] = canonical_digest(trace["update_facts"])


def _mutate_outer_hyperparameters(writer, syncer, config):
    trace = syncer["numeric"]["traces"][0]
    trace["update_facts"]["outer_hyperparameters"]["momentum"] = 0.5
    trace["update_identity"] = canonical_digest(trace["update_facts"])


def _mutate_order_output(writer, syncer, config):
    syncer["order"]["outputs"][3][0] += 0.1


def _mutate_cache_all(writer, syncer, config):
    syncer["memory"]["runs"]["8"]["source_metrics"]["maximum_active_payloads"] = 8


def _mutate_rss(writer, syncer, config):
    gate = config["memory_gates"]["maximum_m8_minus_m1_peak_rss_bytes"]
    syncer["memory"]["runs"]["8"]["peak_over_baseline_rss_bytes"] = gate + 1
    syncer["memory"]["m8_minus_m1_peak_rss_bytes"] = gate + 1


def _mutate_wrong_base_output(writer, syncer, config):
    syncer["mixed_base"]["production_parameters"] = syncer["mixed_base"][
        "wrong_current_relative_parameters"
    ]


def _mutate_writer_torch(writer, syncer, config):
    writer["torch_imported_after"] = True


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_missing_update_fact,
        _mutate_numeric_output,
        _mutate_weight_correspondence,
        _mutate_outer_hyperparameters,
        _mutate_order_output,
        _mutate_cache_all,
        _mutate_rss,
        _mutate_wrong_base_output,
        _mutate_writer_torch,
    ],
)
def test_analyzer_rejects_semantic_mutations(analyzer_case, mutation) -> None:
    writer, syncer, config = copy.deepcopy(analyzer_case)
    mutation(writer, syncer, config)
    with pytest.raises((MergeStressError, KeyError)):
        analyze_roles(writer, syncer, config)
