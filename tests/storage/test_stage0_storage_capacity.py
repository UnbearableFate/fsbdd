"""Contracts for the S0B-03 metadata and bandwidth benchmark."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from fsbdd_stage0.storage_bench import atomic_write_json, validate_storage_config
from fsbdd_stage0.storage_capacity import (
    StorageHarnessError,
    _bandwidth_summary,
    _history_summary,
    _metadata_phase_id,
    _metadata_summary,
    _phase_id,
    bandwidth_matrix,
    deterministic_chunk,
    metadata_matrix,
    summarize_capacity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/stage0/storage_benchmark.json"
PBS_PATH = PROJECT_ROOT / "pbs/stage0_storage_capacity.pbs"


def _config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_storage_config(config)
    return config


def _synthetic_roles(config: dict, run_id: str = "capacity-test") -> list[dict]:
    roles = [
        {
            "schema_version": 1,
            "run_id": run_id,
            "rank": rank,
            "world_size": 16,
            "hostname": "mg0001" if rank < 8 else "mg0002",
            "pbs_job_id": "1.opbs",
            "status": "passed",
            "metadata": [],
            "metadata_coordinator_rounds": [],
            "history_controls": [],
            "bandwidth": [],
            "coordinator_rounds": [],
            "state": None,
        }
        for rank in range(16)
    ]
    for learners, fragments, layout, repeat, profile in metadata_matrix(config):
        active_ranks = [0] if profile == "readdir" and layout == "flat" else list(range(learners))
        for state in config["metadata"]["cache_states"]:
            phase = _metadata_phase_id(
                learners=learners,
                fragments=fragments,
                layout=layout,
                repeat=repeat,
                profile=profile,
                state=state,
            )
            multiplier = 20 if state == "warm" else 1
            for rank in active_ranks:
                references = (
                    learners * fragments
                    if profile == "readdir" and layout == "flat"
                    else fragments
                ) * multiplier
                syscalls = multiplier if profile == "readdir" else references
                roles[rank]["metadata"].append(
                    {
                        "learners": learners,
                        "fragments": fragments,
                        "layout": layout,
                        "repeat": repeat,
                        "profile": profile,
                        "state": state,
                        "rank": rank,
                        "reference_operations": references,
                        "syscall_operations": syscalls,
                        "elapsed_ns": 1_000_000_000,
                        "latency_samples_ns": [1_000, 2_000, 3_000],
                        "latency_operations_seen": syscalls,
                        "coordinator_phase": phase,
                        "measurement_interval": "filesystem_barrier_coordinator_round",
                    }
                )
            roles[0]["metadata_coordinator_rounds"].append(
                {
                    "phase": phase,
                    "learners": learners,
                    "fragments": fragments,
                    "layout": layout,
                    "repeat": repeat,
                    "profile": profile,
                    "state": state,
                    "round_elapsed_ns": 1_000_000_000,
                    "ready_ranks": list(range(16)),
                    "done_ranks": list(range(16)),
                    "timing_method": "rank0_monotonic_start_to_all_done_after_fs_barrier",
                }
            )
    roles[0]["history_controls"] = [
        {
            "history_objects": size,
            "history_scan_counts": [size] * 20,
            "history_scan_latencies_ns": [100 * (index + 1)] * 20,
            "bounded_fixed_references": 512,
            "bounded_elapsed_ns": 1_000_000,
        }
        for index, size in enumerate((0, 1000, 10000))
    ]
    for size_mb, streams, repeat in bandwidth_matrix(config):
        phase = _phase_id(size_mb, streams, repeat)
        size_bytes = size_mb * 1024 * 1024
        for rank in range(streams):
            digest = f"{rank:064x}"[-64:]
            roles[rank]["bandwidth"].append(
                {
                    "phase": phase,
                    "operation": "write",
                    "size_mb": size_mb,
                    "streams": streams,
                    "repeat": repeat,
                    "rank": rank,
                    "stream": rank,
                    "source_writer_rank": rank,
                    "bytes": size_bytes,
                    "sha256": digest,
                    "elapsed_ns": 1_000_000_000,
                }
            )
        readers = [8] if streams == 1 else list(range(streams))
        for rank in readers:
            source = 0 if streams == 1 else (rank + 8) % 16
            digest = f"{source:064x}"[-64:]
            roles[rank]["bandwidth"].append(
                {
                    "phase": phase,
                    "operation": "read",
                    "size_mb": size_mb,
                    "streams": streams,
                    "repeat": repeat,
                    "rank": rank,
                    "stream": rank if streams > 1 else 0,
                    "source_writer_rank": source,
                    "bytes": size_bytes,
                    "sha256": digest,
                    "elapsed_ns": 1_000_000_000,
                }
            )
        roles[0]["coordinator_rounds"].extend(
            [
                {"phase": phase, "operation": "write", "round_elapsed_ns": 1_000_000_000},
                {"phase": phase, "operation": "read", "round_elapsed_ns": 1_000_000_000},
            ]
        )
    roles[0]["state"] = {
        "fixed_records": 1,
        "history_records": 1,
        "fixed_records_before": 1,
        "fixed_records_after": 1,
        "history_records_before": 1,
        "history_records_after": 1,
        "payload_files_after": 0,
        "temporary_files_after": 0,
        "control_files_after": 35,
        "control_file_limit": 35,
    }
    return roles


class TestStage0StorageCapacity(unittest.TestCase):
    def test_bench_03__config_and_metadata_matrix_are_complete(self) -> None:
        config = _config()
        matrix = metadata_matrix(config)
        self.assertEqual(len(matrix), 3 * 3 * 2 * 2 * 3)
        self.assertIn((16, 32, "per_learner", 1, "readdir"), matrix)
        self.assertEqual(config["thresholds"]["metadata_required_ops_per_second"], 5120.0)

    def test_bench_04__bandwidth_matrix_is_complete(self) -> None:
        config = _config()
        matrix = bandwidth_matrix(config)
        self.assertEqual(len(matrix), 4 * 2 * 2)
        self.assertIn((500, 16, 1), matrix)
        self.assertEqual(deterministic_chunk(64, identity="x"), deterministic_chunk(64, identity="x"))
        self.assertNotEqual(deterministic_chunk(64, identity="x"), deterministic_chunk(64, identity="y"))

    def test_bench_04__launcher_is_two_node_sixteen_rank_and_compute_only(self) -> None:
        subprocess.run(["bash", "-n", str(PBS_PATH)], check=True)
        script = PBS_PATH.read_text(encoding="utf-8")
        self.assertIn("#PBS -l select=2", script)
        self.assertIn("-np 16 --map-by ppr:8:node", script)
        self.assertIn("^mg[0-9]+$", script)
        self.assertIn("application_coordination: filesystem_only_fixed_slots", script)

    def test_bench_03__summary_requires_ten_x_margin_and_complete_matrix(self) -> None:
        config = _config()
        roles = _synthetic_roles(config)
        summary = _metadata_summary(roles, config)
        self.assertTrue(summary["threshold"]["passed"])
        self.assertGreaterEqual(
            summary["threshold"]["measured_minimum_warm_reference_ops_per_second"],
            5120.0,
        )
        broken = copy.deepcopy(roles)
        broken[0]["metadata"].pop()
        with self.assertRaisesRegex(StorageHarnessError, "metadata (matrix|concurrency)"):
            _metadata_summary(broken, config)

        missing_interval = copy.deepcopy(roles)
        missing_interval[0]["metadata_coordinator_rounds"].pop()
        with self.assertRaisesRegex(StorageHarnessError, "coordinator interval"):
            _metadata_summary(missing_interval, config)

    def test_bench_03__disjoint_rank_local_intervals_cannot_prove_capacity(self) -> None:
        config = _config()
        roles = _synthetic_roles(config)
        for interval in roles[0]["metadata_coordinator_rounds"]:
            if (
                interval["learners"] == 16
                and interval["fragments"] == 32
                and interval["state"] == "warm"
            ):
                interval["round_elapsed_ns"] = 16_000_000_000
        summary = _metadata_summary(roles, config)
        self.assertFalse(summary["threshold"]["passed"])
        self.assertEqual(
            summary["threshold"]["measured_minimum_warm_reference_ops_per_second"],
            640.0,
        )

    def test_fs_05__history_scan_grows_but_fixed_discovery_stays_bounded(self) -> None:
        config = _config()
        summary = _history_summary(_synthetic_roles(config), config)
        self.assertTrue(summary["negative_history_scanner_cost_grows"])
        self.assertFalse(summary["normal_discovery_scans_history"])
        self.assertEqual(
            {profile["bounded_fixed_references"] for profile in summary["profiles"]},
            {512},
        )

    def test_bench_04__summary_verifies_digests_and_round_threshold(self) -> None:
        config = _config()
        roles = _synthetic_roles(config)
        summary = _bandwidth_summary(roles, config)
        self.assertEqual(len(summary["matrix"]), 32)
        self.assertTrue(summary["threshold"]["passed"])
        broken = copy.deepcopy(roles)
        next(item for item in broken[8]["bandwidth"] if item["operation"] == "read")[
            "sha256"
        ] = "f" * 64
        with self.assertRaisesRegex(StorageHarnessError, "checksum mismatch"):
            _bandwidth_summary(broken, config)

    def test_fs_06__full_summary_requires_bounded_state_and_two_host_mapping(self) -> None:
        config = _config()
        roles = _synthetic_roles(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for role in roles:
                atomic_write_json(root / f"capacity-rank-{role['rank']}.json", role)
            summary = summarize_capacity(config=config, run_id="capacity-test", result_root=root)
            self.assertEqual(summary["outcome"], "target_lustre_capacity_pass")
            self.assertEqual(summary["decision"]["A-BENCH-03"], "PASS")
            self.assertEqual(summary["decision"]["A-BENCH-04"], "PASS")
            self.assertEqual(summary["topology"]["hosts"], {"mg0001": 8, "mg0002": 8})


if __name__ == "__main__":
    unittest.main()
