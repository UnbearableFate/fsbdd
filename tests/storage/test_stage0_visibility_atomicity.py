"""Contracts for Stage 0 cross-node visibility and atomicity evidence."""

from __future__ import annotations

import copy
import json
import subprocess
import unittest
from pathlib import Path

from fsbdd_stage0.storage_bench import StorageHarnessError, validate_storage_config
from fsbdd_stage0.storage_stress import (
    TIMING_METHOD,
    TIMING_ORIGIN,
    build_atomic_record,
    percentile,
    validate_atomic_record,
    validate_atomicity_results,
    validate_latency_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/stage0/storage_benchmark.json"
PBS_PATH = PROJECT_ROOT / "pbs/stage0_storage_visibility_atomicity.pbs"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def latency_samples(config: dict) -> list[dict]:
    records = []
    sequence = 0
    for profile in config["visibility"]["profiles"]:
        for cache_state in config["visibility"]["cache_states"]:
            for _ in range(config["visibility"]["samples_per_cache_state"]):
                records.append(
                    {
                        "sequence": sequence,
                        "profile": profile,
                        "cache_state": cache_state,
                        "timing_method": TIMING_METHOD,
                        "timing_origin": TIMING_ORIGIN,
                        "writer_ack_roundtrip_ns": 1_000_000,
                        "invalid_record_reads": 0,
                    }
                )
                sequence += 1
    return records


def atomic_results(config: dict) -> tuple[dict, dict]:
    writer_sizes = []
    reader_sizes = []
    for record_size in config["atomicity"]["record_sizes_bytes"]:
        writer_sizes.append(
            {
                "record_size": record_size,
                "replacements": config["atomicity"]["replacements_per_record_size"],
            }
        )
        reader_sizes.append(
            {
                "record_size": record_size,
                "violations": 0,
                "readers": [
                    {
                        "reader_id": reader_id,
                        "observations": config["atomicity"]["minimum_observations_per_reader"],
                        "final_seen": True,
                        "violations": 0,
                        "transcript_sha256": "a" * 64,
                    }
                    for reader_id in range(config["atomicity"]["concurrent_readers"])
                ],
            }
        )
    return {"record_sizes": writer_sizes}, {"record_sizes": reader_sizes, "violations": 0}


class TestVisibilityAtomicityContracts(unittest.TestCase):
    def test_bench_01__frozen_visibility_and_atomicity_matrix(self) -> None:
        config = load_config()
        validate_storage_config(config)
        self.assertEqual(config["visibility"]["profiles"], ["empty", "loaded"])
        self.assertEqual(config["visibility"]["cache_states"], ["cold", "warm"])
        self.assertEqual(config["visibility"]["samples_per_profile"], 1000)
        self.assertEqual(config["atomicity"]["replacements_per_record_size"], 100000)
        self.assertEqual(config["atomicity"]["concurrent_readers"], 4)
        self.assertEqual(config["atomicity"]["record_sizes_bytes"], [256, 4096])

    def test_bench_01__timing_origin_and_loaded_profile_fail_closed(self) -> None:
        config = load_config()
        samples = latency_samples(config)
        summary = validate_latency_samples(samples, config)
        self.assertEqual(summary["clock_skew_handling"], "writer_clock_elapsed_only_no_cross_host_subtraction")
        self.assertIn("max_seconds", summary["profiles"]["loaded"])

        wrong_origin = copy.deepcopy(samples)
        wrong_origin[0]["timing_origin"] = "payload_write_start"
        with self.assertRaisesRegex(StorageHarnessError, "began before"):
            validate_latency_samples(wrong_origin, config)

        missing_loaded = [record for record in samples if record["profile"] == "empty"]
        with self.assertRaisesRegex(StorageHarnessError, "sample count"):
            validate_latency_samples(missing_loaded, config)

    def test_bench_01__percentiles_are_interpolated_and_thresholded(self) -> None:
        self.assertEqual(percentile([0, 10, 20, 30], 50), 15.0)
        self.assertEqual(percentile([0, 10, 20, 30], 100), 30.0)
        config = load_config()
        samples = latency_samples(config)
        for sample in samples[-11:]:
            sample["writer_ack_roundtrip_ns"] = 3_000_000_000
        with self.assertRaisesRegex(StorageHarnessError, "loaded visibility p99"):
            validate_latency_samples(samples, config)

    def test_bench_02__fixed_records_reject_partial_and_corrupt_content(self) -> None:
        for record_size in (256, 4096):
            content = build_atomic_record("run-a", record_size, 7)
            self.assertEqual(len(content), record_size)
            self.assertEqual(validate_atomic_record(content, run_id="run-a", record_size=record_size), 7)
            with self.assertRaisesRegex(StorageHarnessError, "record length"):
                validate_atomic_record(content[: len(content) // 2], run_id="run-a", record_size=record_size)
            corrupt = bytearray(content)
            corrupt[corrupt.index(b"a")] = ord("b")
            with self.assertRaises(StorageHarnessError):
                validate_atomic_record(bytes(corrupt), run_id="run-a", record_size=record_size)

    def test_bench_02__summary_requires_100k_zero_violations_and_four_readers(self) -> None:
        config = load_config()
        writer, reader = atomic_results(config)
        summary = validate_atomicity_results(writer, reader, config)
        self.assertEqual(summary["violations"], 0)
        self.assertEqual(summary["record_sizes"]["256"]["replacements"], 100000)

        too_few = copy.deepcopy(writer)
        too_few["record_sizes"][0]["replacements"] = 99999
        with self.assertRaisesRegex(StorageHarnessError, "too few"):
            validate_atomicity_results(too_few, reader, config)

        torn = copy.deepcopy(reader)
        torn["violations"] = 1
        torn["record_sizes"][0]["readers"][0]["violations"] = 1
        with self.assertRaisesRegex(StorageHarnessError, "torn-record"):
            validate_atomicity_results(writer, torn, config)

        three_readers = copy.deepcopy(reader)
        three_readers["record_sizes"][0]["readers"].pop()
        with self.assertRaisesRegex(StorageHarnessError, "reader count"):
            validate_atomicity_results(writer, three_readers, config)

    def test_bench_02__pbs_is_two_node_compute_only_and_launcher_only(self) -> None:
        subprocess.run(["bash", "-n", str(PBS_PATH)], check=True)
        script = PBS_PATH.read_text(encoding="utf-8")
        self.assertIn("#PBS -l select=2", script)
        self.assertIn("#PBS -l walltime=00:30:00", script)
        self.assertIn("--map-by ppr:1:node", script)
        self.assertIn("application_coordination: filesystem_only", script)
        self.assertIn("MIYABI_SKILL_COMMIT", script)
        self.assertIn("FSBDD_STRESS_MODE", script)
        self.assertNotIn("BENCHMARK_MODE", script)
        self.assertNotIn("mpirun -x", script)


if __name__ == "__main__":
    unittest.main()
