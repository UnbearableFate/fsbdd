"""Unit contracts for the compute-only Stage 0 storage harness."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from fsbdd.auxiliary.stage0.storage_bench import (
    SKILL_COMMIT,
    StorageHarnessError,
    atomic_write_json,
    deterministic_payload,
    read_json_record,
    require_compute_context,
    summarize_smoke,
    validate_environment_manifest,
    validate_storage_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/stage0/storage_benchmark.json"
PBS_PATH = PROJECT_ROOT / "pbs/stage0_storage_smoke.pbs"


class TestStorageHarnessContracts(unittest.TestCase):
    def test_bench_01__resolved_config_is_compute_only_and_frozen(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        validate_storage_config(config)
        self.assertEqual(config["miyabi"]["compute_nodes"], 2)
        self.assertEqual(
            config["miyabi"]["target_shared_run_root"],
            str(PROJECT_ROOT / "runtime_runs"),
        )
        self.assertEqual(config["miyabi"]["required_skill_commit"], SKILL_COMMIT)
        self.assertEqual(config["thresholds"]["visibility_p99_seconds"], 2.5)
        self.assertEqual(
            config["thresholds"]["metadata_required_ops_per_second"], 5120.0
        )

    def test_bench_01__login_host_fails_closed(self) -> None:
        environment = {"PBS_JOBID": "123.miyabi", "PBS_NODEFILE": "/tmp/nodes"}
        with self.assertRaisesRegex(StorageHarnessError, "outside a confirmed"):
            require_compute_context(hostname="miyabi-g1", environ=environment)
        with self.assertRaisesRegex(StorageHarnessError, "outside a confirmed"):
            require_compute_context(hostname="mg0123", environ={})
        self.assertEqual(
            require_compute_context(hostname="mg0123", environ=environment),
            ("mg0123", "123.miyabi", "/tmp/nodes"),
        )

    def test_bench_02__payload_is_fixed_and_reproducible(self) -> None:
        first = deterministic_payload(4096, seed="run-a")
        second = deterministic_payload(4096, seed="run-a")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4096)
        self.assertNotEqual(first, deterministic_payload(4096, seed="run-b"))
        with self.assertRaises(StorageHarnessError):
            deterministic_payload(0)

    def test_bench_03__atomic_json_reader_rejects_invalid_or_incomplete_records(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text('{"run_id":', encoding="utf-8")
            with self.assertRaisesRegex(StorageHarnessError, "invalid JSON"):
                read_json_record(path)
            atomic_write_json(path, {"run_id": "run-a"})
            with self.assertRaisesRegex(StorageHarnessError, "missing"):
                read_json_record(path, required_fields=("run_id", "payload_sha256"))
            self.assertEqual(
                read_json_record(path, required_fields=("run_id",))["run_id"], "run-a"
            )

    def test_bench_04__two_node_pbs_launcher_is_static_safe(self) -> None:
        subprocess.run(["bash", "-n", str(PBS_PATH)], check=True)
        script = PBS_PATH.read_text(encoding="utf-8")
        self.assertIn("#PBS -l select=2", script)
        self.assertIn("^mg[0-9]+$", script)
        self.assertIn("--map-by ppr:1:node", script)
        self.assertIn("application_coordination: filesystem_only", script)
        self.assertNotIn("-x PYTHONPATH", script)

    def test_fs_08__environment_manifest_requires_compute_and_fs_identity(self) -> None:
        complete = {
            "schema_version": 1,
            "skill_repository": "https://github.com/UnbearableFate/miyabi-development",
            "skill_commit": SKILL_COMMIT,
            "initial_hostname": "miyabi-g1",
            "compute_hostname": "mg0001",
            "compute_hosts": ["mg0001", "mg0002"],
            "node_type": "cpu_compute",
            "pbs_job_id": "123.miyabi",
            "pbs_nodefile": ["mg0001", "mg0002"],
            "pbs_queue": "debug-g",
            "pbs_group": "xg24i002",
            "target_run_root": str(PROJECT_ROOT / "runtime_runs"),
            "filesystem_type": "lustre",
            "filesystem_source": "example:/lustre/work",
            "module_list": ["nv-hpcx/25.9"],
            "loop_id": "S0B-01",
            "code_commit": "1" * 40,
            "submission_nonce": "n1a2b",
            "config_digest": "2" * 64,
            "config_paths": ["configs/stage0/storage_benchmark.json"],
            "run_id": f"run-n1a2b-{'2' * 12}",
            "actual_role_mapping": [
                {
                    "rank": 0,
                    "role": "writer",
                    "hostname": "mg0001",
                    "pbs_job_id": "123.miyabi",
                    "status": "passed",
                    "test_root": "/work/run",
                },
                {
                    "rank": 1,
                    "role": "reader",
                    "hostname": "mg0002",
                    "pbs_job_id": "123.miyabi",
                    "status": "passed",
                    "test_root": "/work/run",
                },
            ],
        }
        validate_environment_manifest(complete)
        for field in tuple(complete):
            incomplete = dict(complete)
            del incomplete[field]
            with self.subTest(field=field), self.assertRaises(StorageHarnessError):
                validate_environment_manifest(incomplete)

    def test_stor_01__summary_requires_distinct_hosts_and_expected_visibility(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "schema_version": 1,
                "run_id": "run-a",
                "world_size": 2,
                "path_mode": "shared",
                "status": "passed",
                "payload_bytes": 4096,
                "payload_sha256": "a" * 64,
            }
            atomic_write_json(
                root / "smoke-rank-0.json",
                {
                    **common,
                    "rank": 0,
                    "role": "writer",
                    "hostname": "mg0001",
                    "reader_host": "mg0001",
                },
            )
            atomic_write_json(
                root / "smoke-rank-1.json",
                {
                    **common,
                    "rank": 1,
                    "role": "reader",
                    "hostname": "mg0001",
                    "writer_host": "mg0001",
                },
            )
            with self.assertRaisesRegex(StorageHarnessError, "distinct hosts"):
                summarize_smoke(
                    result_root=root,
                    run_id="run-a",
                    path_mode="shared",
                    expected="pass",
                    required_distinct_hosts=2,
                )
            atomic_write_json(
                root / "smoke-rank-0.json",
                {
                    **common,
                    "rank": 0,
                    "role": "writer",
                    "hostname": "mg0001",
                    "reader_host": "mg0002",
                },
            )
            atomic_write_json(
                root / "smoke-rank-1.json",
                {
                    **common,
                    "rank": 1,
                    "role": "reader",
                    "hostname": "mg0002",
                    "writer_host": "mg0001",
                },
            )
            summary = summarize_smoke(
                result_root=root,
                run_id="run-a",
                path_mode="shared",
                expected="pass",
                required_distinct_hosts=2,
            )
            self.assertEqual(summary["outcome"], "shared_visibility_confirmed")
            self.assertEqual(summary["distinct_hosts"], ["mg0001", "mg0002"])

    def test_stor_01__negative_control_must_fail_for_cross_node_visibility(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {
                "schema_version": 1,
                "run_id": "negative-a",
                "world_size": 2,
                "path_mode": "local-negative",
                "status": "failed",
            }
            atomic_write_json(
                root / "smoke-rank-0.json",
                {
                    **base,
                    "rank": 0,
                    "role": "writer",
                    "hostname": "mg0001",
                    "failure_kind": "acknowledgement_timeout",
                },
            )
            atomic_write_json(
                root / "smoke-rank-1.json",
                {
                    **base,
                    "rank": 1,
                    "role": "reader",
                    "hostname": "mg0002",
                    "failure_kind": "visibility_timeout",
                },
            )
            summary = summarize_smoke(
                result_root=root,
                run_id="negative-a",
                path_mode="local-negative",
                expected="failure",
                required_distinct_hosts=2,
            )
            self.assertEqual(summary["outcome"], "node_local_visibility_rejected")


if __name__ == "__main__":
    unittest.main()
