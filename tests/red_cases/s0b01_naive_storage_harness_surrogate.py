"""Deliberately unsafe storage-harness assumptions that must fail S0B-01 RED."""

from __future__ import annotations

import unittest


class TestNaiveStorageHarnessSurrogate(unittest.TestCase):
    def test_fs_08__manifest_cannot_omit_compute_and_fs_identity(self) -> None:
        required = {
            "skill_commit",
            "initial_hostname",
            "compute_hosts",
            "pbs_job_id",
            "pbs_nodefile",
            "target_run_root",
            "filesystem_type",
            "filesystem_source",
            "module_list",
        }
        naive_manifest = {"target_run_root", "pbs_job_id"}
        self.assertEqual(naive_manifest, required)

    def test_fs_08__login_host_cannot_run_benchmark(self) -> None:
        hostname = "miyabi-g1"
        naive_allowed = hostname.startswith("miyabi-")
        self.assertFalse(naive_allowed, "login host was treated as a compute node")

    def test_stor_01__node_local_tmp_cannot_be_assumed_shared(self) -> None:
        writer_host = "mg0001"
        reader_host = "mg0002"
        path = "/tmp/fsbdd-stage0-smoke"
        naive_shared = path.startswith("/")
        safe_shared = writer_host == reader_host or path.startswith("/work/")
        self.assertEqual(naive_shared, safe_shared)


if __name__ == "__main__":
    unittest.main()
