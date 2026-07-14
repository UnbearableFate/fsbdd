"""Pre-implementation counterexamples for the S0B-03 capacity contract."""

from __future__ import annotations

import unittest


class TestIncompleteCapacitySurrogate(unittest.TestCase):
    def test_bench_03__discovery_must_not_scan_history(self) -> None:
        fixed_slots = 16 * 32
        history_objects = 10_000
        naive_discovery_operations = fixed_slots + history_objects
        self.assertEqual(naive_discovery_operations, fixed_slots)

    def test_bench_03__matrix_requires_all_fixed_slot_profiles(self) -> None:
        required = {
            (profile, cache, layout)
            for profile in ("stat", "read", "readdir")
            for cache in ("first_touch", "warm")
            for layout in ("flat", "per_learner")
        }
        naive = {("stat", "warm", "flat")}
        self.assertEqual(naive, required)

    def test_bench_03__threshold_report_requires_steady_demand_and_margin(self) -> None:
        learners = 16
        fragments = 32
        polling_interval_seconds = 1.0
        steady_demand = learners * fragments / polling_interval_seconds
        naive_report = {"measured_ops_per_second": 6000.0}
        self.assertIn("steady_demand_ops_per_second", naive_report)
        self.assertEqual(naive_report["steady_demand_ops_per_second"], steady_demand)
        self.assertGreaterEqual(
            naive_report["measured_ops_per_second"], 10.0 * steady_demand
        )

    def test_bench_04__matrix_requires_all_sizes_and_stream_counts(self) -> None:
        required = {
            (size_mb, streams, operation)
            for size_mb in (50, 100, 250, 500)
            for streams in (1, 16)
            for operation in ("write", "read")
        }
        naive = {
            (size_mb, 1, operation)
            for size_mb in (50, 100, 250)
            for operation in ("write", "read")
        }
        self.assertEqual(naive, required)

    def test_bench_04__concurrent_round_must_be_compared_with_ten_percent_h(self) -> None:
        synchronization_period_seconds = 50.0
        naive_report = {"aggregate_megabytes_per_second": 2000.0}
        self.assertIn("concurrent_round_seconds", naive_report)
        self.assertLessEqual(
            naive_report["concurrent_round_seconds"],
            0.10 * synchronization_period_seconds,
        )


if __name__ == "__main__":
    unittest.main()
