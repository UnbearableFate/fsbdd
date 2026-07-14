"""Deliberately invalid visibility/atomicity claims for S0B-02 RED."""

from __future__ import annotations

import json
import unittest


class TestUnsafeVisibilityAtomicitySurrogate(unittest.TestCase):
    def test_bench_01__payload_write_start_is_not_publication_complete(self) -> None:
        naive_timing_origin = "payload_write_start"
        required_timing_origin = "visibility_record_replace_complete"
        self.assertEqual(naive_timing_origin, required_timing_origin)

    def test_bench_01__loaded_profile_cannot_be_omitted(self) -> None:
        naive_profiles = {"empty"}
        required_profiles = {"empty", "loaded"}
        self.assertEqual(naive_profiles, required_profiles)

    def test_bench_02__partial_direct_overwrite_is_a_violation(self) -> None:
        observations = [b'{"sequence":1', b'{"sequence":2,"complete":true}']
        naive_violations = 0
        actual_violations = 0
        for observation in observations:
            try:
                value = json.loads(observation)
            except json.JSONDecodeError:
                actual_violations += 1
                continue
            if not isinstance(value, dict) or value.get("complete") is not True:
                actual_violations += 1
        self.assertEqual(naive_violations, actual_violations)

    def test_bench_02__replacement_count_cannot_be_rounded_down(self) -> None:
        naive_replacements = 99_999
        required_replacements = 100_000
        self.assertGreaterEqual(naive_replacements, required_replacements)


if __name__ == "__main__":
    unittest.main()
