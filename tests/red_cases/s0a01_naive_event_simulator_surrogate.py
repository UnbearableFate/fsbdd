"""Deliberately naive event decisions that must fail S0A-01 RED."""

from __future__ import annotations

import unittest


class TestNaiveEventSimulatorSurrogate(unittest.TestCase):
    def test_sim_02__duplicate_files_are_not_distinct_learners(self) -> None:
        visible_files = ["a/seq1", "a/seq2", "b/seq1"]
        naive_ready = len(visible_files) >= 3
        self.assertFalse(naive_ready, "file count incorrectly formed distinct quorum")

    def test_stale_01__snapshot_base_is_not_rewritten_at_visibility(self) -> None:
        snapshot_base = 3
        current_at_visibility = 4
        naive_base = current_at_visibility
        self.assertEqual(
            naive_base,
            snapshot_base,
            "in-flight proposal was incorrectly rewritten from stale to fresh",
        )

    def test_stale_07__insertion_order_cannot_choose_the_winner(self) -> None:
        first_order = ["learner-b", "learner-a"]
        second_order = list(reversed(first_order))
        naive_first_winner = first_order[0]
        naive_second_winner = second_order[0]
        self.assertEqual(
            naive_first_winner,
            naive_second_winner,
            "candidate selection depends on event insertion order",
        )


if __name__ == "__main__":
    unittest.main()
