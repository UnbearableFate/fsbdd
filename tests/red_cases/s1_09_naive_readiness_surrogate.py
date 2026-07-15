"""Deliberately wrong readiness/scheduler used to preserve S1-09 RED evidence."""

from __future__ import annotations

import unittest


class NaiveReadiness:
    def __init__(self, *, quorum: int) -> None:
        self.quorum = quorum
        self.polls = 0

    def ready(self, files: list[tuple[str, int]]) -> bool:
        self.polls += 1
        return len(files) >= self.quorum or self.polls >= self.quorum

    def select(self, files: list[tuple[str, int]]) -> list[tuple[str, int]]:
        return list(files)

    def claim(self, ready_fragments: list[int]) -> int:
        return min(ready_fragments)


class S109RedCases(unittest.TestCase):
    def test_quorum_counts_distinct_learners_not_files(self) -> None:
        naive = NaiveReadiness(quorum=2)
        duplicate_learner_files = [("learner-0", 1), ("learner-0", 2)]
        self.assertFalse(naive.ready(duplicate_learner_files))

    def test_poll_iterations_cannot_create_readiness(self) -> None:
        naive = NaiveReadiness(quorum=3)
        self.assertFalse(naive.ready([]))
        self.assertFalse(naive.ready([]))
        self.assertFalse(naive.ready([]))

    def test_hot_fragment_cannot_starve_another_ready_fragment(self) -> None:
        naive = NaiveReadiness(quorum=1)
        claims = [naive.claim([0, 1]) for _ in range(8)]
        self.assertIn(1, claims[:2])

    def test_selection_is_independent_of_input_order(self) -> None:
        naive = NaiveReadiness(quorum=2)
        first = [("learner-0", 1), ("learner-1", 1)]
        second = list(reversed(first))
        self.assertEqual(naive.select(first), naive.select(second))


if __name__ == "__main__":
    unittest.main(verbosity=2)
