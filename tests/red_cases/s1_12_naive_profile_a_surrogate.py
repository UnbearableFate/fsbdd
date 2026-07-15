from __future__ import annotations

import unittest


def unsafe_average(locals_: list[list[float]]) -> list[float]:
    return [sum(column) / len(locals_) for column in zip(*locals_, strict=True)]


class ChasingLatestEvaluation:
    def __init__(self) -> None:
        self.latest = [[0.0], [10.0]]
        self.versions = [0, 0]

    def load(self) -> tuple[tuple[int, ...], list[list[float]]]:
        first = list(self.latest[0])
        first_version = self.versions[0]
        self.latest[1] = [11.0]
        self.versions[1] = 1
        return (first_version, self.versions[1]), [first, list(self.latest[1])]


def barrier_coupled_rates(control: list[float], slowed_index: int) -> list[float]:
    common = min(
        value * (0.5 if index == slowed_index else 1.0)
        for index, value in enumerate(control)
    )
    return [common for _ in control]


class S112SemanticRed(unittest.TestCase):
    def test_a_alg_01__naive_unweighted_e2e_differs_from_token_oracle(self) -> None:
        current = [10.0, 0.0]
        locals_ = [[8.0, 4.0], [6.0, 2.0], [9.0, -1.0], [5.0, 3.0]]
        tokens = [1.0, 3.0, 2.0, 10.0]
        expected = [
            sum(
                local[column] * token
                for local, token in zip(locals_, tokens, strict=True)
            )
            / sum(tokens)
            for column in range(len(current))
        ]
        self.assertEqual(
            unsafe_average(locals_),
            expected,
            "A-ALG-01 requires token-weighted production-path alignment",
        )

    def test_global_05__chasing_latest_does_not_produce_a_frozen_snapshot(self) -> None:
        evaluator = ChasingLatestEvaluation()
        captured_vector = tuple(evaluator.versions)
        loaded_vector, _ = evaluator.load()
        self.assertEqual(
            loaded_vector,
            captured_vector,
            "GLOBAL-05 requires an immutable pre-load version vector",
        )

    def test_a_learn_01__slow_learner_must_not_throttle_peers(self) -> None:
        control = [100.0, 100.0, 100.0, 100.0]
        injected = barrier_coupled_rates(control, 3)
        for index in range(3):
            change = abs(injected[index] / control[index] - 1.0)
            self.assertLessEqual(change, 0.02, "A-LEARN-01 forbids peer-step barriers")


if __name__ == "__main__":
    unittest.main(verbosity=2)
