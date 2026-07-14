from __future__ import annotations

import json
import math
import random
import unittest
from pathlib import Path

from fsbdd_stage0.oracle import (
    OracleInputError,
    inverse_staleness_weights,
    pseudo_gradient,
    validated_pseudo_gradient,
)


FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "oracle_golden_cases.json"
)


class TestAAlg00PseudoGradientWeighting(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.tolerance = cls.fixture["tolerance"]

    def assert_vector_close(self, actual: list[float], expected: list[float]) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected, strict=True):
            self.assertAlmostEqual(actual_value, expected_value, delta=self.tolerance)

    def test_oracle_01__golden_fresh_and_stale_cases(self) -> None:
        for case in self.fixture["pseudo_gradient_cases"]:
            with self.subTest(case=case["id"]):
                actual = validated_pseudo_gradient(
                    base=case["base"],
                    local=case["local"],
                    base_version=case["base_version"],
                    current_version=case["current_version"],
                    s_max=case["s_max"],
                    base_identity_matches=True,
                )
                self.assert_vector_close(actual, case["expected"])

    def test_inv_07__current_minus_local_stale_surrogate_is_rejected(self) -> None:
        case = next(
            case
            for case in self.fixture["pseudo_gradient_cases"]
            if case["id"] == "stale-s1-counterexample"
        )
        reference = pseudo_gradient(case["base"], case["local"])
        wrong = pseudo_gradient(case["current"], case["local"])
        self.assert_vector_close(reference, case["expected"])
        self.assert_vector_close(wrong, case["forbidden_current_minus_local"])
        with self.assertRaises(AssertionError):
            self.assert_vector_close(wrong, case["expected"])

    def test_oracle_02__weight_golden_cases(self) -> None:
        for case in self.fixture["weight_cases"]:
            with self.subTest(case=case["id"]):
                weights = inverse_staleness_weights(
                    case["tokens"], case["staleness"], case["lambda_s"]
                )
                self.assert_vector_close(weights, case["expected"])

    def test_oracle_02__weight_properties(self) -> None:
        rng = random.Random(20260714)
        for _ in range(500):
            width = rng.randint(1, 16)
            tokens = [rng.uniform(1.0, 1_000_000.0) for _ in range(width)]
            staleness = [rng.randint(0, 8) for _ in range(width)]
            lambda_s = rng.uniform(0.0, 4.0)
            weights = inverse_staleness_weights(tokens, staleness, lambda_s)
            self.assertTrue(all(math.isfinite(weight) and weight >= 0 for weight in weights))
            self.assertAlmostEqual(sum(weights), 1.0, delta=2e-6)

    def test_oracle_02__fresh_is_undiscounted_and_staleness_is_monotone(self) -> None:
        fresh = inverse_staleness_weights([7.0, 13.0], [0, 0], 4.0)
        self.assert_vector_close(fresh, [0.35, 0.65])
        for lambda_s in (0.0, 0.5, 1.0, 2.0):
            implementation_mass = []
            for stale in range(8):
                weights = inverse_staleness_weights(
                    [100.0, 100.0], [stale, 0], lambda_s
                )
                implementation_mass.append(weights[0])
            self.assertTrue(
                all(
                    left >= right
                    for left, right in zip(
                        implementation_mass, implementation_mass[1:]
                    )
                )
            )
        self.assert_vector_close(
            inverse_staleness_weights([100.0, 100.0], [0, 3], 1.0),
            [0.8, 0.2],
        )

    def test_oracle_02__randomized_values_match_independent_formula(self) -> None:
        rng = random.Random(7182818)
        for _ in range(500):
            width = rng.randint(1, 12)
            tokens = [rng.uniform(1.0, 1_000_000.0) for _ in range(width)]
            staleness = [rng.randint(0, 32) for _ in range(width)]
            lambda_s = rng.uniform(0.0, 4.0)
            raw = [
                token / (1.0 + lambda_s * stale)
                for token, stale in zip(tokens, staleness, strict=True)
            ]
            expected = [value / sum(raw) for value in raw]
            self.assert_vector_close(
                inverse_staleness_weights(tokens, staleness, lambda_s), expected
            )

    def test_oracle_02__single_extreme_and_lambda_zero_boundaries(self) -> None:
        self.assert_vector_close(inverse_staleness_weights([17.0], [7], 2.0), [1.0])
        no_decay = inverse_staleness_weights([1.0, 1_000_000.0], [0, 99], 0.0)
        self.assertAlmostEqual(no_decay[1] / no_decay[0], 1_000_000.0, delta=1.0)
        mixed = inverse_staleness_weights([1.0, 1_000_000.0], [0, 1], 1.0)
        self.assertTrue(0.0 < mixed[0] < mixed[1] < 1.0)

    def test_oracle_01__invalid_identity_future_and_too_stale_are_rejected(self) -> None:
        for case in self.fixture["invalid_base_cases"]:
            with self.subTest(case=case["id"]):
                arguments = {key: value for key, value in case.items() if key not in {"id", "error"}}
                with self.assertRaisesRegex(OracleInputError, case["error"]):
                    validated_pseudo_gradient(**arguments)

    def test_oracle_02__invalid_weight_inputs_are_rejected(self) -> None:
        invalid = (
            ([0.0], [0], 1.0),
            ([1.0], [-1], 1.0),
            ([1.0], [0], -1.0),
            ([1.0, 2.0], [0], 1.0),
        )
        for tokens, staleness, lambda_s in invalid:
            with self.subTest(tokens=tokens, staleness=staleness, lambda_s=lambda_s):
                with self.assertRaises(OracleInputError):
                    inverse_staleness_weights(tokens, staleness, lambda_s)


if __name__ == "__main__":
    unittest.main()
