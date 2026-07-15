"""Deliberately broken decisions/transitions proving S0C-02 RED sensitivity."""

from __future__ import annotations

import unittest


class TestNaiveConsumptionTransitionSurrogate(unittest.TestCase):
    def test_inv_05__file_count_is_not_distinct_learner_quorum(self) -> None:
        proposals = ["learner-a/seq-1", "learner-a/seq-2", "learner-b/seq-1"]
        naive_ready = len(proposals) >= 3
        self.assertFalse(
            naive_ready,
            "INV-05 violation: two files from learner-a were counted as two votes",
        )

    def test_prop_06__new_sequence_on_consumed_base_is_not_eligible(self) -> None:
        frontier = {"last_sequence": 2, "last_base_version": 2}
        proposal = {"sequence": 3, "base_version": 2}
        naive_eligible = proposal["sequence"] > frontier["last_sequence"]
        self.assertFalse(
            naive_eligible,
            "PROP-06 violation: sequence advanced but the base trajectory was consumed",
        )

    def test_opt_04__nesterov_must_use_the_updated_buffer(self) -> None:
        parameters = [10.0, -5.0]
        gradient = [2.0, -4.0]
        learning_rate = 0.1
        momentum = 0.9
        old_buffer = [0.0, 0.0]
        wrong = [
            value - learning_rate * (grad + momentum * old)
            for value, grad, old in zip(parameters, gradient, old_buffer, strict=True)
        ]
        self.assertEqual(
            wrong,
            [9.62, -4.24],
            "OPT-04 violation: Nesterov direction used the old momentum buffer",
        )

    def test_oracle_05__missing_requirement_mapping_is_a_failure(self) -> None:
        mapped = {"ORACLE-01", "ORACLE-02", "ORACLE-03", "ORACLE-04"}
        self.assertIn("ORACLE-05", mapped, "ORACLE-05 has no evidence mapping")


if __name__ == "__main__":
    unittest.main()
