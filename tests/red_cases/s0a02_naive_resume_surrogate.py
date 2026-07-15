"""Deliberately unsafe resume decisions that must fail corrected S0A-02 RED."""

from __future__ import annotations

import unittest


class TestNaiveResumeSurrogate(unittest.TestCase):
    def test_resume__existing_checkpoint_must_not_be_rejected_unconditionally(
        self,
    ) -> None:
        run_directory_exists = True
        naive_entrypoint_exits_73 = run_directory_exists
        self.assertFalse(
            naive_entrypoint_exits_73,
            "interrupted checkpoint cannot be resumed through the PBS entrypoint",
        )

    def test_resume__old_generator_row_must_not_be_trusted(self) -> None:
        frozen_row_key_matches = True
        retained_generator_commit = "old-code"
        current_generator_commit = "checked-code"
        naive_accepts = frozen_row_key_matches
        safe_accepts = (
            frozen_row_key_matches
            and retained_generator_commit == current_generator_commit
        )
        self.assertEqual(
            naive_accepts,
            safe_accepts,
            "row_key alone silently accepts a row from different code",
        )

    def test_resume__corrupted_metric_row_must_not_be_trusted(self) -> None:
        retained = {
            "accepted_tokens": 100,
            "token_opportunities": 200,
            "accepted_token_efficiency": 0.9,
        }
        naive_accepts = True
        formula_is_valid = retained["accepted_token_efficiency"] == (
            retained["accepted_tokens"] / retained["token_opportunities"]
        )
        self.assertEqual(
            naive_accepts,
            formula_is_valid,
            "retained metrics were not revalidated before resume",
        )


if __name__ == "__main__":
    unittest.main()
