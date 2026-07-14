"""Deliberately incomplete sweep/report logic that must fail S0A-02 RED."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stage0" / "simulation_sweep.json"


class TestIncompleteSweepSurrogate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_sim_03__omitting_one_visibility_axis_fails_matrix_cardinality(self) -> None:
        axes = self.config["axes"]
        deliberately_incomplete_lengths = [
            len(axes["learners"]),
            len(axes["quorum_fraction"]),
            len(axes["heterogeneity_ratio"]),
            len(axes["grace_fraction_of_h"]),
            len(axes["visibility_delay_seconds"][:-1]),
            len(axes["s_max"]),
            len(axes["speed_model"]),
            len(axes["seed"]),
        ]
        naive_rows = math.prod(deliberately_incomplete_lengths)
        self.assertEqual(naive_rows, self.config["execution"]["expected_rows"])

    def test_sim_04__report_missing_required_metric_fails_schema(self) -> None:
        naive_report_fields = {
            "token_weighted_discard_rate",
            "accepted_token_efficiency",
            "stale_acceptance_rate",
        }
        self.assertEqual(naive_report_fields, set(self.config["outputs"]))

    def test_sim_04__symmetric_interval_cannot_scale_with_learner_count(self) -> None:
        h_steps = self.config["fixed"]["h_steps"]
        learners = self.config["axes"]["learners"][0]
        naive_interval = h_steps * learners
        self.assertEqual(naive_interval, h_steps)


if __name__ == "__main__":
    unittest.main()
