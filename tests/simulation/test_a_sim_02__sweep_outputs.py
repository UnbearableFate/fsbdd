from __future__ import annotations

import csv
import json
import os
import statistics
import tempfile
import unittest
from pathlib import Path

from fsbdd_stage0.sweep import (
    AXIS_ORDER,
    EXPECTED_SIM03_AXES,
    SweepInputError,
    file_sha256,
    load_sweep_config,
    matrix_specs,
    run_matrix_row,
    run_shard,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "stage0" / "simulation_sweep.json"


class TestASim02SweepOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_sweep_config(CONFIG_PATH)
        run_dir_value = os.environ.get("FSBDD_SWEEP_RUN_DIR")
        cls.run_dir = Path(run_dir_value) if run_dir_value else None
        if cls.run_dir is not None:
            cls.manifest_path = cls.run_dir / "results" / "simulation_sweep_manifest.json"
            cls.summary_path = cls.run_dir / "results" / "simulation_summary.json"
            cls.recommendation_path = cls.run_dir / "results" / "simulation_recommendation.md"
        else:
            cls.manifest_path = ROOT / "reports" / "stage0" / "simulation_sweep_manifest.json"
            cls.summary_path = ROOT / "reports" / "stage0" / "simulation_summary.json"
            cls.recommendation_path = ROOT / "reports" / "stage0" / "simulation_recommendation.md"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.summary = json.loads(cls.summary_path.read_text(encoding="utf-8"))
        cls.raw_path = Path(cls.manifest["artifacts"]["raw_csv"]["path"])
        cls.aggregate_path = Path(cls.manifest["artifacts"]["aggregate_csv"]["path"])
        with cls.raw_path.open("r", encoding="utf-8", newline="") as handle:
            cls.raw_rows = list(csv.DictReader(handle))

    def test_sim_03__frozen_matrix_is_exact_and_unique(self) -> None:
        specs = matrix_specs(self.config)
        self.assertEqual(len(specs), 7776)
        identities = {
            tuple(spec[name] for name in AXIS_ORDER)
            for spec in specs
        }
        self.assertEqual(len(identities), 7776)
        for axis, expected in EXPECTED_SIM03_AXES.items():
            self.assertEqual(self.config["axes"][axis], expected)
        self.assertEqual(self.config["axes"]["seed"], [17, 29, 43])
        for spec in specs:
            self.assertEqual(spec["q"], int(spec["learners"] * spec["quorum_fraction"]))

    def test_sim_04__symmetric_fresh_case_has_analytic_interval(self) -> None:
        spec = next(
            item
            for item in matrix_specs(self.config)
            if item["learners"] == 4
            and item["quorum_fraction"] == 1.0
            and item["heterogeneity_ratio"] == 1.0
            and item["grace_fraction_of_h"] == 0.0
            and item["visibility_delay_seconds"] == 0.0
            and item["s_max"] == 0
            and item["speed_model"] == "constant_ratio"
            and item["seed"] == 17
        )
        row = run_matrix_row(0, spec)
        self.assertEqual(row["stale_accepted_proposals"], 0)
        self.assertEqual(row["discarded_tokens"], 0)
        self.assertGreater(row["accepted_token_efficiency"], 0.9)
        self.assertAlmostEqual(row["update_interval_mean"], spec["h_steps"], delta=1e-9)

    def test_a_sim_02__raw_matrix_and_manifest_are_complete(self) -> None:
        expected = self.config["execution"]["expected_rows"]
        self.assertTrue(self.manifest["matrix"]["complete"])
        self.assertEqual(self.manifest["matrix"]["expected_rows"], expected)
        self.assertEqual(self.manifest["matrix"]["actual_rows"], expected)
        self.assertTrue(self.manifest["resumed_from_checkpoint"])
        self.assertGreaterEqual(self.manifest["execution_sessions"], 2)
        self.assertNotEqual(
            self.manifest["initial_pbs_job_id"],
            self.manifest["pbs_job_id"],
        )
        self.assertEqual(len(self.raw_rows), expected)
        self.assertEqual(
            len({row["row_key"] for row in self.raw_rows}),
            expected,
        )
        self.assertEqual(
            {int(row["matrix_index"]) for row in self.raw_rows},
            set(range(expected)),
        )
        for artifact in self.manifest["artifacts"].values():
            path = Path(artifact["path"])
            self.assertEqual(file_sha256(path), artifact["sha256"])
        for shard in self.manifest["shards"]:
            self.assertEqual(file_sha256(Path(shard["path"])), shard["sha256"])

    def test_a_sim_02__every_axis_value_and_pair_is_present(self) -> None:
        converters = {
            "learners": int,
            "quorum_fraction": float,
            "heterogeneity_ratio": float,
            "grace_fraction_of_h": float,
            "visibility_delay_seconds": float,
            "s_max": int,
            "speed_model": str,
            "seed": int,
        }
        for axis in AXIS_ORDER:
            observed = sorted({converters[axis](row[axis]) for row in self.raw_rows}, key=str)
            expected = sorted(self.config["axes"][axis], key=str)
            self.assertEqual(observed, expected, axis)
        paired = {}
        for row in self.raw_rows:
            key = tuple(row[name] for name in AXIS_ORDER if name != "s_max")
            paired.setdefault(key, {})[int(row["s_max"])] = float(
                row["accepted_token_efficiency"]
            )
        self.assertEqual(len(paired), 2592)
        self.assertTrue(all(set(arms) == {0, 1, 2} for arms in paired.values()))
        gains = [arms[1] - arms[0] for arms in paired.values()]
        self.assertAlmostEqual(
            statistics.fmean(gains),
            self.summary["recovery"]["paired_seed_level"]["mean"],
            delta=1e-15,
        )

    def test_a_sim_03__summary_and_recommendation_schema_is_actionable(self) -> None:
        matrix = self.summary["matrix"]
        self.assertEqual(matrix["aggregate_profile_rows"], 2592)
        self.assertEqual(matrix["paired_recovery_rows"], 2592)
        self.assertEqual(matrix["seed_count"], 3)
        self.assertEqual(
            sum(self.summary["recovery"]["paper_classification_counts"].values()),
            864,
        )
        symmetric = self.summary["symmetric_fresh_sanity"]
        self.assertEqual(symmetric["rows"], 9)
        self.assertAlmostEqual(
            symmetric["metrics"]["update_interval_mean"]["mean"],
            self.config["fixed"]["h_steps"],
            delta=1e-9,
        )
        anchor = self.summary["stage4_prediction_anchor"]
        self.assertEqual(anchor["seed_count"], 3)
        self.assertEqual(anchor["profile"]["s_max"], 1)
        self.assertLess(anchor["profile"]["q"], anchor["profile"]["learners"])
        self.assertGreaterEqual(anchor["profile"]["q"], 2)
        self.assertEqual(anchor["profile"]["q_fresh"], 1)
        self.assertAlmostEqual(
            anchor["profile"]["visibility_delay_over_h"],
            anchor["profile"]["visibility_delay_seconds"] / anchor["profile"]["h_steps"],
        )
        self.assertIn(
            anchor["paper_classification"],
            {"full_second_contribution", "compressed_section", "ablation_or_discussion"},
        )
        recommendation = self.recommendation_path.read_text(encoding="utf-8")
        self.assertIn("Stage 1 Profile A recommendation", recommendation)
        self.assertIn("Stage 4 remains mandatory", recommendation)
        self.assertIn("0/1/5/30s", recommendation)
        self.assertIn("runtime correctness", recommendation)

    def test_sim_04__aggregate_units_and_state_bounds_are_explicit(self) -> None:
        with self.aggregate_path.open("r", encoding="utf-8", newline="") as handle:
            aggregates = list(csv.DictReader(handle))
        self.assertEqual(len(aggregates), 2592)
        self.assertTrue(all(int(row["seed_count"]) == 3 for row in aggregates))
        for row in self.raw_rows[::257]:
            learners = int(row["learners"])
            fragments = int(row["fragments"])
            accepted_tokens = int(row["accepted_tokens"])
            discarded_tokens = int(row["discarded_tokens"])
            token_opportunities = int(row["token_opportunities"])
            self.assertAlmostEqual(
                float(row["visibility_delay_over_h"]),
                float(row["visibility_delay_seconds"])
                / (
                    float(row["h_steps"])
                    * float(row["nominal_fastest_step_seconds"])
                ),
            )
            self.assertAlmostEqual(
                float(row["accepted_token_efficiency"]),
                accepted_tokens / token_opportunities,
                delta=1e-15,
            )
            discard_denominator = accepted_tokens + discarded_tokens
            expected_discard = (
                discarded_tokens / discard_denominator if discard_denominator else 0.0
            )
            self.assertAlmostEqual(
                float(row["token_weighted_discard_rate"]),
                expected_discard,
                delta=1e-15,
            )
            self.assertLessEqual(int(row["max_latest_slots"]), learners * fragments)
            self.assertLessEqual(int(row["max_frontier_slots"]), learners * fragments)
            self.assertLessEqual(
                int(row["max_rejection_tracking_slots"]), learners * fragments
            )
            self.assertLessEqual(
                int(row["max_accepted_tracking_slots"]), learners * fragments
            )
        self.assertEqual(
            self.summary["recovery"]["unit"],
            "accepted-token efficiency fraction; multiply by 100 for percentage points",
        )

    def test_sim_04__lognormal_default_region_records_seed_sensitivity(self) -> None:
        rows = [
            row
            for row in self.raw_rows
            if int(row["learners"]) == 8
            and float(row["quorum_fraction"]) == 0.5
            and float(row["heterogeneity_ratio"]) == 2.0
            and float(row["grace_fraction_of_h"]) == 0.1
            and float(row["visibility_delay_seconds"]) == 1.0
            and int(row["s_max"]) == 1
            and row["speed_model"] == "lognormal_jitter"
        ]
        self.assertEqual({int(row["seed"]) for row in rows}, {17, 29, 43})
        outcomes = {
            (
                float(row["accepted_token_efficiency"]),
                float(row["stale_acceptance_rate"]),
                float(row["update_interval_mean"]),
            )
            for row in rows
        }
        self.assertGreater(len(outcomes), 1)

    def test_sim_03__partial_shard_resume_rejects_mixed_or_corrupt_rows(self) -> None:
        generator_commit = "resume-test-commit"
        generator_digest = "resume-test-digest"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "partial.csv"
            partial = run_shard(
                CONFIG_PATH,
                output,
                0,
                self.config["execution"]["shard_count"],
                generator_commit,
                generator_digest,
                max_new_rows=1,
            )
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["rows"], 1)
            retained = run_shard(
                CONFIG_PATH,
                output,
                0,
                self.config["execution"]["shard_count"],
                generator_commit,
                generator_digest,
                max_new_rows=0,
            )
            self.assertEqual(retained["rows"], 1)
            with self.assertRaises(SweepInputError):
                run_shard(
                    CONFIG_PATH,
                    output,
                    0,
                    self.config["execution"]["shard_count"],
                    "different-code",
                    generator_digest,
                    max_new_rows=0,
                )
            with output.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                rows = list(reader)
            rows[0]["accepted_token_efficiency"] = "0.9"
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(SweepInputError):
                run_shard(
                    CONFIG_PATH,
                    output,
                    0,
                    self.config["execution"]["shard_count"],
                    generator_commit,
                    generator_digest,
                    max_new_rows=0,
                )


if __name__ == "__main__":
    unittest.main()
