from __future__ import annotations

import json
import unittest
from pathlib import Path

from fsbdd_stage0.simulation import (
    SimulationConfig,
    evaluate_decision_case,
    simulate,
)


ROOT = Path(__file__).resolve().parents[2]
DECISIONS = ROOT / "tests" / "fixtures" / "stage0_decision_vectors.json"


class TestASim01EventSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.decisions = json.loads(DECISIONS.read_text(encoding="utf-8"))

    def test_sim_02__all_shared_oracle_decision_vectors_conform(self) -> None:
        for case in self.decisions["decision_cases"]:
            with self.subTest(case=case["id"]):
                actual = evaluate_decision_case(case, self.decisions["lambda_s"])
                self.assertEqual(actual["ready"], case["expected_ready"])
                self.assertEqual(actual["selected"], case["expected_selected"])
                for proposal_id, reason in case["expected_rejections"].items():
                    self.assertEqual(actual["rejections"][proposal_id], reason)

    def test_sim_01__symmetric_fresh_profile_has_periodic_updates(self) -> None:
        result = simulate(
            SimulationConfig(
                learners=4,
                fragments=4,
                q=4,
                q_fresh=4,
                s_max=0,
                lambda_s=1.0,
                h_steps=10,
                grace_fraction_of_h=0.0,
                upload_delay_seconds=0.0,
                visibility_delay_seconds=0.0,
                duration_seconds=220.0,
                heterogeneity_ratio=1.0,
                speed_model="constant_ratio",
                tokens_per_step=32,
                seed=7,
                max_trace_events=64,
            )
        )
        self.assertGreaterEqual(result.global_cycle, 20)
        self.assertEqual(result.stale_accepted_proposals, 0)
        self.assertEqual(result.discarded_tokens, 0)
        self.assertGreater(result.accepted_token_efficiency, 0.8)
        self.assertAlmostEqual(result.update_intervals["mean"], 10.0, delta=1e-9)
        print(
            "SIMULATOR_EVENT_TRACE_SAMPLE="
            + json.dumps(
                {
                    "config_digest": result.config_digest,
                    "global_cycle": result.global_cycle,
                    "trace": list(result.trace[:12]),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def test_stale_01__visibility_and_heterogeneity_create_real_stale_events(self) -> None:
        common = dict(
            learners=4,
            fragments=2,
            q=2,
            q_fresh=1,
            lambda_s=1.0,
            h_steps=10,
            grace_fraction_of_h=0.1,
            upload_delay_seconds=0.5,
            visibility_delay_seconds=5.0,
            duration_seconds=400.0,
            heterogeneity_ratio=3.0,
            speed_model="constant_ratio",
            tokens_per_step=32,
            seed=19,
            max_trace_events=128,
        )
        stale_aware = simulate(SimulationConfig(s_max=1, **common))
        fresh_only = simulate(SimulationConfig(s_max=0, **common))
        self.assertGreater(stale_aware.stale_accepted_proposals, 0)
        self.assertGreater(stale_aware.stale_acceptance_rate, 0.0)
        self.assertGreater(fresh_only.discarded_tokens, 0)

    def test_sim_01__seed_and_insertion_order_are_deterministic(self) -> None:
        config = SimulationConfig(
            learners=4,
            fragments=3,
            q=3,
            q_fresh=1,
            s_max=1,
            lambda_s=1.0,
            h_steps=12,
            grace_fraction_of_h=0.1,
            upload_delay_seconds=0.2,
            visibility_delay_seconds=1.0,
            duration_seconds=180.0,
            heterogeneity_ratio=2.0,
            speed_model="lognormal_jitter",
            tokens_per_step=16,
            seed=101,
            max_trace_events=80,
        )
        first = simulate(config)
        repeated = simulate(config)
        reversed_initialization = simulate(config, learner_order=(3, 2, 1, 0))
        self.assertEqual(first.as_dict(), repeated.as_dict())
        self.assertEqual(first.as_dict(), reversed_initialization.as_dict())
        changed_seed = simulate(
            SimulationConfig(**{**config.as_dict(), "seed": 102})
        )
        self.assertNotEqual(first.as_dict(), changed_seed.as_dict())

    def test_inv_08__operational_state_and_trace_are_bounded(self) -> None:
        config = SimulationConfig(
            learners=4,
            fragments=4,
            q=2,
            q_fresh=1,
            s_max=2,
            lambda_s=1.0,
            h_steps=5,
            grace_fraction_of_h=0.0,
            upload_delay_seconds=0.1,
            visibility_delay_seconds=1.0,
            duration_seconds=2_000.0,
            heterogeneity_ratio=2.0,
            speed_model="lognormal_jitter",
            tokens_per_step=8,
            seed=313,
            max_trace_events=32,
        )
        result = simulate(config)
        self.assertLessEqual(result.max_latest_slots, config.learners * config.fragments)
        self.assertLessEqual(result.max_frontier_slots, config.learners * config.fragments)
        self.assertLessEqual(
            result.max_rejection_tracking_slots,
            config.learners * config.fragments,
        )
        self.assertLessEqual(
            result.max_accepted_tracking_slots,
            config.learners * config.fragments,
        )
        self.assertLessEqual(len(result.trace), config.max_trace_events)
        self.assertLessEqual(result.max_event_queue, config.learners * config.fragments * 4)
        json.dumps(result.as_dict(), sort_keys=True, allow_nan=False)

    def test_sim_01__rejects_profiles_outside_stage0_bounds(self) -> None:
        valid = dict(
            learners=4,
            fragments=4,
            q=2,
            q_fresh=1,
            s_max=1,
            lambda_s=1.0,
            h_steps=8,
            grace_fraction_of_h=0.1,
            upload_delay_seconds=0.1,
            visibility_delay_seconds=1.0,
            duration_seconds=20.0,
            heterogeneity_ratio=1.0,
            speed_model="constant_ratio",
            tokens_per_step=8,
            seed=1,
            max_trace_events=0,
        )
        invalid_profiles = (
            {"s_max": 3},
            {"q": 5},
            {"q_fresh": 3},
            {"fragments": 9},
            {"speed_model": "unspecified"},
            {"duration_seconds": float("inf")},
        )
        for replacement in invalid_profiles:
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    SimulationConfig(**{**valid, **replacement})

    def test_inv_08__identity_accounting_is_fixed_size_as_duration_scales(self) -> None:
        base = dict(
            learners=1,
            fragments=1,
            q=1,
            q_fresh=1,
            s_max=1,
            lambda_s=1.0,
            h_steps=4,
            grace_fraction_of_h=0.1,
            upload_delay_seconds=0.5,
            visibility_delay_seconds=2.0,
            heterogeneity_ratio=1.0,
            tokens_per_step=8,
            seed=41,
            max_trace_events=0,
        )
        for speed_model in ("constant_ratio", "lognormal_jitter"):
            with self.subTest(speed_model=speed_model):
                short = simulate(
                    SimulationConfig(
                        duration_seconds=80.0,
                        speed_model=speed_model,
                        **base,
                    )
                )
                long = simulate(
                    SimulationConfig(
                        duration_seconds=2_000.0,
                        speed_model=speed_model,
                        **base,
                    )
                )
                self.assertGreater(long.accepted_proposals, short.accepted_proposals)
                self.assertEqual(short.max_accepted_tracking_slots, 1)
                self.assertEqual(long.max_accepted_tracking_slots, 1)
                self.assertLessEqual(short.max_rejection_tracking_slots, 1)
                self.assertLessEqual(long.max_rejection_tracking_slots, 1)


if __name__ == "__main__":
    unittest.main()
