from __future__ import annotations

import csv
import itertools
import json
import unittest
from pathlib import Path

from fsbdd_stage0.oracle import (
    CandidatePolicy,
    ConsumptionFrontier,
    OuterSGDState,
    Proposal,
    commit_consumption,
    outer_sgd_step,
    select_proposals,
    validate_requirement_evidence,
    weighted_direct_merge,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "stage0_decision_vectors.json"
TRACEABILITY = ROOT / "reports" / "stage0" / "requirement_to_evidence.csv"


class TestAAlg00ConsumptionOuterTransitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    @staticmethod
    def make_proposals(case: dict[str, object]) -> list[Proposal]:
        return [Proposal(**proposal) for proposal in case["proposals"]]

    @staticmethod
    def make_frontiers(case: dict[str, object]) -> dict[str, ConsumptionFrontier]:
        return {
            learner: ConsumptionFrontier(**frontier)
            for learner, frontier in case["frontiers"].items()
        }

    def policy(self, case: dict[str, object]) -> CandidatePolicy:
        return CandidatePolicy(
            current_version=case["current_version"],
            s_max=case["s_max"],
            q=case["q"],
            q_fresh=case["q_fresh"],
            max_contributors=case["max_contributors"],
            lambda_s=self.fixture["lambda_s"],
        )

    def assert_case(self, case: dict[str, object], proposals: list[Proposal]) -> None:
        result = select_proposals(
            proposals, self.make_frontiers(case), self.policy(case)
        )
        self.assertEqual(result.ready, case["expected_ready"])
        self.assertEqual(
            [proposal.proposal_id for proposal in result.selected],
            case["expected_selected"],
        )
        for proposal_id, reason in case["expected_rejections"].items():
            self.assertEqual(result.rejections[proposal_id], reason)

    def test_oracle_03__shared_decision_vectors(self) -> None:
        for case in self.fixture["decision_cases"]:
            with self.subTest(case=case["id"]):
                self.assert_case(case, self.make_proposals(case))

    def test_inv_05__proposal_order_does_not_change_selection(self) -> None:
        case = next(
            item
            for item in self.fixture["decision_cases"]
            if item["id"] == "one-per-learner-prefers-fresh-before-token-mass"
        )
        proposals = self.make_proposals(case)
        for order in itertools.permutations(proposals):
            self.assert_case(case, list(order))

    def test_prop_06__frontier_commits_only_after_successful_publication(self) -> None:
        case = next(
            item
            for item in self.fixture["decision_cases"]
            if item["id"] == "sequence-jump-with-new-base-is-eligible"
        )
        initial = self.make_frontiers(case)
        result = select_proposals(self.make_proposals(case), initial, self.policy(case))
        self.assertEqual(commit_consumption(initial, result.selected, False), initial)
        committed = commit_consumption(initial, result.selected, True)
        self.assertEqual(committed["a"], ConsumptionFrontier(9, 4))
        repeated = select_proposals(self.make_proposals(case), committed, self.policy(case))
        self.assertFalse(repeated.ready)
        self.assertEqual(repeated.rejections["a-seq9-base4"], "consumed_sequence")

    def test_prop_06__frontier_remains_fixed_size_across_many_updates(self) -> None:
        frontiers: dict[str, ConsumptionFrontier] = {}
        for version in range(1_000):
            proposal = Proposal(
                proposal_id=f"a-{version}",
                learner_id="a",
                sequence=version,
                base_version=version,
                tokens=1,
                local_steps=1,
                base_identity_matches=True,
            )
            policy = CandidatePolicy(
                current_version=version,
                s_max=0,
                q=1,
                q_fresh=1,
                max_contributors=1,
                lambda_s=1.0,
            )
            result = select_proposals([proposal], frontiers, policy)
            self.assertTrue(result.ready)
            frontiers = commit_consumption(frontiers, result.selected, True)
        self.assertEqual(frontiers, {"a": ConsumptionFrontier(999, 999)})

    def test_oracle_04__nesterov_reference_transitions(self) -> None:
        case = self.fixture["outer_transition_cases"][0]
        parameters = case["initial_parameters"]
        state = OuterSGDState(momentum_buffer=None)
        for index, gradient in enumerate(case["gradients"]):
            parameters, state = outer_sgd_step(
                parameters,
                gradient,
                state,
                learning_rate=case["learning_rate"],
                momentum=case["momentum"],
                nesterov=case["nesterov"],
            )
            for actual, expected in zip(
                parameters, case["expected_parameters"][index], strict=True
            ):
                self.assertAlmostEqual(actual, expected, delta=1e-6)
            for actual, expected in zip(
                state.momentum_buffer, case["expected_buffers"][index], strict=True
            ):
                self.assertAlmostEqual(actual, expected, delta=1e-6)

    def test_oracle_04__lr_one_no_momentum_equals_direct_averaging(self) -> None:
        current = [10.0, 0.0]
        merged = weighted_direct_merge(
            current=current,
            bases=[current, current],
            locals_=[[8.0, 4.0], [6.0, 2.0]],
            weights=[0.25, 0.75],
        )
        parameters, state = outer_sgd_step(
            current,
            merged,
            OuterSGDState(momentum_buffer=None),
            learning_rate=1.0,
            momentum=0.0,
            nesterov=False,
        )
        self.assertEqual(state, OuterSGDState(momentum_buffer=None))
        for actual, expected in zip(parameters, [6.5, 2.5], strict=True):
            self.assertAlmostEqual(actual, expected, delta=1e-6)

    def test_opt_01__mixed_fresh_stale_merge_is_applied_to_current(self) -> None:
        current = [12.0, -2.0]
        merged = weighted_direct_merge(
            current=current,
            bases=[[3.0, 5.0], current],
            locals_=[[1.0, 1.0], [8.0, -4.0]],
            weights=[0.5, 0.5],
        )
        self.assertEqual(merged, [3.0, 3.0])
        parameters, _ = outer_sgd_step(
            current,
            merged,
            OuterSGDState(momentum_buffer=None),
            learning_rate=1.0,
            momentum=0.0,
            nesterov=False,
        )
        self.assertEqual(parameters, [9.0, -5.0])

    def test_oracle_05__initial_requirement_evidence_map_is_complete(self) -> None:
        with TRACEABILITY.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {
            *(f"SIM-{index:02d}" for index in range(1, 6)),
            *(f"BENCH-{index:02d}" for index in range(1, 6)),
            *(f"ORACLE-{index:02d}" for index in range(1, 6)),
            "A-SIM-01",
            "A-SIM-02",
            "A-SIM-03",
            "A-BENCH-01",
            "A-BENCH-02",
            "A-BENCH-03",
            "A-BENCH-04",
            "A-ALG-00",
        }
        validate_requirement_evidence(rows, required)


if __name__ == "__main__":
    unittest.main()
