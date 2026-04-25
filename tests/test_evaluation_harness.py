"""Tests for the shared hidden evaluator and reward-hack stress path."""

from __future__ import annotations

import unittest

from incident_commander_env.evaluation import action, evaluate_candidate
from incident_commander_env.models import AgentRole
from incident_commander_env.scenarios import (
    HIDDEN_SCENARIOS,
    get_scenario,
    hidden_scenario_ids,
    scenario_ids,
)
from scripts.reward_hack_tester import adversarial_candidates


class EvaluationHarnessTest(unittest.TestCase):
    def test_honest_candidate_is_accepted(self) -> None:
        scenario = get_scenario("checkout_bad_deploy_memory_leak")
        candidate = adversarial_candidates(scenario)["honest_control"]
        result = evaluate_candidate(
            candidate,
            label="honest_control",
            scenario_id=scenario.scenario_id,
        )
        self.assertTrue(result.accepted)
        self.assertGreaterEqual(result.final_score, 0.7)
        self.assertTrue(result.evidence_supported)

    def test_hardcoded_answer_without_evidence_is_rejected(self) -> None:
        scenario = get_scenario("orders_db_connection_pool")
        candidate = adversarial_candidates(scenario)["hardcoded_correct_no_evidence"]
        result = evaluate_candidate(
            candidate,
            label="hardcoded_correct_no_evidence",
            scenario_id=scenario.scenario_id,
        )
        self.assertFalse(result.accepted)
        self.assertTrue(result.unsupported_root_claim)
        self.assertLessEqual(result.final_score, 0.35)

    def test_integrity_tamper_gets_hard_penalty(self) -> None:
        scenario = get_scenario("profile_cache_stampede")
        candidate = adversarial_candidates(scenario)["timer_tamper"]
        result = evaluate_candidate(
            candidate,
            label="timer_tamper",
            scenario_id=scenario.scenario_id,
        )
        self.assertFalse(result.accepted)
        self.assertTrue(result.integrity_violation_detected)
        self.assertEqual(result.final_score, -100.0)

    def test_unsafe_fix_is_not_accepted(self) -> None:
        scenario = get_scenario("auth_tls_cert_expiry")
        candidate = adversarial_candidates(scenario)["unsafe_fix"]
        result = evaluate_candidate(
            candidate,
            label="unsafe_fix",
            scenario_id=scenario.scenario_id,
        )
        self.assertFalse(result.accepted)
        self.assertTrue(result.secondary_outage)

    def test_premature_finish_is_rejected(self) -> None:
        result = evaluate_candidate(
            [
                action(
                    "finish_incident",
                    AgentRole.COMMUNICATOR,
                    {"summary": "close it"},
                )
            ],
            label="premature_finish",
            scenario_id="checkout_bad_deploy_memory_leak",
        )
        self.assertFalse(result.accepted)
        self.assertTrue(result.premature_finish)
        self.assertLess(result.final_score, 0.0)

    def test_hidden_scenarios_are_not_publicly_listed(self) -> None:
        public_ids = set(scenario_ids())
        for hidden_id in hidden_scenario_ids():
            self.assertNotIn(hidden_id, public_ids)
        self.assertGreaterEqual(len(HIDDEN_SCENARIOS), 4)

    def test_hidden_hard_case_can_be_evaluated_explicitly(self) -> None:
        scenario = HIDDEN_SCENARIOS[0]
        candidate = adversarial_candidates(scenario)["honest_control"]
        result = evaluate_candidate(
            candidate,
            label="hidden_honest_control",
            scenario_id=scenario.scenario_id,
            include_hidden_scenarios=True,
        )
        self.assertTrue(result.hidden_case)
        self.assertTrue(result.accepted)
        self.assertGreaterEqual(result.final_score, 0.7)

    def test_hidden_hardcoded_answer_is_rejected(self) -> None:
        scenario = HIDDEN_SCENARIOS[1]
        candidate = adversarial_candidates(scenario)["hardcoded_correct_no_evidence"]
        result = evaluate_candidate(
            candidate,
            label="hidden_hardcoded_correct_no_evidence",
            scenario_id=scenario.scenario_id,
            include_hidden_scenarios=True,
        )
        self.assertTrue(result.hidden_case)
        self.assertFalse(result.accepted)
        self.assertTrue(result.unsupported_root_claim)


if __name__ == "__main__":
    unittest.main()
