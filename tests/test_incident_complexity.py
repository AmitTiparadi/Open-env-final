"""Tests for red herrings, causal chains, and communication accuracy."""

from __future__ import annotations

import unittest

from incident_commander_env.models import AgentRole, IncidentAction
from incident_commander_env.scenarios import get_scenario, render_logs, render_metrics
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment


class IncidentComplexityTest(unittest.TestCase):
    def test_red_herring_logs_and_metrics_are_injected(self) -> None:
        scenario = get_scenario("orders_db_connection_pool")
        logs = render_logs(scenario, service="checkout-api", limit=10)
        metrics = render_metrics(scenario, service="checkout-api")
        self.assertTrue(any("heap_growth" in line for line in logs))
        self.assertIn("memory_percent", metrics)

    def test_chasing_red_herring_is_penalized(self) -> None:
        env = IncidentCommanderEnvironment()
        env.reset(scenario_id="orders_db_connection_pool")
        obs = env.step(
            IncidentAction(
                tool_name="submit_root_cause",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={
                    "root_cause": "checkout memory leak",
                    "evidence": "checkout-api heap_growth warning looked suspicious",
                },
            )
        )
        self.assertTrue(env.state.red_herring_chased)
        self.assertLess(obs.reward, -0.3)
        self.assertLessEqual(obs.rubric_scores.red_herring_penalty, -0.15)

    def test_causal_chain_evidence_gets_reward_credit(self) -> None:
        env = IncidentCommanderEnvironment()
        scenario = get_scenario("checkout_bad_deploy_memory_leak")
        env.reset(scenario_id=scenario.scenario_id)
        evidence = " ".join(scenario.evidence_terms + scenario.causal_chain)
        obs = env.step(
            IncidentAction(
                tool_name="submit_root_cause",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={
                    "root_cause": scenario.root_cause,
                    "evidence": evidence,
                },
            )
        )
        self.assertTrue(env.state.causal_chain_traced)
        self.assertGreater(obs.reward, 0.4)
        self.assertEqual(obs.rubric_scores.causal_chain, 0.08)

    def test_communicator_false_claim_before_fix_is_penalized(self) -> None:
        env = IncidentCommanderEnvironment()
        env.reset(scenario_id="auth_tls_cert_expiry")
        obs = env.step(
            IncidentAction(
                tool_name="send_update",
                agent_role=AgentRole.COMMUNICATOR,
                arguments={
                    "message": (
                        "auth-service is resolved. Root cause was oauth database timeout."
                    )
                },
            )
        )
        self.assertTrue(env.state.communication_mismatch_detected)
        self.assertTrue(env.state.hallucination_detected)
        self.assertLess(obs.reward, 0.0)
        self.assertLessEqual(obs.rubric_scores.communication_accuracy, -0.12)


if __name__ == "__main__":
    unittest.main()
