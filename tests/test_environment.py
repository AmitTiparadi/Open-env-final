"""Smoke tests for Incident Commander."""

from __future__ import annotations

import unittest

from incident_commander_env.models import AgentRole, IncidentAction
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment


class IncidentCommanderEnvironmentTest(unittest.TestCase):
    def test_reset_returns_alerts_and_budget(self) -> None:
        env = IncidentCommanderEnvironment()
        obs = env.reset(seed=0, difficulty="easy")
        self.assertGreater(len(obs.visible_alerts), 0)
        self.assertEqual(obs.turn_budget_remaining, 14)
        self.assertFalse(obs.done)

    def test_role_permissions_are_enforced(self) -> None:
        env = IncidentCommanderEnvironment()
        env.reset(seed=0)
        obs = env.step(
            IncidentAction(
                tool_name="deploy_fix",
                agent_role=AgentRole.MONITOR,
                arguments={"fix_id": "rollback_checkout_api_v42"},
            )
        )
        self.assertLess(obs.reward or 0.0, 0.0)
        self.assertIn("not allowed", obs.message)

    def test_correct_workflow_scores_high(self) -> None:
        env = IncidentCommanderEnvironment()
        env.reset(scenario_id="checkout_bad_deploy_memory_leak")
        env.step(
            IncidentAction(
                tool_name="query_logs",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={"service": "checkout-api"},
            )
        )
        env.step(
            IncidentAction(
                tool_name="share_note",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={"note": "checkout-api:v42 heap_growth_mb gc_pause_ms evidence"},
            )
        )
        obs = env.step(
            IncidentAction(
                tool_name="submit_root_cause",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={
                    "root_cause": "bad deploy memory leak",
                    "evidence": "checkout-api:v42 heap_growth_mb gc_pause_ms",
                },
            )
        )
        self.assertGreaterEqual(obs.rubric_scores.root_cause, 0.4)
        env.step(
            IncidentAction(
                tool_name="deploy_fix",
                agent_role=AgentRole.REMEDIATOR,
                arguments={"fix_id": "rollback_checkout_api_v42"},
            )
        )
        env.step(
            IncidentAction(
                tool_name="send_update",
                agent_role=AgentRole.COMMUNICATOR,
                arguments={
                    "message": (
                        "checkout-api user impact is mitigated after rollback. "
                        "Root cause was bad deploy memory leak. Next update after monitoring."
                    )
                },
            )
        )
        obs = env.step(
            IncidentAction(
                tool_name="finish_incident",
                agent_role=AgentRole.COMMUNICATOR,
                arguments={"summary": "done"},
            )
        )
        self.assertTrue(obs.done)
        self.assertTrue(env.state.resolved)
        self.assertGreaterEqual(obs.rubric_scores.total, 0.8)

    def test_hallucinated_root_cause_penalized(self) -> None:
        env = IncidentCommanderEnvironment()
        env.reset(scenario_id="checkout_bad_deploy_memory_leak")
        obs = env.step(
            IncidentAction(
                tool_name="submit_root_cause",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={"root_cause": "DNS outage caused a network partition"},
            )
        )
        self.assertLess(obs.reward or 0.0, -0.3)
        self.assertTrue(env.state.hallucination_detected)


if __name__ == "__main__":
    unittest.main()
