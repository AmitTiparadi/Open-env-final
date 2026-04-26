"""Tests for structured execution logs and prompt updates."""

from __future__ import annotations

import unittest

from incident_commander_env.dynamic_prompting import build_prompt_update
from incident_commander_env.execution_logging import ERROR_IMPORT, ERROR_LOGIC
from incident_commander_env.models import AgentRole, IncidentAction
from incident_commander_env.scenarios import HIDDEN_SCENARIOS
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment


class ExecutionLoggingTest(unittest.TestCase):
    def test_python_failures_are_structured_and_prompted(self) -> None:
        env = IncidentCommanderEnvironment()
        env.reset(scenario_id="checkout_bad_deploy_memory_leak")
        obs = env.step(
            IncidentAction(
                tool_name="python_exec",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={"code": "import os\nos.listdir('.')"},
            )
        )
        latest = obs.metadata["latest_execution_log"]
        self.assertFalse(latest["success"])
        self.assertEqual(latest["exit_code"], 1)
        self.assertEqual(latest["error"]["type"], ERROR_IMPORT)
        self.assertIn("stderr", latest)
        prompt_update = obs.metadata["prompt_update"]
        self.assertIn("import_error", prompt_update["failure_counts"])
        self.assertTrue(prompt_update["retry_guidance"])

    def test_logic_failures_are_classified_without_sentry_exposure(self) -> None:
        env = IncidentCommanderEnvironment()
        env.reset(scenario_id="checkout_bad_deploy_memory_leak")
        obs = env.step(
            IncidentAction(
                tool_name="deploy_fix",
                agent_role=AgentRole.MONITOR,
                arguments={"fix_id": "rollback_checkout_api_v42"},
            )
        )
        latest = obs.metadata["latest_execution_log"]
        self.assertEqual(latest["error"]["type"], ERROR_LOGIC)
        self.assertFalse(latest["success"])
        self.assertNotIn("sentry", str(obs.metadata).lower())

    def test_hidden_case_execution_context_masks_real_id(self) -> None:
        hidden = HIDDEN_SCENARIOS[0]
        env = IncidentCommanderEnvironment()
        env.reset(scenario_id=hidden.scenario_id, include_hidden_scenarios=True)
        obs = env.step(
            IncidentAction(
                tool_name="check_metrics",
                agent_role=AgentRole.MONITOR,
                arguments={"service": hidden.affected_service},
            )
        )
        latest = obs.metadata["latest_execution_log"]
        self.assertEqual(latest["context"]["scenario_id"], "hidden_eval_case")
        self.assertNotIn(hidden.scenario_id, str(obs.metadata["prompt_update"]))

    def test_prompt_update_summarizes_recurring_failures(self) -> None:
        env = IncidentCommanderEnvironment()
        env.reset(scenario_id="checkout_bad_deploy_memory_leak")
        for _ in range(2):
            env.step(
                IncidentAction(
                    tool_name="python_exec",
                    agent_role=AgentRole.INVESTIGATOR,
                    arguments={"code": "import requests"},
                )
            )
        update = build_prompt_update(env.execution_logs)
        self.assertIn("import_error", update.failure_counts)
        self.assertTrue(update.recurring_patterns)


if __name__ == "__main__":
    unittest.main()
