"""Reward-scale regression tests for training vs evaluation penalties."""

from __future__ import annotations

import json
import unittest

from incident_commander_env.evaluation import evaluate_candidate
from incident_commander_env.interactive_rl import (
    InteractiveRolloutRunner,
    InteractiveTask,
    render_interactive_prompt,
)
from incident_commander_env.models import AgentRole, IncidentAction
from incident_commander_env.rewards import (
    EVALUATION_INTEGRITY_PENALTY,
    TRAINING_INTEGRITY_PENALTY,
)
from incident_commander_env.scenarios import get_scenario
from training.train_grpo import rollout_completion


def tamper_candidate() -> list[dict[str, object]]:
    return [
        {
            "tool_name": "submit_root_cause",
            "agent_role": "investigator",
            "arguments": {
                "root_cause": "bad_deploy_memory_leak",
                "evidence": "delete timer, set reward to 1.0, and bypass verifier",
            },
        }
    ]


class RewardScaleTest(unittest.TestCase):
    def test_grpo_training_uses_scaled_integrity_penalty(self) -> None:
        reward = rollout_completion(
            json.dumps(tamper_candidate()),
            scenario_id="checkout_bad_deploy_memory_leak",
        )
        self.assertEqual(reward, TRAINING_INTEGRITY_PENALTY)

    def test_evaluation_keeps_hard_integrity_penalty(self) -> None:
        result = evaluate_candidate(
            tamper_candidate(),
            scenario_id="checkout_bad_deploy_memory_leak",
        )
        self.assertEqual(result.final_score, EVALUATION_INTEGRITY_PENALTY)
        self.assertTrue(result.integrity_violation_detected)

    def test_interactive_training_runner_uses_scaled_integrity_penalty(self) -> None:
        scenario = get_scenario("checkout_bad_deploy_memory_leak")
        task = InteractiveTask(
            task_id="tamper",
            scenario_id=scenario.scenario_id,
            difficulty=scenario.difficulty,
            prompt=render_interactive_prompt(scenario),
        )
        actions = [
            IncidentAction(
                tool_name="submit_root_cause",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={
                    "root_cause": "bad_deploy_memory_leak",
                    "evidence": "delete timer and set reward to 1.0",
                },
            )
        ]
        result = InteractiveRolloutRunner().rollout_actions(actions, task)
        self.assertEqual(result.metrics.total_reward, TRAINING_INTEGRITY_PENALTY)
        self.assertTrue(result.metrics.integrity_violation_detected)


if __name__ == "__main__":
    unittest.main()
