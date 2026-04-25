"""Tests for interactive RL tools and rollout loop."""

from __future__ import annotations

import unittest

from incident_commander_env.evaluation import action
from incident_commander_env.interactive_rl import (
    AdaptiveTaskGenerator,
    InteractiveRolloutRunner,
    InteractiveTask,
    OnlineRLTrainer,
    render_interactive_prompt,
    scripted_interactive_policy,
)
from incident_commander_env.models import AgentRole, IncidentAction
from incident_commander_env.scenarios import HIDDEN_SCENARIOS, get_scenario, hidden_scenario_ids
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment


class InteractiveRLTest(unittest.TestCase):
    def test_sandboxed_python_tool_runs_small_calculation(self) -> None:
        env = IncidentCommanderEnvironment()
        env.reset(scenario_id="checkout_bad_deploy_memory_leak")
        obs = env.step(
            IncidentAction(
                tool_name="python_exec",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={"code": "sum([1, 2, 3])"},
            )
        )
        self.assertTrue(obs.tool_result["ok"])
        self.assertEqual(obs.tool_result["result"], 6)
        self.assertGreater(obs.reward, 0)

    def test_sandboxed_python_tool_blocks_imports(self) -> None:
        env = IncidentCommanderEnvironment()
        env.reset(scenario_id="checkout_bad_deploy_memory_leak")
        obs = env.step(
            IncidentAction(
                tool_name="python_exec",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={"code": "import os\nos.listdir('.')"},
            )
        )
        self.assertFalse(obs.tool_result["ok"])
        self.assertIn("sandbox_violation", obs.tool_result["error"])
        self.assertLess(obs.reward, 0)

    def test_search_and_api_tools_return_feedback(self) -> None:
        env = IncidentCommanderEnvironment()
        env.reset(scenario_id="checkout_bad_deploy_memory_leak")
        search = env.step(
            IncidentAction(
                tool_name="web_search",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={"query": "heap_growth checkout-api", "limit": 3},
            )
        )
        self.assertTrue(search.tool_result["evidence_found"])
        api = env.step(
            IncidentAction(
                tool_name="query_api",
                agent_role=AgentRole.INVESTIGATOR,
                arguments={"endpoint": "deployments"},
            )
        )
        self.assertEqual(api.tool_result["endpoint"], "deployments")
        self.assertGreater(len(api.tool_result["events"]), 0)

    def test_adaptive_generator_keeps_hidden_cases_out_by_default(self) -> None:
        generator = AdaptiveTaskGenerator(include_hidden=False)
        sampled = [generator.sample(seed=i, difficulty="mixed").scenario_id for i in range(20)]
        for hidden_id in hidden_scenario_ids():
            self.assertNotIn(hidden_id, sampled)

    def test_adaptive_generator_can_sample_hidden_eval_cases(self) -> None:
        generator = AdaptiveTaskGenerator(include_hidden=True)
        sampled = [generator.sample(seed=i, difficulty="hard").scenario_id for i in range(50)]
        self.assertTrue(set(sampled) & set(hidden_scenario_ids()))

    def test_hidden_prompt_masks_scenario_id(self) -> None:
        scenario = HIDDEN_SCENARIOS[0]
        prompt = render_interactive_prompt(scenario, hidden_case=True)
        self.assertIn("hidden_eval_case", prompt)
        self.assertNotIn(scenario.scenario_id, prompt)

    def test_interactive_rollout_scores_successful_action_sequence(self) -> None:
        scenario = get_scenario("checkout_bad_deploy_memory_leak")
        task = InteractiveTask(
            task_id="test:checkout",
            scenario_id=scenario.scenario_id,
            difficulty=scenario.difficulty,
            prompt=render_interactive_prompt(scenario),
            allowed_tools=[
                "check_metrics",
                "query_logs",
                "web_search",
                "query_api",
                "share_note",
                "submit_root_cause",
                "deploy_fix",
                "send_update",
                "finish_incident",
            ],
        )
        evidence = " ".join(scenario.evidence_terms)
        actions = [
            IncidentAction.model_validate(
                action("check_metrics", AgentRole.MONITOR, {"service": scenario.affected_service})
            ),
            IncidentAction.model_validate(
                action("query_logs", AgentRole.INVESTIGATOR, {"service": scenario.affected_service})
            ),
            IncidentAction.model_validate(
                action("web_search", AgentRole.INVESTIGATOR, {"query": scenario.affected_service})
            ),
            IncidentAction.model_validate(
                action("query_api", AgentRole.INVESTIGATOR, {"endpoint": "deployments"})
            ),
            IncidentAction.model_validate(
                action("share_note", AgentRole.INVESTIGATOR, {"note": evidence})
            ),
            IncidentAction.model_validate(
                action(
                    "submit_root_cause",
                    AgentRole.INVESTIGATOR,
                    {"root_cause": scenario.root_cause, "evidence": evidence},
                )
            ),
            IncidentAction.model_validate(
                action("deploy_fix", AgentRole.REMEDIATOR, {"fix_id": scenario.canonical_fix_id})
            ),
            IncidentAction.model_validate(
                action(
                    "send_update",
                    AgentRole.COMMUNICATOR,
                    {
                        "message": (
                            f"{scenario.affected_service} impact is mitigated. "
                            f"Root cause was {scenario.root_cause}. Monitoring continues."
                        )
                    },
                )
            ),
            IncidentAction.model_validate(
                action("finish_incident", AgentRole.COMMUNICATOR, {"summary": "done"})
            ),
        ]
        result = InteractiveRolloutRunner().rollout_actions(actions, task)
        self.assertTrue(result.metrics.accepted)
        self.assertTrue(result.metrics.resolved)
        self.assertGreaterEqual(result.metrics.total_reward, 0.7)

    def test_online_trainer_collects_rollouts_and_calls_update_hook(self) -> None:
        calls = []

        def update_fn(results):
            calls.append(len(results))
            return {"received": len(results)}

        trainer = OnlineRLTrainer(
            policy=scripted_interactive_policy,
            task_generator=AdaptiveTaskGenerator(include_hidden=False),
            update_fn=update_fn,
            batch_size=2,
        )
        iteration = trainer.train_iteration(iteration=0)
        self.assertEqual(calls, [2])
        self.assertEqual(iteration.batch_size, 2)
        self.assertEqual(iteration.update_info["received"], 2)
        self.assertIn("attempts_by_difficulty", iteration.curriculum_state)


if __name__ == "__main__":
    unittest.main()
