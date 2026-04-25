"""Regression tests for GRPO completion parsing and reward shaping."""

from __future__ import annotations

import json

from training.train_grpo import parse_actions, rollout_completion


def test_parse_actions_accepts_aliases_and_fenced_json() -> None:
    completion = """```json
{"actions": [
  {"tool": "metrics", "role": "monitor", "args": {"service": "checkout-api"}},
  {"tool": "logs", "role": "investigator", "args": {"service": "checkout-api"}},
  {"tool": "root_cause", "role": "investigator", "args": {"root_cause": "bad_deploy_memory_leak", "evidence": "checkout-api:v42 heap_growth_mb gc_pause_ms"}}
]}
```"""

    actions = parse_actions(completion)

    assert [action.tool_name for action in actions] == [
        "check_metrics",
        "query_logs",
        "submit_root_cause",
    ]


def test_parse_actions_accepts_openenv_style_role_in_arguments() -> None:
    completion = json.dumps(
        {
            "actions": [
                {
                    "type": "call_tool",
                    "tool_name": "submit_root_cause",
                    "arguments": {
                        "agent_role": "investigator",
                        "root_cause": "bad_deploy_memory_leak",
                        "evidence": "checkout-api:v42 heap_growth_mb gc_pause_ms",
                    },
                }
            ]
        }
    )

    actions = parse_actions(completion)

    assert len(actions) == 1
    assert actions[0].agent_role == "investigator"
    assert "agent_role" not in actions[0].arguments


def test_rollout_rewards_vary_for_format_quality() -> None:
    valid_completion = json.dumps(
        [
            {
                "tool_name": "check_metrics",
                "agent_role": "monitor",
                "arguments": {"service": "checkout-api"},
            },
            {
                "tool_name": "query_logs",
                "agent_role": "investigator",
                "arguments": {"service": "checkout-api"},
            },
            {
                "tool_name": "submit_root_cause",
                "agent_role": "investigator",
                "arguments": {
                    "root_cause": "bad_deploy_memory_leak",
                    "evidence": "checkout-api:v42 heap_growth_mb gc_pause_ms",
                },
            },
        ]
    )
    prose_completion = "I will check_metrics and query_logs before submit_root_cause."

    valid_reward = rollout_completion(
        valid_completion,
        scenario_id="checkout_bad_deploy_memory_leak",
    )
    prose_reward = rollout_completion(
        prose_completion,
        scenario_id="checkout_bad_deploy_memory_leak",
    )

    assert valid_reward > 0.5
    assert prose_reward > -0.05
    assert valid_reward != prose_reward
