"""Run a local smoke test of the interactive RL action-observation loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from incident_commander_env.evaluation import action
from incident_commander_env.interactive_rl import (
    AdaptiveTaskGenerator,
    InteractiveRolloutRunner,
)
from incident_commander_env.models import AgentRole, IncidentAction
from incident_commander_env.scenarios import get_scenario


def ideal_actions(scenario_id: str) -> list[IncidentAction]:
    scenario = get_scenario(scenario_id)
    evidence = " ".join(scenario.evidence_terms)
    return [
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
                {
                    "root_cause": scenario.root_cause,
                    "confidence": 0.9,
                    "evidence": evidence,
                },
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


def main() -> None:
    generator = AdaptiveTaskGenerator(include_hidden=False)
    task = generator.sample(seed=0, difficulty="mixed")
    runner = InteractiveRolloutRunner()
    result = runner.rollout_actions(ideal_actions(task.scenario_id), task)
    generator.record_result(result)
    print(
        json.dumps(
            {
                "task": task.model_dump(mode="json"),
                "metrics": result.metrics.model_dump(mode="json"),
                "adaptive_state": generator.export_state(),
                "trajectory_preview": [
                    {
                        "turn": step.turn,
                        "tool": step.action.get("tool_name"),
                        "reward": step.reward,
                        "message": step.observation.get("message"),
                    }
                    for step in result.trajectory[:5]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
