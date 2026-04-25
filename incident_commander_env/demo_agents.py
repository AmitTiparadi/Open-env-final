"""Baseline and scripted policies for demos and quick evaluations."""

from __future__ import annotations

import random
from typing import Any, Optional

from incident_commander_env.models import AgentRole, IncidentAction
from incident_commander_env.rewards import ROOT_CAUSE_ALIASES
from incident_commander_env.scenarios import SCENARIOS
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment


def infer_root_cause(text: str) -> str:
    normalized = text.lower()
    for root, aliases in ROOT_CAUSE_ALIASES.items():
        if root in normalized or any(alias.lower() in normalized for alias in aliases):
            return root
    return "unknown"


def fix_for_root_cause(root_cause: str) -> str:
    for scenario in SCENARIOS:
        if scenario.root_cause == root_cause:
            return scenario.canonical_fix_id
    return "restart_everything"


def run_scripted_response(
    seed: Optional[int] = None,
    difficulty: str = "mixed",
    scenario_id: Optional[str] = None,
) -> dict[str, Any]:
    env = IncidentCommanderEnvironment()
    obs = env.reset(seed=seed, difficulty=difficulty, scenario_id=scenario_id)
    transcript: list[dict[str, Any]] = [obs.model_dump(mode="json")]

    service = obs.metadata["affected_service"]
    actions = [
        IncidentAction(
            tool_name="check_metrics",
            agent_role=AgentRole.MONITOR,
            arguments={"service": service},
        ),
        IncidentAction(
            tool_name="query_logs",
            agent_role=AgentRole.INVESTIGATOR,
            arguments={"service": service, "limit": 5},
        ),
    ]
    evidence_text = ""
    for action in actions:
        obs = env.step(action)
        transcript.append(obs.model_dump(mode="json"))
        evidence_text += " " + str(obs.tool_result)

    root = infer_root_cause(evidence_text)
    obs = env.step(
        IncidentAction(
            tool_name="share_note",
            agent_role=AgentRole.INVESTIGATOR,
            arguments={"note": f"Evidence points to {root}: {evidence_text[:350]}"},
        )
    )
    transcript.append(obs.model_dump(mode="json"))
    obs = env.step(
        IncidentAction(
            tool_name="submit_root_cause",
            agent_role=AgentRole.INVESTIGATOR,
            arguments={
                "root_cause": root,
                "confidence": 0.82,
                "evidence": evidence_text,
            },
        )
    )
    transcript.append(obs.model_dump(mode="json"))
    obs = env.step(
        IncidentAction(
            tool_name="deploy_fix",
            agent_role=AgentRole.REMEDIATOR,
            arguments={"fix_id": fix_for_root_cause(root)},
        )
    )
    transcript.append(obs.model_dump(mode="json"))
    obs = env.step(
        IncidentAction(
            tool_name="send_update",
            agent_role=AgentRole.COMMUNICATOR,
            arguments={
                "message": (
                    f"{service} impact is mitigated. Root cause was {root}; "
                    "we applied the targeted remediation and are monitoring. "
                    "Next update after stability checks."
                )
            },
        )
    )
    transcript.append(obs.model_dump(mode="json"))
    obs = env.step(
        IncidentAction(
            tool_name="finish_incident",
            agent_role=AgentRole.COMMUNICATOR,
            arguments={"summary": "Mitigated and communicated."},
        )
    )
    transcript.append(obs.model_dump(mode="json"))
    return {
        "scenario_id": env.state.scenario_id,
        "final_score": obs.rubric_scores.total,
        "resolved": env.state.resolved,
        "transcript": transcript,
    }


def run_random_response(
    seed: Optional[int] = None,
    difficulty: str = "mixed",
) -> dict[str, Any]:
    rng = random.Random(seed)
    env = IncidentCommanderEnvironment()
    obs = env.reset(seed=seed, difficulty=difficulty)
    transcript = [obs.model_dump(mode="json")]
    services = ["checkout-api", "orders-api", "profile-service", "auth-service", "edge-gateway"]
    fix_ids = [
        "rollback_checkout_api_v42",
        "increase_orders_pool_and_restart_workers",
        "restore_profile_cache_ttl_singleflight",
        "rotate_auth_service_certificate",
        "restart_everything",
        "flush_all_profile_cache",
    ]
    for _ in range(8):
        action = rng.choice(
            [
                IncidentAction(
                    tool_name="check_metrics",
                    agent_role=AgentRole.MONITOR,
                    arguments={"service": rng.choice(services)},
                ),
                IncidentAction(
                    tool_name="query_logs",
                    agent_role=AgentRole.INVESTIGATOR,
                    arguments={"service": rng.choice(services)},
                ),
                IncidentAction(
                    tool_name="submit_root_cause",
                    agent_role=AgentRole.INVESTIGATOR,
                    arguments={"root_cause": rng.choice(["dns outage", "memory leak", "unknown"])},
                ),
                IncidentAction(
                    tool_name="deploy_fix",
                    agent_role=AgentRole.REMEDIATOR,
                    arguments={"fix_id": rng.choice(fix_ids)},
                ),
                IncidentAction(
                    tool_name="send_update",
                    agent_role=AgentRole.COMMUNICATOR,
                    arguments={"message": "We are investigating and will update soon."},
                ),
            ]
        )
        obs = env.step(action)
        transcript.append(obs.model_dump(mode="json"))
        if obs.done:
            break
    return {
        "scenario_id": env.state.scenario_id,
        "final_score": obs.rubric_scores.total,
        "resolved": env.state.resolved,
        "transcript": transcript,
    }
