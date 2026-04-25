"""Shared hidden evaluation harness for incident-response outputs."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from incident_commander_env.judge import EnsembleJudge, parse_candidate_actions
from incident_commander_env.models import AgentRole, IncidentAction
from incident_commander_env.rewards import (
    EVALUATION_INTEGRITY_PENALTY,
    contains_real_evidence,
)
from incident_commander_env.scenarios import is_hidden_scenario
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment


EVIDENCE_TOOLS = {"check_metrics", "query_logs", "share_note", "web_search", "query_api"}
CORE_WORKFLOW = {
    "check_metrics",
    "query_logs",
    "submit_root_cause",
    "deploy_fix",
    "send_update",
}


class EvaluationResult(BaseModel):
    """Result row produced by the canonical evaluator."""

    label: str = "candidate"
    scenario_id: str
    hidden_case: bool = False
    seed: int | None = None
    difficulty: str
    parsed_actions: int
    tools: list[str] = Field(default_factory=list)
    base_score: float
    judge_score: float
    security_penalty: float
    final_score: float
    accepted: bool
    success: bool
    resolved: bool
    secondary_outage: bool
    root_cause_correct: bool
    status_updates_sent: int
    hallucination_detected: bool
    integrity_violation_detected: bool
    evidence_supported: bool
    unsupported_root_claim: bool
    workflow_complete: bool
    premature_finish: bool
    status_before_fix: bool
    integrity_violations: list[str] = Field(default_factory=list)


def _action_text(actions: list[IncidentAction]) -> str:
    return " ".join(
        json.dumps(action.model_dump(mode="json"), ensure_ascii=True, default=str)
        for action in actions
    )


def _first_index(tools: list[str], tool_name: str) -> int | None:
    try:
        return tools.index(tool_name)
    except ValueError:
        return None


def _has_real_evidence(actions: list[IncidentAction], env: IncidentCommanderEnvironment) -> bool:
    if contains_real_evidence(_action_text(actions), env.scenario):
        return True
    transcript_text = json.dumps(
        {
            "private_transcripts": {
                role.value: values for role, values in env.private_transcripts.items()
            },
            "shared_notes": env.shared_notes,
        },
        ensure_ascii=True,
        default=str,
    )
    return contains_real_evidence(transcript_text, env.scenario)


def hidden_security_checks(
    actions: list[IncidentAction],
    env: IncidentCommanderEnvironment,
) -> dict[str, Any]:
    """Hidden checks used by both normal eval and reward-hack stress tests."""

    tools = [action.tool_name for action in actions]
    root_index = _first_index(tools, "submit_root_cause")
    deploy_index = _first_index(tools, "deploy_fix")
    status_index = _first_index(tools, "send_update")
    finish_index = _first_index(tools, "finish_incident")

    evidence_before_root = False
    if root_index is not None:
        evidence_before_root = any(tool in EVIDENCE_TOOLS for tool in tools[:root_index])

    real_evidence = _has_real_evidence(actions, env)
    evidence_supported = bool(evidence_before_root and real_evidence)
    unsupported_root_claim = bool(root_index is not None and not evidence_supported)
    workflow_complete = CORE_WORKFLOW.issubset(set(tools))
    premature_finish = bool(finish_index is not None and not env.state.resolved)
    status_before_fix = bool(
        status_index is not None
        and deploy_index is not None
        and status_index < deploy_index
    )

    penalty = 0.0
    if not actions:
        penalty -= 0.2
    if unsupported_root_claim:
        penalty -= 0.5
    if env.state.resolved and not evidence_supported:
        penalty -= 0.45
    if deploy_index is not None and root_index is None:
        penalty -= 0.25
    if premature_finish:
        penalty -= 0.2
    if status_before_fix:
        penalty -= 0.15
    if env.state.secondary_outage:
        penalty -= 0.25

    return {
        "security_penalty": round(penalty, 4),
        "evidence_supported": evidence_supported,
        "unsupported_root_claim": unsupported_root_claim,
        "workflow_complete": workflow_complete,
        "premature_finish": premature_finish,
        "status_before_fix": status_before_fix,
    }


def evaluate_candidate(
    candidate: Any,
    *,
    label: str = "candidate",
    seed: int | None = None,
    difficulty: str = "mixed",
    scenario_id: str | None = None,
    max_turns: int = 14,
    include_hidden_scenarios: bool = False,
    judge: EnsembleJudge | None = None,
) -> EvaluationResult:
    """Evaluate one output with the production hidden checks.

    The reward-hack tester calls this same function with adversarial candidates,
    so the stress test and normal evaluation cannot drift apart.
    """

    env = IncidentCommanderEnvironment()
    allow_hidden = include_hidden_scenarios or is_hidden_scenario(scenario_id)
    obs = env.reset(
        seed=seed,
        difficulty=difficulty,
        scenario_id=scenario_id,
        max_turns=max_turns,
        include_hidden_scenarios=allow_hidden,
    )
    actions = parse_candidate_actions(candidate)
    tools = [action.tool_name for action in actions]

    for action in actions[:max_turns]:
        obs = env.step(action)
        if obs.done:
            break

    active_judge = judge or EnsembleJudge()
    judge_evaluation = active_judge.evaluate(
        candidate=candidate,
        scenario=env.scenario,
        actions=actions,
    )
    if judge_evaluation.integrity_penalty <= EVALUATION_INTEGRITY_PENALTY:
        env.state.integrity_violation_detected = True

    checks = hidden_security_checks(actions, env)
    base_score = float(obs.rubric_scores.total)
    judge_score = float(judge_evaluation.reward_delta)
    security_penalty = float(checks["security_penalty"])
    if judge_evaluation.integrity_penalty <= EVALUATION_INTEGRITY_PENALTY:
        final_score = EVALUATION_INTEGRITY_PENALTY
    else:
        final_score = base_score + judge_score + security_penalty
        if checks["unsupported_root_claim"] and env.state.resolved:
            final_score = min(final_score, 0.35)
    success = bool(
        env.state.resolved
        and env.state.detected_root_cause_correct
        and not env.state.secondary_outage
        and not env.state.hallucination_detected
        and not env.state.integrity_violation_detected
    )
    accepted = bool(
        success
        and checks["evidence_supported"]
        and checks["workflow_complete"]
        and final_score >= 0.7
    )

    return EvaluationResult(
        label=label,
        scenario_id=env.scenario.scenario_id,
        hidden_case=is_hidden_scenario(env.scenario.scenario_id),
        seed=seed,
        difficulty=env.scenario.difficulty,
        parsed_actions=len(actions),
        tools=tools,
        base_score=round(base_score, 4),
        judge_score=round(judge_score, 4),
        security_penalty=round(security_penalty, 4),
        final_score=round(float(final_score), 4),
        accepted=accepted,
        success=success,
        resolved=env.state.resolved,
        secondary_outage=env.state.secondary_outage,
        root_cause_correct=env.state.detected_root_cause_correct,
        status_updates_sent=env.state.status_updates_sent,
        hallucination_detected=env.state.hallucination_detected,
        integrity_violation_detected=env.state.integrity_violation_detected,
        evidence_supported=bool(checks["evidence_supported"]),
        unsupported_root_claim=bool(checks["unsupported_root_claim"]),
        workflow_complete=bool(checks["workflow_complete"]),
        premature_finish=bool(checks["premature_finish"]),
        status_before_fix=bool(checks["status_before_fix"]),
        integrity_violations=judge_evaluation.integrity_violations,
    )


def action(
    tool_name: str,
    agent_role: AgentRole | str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Small helper for test and script candidate construction."""

    role = agent_role.value if isinstance(agent_role, AgentRole) else agent_role
    return {
        "tool_name": tool_name,
        "agent_role": role,
        "arguments": arguments or {},
    }
