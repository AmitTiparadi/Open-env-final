"""Core incident simulator and OpenEnv-compatible step loop."""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from incident_commander_env.agent_config import (
    AGENT_MODEL_BY_ROLE,
    DEFAULT_JUDGE_ENSEMBLE_SIZE,
)
from incident_commander_env.compat import OpenEnvEnvironment
from incident_commander_env.judge import EnsembleJudge, parse_candidate_actions
from incident_commander_env.models import (
    AgentRole,
    IncidentAction,
    IncidentObservation,
    IncidentState,
    RubricBreakdown,
    ToolSpec,
)
from incident_commander_env.rewards import (
    best_evidence_overlap,
    contains_real_evidence,
    detects_false_root_cause,
    matches_root_cause,
    score_status_update,
    speed_bonus,
)
from incident_commander_env.scenarios import (
    IncidentScenario,
    generate_scenario,
    get_scenario,
    render_logs,
    render_metrics,
    is_hidden_scenario,
    scenario_ids,
)


INCIDENT_RESPONSE_ROLES = [
    AgentRole.MONITOR,
    AgentRole.INVESTIGATOR,
    AgentRole.REMEDIATOR,
    AgentRole.COMMUNICATOR,
]


class IncidentCommanderEnvironment(
    OpenEnvEnvironment[IncidentAction, IncidentObservation, IncidentState]
):
    """Multi-agent incident response environment.

    The environment is deliberately role-scoped: agents share a scratchpad, but
    tool outputs are returned only to the role that called the tool.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state = IncidentState(episode_id=str(uuid4()), step_count=0)
        self.scenario: IncidentScenario = generate_scenario(seed=0)
        self.shared_notes: list[str] = []
        self.private_transcripts: dict[AgentRole, list[dict[str, Any]]] = {
            role: [] for role in AgentRole
        }
        self.status_update_scores: list[float] = []
        self.status_update_reasons: list[list[str]] = []
        self.judge = EnsembleJudge()
        self.judge_evaluations: list[dict[str, Any]] = []
        self.tool_history: list[dict[str, Any]] = []
        self.max_turns = 14

    @property
    def state(self) -> IncidentState:
        return self._state

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: str = "easy",
        scenario_id: Optional[str] = None,
        preferred_root_cause: Optional[str] = None,
        max_turns: int = 14,
        include_hidden_scenarios: bool = False,
        **_: Any,
    ) -> IncidentObservation:
        self.max_turns = max_turns
        self.scenario = (
            get_scenario(scenario_id, include_hidden=include_hidden_scenarios)
            if scenario_id
            else generate_scenario(
                seed=seed,
                difficulty=difficulty,
                preferred_root_cause=preferred_root_cause,
                include_hidden=include_hidden_scenarios,
            )
        )
        self.shared_notes = []
        self.private_transcripts = {role: [] for role in AgentRole}
        self.status_update_scores = []
        self.status_update_reasons = []
        self.judge_evaluations = []
        self.tool_history = []
        self._state = IncidentState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            scenario_id=self.scenario.scenario_id,
            difficulty=self.scenario.difficulty,
            max_turns=max_turns,
            detected_root_cause=None,
            detected_root_cause_correct=False,
            deployed_fix_id=None,
            resolved=False,
            secondary_outage=False,
            status_updates_sent=0,
            hallucination_detected=False,
            integrity_violation_detected=False,
            shared_note_count=0,
            judge_evaluations=0,
            last_actor=None,
        )
        return self._observation(
            message=(
                "Incident started. Use role-scoped tools to triage, share "
                "evidence, remediate, and communicate."
            ),
            visible_alerts=list(self.scenario.alerts),
            reward=0.0,
            role=None,
        )

    def step(
        self,
        action: IncidentAction,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> IncidentObservation:
        del timeout_s
        if self._state.step_count >= self.max_turns or self._state.secondary_outage:
            return self._observation(
                message="Episode already terminal.",
                reward=0.0,
                done=True,
                role=getattr(action, "agent_role", None),
            )

        action = self._coerce_action(action)
        self._state.step_count += 1
        self._state.last_actor = action.agent_role
        handler = {
            "list_tools": self._tool_list_tools,
            "query_logs": self._tool_query_logs,
            "check_metrics": self._tool_check_metrics,
            "share_note": self._tool_share_note,
            "submit_root_cause": self._tool_submit_root_cause,
            "deploy_fix": self._tool_deploy_fix,
            "send_update": self._tool_send_update,
            "finish_incident": self._tool_finish_incident,
            "judge_response": self._tool_judge_response,
        }.get(action.tool_name)

        if handler is None:
            observation = self._observation(
                message=f"Unknown tool: {action.tool_name}",
                reward=-0.03,
                role=action.agent_role,
                tool_result={"error": "unknown_tool", "known_tools": list(self.tool_names())},
            )
        elif not self._role_allowed(action.tool_name, action.agent_role):
            observation = self._observation(
                message=f"{action.agent_role.value} is not allowed to use {action.tool_name}.",
                reward=-0.02,
                role=action.agent_role,
                tool_result={"error": "role_not_allowed"},
            )
        else:
            observation = handler(action)

        if self._state.step_count >= self.max_turns and not observation.done:
            observation.done = True
            observation.message = f"{observation.message} Turn budget exhausted."
            if not self._state.resolved:
                observation.reward = float(observation.reward or 0.0) - 0.1

        self.tool_history.append(
            {
                "step": self._state.step_count,
                "role": action.agent_role.value,
                "tool": action.tool_name,
                "reward": observation.reward,
                "done": observation.done,
            }
        )
        return observation

    def tool_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.tool_specs())

    def tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="list_tools",
                description="List available incident-response tools.",
                allowed_roles=list(AgentRole),
                input_schema={},
            ),
            ToolSpec(
                name="check_metrics",
                description="Inspect noisy metrics for a service.",
                allowed_roles=[AgentRole.MONITOR, AgentRole.INVESTIGATOR],
                input_schema={
                    "service": "service name, for example checkout-api",
                    "metric": "optional metric name",
                },
            ),
            ToolSpec(
                name="query_logs",
                description="Query service logs with optional search terms.",
                allowed_roles=[AgentRole.INVESTIGATOR, AgentRole.REMEDIATOR],
                input_schema={
                    "service": "service name",
                    "query": "optional search terms",
                    "limit": "max log lines",
                },
            ),
            ToolSpec(
                name="share_note",
                description="Publish evidence or a hypothesis to the shared scratchpad.",
                allowed_roles=INCIDENT_RESPONSE_ROLES,
                input_schema={"note": "short evidence-backed note"},
            ),
            ToolSpec(
                name="submit_root_cause",
                description="Submit the team's current root-cause hypothesis.",
                allowed_roles=[AgentRole.INVESTIGATOR, AgentRole.REMEDIATOR],
                input_schema={
                    "root_cause": "canonical or natural-language root cause",
                    "confidence": "0.0 to 1.0",
                    "evidence": "evidence supporting the claim",
                },
            ),
            ToolSpec(
                name="deploy_fix",
                description="Deploy a remediation by fix_id.",
                allowed_roles=[AgentRole.REMEDIATOR],
                input_schema={"fix_id": "candidate remediation id"},
            ),
            ToolSpec(
                name="send_update",
                description="Send a stakeholder status update.",
                allowed_roles=[AgentRole.COMMUNICATOR],
                input_schema={"message": "concise status update", "audience": "optional"},
            ),
            ToolSpec(
                name="finish_incident",
                description="End the episode after fix and communication are complete.",
                allowed_roles=[AgentRole.REMEDIATOR, AgentRole.COMMUNICATOR],
                input_schema={"summary": "final incident summary"},
            ),
            ToolSpec(
                name="judge_response",
                description=(
                    "Evaluate a candidate incident response with a 10-call "
                    "LLM-as-a-Judge ensemble and hard anti-reward-hacking checks."
                ),
                allowed_roles=[AgentRole.JUDGE],
                input_schema={
                    "candidate_response": "JSON tool-call list or natural-language response",
                    "ensemble_size": "optional judge calls, defaults to 10",
                },
            ),
        ]

    def _tool_list_tools(self, action: IncidentAction) -> IncidentObservation:
        tools = [
            spec
            for spec in self.tool_specs()
            if action.agent_role in spec.allowed_roles
        ]
        return self._observation(
            message=f"{action.agent_role.value} can use {len(tools)} tools.",
            reward=0.0,
            role=action.agent_role,
            tool_result={"tools": [tool.model_dump(mode="json") for tool in tools]},
        )

    def _tool_query_logs(self, action: IncidentAction) -> IncidentObservation:
        service = str(action.arguments.get("service") or self.scenario.affected_service)
        query = str(action.arguments.get("query") or "")
        limit = int(action.arguments.get("limit") or 5)
        logs = render_logs(self.scenario, service=service, query=query, limit=limit)
        result = {"service": service, "query": query, "logs": logs}
        self.private_transcripts[action.agent_role].append(result)
        reward = 0.015 if any(contains_real_evidence(line, self.scenario) for line in logs) else 0.0
        return self._observation(
            message=f"Returned {len(logs)} log lines for {service}.",
            reward=reward,
            role=action.agent_role,
            private_output=result,
        )

    def _tool_check_metrics(self, action: IncidentAction) -> IncidentObservation:
        service = str(action.arguments.get("service") or self.scenario.affected_service)
        metric = str(action.arguments.get("metric") or "")
        metrics = render_metrics(self.scenario, service=service, metric=metric)
        result = {
            "service": service,
            "metric": metric or "all",
            "series": metrics,
            "hint": self._metric_hint(metrics),
        }
        self.private_transcripts[action.agent_role].append(result)
        reward = 0.015 if service in {self.scenario.affected_service, "edge-gateway"} else 0.0
        return self._observation(
            message=f"Returned metrics for {service}.",
            reward=reward,
            role=action.agent_role,
            private_output=result,
        )

    def _tool_share_note(self, action: IncidentAction) -> IncidentObservation:
        note = str(action.arguments.get("note") or "").strip()
        if not note:
            return self._observation(
                message="No note was shared.",
                reward=-0.02,
                role=action.agent_role,
                tool_result={"error": "empty_note"},
            )
        self.shared_notes.append(f"{action.agent_role.value}: {note}")
        self._state.shared_note_count = len(self.shared_notes)
        hallucinated = detects_false_root_cause(note, self.scenario)
        if hallucinated:
            self._state.hallucination_detected = True
        reward = 0.02 if contains_real_evidence(note, self.scenario) else 0.005
        if hallucinated:
            reward -= 0.08
        return self._observation(
            message="Shared note added.",
            reward=reward,
            role=action.agent_role,
            tool_result={"accepted": True, "hallucinated": hallucinated},
        )

    def _tool_submit_root_cause(self, action: IncidentAction) -> IncidentObservation:
        root_cause = str(action.arguments.get("root_cause") or "")
        evidence = str(action.arguments.get("evidence") or "")
        correct = matches_root_cause(f"{root_cause} {evidence}", self.scenario)
        hallucinated = detects_false_root_cause(f"{root_cause} {evidence}", self.scenario)
        self._state.detected_root_cause = root_cause or None
        self._state.detected_root_cause_correct = correct
        if hallucinated:
            self._state.hallucination_detected = True
        if correct:
            reward = 0.4
            message = "Correct root cause submitted."
        else:
            reward = -0.3
            message = "Root cause submission is not supported by the incident evidence."
        if hallucinated:
            reward -= 0.1
        return self._observation(
            message=message,
            reward=reward,
            role=action.agent_role,
            tool_result={
                "correct": correct,
                "canonical_root_cause": self.scenario.root_cause if correct else None,
                "hallucinated": hallucinated,
            },
        )

    def _tool_deploy_fix(self, action: IncidentAction) -> IncidentObservation:
        fix_id = str(action.arguments.get("fix_id") or "")
        self._state.deployed_fix_id = fix_id or None
        if fix_id == self.scenario.canonical_fix_id or fix_id in self.scenario.safe_fix_ids:
            self._state.resolved = True
            reward = 0.3
            message = "Fix deployed and incident is mitigated."
            result = {"resolved": True, "secondary_outage": False}
        elif fix_id in self.scenario.dangerous_fix_ids:
            self._state.secondary_outage = True
            reward = -0.2
            message = "Fix caused a secondary outage."
            result = {"resolved": False, "secondary_outage": True}
        else:
            reward = -0.05
            message = "Fix did not address the incident."
            result = {"resolved": False, "secondary_outage": False}
        return self._observation(
            message=message,
            reward=reward,
            role=action.agent_role,
            tool_result={"fix_id": fix_id, **result},
            done=self._state.secondary_outage,
        )

    def _tool_send_update(self, action: IncidentAction) -> IncidentObservation:
        message = str(action.arguments.get("message") or "")
        score, hallucinated, reasons = score_status_update(
            message=message,
            scenario=self.scenario,
            root_cause_correct=self._state.detected_root_cause_correct,
            resolved=self._state.resolved,
        )
        self.status_update_scores.append(score)
        self.status_update_reasons.append(reasons)
        self._state.status_updates_sent += 1
        if hallucinated:
            self._state.hallucination_detected = True
        return self._observation(
            message="Stakeholder update sent.",
            reward=score - (0.1 if hallucinated else 0.0),
            role=action.agent_role,
            tool_result={
                "score": score,
                "reasons": reasons,
                "hallucinated": hallucinated,
            },
        )

    def _tool_finish_incident(self, action: IncidentAction) -> IncidentObservation:
        del action
        done_reward = speed_bonus(
            self._state.step_count,
            self.max_turns,
            self._state.resolved and not self._state.secondary_outage,
        )
        if not self._state.resolved:
            done_reward -= 0.1
            message = "Incident closed before mitigation."
        elif self._state.status_updates_sent == 0:
            done_reward -= 0.05
            message = "Incident mitigated but closed without stakeholder update."
        else:
            message = "Incident closed with mitigation and stakeholder update."
        return self._observation(
            message=message,
            reward=done_reward,
            done=True,
            role=self._state.last_actor,
            tool_result={"speed_bonus": max(0.0, done_reward)},
        )

    def _tool_judge_response(self, action: IncidentAction) -> IncidentObservation:
        candidate = action.arguments.get("candidate_response")
        if candidate is None:
            candidate = action.arguments.get("response") or action.arguments.get("output") or []
        ensemble_size = int(action.arguments.get("ensemble_size") or DEFAULT_JUDGE_ENSEMBLE_SIZE)
        parsed_actions = parse_candidate_actions(candidate)
        evaluation = self.judge.evaluate(
            candidate=candidate,
            scenario=self.scenario,
            actions=parsed_actions,
            ensemble_size=ensemble_size,
        )
        dumped = evaluation.model_dump(mode="json")
        self.judge_evaluations.append(dumped)
        self._state.judge_evaluations = len(self.judge_evaluations)
        if evaluation.integrity_penalty <= -100:
            self._state.integrity_violation_detected = True
            reward = -100.0
            message = "Judge detected a hard integrity violation."
        else:
            reward = evaluation.reward_delta
            message = "Judge completed ensemble evaluation."
        return self._observation(
            message=message,
            reward=reward,
            role=action.agent_role,
            tool_result={
                **dumped,
                "parsed_action_count": len(parsed_actions),
            },
        )

    def _observation(
        self,
        message: str,
        reward: float,
        done: bool = False,
        role: Optional[AgentRole] = None,
        visible_alerts: Optional[list[str]] = None,
        tool_result: Optional[dict[str, Any]] = None,
        private_output: Optional[dict[str, Any]] = None,
    ) -> IncidentObservation:
        rubric = self._rubric_breakdown(process_reward=reward)
        return IncidentObservation(
            message=message,
            reward=round(float(reward), 4),
            done=bool(done),
            role=role,
            visible_alerts=visible_alerts or [],
            shared_notes=list(self.shared_notes),
            tool_result=tool_result or private_output or {},
            private_output_for=role if private_output else None,
            rubric_scores=rubric,
            available_tools=self.tool_specs(),
            turn_budget_remaining=max(0, self.max_turns - self._state.step_count),
            metadata={
                "scenario_id": self._public_scenario_id(),
                "difficulty": self.scenario.difficulty,
                "affected_service": self.scenario.affected_service,
                "scenario_ids": scenario_ids(),
                "tool_history": self.tool_history[-5:],
                "agent_models": AGENT_MODEL_BY_ROLE,
                "judge_ensemble_size": DEFAULT_JUDGE_ENSEMBLE_SIZE,
            },
        )

    def _rubric_breakdown(self, process_reward: float = 0.0) -> RubricBreakdown:
        root = 0.4 if self._state.detected_root_cause_correct else 0.0
        if self._state.resolved and not self._state.secondary_outage:
            fix = 0.3
        elif self._state.secondary_outage:
            fix = -0.2
        else:
            fix = 0.0
        update = max(self.status_update_scores) if self.status_update_scores else 0.0
        hallucination = -0.3 if self._state.hallucination_detected else 0.0
        integrity = -100.0 if self._state.integrity_violation_detected else 0.0
        judge_score = max(
            (float(item.get("reward_delta", 0.0)) for item in self.judge_evaluations),
            default=0.0,
        )
        speed = speed_bonus(
            self._state.step_count,
            self.max_turns,
            self._state.resolved and not self._state.secondary_outage,
        )
        process = best_evidence_overlap(self.shared_notes, self.scenario) + process_reward
        total = root + fix + update + speed + hallucination + judge_score + integrity
        return RubricBreakdown(
            root_cause=round(root, 4),
            fix_safety=round(fix, 4),
            status_update=round(update, 4),
            speed_bonus=round(speed, 4),
            hallucination_penalty=round(hallucination, 4),
            process_reward=round(process, 4),
            judge_score=round(judge_score, 4),
            integrity_penalty=round(integrity, 4),
            total=round(total, 4),
        )

    def _public_scenario_id(self) -> str:
        if is_hidden_scenario(self.scenario.scenario_id):
            return "hidden_eval_case"
        return self.scenario.scenario_id

    def _metric_hint(self, metrics: dict[str, list[float]]) -> str:
        if not metrics:
            return "No metrics found for service."
        rising = [
            name
            for name, values in metrics.items()
            if len(values) >= 2 and values[-1] > values[0] * 1.5
        ]
        falling = [
            name
            for name, values in metrics.items()
            if len(values) >= 2 and values[-1] < values[0] * 0.75
        ]
        parts = []
        if rising:
            parts.append(f"rising: {', '.join(rising)}")
        if falling:
            parts.append(f"falling: {', '.join(falling)}")
        return "; ".join(parts) or "No sharp movement detected."

    def _role_allowed(self, tool_name: str, role: AgentRole) -> bool:
        for spec in self.tool_specs():
            if spec.name == tool_name:
                return role in spec.allowed_roles
        return False

    def _coerce_action(self, action: Any) -> IncidentAction:
        if isinstance(action, IncidentAction):
            return action
        if isinstance(action, dict):
            return IncidentAction.model_validate(action)
        arguments = dict(getattr(action, "arguments", {}) or {})
        tool_name = getattr(action, "tool_name", None) or arguments.pop("tool_name", None)
        role = getattr(action, "agent_role", None) or arguments.pop("agent_role", None)
        if tool_name is None and action.__class__.__name__ == "ListToolsAction":
            tool_name = "list_tools"
        return IncidentAction(
            tool_name=tool_name or "unknown",
            agent_role=role or AgentRole.MONITOR,
            arguments=arguments,
        )
