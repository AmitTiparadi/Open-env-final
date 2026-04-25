"""Interactive RL rollout system for action-observation training."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

from incident_commander_env.evaluation import hidden_security_checks
from incident_commander_env.judge import EnsembleJudge, parse_candidate_actions
from incident_commander_env.models import IncidentAction, IncidentObservation
from incident_commander_env.rewards import (
    EVALUATION_INTEGRITY_PENALTY,
    TRAINING_INTEGRITY_PENALTY,
)
from incident_commander_env.scenarios import (
    IncidentScenario,
    evaluation_scenarios,
    generate_scenario,
    is_hidden_scenario,
)
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment


class InteractiveTask(BaseModel):
    """A task instance sampled for online RL."""

    task_id: str
    scenario_id: str
    difficulty: str
    prompt: str
    max_turns: int = 14
    hidden_case: bool = False
    allowed_tools: list[str] = Field(default_factory=list)


class TrajectoryStep(BaseModel):
    """One action-observation transition."""

    turn: int
    action: dict[str, Any]
    observation: dict[str, Any]
    reward: float
    done: bool


class RolloutMetrics(BaseModel):
    """Aggregated reward signals for RL updates and dashboards."""

    total_reward: float
    environment_reward: float
    judge_reward: float
    security_penalty: float
    efficiency_bonus: float
    success: bool
    accepted: bool
    root_cause_correct: bool
    resolved: bool
    hallucination_detected: bool
    integrity_violation_detected: bool
    tool_errors: int = 0
    steps: int = 0


class InteractiveRolloutResult(BaseModel):
    """Full trajectory returned by the interactive runner."""

    task: InteractiveTask
    trajectory: list[TrajectoryStep]
    metrics: RolloutMetrics
    final_observation: dict[str, Any]


class TrainingIteration(BaseModel):
    """Metrics emitted by one online RL collection/update iteration."""

    iteration: int
    batch_size: int
    mean_reward: float
    success_rate: float
    accepted_rate: float
    mean_steps: float
    update_info: dict[str, Any] = Field(default_factory=dict)
    curriculum_state: dict[str, Any] = Field(default_factory=dict)


class InteractivePolicy(Protocol):
    """Callable policy interface used by the rollout runner."""

    def __call__(
        self,
        history: list[dict[str, Any]],
        task: InteractiveTask,
    ) -> Any:
        ...


PolicyUpdater = Callable[[list[InteractiveRolloutResult]], dict[str, Any] | None]


@dataclass
class AdaptiveTaskGenerator:
    """Samples scenarios by weakness instead of fixed dataset order."""

    include_hidden: bool = False
    max_turns: int = 14
    rng: Random = field(default_factory=Random)
    attempts_by_root: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failures_by_root: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    attempts_by_difficulty: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failures_by_difficulty: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def sample(
        self,
        seed: int | None = None,
        difficulty: str = "mixed",
        preferred_root_cause: str | None = None,
    ) -> InteractiveTask:
        rng = Random(seed) if seed is not None else self.rng
        scenarios = list(evaluation_scenarios(include_hidden=self.include_hidden))
        if preferred_root_cause:
            scenarios = [s for s in scenarios if s.root_cause == preferred_root_cause] or scenarios
        if difficulty != "mixed":
            scenarios = [s for s in scenarios if s.difficulty == difficulty] or scenarios

        weighted: list[tuple[float, IncidentScenario]] = []
        for scenario in scenarios:
            root_attempts = max(1, self.attempts_by_root[scenario.root_cause])
            difficulty_attempts = max(1, self.attempts_by_difficulty[scenario.difficulty])
            root_weakness = self.failures_by_root[scenario.root_cause] / root_attempts
            difficulty_weakness = self.failures_by_difficulty[scenario.difficulty] / difficulty_attempts
            hard_bonus = 0.25 if scenario.difficulty == "hard" else 0.0
            hidden_bonus = 0.2 if is_hidden_scenario(scenario.scenario_id) else 0.0
            weighted.append((1.0 + root_weakness + difficulty_weakness + hard_bonus + hidden_bonus, scenario))

        total = sum(weight for weight, _ in weighted)
        pick = rng.random() * total
        cumulative = 0.0
        scenario = weighted[-1][1]
        for weight, candidate in weighted:
            cumulative += weight
            if pick <= cumulative:
                scenario = candidate
                break

        hidden = is_hidden_scenario(scenario.scenario_id)
        return InteractiveTask(
            task_id=f"{'hidden' if hidden else 'public'}:{scenario.scenario_id}:{seed or 0}",
            scenario_id=scenario.scenario_id,
            difficulty=scenario.difficulty,
            prompt=render_interactive_prompt(scenario, hidden_case=hidden),
            max_turns=self.max_turns,
            hidden_case=hidden,
            allowed_tools=[
                "check_metrics",
                "query_logs",
                "share_note",
                "submit_root_cause",
                "deploy_fix",
                "send_update",
                "finish_incident",
                "python_exec",
                "web_search",
                "query_api",
            ],
        )

    def record_result(self, result: InteractiveRolloutResult) -> None:
        scenario = generate_scenario(
            difficulty=result.task.difficulty,
            preferred_root_cause=None,
            include_hidden=result.task.hidden_case,
        )
        # Use the task scenario id if available in the current scenario pool.
        for candidate in evaluation_scenarios(include_hidden=result.task.hidden_case):
            if candidate.scenario_id == result.task.scenario_id:
                scenario = candidate
                break
        self.attempts_by_root[scenario.root_cause] += 1
        self.attempts_by_difficulty[scenario.difficulty] += 1
        if not result.metrics.accepted:
            self.failures_by_root[scenario.root_cause] += 1
            self.failures_by_difficulty[scenario.difficulty] += 1

    def export_state(self) -> dict[str, Any]:
        return {
            "include_hidden": self.include_hidden,
            "attempts_by_root": dict(self.attempts_by_root),
            "failures_by_root": dict(self.failures_by_root),
            "attempts_by_difficulty": dict(self.attempts_by_difficulty),
            "failures_by_difficulty": dict(self.failures_by_difficulty),
        }


def render_interactive_prompt(
    scenario: IncidentScenario,
    hidden_case: bool = False,
) -> str:
    """Render the first prompt for an interactive action loop."""

    scenario_label = "hidden_eval_case" if hidden_case else scenario.scenario_id
    alerts = "\n".join(f"- {alert}" for alert in scenario.alerts)
    return f"""You are an incident-response agent in an interactive environment.
You must solve the task by repeatedly taking one tool action, reading the
observation, and deciding the next action.

Scenario id: {scenario_label}
Difficulty: {scenario.difficulty}
Affected service: {scenario.affected_service}
Visible alerts:
{alerts}

Return exactly one JSON tool call per turn:
{{"tool_name":"check_metrics","agent_role":"monitor","arguments":{{"service":"..."}}}}

Useful tools include check_metrics, query_logs, web_search, query_api,
python_exec, share_note, submit_root_cause, deploy_fix, send_update, and
finish_incident. Use evidence before claiming a root cause.
"""


def _coerce_policy_action(value: Any) -> IncidentAction | None:
    if isinstance(value, IncidentAction):
        return value
    actions = parse_candidate_actions(value)
    if actions:
        return actions[0]
    if isinstance(value, dict):
        try:
            return IncidentAction.model_validate(value)
        except Exception:
            return None
    return None


class InteractiveRolloutRunner:
    """Runs a policy through an action-observation loop and scores it."""

    def __init__(
        self,
        judge: EnsembleJudge | None = None,
        invalid_action_penalty: float = -0.2,
        integrity_penalty: float = TRAINING_INTEGRITY_PENALTY,
    ) -> None:
        self.judge = judge or EnsembleJudge()
        self.invalid_action_penalty = invalid_action_penalty
        self.integrity_penalty = integrity_penalty

    def rollout(
        self,
        policy: InteractivePolicy,
        task: InteractiveTask,
    ) -> InteractiveRolloutResult:
        env = IncidentCommanderEnvironment()
        obs = env.reset(
            difficulty=task.difficulty,
            scenario_id=task.scenario_id,
            max_turns=task.max_turns,
            include_hidden_scenarios=task.hidden_case,
        )
        history: list[dict[str, Any]] = [
            {"type": "prompt", "content": task.prompt},
            {"type": "observation", "content": obs.model_dump(mode="json")},
        ]
        trajectory: list[TrajectoryStep] = []
        actions: list[IncidentAction] = []
        environment_reward = 0.0
        tool_errors = 0

        for turn in range(1, task.max_turns + 1):
            raw_action = policy(history, task)
            action = _coerce_policy_action(raw_action)
            if action is None:
                tool_errors += 1
                environment_reward += self.invalid_action_penalty
                trajectory.append(
                    TrajectoryStep(
                        turn=turn,
                        action={"invalid": raw_action},
                        observation={"message": "Invalid action JSON."},
                        reward=self.invalid_action_penalty,
                        done=True,
                    )
                )
                break

            obs = env.step(action)
            actions.append(action)
            environment_reward += float(obs.reward or 0.0)
            if obs.tool_result.get("error"):
                tool_errors += 1
            step = TrajectoryStep(
                turn=turn,
                action=action.model_dump(mode="json"),
                observation=obs.model_dump(mode="json"),
                reward=float(obs.reward or 0.0),
                done=obs.done,
            )
            trajectory.append(step)
            history.append({"type": "action", "content": step.action})
            history.append({"type": "observation", "content": step.observation})
            if obs.done:
                break

        judge_eval = self.judge.evaluate(
            candidate=[action.model_dump(mode="json") for action in actions],
            scenario=env.scenario,
            actions=actions,
        )
        if judge_eval.integrity_penalty <= EVALUATION_INTEGRITY_PENALTY:
            env.state.integrity_violation_detected = True

        checks = hidden_security_checks(actions, env)
        security_penalty = float(checks["security_penalty"])
        if judge_eval.integrity_penalty <= EVALUATION_INTEGRITY_PENALTY:
            total_reward = self.integrity_penalty
        else:
            total_reward = (
                environment_reward
                + float(obs.rubric_scores.total)
                + float(judge_eval.reward_delta)
                + security_penalty
            )

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
            and total_reward >= 0.7
        )
        metrics = RolloutMetrics(
            total_reward=round(float(total_reward), 4),
            environment_reward=round(float(environment_reward), 4),
            judge_reward=float(judge_eval.reward_delta),
            security_penalty=round(security_penalty, 4),
            efficiency_bonus=float(obs.rubric_scores.speed_bonus),
            success=success,
            accepted=accepted,
            root_cause_correct=env.state.detected_root_cause_correct,
            resolved=env.state.resolved,
            hallucination_detected=env.state.hallucination_detected,
            integrity_violation_detected=env.state.integrity_violation_detected,
            tool_errors=tool_errors,
            steps=len(trajectory),
        )
        return InteractiveRolloutResult(
            task=task,
            trajectory=trajectory,
            metrics=metrics,
            final_observation=obs.model_dump(mode="json"),
        )

    def rollout_actions(
        self,
        actions: list[IncidentAction],
        task: InteractiveTask,
    ) -> InteractiveRolloutResult:
        index = 0

        def policy(_: list[dict[str, Any]], __: InteractiveTask) -> Any:
            nonlocal index
            if index >= len(actions):
                return {
                    "tool_name": "finish_incident",
                    "agent_role": "communicator",
                    "arguments": {"summary": "No more actions."},
                }
            action = actions[index]
            index += 1
            return action

        return self.rollout(policy, task)


class OnlineRLTrainer:
    """Collects interactive rollouts and calls a policy-update hook.

    The updater can wrap TRL/GRPO, a supervised replay buffer, or any custom
    weight-update mechanism. Keeping it as a hook makes the environment loop
    testable without requiring a GPU runtime.
    """

    def __init__(
        self,
        policy: InteractivePolicy,
        task_generator: AdaptiveTaskGenerator | None = None,
        runner: InteractiveRolloutRunner | None = None,
        update_fn: PolicyUpdater | None = None,
        batch_size: int = 4,
    ) -> None:
        self.policy = policy
        self.task_generator = task_generator or AdaptiveTaskGenerator()
        self.runner = runner or InteractiveRolloutRunner()
        self.update_fn = update_fn
        self.batch_size = batch_size

    def train_iteration(self, iteration: int = 0) -> TrainingIteration:
        results: list[InteractiveRolloutResult] = []
        for item in range(self.batch_size):
            task = self.task_generator.sample(seed=iteration * self.batch_size + item)
            result = self.runner.rollout(self.policy, task)
            self.task_generator.record_result(result)
            results.append(result)

        update_info = self.update_fn(results) if self.update_fn else {}
        rewards = [result.metrics.total_reward for result in results]
        successes = [result.metrics.success for result in results]
        accepted = [result.metrics.accepted for result in results]
        steps = [result.metrics.steps for result in results]
        return TrainingIteration(
            iteration=iteration,
            batch_size=len(results),
            mean_reward=round(sum(rewards) / max(1, len(rewards)), 4),
            success_rate=round(sum(bool(value) for value in successes) / max(1, len(successes)), 4),
            accepted_rate=round(sum(bool(value) for value in accepted) / max(1, len(accepted)), 4),
            mean_steps=round(sum(steps) / max(1, len(steps)), 4),
            update_info=update_info or {},
            curriculum_state=self.task_generator.export_state(),
        )

    def train(self, iterations: int = 1) -> list[TrainingIteration]:
        return [self.train_iteration(iteration=i) for i in range(iterations)]


def scripted_interactive_policy(history: list[dict[str, Any]], task: InteractiveTask) -> Any:
    """Small deterministic policy useful for smoke tests."""

    observations = [item for item in history if item["type"] == "observation"]
    turn = max(0, len(observations) - 1)
    service = ""
    if observations:
        service = observations[0]["content"].get("metadata", {}).get("affected_service", "")
    plan = [
        {"tool_name": "check_metrics", "agent_role": "monitor", "arguments": {"service": service}},
        {"tool_name": "query_logs", "agent_role": "investigator", "arguments": {"service": service, "limit": 5}},
        {"tool_name": "web_search", "agent_role": "investigator", "arguments": {"query": service, "limit": 5}},
        {"tool_name": "query_api", "agent_role": "investigator", "arguments": {"endpoint": "deployments"}},
    ]
    if turn < len(plan):
        return plan[turn]

    action_blob = json.dumps(
        [item["content"] for item in history if item["type"] == "action"],
        ensure_ascii=True,
    )
    evidence_blob = json.dumps(
        [item["content"] for item in history if item["type"] == "observation"],
        ensure_ascii=True,
    )
    if "submit_root_cause" not in action_blob:
        return {
            "tool_name": "submit_root_cause",
            "agent_role": "investigator",
            "arguments": {"root_cause": evidence_blob[:500], "evidence": evidence_blob[:1200]},
        }
    if "deploy_fix" not in action_blob:
        # The scripted smoke policy intentionally cannot see hidden answer keys.
        return {
            "tool_name": "deploy_fix",
            "agent_role": "remediator",
            "arguments": {"fix_id": "unknown_fix"},
        }
    if "send_update" not in action_blob:
        return {
            "tool_name": "send_update",
            "agent_role": "communicator",
            "arguments": {"message": f"{service} incident is being mitigated and monitored."},
        }
    return {
        "tool_name": "finish_incident",
        "agent_role": "communicator",
        "arguments": {"summary": "Interactive rollout complete."},
    }
