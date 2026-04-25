"""Pydantic models for the Incident Commander environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from incident_commander_env.compat import Action, Observation, State


class AgentRole(str, Enum):
    MONITOR = "monitor"
    INVESTIGATOR = "investigator"
    REMEDIATOR = "remediator"
    COMMUNICATOR = "communicator"


class IncidentAction(Action):
    """A role-scoped tool call used by the OpenEnv step loop."""

    tool_name: str = Field(
        description=(
            "Tool to call. Reserved OpenEnv names reset, step, state, and close "
            "are intentionally not used as tools."
        )
    )
    agent_role: AgentRole = Field(description="Specialized incident-response role.")
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    """Human-readable description of an available role-scoped tool."""

    name: str
    description: str
    allowed_roles: List[AgentRole]
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class RubricBreakdown(BaseModel):
    """Composable scoring rubric aligned with the hackathon judging brief."""

    root_cause: float = 0.0
    fix_safety: float = 0.0
    status_update: float = 0.0
    speed_bonus: float = 0.0
    hallucination_penalty: float = 0.0
    process_reward: float = 0.0
    total: float = 0.0


class IncidentObservation(Observation):
    """Observation returned after reset or a tool call."""

    message: str = ""
    role: Optional[AgentRole] = None
    visible_alerts: List[str] = Field(default_factory=list)
    shared_notes: List[str] = Field(default_factory=list)
    tool_result: Dict[str, Any] = Field(default_factory=dict)
    private_output_for: Optional[AgentRole] = None
    rubric_scores: RubricBreakdown = Field(default_factory=RubricBreakdown)
    available_tools: List[ToolSpec] = Field(default_factory=list)
    turn_budget_remaining: int = 0


class IncidentState(State):
    """Internal episode state exposed through the OpenEnv state endpoint."""

    scenario_id: Optional[str] = None
    difficulty: str = "easy"
    max_turns: int = 14
    detected_root_cause: Optional[str] = None
    detected_root_cause_correct: bool = False
    deployed_fix_id: Optional[str] = None
    resolved: bool = False
    secondary_outage: bool = False
    status_updates_sent: int = 0
    hallucination_detected: bool = False
    shared_note_count: int = 0
    last_actor: Optional[AgentRole] = None
