"""OpenEnv MCP wrapper around the Incident Commander simulator."""

from __future__ import annotations

from typing import Any, Optional

from incident_commander_env.models import AgentRole, IncidentAction, IncidentObservation
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment

try:
    from fastmcp import FastMCP
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from openenv.core.env_server.types import Action, Observation
except Exception as exc:  # pragma: no cover - import guard for fallback installs
    FastMCP = None
    MCPEnvironment = object  # type: ignore
    CallToolAction = None  # type: ignore
    CallToolObservation = None  # type: ignore
    Action = object  # type: ignore
    Observation = object  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class IncidentCommanderMCPEnvironment(MCPEnvironment):  # type: ignore[misc]
    """MCPEnvironment exposing role-scoped incident response tools."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError("OpenEnv MCP dependencies are not installed") from _IMPORT_ERROR

        self.core = IncidentCommanderEnvironment()
        mcp = FastMCP("incident_commander_env")

        @mcp.tool
        def list_incident_tools(agent_role: str = "monitor") -> dict[str, Any]:
            """List tools available to a role."""

            return self._call_core(
                "list_tools",
                agent_role=agent_role,
                arguments={},
            )

        @mcp.tool
        def check_metrics(
            agent_role: str,
            service: str,
            metric: str = "",
        ) -> dict[str, Any]:
            """Inspect noisy metrics for a service."""

            return self._call_core(
                "check_metrics",
                agent_role=agent_role,
                arguments={"service": service, "metric": metric},
            )

        @mcp.tool
        def query_logs(
            agent_role: str,
            service: str,
            query: str = "",
            limit: int = 5,
        ) -> dict[str, Any]:
            """Query service logs with optional search terms."""

            return self._call_core(
                "query_logs",
                agent_role=agent_role,
                arguments={"service": service, "query": query, "limit": limit},
            )

        @mcp.tool
        def share_note(agent_role: str, note: str) -> dict[str, Any]:
            """Publish evidence or a hypothesis to the shared scratchpad."""

            return self._call_core(
                "share_note",
                agent_role=agent_role,
                arguments={"note": note},
            )

        @mcp.tool
        def submit_root_cause(
            agent_role: str,
            root_cause: str,
            confidence: float = 0.5,
            evidence: str = "",
        ) -> dict[str, Any]:
            """Submit the team's current root-cause hypothesis."""

            return self._call_core(
                "submit_root_cause",
                agent_role=agent_role,
                arguments={
                    "root_cause": root_cause,
                    "confidence": confidence,
                    "evidence": evidence,
                },
            )

        @mcp.tool
        def deploy_fix(agent_role: str, fix_id: str) -> dict[str, Any]:
            """Deploy a remediation by fix_id."""

            return self._call_core(
                "deploy_fix",
                agent_role=agent_role,
                arguments={"fix_id": fix_id},
            )

        @mcp.tool
        def send_update(
            agent_role: str,
            message: str,
            audience: str = "stakeholders",
        ) -> dict[str, Any]:
            """Send a stakeholder status update."""

            return self._call_core(
                "send_update",
                agent_role=agent_role,
                arguments={"message": message, "audience": audience},
            )

        @mcp.tool
        def finish_incident(agent_role: str, summary: str = "") -> dict[str, Any]:
            """End the episode after fix and communication are complete."""

            return self._call_core(
                "finish_incident",
                agent_role=agent_role,
                arguments={"summary": summary},
            )

        super().__init__(mcp)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        return self.core.reset(seed=seed, episode_id=episode_id, **kwargs)

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        observation = super().step(action, timeout_s=timeout_s, **kwargs)
        return self._copy_reward_from_tool_payload(observation)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        observation = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        return self._copy_reward_from_tool_payload(observation)

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        del timeout_s, kwargs
        if isinstance(action, IncidentAction):
            return self.core.step(action)
        return IncidentObservation(
            message=f"Unsupported non-MCP action: {type(action).__name__}",
            reward=-0.03,
            done=False,
        )

    @property
    def state(self):  # type: ignore[no-untyped-def]
        return self.core.state

    def close(self) -> None:
        self.core.close()
        super().close()

    def _call_core(
        self,
        tool_name: str,
        agent_role: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        role = AgentRole(agent_role)
        observation = self.core.step(
            IncidentAction(
                tool_name=tool_name,
                agent_role=role,
                arguments=arguments,
            )
        )
        return observation.model_dump(mode="json")

    def _copy_reward_from_tool_payload(self, observation: Observation) -> Observation:
        if not CallToolObservation or not isinstance(observation, CallToolObservation):
            return observation
        payload = self._extract_payload(observation.result)
        if isinstance(payload, dict):
            observation.reward = payload.get("reward")
            observation.done = bool(payload.get("done", False))
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                observation.metadata.update(metadata)
        return observation

    def _extract_payload(self, result: Any) -> Any:
        if hasattr(result, "data"):
            return result.data
        if hasattr(result, "structured_content"):
            content = result.structured_content
            if isinstance(content, dict) and "result" in content:
                return content["result"]
        if isinstance(result, dict):
            return result
        return None
