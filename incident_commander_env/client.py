"""Small synchronous client for local demos and notebooks."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Optional

from incident_commander_env.models import AgentRole, IncidentAction
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment


class IncidentCommanderEnv:
    """Use local in-process mode by default, or HTTP mode with base_url."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url.rstrip("/") if base_url else None
        self._local_env = None if base_url else IncidentCommanderEnvironment()

    def __enter__(self) -> "IncidentCommanderEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def sync(self) -> "IncidentCommanderEnv":
        return self

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        if self._local_env:
            obs = self._local_env.reset(**kwargs)
            return {"observation": obs.model_dump(mode="json"), "reward": obs.reward, "done": obs.done}
        return self._post("/reset", kwargs)

    def step(
        self,
        tool_name: str | IncidentAction,
        agent_role: AgentRole | str | None = None,
        **arguments: Any,
    ) -> dict[str, Any]:
        if isinstance(tool_name, IncidentAction):
            action = tool_name
        else:
            action = IncidentAction(
                tool_name=tool_name,
                agent_role=agent_role or AgentRole.MONITOR,
                arguments=arguments,
            )
        if self._local_env:
            obs = self._local_env.step(action)
            return {"observation": obs.model_dump(mode="json"), "reward": obs.reward, "done": obs.done}
        return self._post("/step", {"action": action.model_dump(mode="json")})

    def state(self) -> dict[str, Any]:
        if self._local_env:
            return self._local_env.state.model_dump(mode="json")
        with urllib.request.urlopen(f"{self.base_url}/state") as response:
            return json.loads(response.read().decode("utf-8"))

    def close(self) -> None:
        self._local_env = None

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.base_url is not None
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
