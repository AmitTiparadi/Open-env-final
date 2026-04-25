"""FastAPI/OpenEnv server entry point."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import Body, FastAPI

from incident_commander_env.compat import OPENENV_AVAILABLE
from incident_commander_env.models import IncidentAction, IncidentObservation
from incident_commander_env.server.incident_environment import IncidentCommanderEnvironment


def _create_fallback_app() -> FastAPI:
    app = FastAPI(
        title="Incident Commander Environment",
        description="OpenEnv-style multi-agent SRE incident simulator.",
        version="0.1.0",
    )
    env = IncidentCommanderEnvironment()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "openenv": str(OPENENV_AVAILABLE).lower()}

    @app.get("/tools")
    def tools(role: Optional[str] = None) -> dict[str, Any]:
        specs = env.tool_specs()
        if role:
            specs = [spec for spec in specs if role in [r.value for r in spec.allowed_roles]]
        return {"tools": [spec.model_dump(mode="json") for spec in specs]}

    @app.post("/reset")
    def reset(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        obs = env.reset(**payload)
        return {
            "observation": obs.model_dump(mode="json"),
            "reward": obs.reward,
            "done": obs.done,
        }

    @app.post("/step")
    def step(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        action_payload = payload.get("action", payload)
        action = IncidentAction.model_validate(action_payload)
        obs = env.step(action)
        return {
            "observation": obs.model_dump(mode="json"),
            "reward": obs.reward,
            "done": obs.done,
        }

    @app.get("/state")
    def state() -> dict[str, Any]:
        return env.state.model_dump(mode="json")

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "name": "incident_commander_env",
            "docs": "/docs",
            "health": "/health",
            "reset": "POST /reset",
            "step": "POST /step",
        }

    return app


def _create_app() -> FastAPI:
    try:
        from openenv.core.env_server.http_server import create_app
        from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

        from incident_commander_env.server.mcp_environment import (
            IncidentCommanderMCPEnvironment,
        )

        return create_app(
            IncidentCommanderMCPEnvironment,
            CallToolAction,
            CallToolObservation,
            env_name="incident_commander_env",
        )
    except Exception:
        return _create_fallback_app()


app = _create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
