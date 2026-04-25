"""Compatibility layer for running with or without openenv-core installed."""

from __future__ import annotations

from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.interfaces import Environment as OpenEnvEnvironment
    from openenv.core.env_server.types import (  # type: ignore
        Action,
        Observation,
        State,
    )

    OPENENV_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when openenv-core is absent
    OPENENV_AVAILABLE = False

    class Action(BaseModel):
        model_config = ConfigDict(
            extra="allow",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        model_config = ConfigDict(
            extra="allow",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        done: bool = False
        reward: float | None = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        model_config = ConfigDict(
            extra="allow",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        episode_id: Optional[str] = None
        step_count: int = 0

    ActT = TypeVar("ActT", bound=Action)
    ObsT = TypeVar("ObsT", bound=Observation)
    StateT = TypeVar("StateT", bound=State)

    class OpenEnvEnvironment(Generic[ActT, ObsT, StateT]):
        SUPPORTS_CONCURRENT_SESSIONS = True

        def reset(
            self,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
            **kwargs: Any,
        ) -> ObsT:
            raise NotImplementedError

        async def reset_async(
            self,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
            **kwargs: Any,
        ) -> ObsT:
            return self.reset(seed=seed, episode_id=episode_id, **kwargs)

        def step(
            self,
            action: ActT,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> ObsT:
            raise NotImplementedError

        async def step_async(
            self,
            action: ActT,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> ObsT:
            return self.step(action, timeout_s=timeout_s, **kwargs)

        @property
        def state(self) -> StateT:
            raise NotImplementedError

        def close(self) -> None:
            return None
