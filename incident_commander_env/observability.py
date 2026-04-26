"""Optional Sentry integration for system reliability.

Sentry is deliberately kept outside the agent-facing observation surface. The
agent may receive sanitized failure summaries from dynamic_prompting, but never
DSNs, event ids, stack traces, or Sentry APIs.
"""

from __future__ import annotations

import os
from typing import Any

from incident_commander_env.execution_logging import ExecutionLog


class SentryMonitor:
    """Thin optional wrapper around sentry-sdk."""

    def __init__(
        self,
        *,
        dsn: str | None = None,
        environment: str | None = None,
        release: str | None = None,
    ) -> None:
        self.dsn = dsn if dsn is not None else os.getenv("SENTRY_DSN")
        self.enabled = bool(self.dsn)
        self._sdk: Any | None = None
        if not self.enabled:
            return
        try:
            import sentry_sdk  # type: ignore
        except Exception:
            self.enabled = False
            return
        self._sdk = sentry_sdk
        sentry_sdk.init(
            dsn=self.dsn,
            environment=environment or os.getenv("SENTRY_ENVIRONMENT", "local"),
            release=release or os.getenv("SENTRY_RELEASE", "incident-commander-env"),
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
        )

    def capture_exception(
        self,
        exc: BaseException,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self._sdk is None:
            return
        with self._sdk.push_scope() as scope:
            if context:
                scope.set_context("incident_execution", context)
            self._sdk.capture_exception(exc)

    def capture_execution_log(
        self,
        log: ExecutionLog,
        *,
        task_context: dict[str, Any] | None = None,
        recurring_pattern: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self._sdk is None or log.success:
            return
        with self._sdk.push_scope() as scope:
            scope.set_tag("tool_name", log.tool_name)
            if log.error.type:
                scope.set_tag("error_type", log.error.type)
            if log.role:
                scope.set_tag("agent_role", log.role.value)
            scope.set_context(
                "execution_log",
                {
                    "step": log.step,
                    "success": log.success,
                    "exit_code": log.exit_code,
                    "reward": log.reward,
                    "error": log.error.model_dump(mode="json"),
                    "context": task_context or {},
                    "recurring_pattern": recurring_pattern or {},
                },
            )
            message = f"{log.tool_name} failed with {log.error.type or 'unknown_error'}"
            self._sdk.capture_message(message, level="warning")
