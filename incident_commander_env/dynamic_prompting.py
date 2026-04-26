"""Failure-aware prompt update helpers.

The dynamic prompting layer consumes structured execution logs only. It never
receives hidden test details or Sentry internals.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from incident_commander_env.execution_logging import (
    ERROR_IMPORT,
    ERROR_LOGIC,
    ERROR_RUNTIME,
    ERROR_SYNTAX,
    ExecutionLog,
)


GUIDANCE_BY_ERROR_TYPE = {
    ERROR_SYNTAX: "Fix syntax before retrying; emit valid JSON/tool arguments only.",
    ERROR_IMPORT: "Avoid unavailable imports and use built-in tools instead.",
    ERROR_RUNTIME: "Check tool argument names and expected input shapes before retrying.",
    ERROR_LOGIC: "Re-check evidence, role permissions, root cause, fix id, and workflow order.",
}


class PromptUpdate(BaseModel):
    """Sanitized retry guidance produced from structured execution logs."""

    summary: str = ""
    retry_guidance: list[str] = Field(default_factory=list)
    failure_counts: dict[str, int] = Field(default_factory=dict)
    recurring_patterns: list[str] = Field(default_factory=list)


def build_prompt_update(
    logs: list[ExecutionLog],
    *,
    max_guidance: int = 3,
) -> PromptUpdate:
    """Summarize recent failures without exposing hidden test or Sentry data."""

    failures = [log for log in logs if not log.success and log.error.type]
    if not failures:
        return PromptUpdate(summary="No recent execution failures.")

    type_counts = Counter(str(log.error.type) for log in failures)
    tool_counts = Counter(f"{log.tool_name}:{log.error.type}" for log in failures)
    most_common_type, most_common_count = type_counts.most_common(1)[0]
    guidance: list[str] = []
    for error_type, _ in type_counts.most_common(max_guidance):
        item = GUIDANCE_BY_ERROR_TYPE.get(error_type)
        if item:
            guidance.append(item)
    recurring = [
        f"{pattern} occurred {count} times"
        for pattern, count in tool_counts.most_common(max_guidance)
        if count > 1
    ]
    summary = (
        f"Recent executions show {most_common_count} {most_common_type} "
        f"failure{'s' if most_common_count != 1 else ''}."
    )
    return PromptUpdate(
        summary=summary,
        retry_guidance=guidance,
        failure_counts=dict(type_counts),
        recurring_patterns=recurring,
    )


def prompt_update_metadata(logs: list[ExecutionLog]) -> dict[str, Any]:
    """JSON-ready metadata for observations or training dashboards."""

    update = build_prompt_update(logs)
    return update.model_dump(mode="json")
