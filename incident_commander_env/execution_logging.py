"""Structured execution logging for environment actions."""

from __future__ import annotations

import json
import traceback
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from incident_commander_env.models import AgentRole, IncidentAction, IncidentObservation


ERROR_SYNTAX = "syntax_error"
ERROR_IMPORT = "import_error"
ERROR_RUNTIME = "runtime_error"
ERROR_LOGIC = "logic_error"


class StructuredError(BaseModel):
    """Normalized error details for prompting, eval, and observability."""

    type: str | None = None
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ExecutionLog(BaseModel):
    """One structured record captured after an environment action."""

    step: int
    tool_name: str
    role: AgentRole | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    success: bool = True
    error: StructuredError = Field(default_factory=StructuredError)
    reward: float = 0.0
    done: bool = False
    context: dict[str, Any] = Field(default_factory=dict)


def classify_python_failure(result: dict[str, Any]) -> StructuredError:
    """Classify sandboxed Python failures into stable error categories."""

    violations = [str(item) for item in result.get("violations", [])]
    error = str(result.get("error") or "")
    message = str(result.get("message") or error or "python execution failed")
    lower_blob = " ".join([error, message, *violations]).lower()
    if "syntax_error" in lower_blob or error == "SyntaxError":
        error_type = ERROR_SYNTAX
    elif (
        "import" in lower_blob
        or "forbidden_module" in lower_blob
        or error in {"ImportError", "ModuleNotFoundError"}
    ):
        error_type = ERROR_IMPORT
    else:
        error_type = ERROR_RUNTIME
    return StructuredError(
        type=error_type,
        message=message[:400],
        details={"python_error": error, "violations": violations},
    )


def classify_observation_failure(
    action: IncidentAction,
    observation: IncidentObservation,
) -> StructuredError:
    """Classify an unsuccessful action without exposing hidden answer data."""

    result = observation.tool_result or {}
    if action.tool_name == "python_exec" and not result.get("ok", True):
        return classify_python_failure(result)

    error = str(result.get("error") or "")
    if error:
        if error in {"unknown_tool", "role_not_allowed", "empty_note", "unknown_endpoint"}:
            error_type = ERROR_LOGIC
        else:
            error_type = ERROR_RUNTIME
        return StructuredError(
            type=error_type,
            message=error,
            details={"tool_error": error},
        )

    if result.get("hallucinated") or result.get("red_herring_chased"):
        return StructuredError(
            type=ERROR_LOGIC,
            message="claim was not supported by incident evidence",
            details={
                "hallucinated": bool(result.get("hallucinated")),
                "red_herring_chased": bool(result.get("red_herring_chased")),
            },
        )
    if result.get("communication_mismatch"):
        return StructuredError(
            type=ERROR_LOGIC,
            message="stakeholder update exceeded established facts",
            details={"communication_mismatch": True},
        )
    if "correct" in result and result.get("correct") is False:
        return StructuredError(
            type=ERROR_LOGIC,
            message="root cause was not supported by evidence",
            details={"root_cause_correct": False},
        )
    if result.get("secondary_outage"):
        return StructuredError(
            type=ERROR_LOGIC,
            message="remediation caused a secondary outage",
            details={"secondary_outage": True},
        )
    if "resolved" in result and result.get("resolved") is False and action.tool_name == "deploy_fix":
        return StructuredError(
            type=ERROR_LOGIC,
            message="fix did not resolve the incident",
            details={"resolved": False},
        )
    if float(observation.reward or 0.0) < 0:
        return StructuredError(
            type=ERROR_LOGIC,
            message=observation.message,
            details={"negative_reward": float(observation.reward or 0.0)},
        )
    return StructuredError()


def classify_exception(exc: BaseException) -> StructuredError:
    """Classify unexpected tool/runtime exceptions for Sentry and logs."""

    if isinstance(exc, SyntaxError):
        error_type = ERROR_SYNTAX
    elif isinstance(exc, (ImportError, ModuleNotFoundError)):
        error_type = ERROR_IMPORT
    else:
        error_type = ERROR_RUNTIME
    return StructuredError(
        type=error_type,
        message=str(exc)[:400],
        details={
            "exception_type": type(exc).__name__,
            "traceback": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        },
    )


def _compact_json(value: Any, limit: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, default=str, sort_keys=True)
    except TypeError:
        text = str(value)
    return text[:limit]


def execution_log_from_observation(
    *,
    step: int,
    action: IncidentAction,
    observation: IncidentObservation,
    context: dict[str, Any] | None = None,
) -> ExecutionLog:
    """Create a structured log row after an action has run."""

    result = observation.tool_result or {}
    error = classify_observation_failure(action, observation)
    success = error.type is None and not bool(result.get("error"))
    stdout = ""
    stderr = ""
    if action.tool_name == "python_exec":
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or result.get("message") or "")
    elif success:
        stdout = _compact_json({"message": observation.message, "result": result})
    else:
        stderr = error.message or observation.message
    return ExecutionLog(
        step=step,
        tool_name=action.tool_name,
        role=action.agent_role,
        stdout=stdout,
        stderr=stderr,
        exit_code=0 if success else 1,
        success=success,
        error=error,
        reward=float(observation.reward or 0.0),
        done=bool(observation.done),
        context=context or {},
    )


def execution_log_from_exception(
    *,
    step: int,
    action: IncidentAction,
    exc: BaseException,
    context: dict[str, Any] | None = None,
) -> ExecutionLog:
    """Create a structured log row when a tool handler raises."""

    error = classify_exception(exc)
    return ExecutionLog(
        step=step,
        tool_name=action.tool_name,
        role=action.agent_role,
        stdout="",
        stderr=error.message,
        exit_code=1,
        success=False,
        error=error,
        reward=-0.2,
        done=False,
        context=context or {},
    )


class FailurePatternTracker:
    """Small in-memory counter for recurring execution failure patterns."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def record(self, log: ExecutionLog) -> dict[str, Any] | None:
        if log.success or not log.error.type:
            return None
        key = f"{log.tool_name}:{log.error.type}"
        self._counts[key] += 1
        return {
            "pattern": key,
            "count": self._counts[key],
            "tool_name": log.tool_name,
            "error_type": log.error.type,
        }

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)
