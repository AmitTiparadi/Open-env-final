"""Auxiliary tools for interactive RL rollouts.

These tools are intentionally deterministic and sandboxed so they can be used
inside RL jobs without giving the model file, network, or reward-state access.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import statistics
from typing import Any

from incident_commander_env.rewards import contains_real_evidence, normalize_text
from incident_commander_env.scenarios import IncidentScenario


FORBIDDEN_PYTHON_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
FORBIDDEN_MODULE_NAMES = {
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "threading",
    "urllib",
}


def _merged_metrics(scenario: IncidentScenario) -> dict[str, dict[str, tuple[float, ...]]]:
    merged = {service: dict(metrics) for service, metrics in scenario.metrics.items()}
    for service, metrics in scenario.red_herring_metrics.items():
        merged.setdefault(service, {}).update(metrics)
    return merged


def _safe_builtins() -> dict[str, Any]:
    return {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }


def validate_python_code(code: str) -> list[str]:
    """Return sandbox violations for code submitted by an agent."""

    if len(code) > 1500:
        return ["code_too_long"]
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return [f"syntax_error:{exc.msg}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append("imports_disabled")
            names = [alias.name.split(".")[0] for alias in getattr(node, "names", [])]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".")[0])
            for name in names:
                if name in FORBIDDEN_MODULE_NAMES:
                    violations.append(f"forbidden_module:{name}")
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_PYTHON_NAMES or node.id in FORBIDDEN_MODULE_NAMES:
                violations.append(f"forbidden_name:{node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                violations.append("dunder_attribute")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_PYTHON_NAMES:
                violations.append(f"forbidden_call:{node.func.id}")
    return sorted(set(violations))


def execute_python(code: str) -> dict[str, Any]:
    """Run tiny Python snippets in a restricted namespace."""

    violations = validate_python_code(code)
    if violations:
        return {
            "ok": False,
            "success": False,
            "error": "sandbox_violation",
            "violations": violations,
            "stdout": "",
            "stderr": "; ".join(violations),
            "exit_code": 1,
        }

    env = {
        "__builtins__": _safe_builtins(),
        "json": json,
        "math": math,
        "statistics": statistics,
    }
    stdout = io.StringIO()
    try:
        parsed = ast.parse(code, mode="exec")
        with contextlib.redirect_stdout(stdout):
            if len(parsed.body) == 1 and isinstance(parsed.body[0], ast.Expr):
                result = eval(compile(ast.Expression(parsed.body[0].value), "<agent-python>", "eval"), env, {})
            else:
                exec(compile(parsed, "<agent-python>", "exec"), env, {})
                result = None
    except Exception as exc:
        return {
            "ok": False,
            "success": False,
            "error": type(exc).__name__,
            "message": str(exc)[:300],
            "stdout": stdout.getvalue()[:1000],
            "stderr": str(exc)[:1000],
            "exit_code": 1,
        }
    return {
        "ok": True,
        "success": True,
        "stdout": stdout.getvalue()[:1000],
        "stderr": "",
        "exit_code": 0,
        "result": _jsonable_result(result),
    }


def _jsonable_result(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _scenario_corpus(scenario: IncidentScenario) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for alert in scenario.alerts:
        rows.append({"source": "alert", "text": alert})
    for service, lines in scenario.logs.items():
        for line in lines:
            rows.append({"source": f"logs:{service}", "text": line})
    for service, lines in scenario.red_herring_logs.items():
        for line in lines:
            rows.append({"source": f"logs:{service}:red_herring", "text": line})
    for service, metrics in scenario.metrics.items():
        for metric, values in metrics.items():
            if values:
                rows.append(
                    {
                        "source": f"metrics:{service}",
                        "text": f"{metric} moved from {values[0]} to {values[-1]}",
                    }
                )
    for service, metrics in scenario.red_herring_metrics.items():
        for metric, values in metrics.items():
            if values:
                rows.append(
                    {
                        "source": f"metrics:{service}:red_herring",
                        "text": f"{metric} moved from {values[0]} to {values[-1]}",
                    }
                )
    for index, link in enumerate(scenario.causal_chain, 1):
        rows.append({"source": f"causal_chain:{index}", "text": link})
    rows.append({"source": "impact", "text": scenario.stakeholder_impact})
    for red_herring in scenario.red_herrings:
        rows.append({"source": "red_herring", "text": red_herring})
    return rows


def search_knowledge_base(
    scenario: IncidentScenario,
    query: str,
    limit: int = 5,
) -> dict[str, Any]:
    """Search the incident-local knowledge base."""

    normalized_terms = [
        term for term in normalize_text(query).split() if len(term) > 2
    ]
    scored: list[tuple[int, dict[str, str]]] = []
    for row in _scenario_corpus(scenario):
        text = normalize_text(row["text"])
        score = sum(1 for term in normalized_terms if term in text)
        if not normalized_terms and row["source"] == "alert":
            score = 1
        if score:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    snippets = [
        {"source": row["source"], "text": row["text"]}
        for _, row in scored[: max(1, limit)]
    ]
    return {
        "query": query,
        "results": snippets,
        "evidence_found": contains_real_evidence(
            " ".join(item["text"] for item in snippets),
            scenario,
        ),
    }


def query_incident_api(scenario: IncidentScenario, endpoint: str) -> dict[str, Any]:
    """Return structured incident data without exposing hidden labels."""

    endpoint_name = normalize_text(endpoint).replace(" ", "_")
    if endpoint_name in {"service_graph", "dependencies", "service_dependencies"}:
        services = sorted(
            set(scenario.logs)
            | set(scenario.metrics)
            | set(scenario.red_herring_logs)
            | set(scenario.red_herring_metrics)
        )
        return {
            "endpoint": "service_graph",
            "affected_service": scenario.affected_service,
            "origin_service_candidate": scenario.origin_service or scenario.affected_service,
            "observed_services": services,
            "possible_downstream_symptoms": list(scenario.red_herrings),
        }
    if endpoint_name in {"deployments", "config_changes", "changes"}:
        change_lines = [
            line
            for lines in [*scenario.logs.values(), *scenario.red_herring_logs.values()]
            for line in lines
            if any(term in line.lower() for term in ("deploy", "config", "canary", "rollback"))
        ]
        return {"endpoint": "deployments", "events": change_lines[:8]}
    if endpoint_name in {"metrics_summary", "metrics", "slo"}:
        return {
            "endpoint": "metrics_summary",
            "metrics": {
                service: {
                    name: {"first": values[0], "last": values[-1]}
                    for name, values in metrics.items()
                    if values
                }
                for service, metrics in _merged_metrics(scenario).items()
            },
        }
    if endpoint_name in {"runbook", "remediation_runbook"}:
        return {
            "endpoint": "runbook",
            "guidance": [
                "confirm root cause with logs or metrics before remediation",
                "avoid broad destructive mitigations when symptoms are downstream",
                "send a stakeholder update only after separating facts from hypotheses",
            ],
            "known_dangerous_actions": list(scenario.dangerous_fix_ids),
        }
    return {
        "endpoint": endpoint,
        "error": "unknown_endpoint",
        "known_endpoints": [
            "service_graph",
            "deployments",
            "metrics_summary",
            "runbook",
        ],
    }
