"""Generate reproducible training datasets from the incident simulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from incident_commander_env.demo_agents import run_scripted_response
from incident_commander_env.scenarios import SCENARIOS, IncidentScenario


SYSTEM_PROMPT = """You are an incident-response policy.
Return a JSON list of tool calls. Each call must contain tool_name, agent_role,
and arguments. Use only evidence from tool outputs and shared notes.
"""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def scenario_services(scenario: IncidentScenario) -> list[str]:
    services = set(scenario.logs) | set(scenario.metrics) | {scenario.affected_service}
    return sorted(services)


def metric_summary(scenario: IncidentScenario) -> str:
    parts: list[str] = []
    for service, metrics in scenario.metrics.items():
        for name, values in metrics.items():
            if not values:
                continue
            parts.append(
                f"{service}.{name} moved from {values[0]} to {values[-1]}"
            )
    return "; ".join(parts)


def make_pretrain_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        alerts = " ".join(scenario.alerts)
        logs = " ".join(
            line
            for service_lines in scenario.logs.values()
            for line in service_lines
        )
        metrics = metric_summary(scenario)
        safe_fixes = ", ".join(scenario.safe_fix_ids)
        dangerous_fixes = ", ".join(scenario.dangerous_fix_ids)

        rows.append(
            {
                "id": f"{scenario.scenario_id}:postmortem",
                "split": "pretrain",
                "source": "synthetic_incident_postmortem",
                "text": (
                    f"Incident postmortem: {scenario.title}. "
                    f"The affected service was {scenario.affected_service}. "
                    f"Alerts included: {alerts}. "
                    f"Relevant logs included: {logs}. "
                    f"Relevant metrics: {metrics}. "
                    f"The root cause was {scenario.root_cause}. "
                    f"Safe remediation options were: {safe_fixes}. "
                    f"Unsafe remediations to avoid were: {dangerous_fixes}. "
                    f"Stakeholder impact: {scenario.stakeholder_impact}"
                ),
            }
        )
        rows.append(
            {
                "id": f"{scenario.scenario_id}:runbook",
                "split": "pretrain",
                "source": "synthetic_sre_runbook",
                "text": (
                    f"Runbook for {scenario.affected_service}: first inspect alerts, "
                    f"then check metrics for {', '.join(scenario_services(scenario))}. "
                    f"Query logs for terms such as {', '.join(scenario.evidence_terms)}. "
                    f"Do not claim a root cause until evidence links symptoms to "
                    f"{scenario.root_cause}. Prefer {scenario.canonical_fix_id} when "
                    f"the evidence matches this incident. Avoid red herrings such as "
                    f"{', '.join(scenario.red_herrings)}."
                ),
            }
        )
        rows.append(
            {
                "id": f"{scenario.scenario_id}:communication",
                "split": "pretrain",
                "source": "synthetic_stakeholder_update_guidance",
                "text": (
                    f"Stakeholder communication for {scenario.title}: name the affected "
                    f"service, describe user impact, separate confirmed facts from "
                    f"hypotheses, mention the mitigation only after it is deployed, and "
                    f"set expectations for monitoring or the next update. For this "
                    f"incident, the impact was: {scenario.stakeholder_impact}"
                ),
            }
        )
    return rows


def assistant_actions_for_scenario(scenario: IncidentScenario) -> str:
    service = scenario.affected_service
    evidence = " ".join(scenario.evidence_terms[:4])
    actions = [
        {
            "tool_name": "check_metrics",
            "agent_role": "monitor",
            "arguments": {"service": service},
        },
        {
            "tool_name": "query_logs",
            "agent_role": "investigator",
            "arguments": {"service": service, "limit": 5},
        },
        {
            "tool_name": "share_note",
            "agent_role": "investigator",
            "arguments": {
                "note": f"Evidence points to {scenario.root_cause}: {evidence}"
            },
        },
        {
            "tool_name": "submit_root_cause",
            "agent_role": "investigator",
            "arguments": {
                "root_cause": scenario.root_cause,
                "confidence": 0.9,
                "evidence": evidence,
            },
        },
        {
            "tool_name": "deploy_fix",
            "agent_role": "remediator",
            "arguments": {"fix_id": scenario.canonical_fix_id},
        },
        {
            "tool_name": "send_update",
            "agent_role": "communicator",
            "arguments": {
                "message": (
                    f"{service} impact is mitigated. Root cause was "
                    f"{scenario.root_cause}. We applied {scenario.canonical_fix_id} "
                    "and are monitoring before the next update."
                )
            },
        },
        {
            "tool_name": "finish_incident",
            "agent_role": "communicator",
            "arguments": {
                "summary": (
                    f"{scenario.title} mitigated. Root cause: {scenario.root_cause}. "
                    f"Fix: {scenario.canonical_fix_id}."
                )
            },
        },
    ]
    return json.dumps(actions, ensure_ascii=False)


def make_sft_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        prompt = (
            f"Incident started. Visible alerts: {' | '.join(scenario.alerts)}. "
            f"Coordinate the monitor, investigator, remediator, and communicator. "
            "Return a JSON list of tool calls for the episode."
        )
        rows.append(
            {
                "id": f"{scenario.scenario_id}:ideal_trajectory",
                "split": "sft",
                "scenario_id": scenario.scenario_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": assistant_actions_for_scenario(scenario),
                    },
                ],
            }
        )
        result = run_scripted_response(scenario_id=scenario.scenario_id)
        rows.append(
            {
                "id": f"{scenario.scenario_id}:scripted_transcript",
                "split": "sft",
                "scenario_id": scenario.scenario_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": assistant_actions_for_scenario(scenario),
                    },
                ],
                "metadata": {
                    "scripted_final_score": result["final_score"],
                    "resolved": result["resolved"],
                },
            }
        )
    return rows


def make_eval_rows(seeds_per_scenario: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for seed in range(seeds_per_scenario):
            rows.append(
                {
                    "id": f"{scenario.scenario_id}:seed_{seed}",
                    "split": "eval",
                    "seed": seed,
                    "scenario_id": scenario.scenario_id,
                    "difficulty": scenario.difficulty,
                    "affected_service": scenario.affected_service,
                    "expected_root_cause": scenario.root_cause,
                    "expected_fix_id": scenario.canonical_fix_id,
                    "unsafe_fix_ids": list(scenario.dangerous_fix_ids),
                    "alerts": list(scenario.alerts),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data", help="Output dataset directory.")
    parser.add_argument("--eval-seeds", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pretrain_rows = make_pretrain_rows()
    sft_rows = make_sft_rows()
    eval_rows = make_eval_rows(seeds_per_scenario=args.eval_seeds)

    write_jsonl(out_dir / "pretrain_corpus.jsonl", pretrain_rows)
    write_jsonl(out_dir / "sft_trajectories.jsonl", sft_rows)
    write_jsonl(out_dir / "eval_scenarios.jsonl", eval_rows)

    print(
        json.dumps(
            {
                "pretrain_rows": len(pretrain_rows),
                "sft_rows": len(sft_rows),
                "eval_rows": len(eval_rows),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
