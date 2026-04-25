"""Generate targeted synthetic data for known Incident Commander failure modes.

This intentionally does not create generic incidents. It expands the existing
hand-authored scenarios into focused examples for:
- compact JSON termination
- red-herring rejection
- cascading-origin tracing
- communicator uncertainty before mitigation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from incident_commander_env.models import AgentRole
from incident_commander_env.scenarios import SCENARIOS, IncidentScenario
from training.prepare_data import (
    SYSTEM_PROMPT,
    make_eval_rows,
    make_pretrain_rows,
    make_sft_rows,
    write_jsonl,
)


TARGETED_SYSTEM_PROMPT = (
    SYSTEM_PROMPT.strip()
    + "\n/no_think\nReturn compact JSON only. Use 6 to 9 calls. "
    "Stop after finish_incident. Do not reveal thinking or `</think>`."
)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def action(tool_name: str, role: AgentRole, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "agent_role": role.value,
        "arguments": arguments,
    }


def scenario_prompt(scenario: IncidentScenario, focus: str) -> str:
    return (
        f"/no_think\nIncident: {scenario.title}. Focus: {focus}. "
        f"Affected service: {scenario.affected_service}. "
        f"Alerts: {' | '.join(scenario.alerts)}. "
        "Return compact JSON tool calls only. First character must be `[`; "
        "last character must be `]`."
    )


def chain_text(scenario: IncidentScenario) -> str:
    return " -> ".join(scenario.causal_chain or (scenario.root_cause,))


def evidence_text(scenario: IncidentScenario, max_terms: int = 5) -> str:
    terms = [*scenario.evidence_terms[:max_terms], *scenario.causal_chain[:2]]
    return " ".join(dict.fromkeys(terms))


def red_herring_service(scenario: IncidentScenario) -> str:
    if scenario.red_herring_logs:
        return next(iter(scenario.red_herring_logs))
    if scenario.red_herring_metrics:
        return next(iter(scenario.red_herring_metrics))
    if scenario.symptom_services:
        return scenario.symptom_services[0]
    return scenario.affected_service


def misleading_cause(scenario: IncidentScenario) -> str:
    if scenario.misleading_root_causes:
        return scenario.misleading_root_causes[0]
    if scenario.red_herrings:
        return scenario.red_herrings[0]
    return "downstream symptom"


def compact_success_actions(scenario: IncidentScenario) -> list[dict[str, Any]]:
    service = scenario.affected_service
    evidence = evidence_text(scenario)
    chain = chain_text(scenario)
    return [
        action("check_metrics", AgentRole.MONITOR, {"service": service}),
        action("query_logs", AgentRole.INVESTIGATOR, {"service": service, "limit": 6}),
        action("query_api", AgentRole.INVESTIGATOR, {"endpoint": "metrics_summary", "service": service}),
        action(
            "share_note",
            AgentRole.INVESTIGATOR,
            {"note": f"Evidence supports {scenario.root_cause}: {evidence}. Chain: {chain}."},
        ),
        action(
            "submit_root_cause",
            AgentRole.INVESTIGATOR,
            {"root_cause": scenario.root_cause, "confidence": 0.9, "evidence": evidence},
        ),
        action("deploy_fix", AgentRole.REMEDIATOR, {"fix_id": scenario.canonical_fix_id}),
        action(
            "send_update",
            AgentRole.COMMUNICATOR,
            {
                "message": (
                    f"{service} impact is mitigated. Confirmed cause: "
                    f"{scenario.root_cause}. Monitoring after {scenario.canonical_fix_id}."
                )
            },
        ),
        action(
            "finish_incident",
            AgentRole.COMMUNICATOR,
            {"summary": f"{service} mitigated; cause {scenario.root_cause}; fix {scenario.canonical_fix_id}."},
        ),
    ]


def red_herring_rejection_actions(scenario: IncidentScenario) -> list[dict[str, Any]]:
    service = scenario.affected_service
    false_service = red_herring_service(scenario)
    false_cause = misleading_cause(scenario)
    evidence = evidence_text(scenario)
    return [
        action("check_metrics", AgentRole.MONITOR, {"service": false_service}),
        action("query_logs", AgentRole.INVESTIGATOR, {"service": false_service, "limit": 4}),
        action(
            "share_note",
            AgentRole.INVESTIGATOR,
            {
                "note": (
                    f"{false_service} suggests {false_cause}, but this is a red herring; "
                    f"it does not explain {service} evidence."
                )
            },
        ),
        action("check_metrics", AgentRole.MONITOR, {"service": service}),
        action("query_logs", AgentRole.INVESTIGATOR, {"service": service, "limit": 6}),
        action(
            "submit_root_cause",
            AgentRole.INVESTIGATOR,
            {"root_cause": scenario.root_cause, "confidence": 0.86, "evidence": evidence},
        ),
        action("deploy_fix", AgentRole.REMEDIATOR, {"fix_id": scenario.canonical_fix_id}),
        action(
            "send_update",
            AgentRole.COMMUNICATOR,
            {"message": f"{service} is mitigated; false lead {false_cause} was ruled out."},
        ),
        action("finish_incident", AgentRole.COMMUNICATOR, {"summary": "Closed after ruling out red herring."}),
    ]


def cascading_origin_actions(scenario: IncidentScenario) -> list[dict[str, Any]]:
    service = scenario.affected_service
    symptom = scenario.symptom_services[0] if scenario.symptom_services else service
    evidence = evidence_text(scenario)
    chain = chain_text(scenario)
    return [
        action("query_api", AgentRole.INVESTIGATOR, {"endpoint": "service_graph", "service": service}),
        action("check_metrics", AgentRole.MONITOR, {"service": symptom}),
        action("check_metrics", AgentRole.MONITOR, {"service": service}),
        action("query_logs", AgentRole.INVESTIGATOR, {"service": service, "limit": 6}),
        action(
            "share_note",
            AgentRole.INVESTIGATOR,
            {"note": f"Downstream symptom {symptom}; origin chain is {chain}."},
        ),
        action(
            "submit_root_cause",
            AgentRole.INVESTIGATOR,
            {"root_cause": scenario.root_cause, "confidence": 0.91, "evidence": f"{evidence}. {chain}"},
        ),
        action("deploy_fix", AgentRole.REMEDIATOR, {"fix_id": scenario.canonical_fix_id}),
        action(
            "send_update",
            AgentRole.COMMUNICATOR,
            {"message": f"Mitigated origin {service}; downstream symptom {symptom} is recovering."},
        ),
        action("finish_incident", AgentRole.COMMUNICATOR, {"summary": f"Origin traced: {chain}."}),
    ]


def communicator_uncertainty_actions(scenario: IncidentScenario) -> list[dict[str, Any]]:
    service = scenario.affected_service
    evidence = evidence_text(scenario)
    return [
        action("check_metrics", AgentRole.MONITOR, {"service": service}),
        action(
            "send_update",
            AgentRole.COMMUNICATOR,
            {
                "message": (
                    f"We are investigating {service} impact. Root cause is not confirmed yet; "
                    "next update after logs and remediation decision."
                )
            },
        ),
        action("query_logs", AgentRole.INVESTIGATOR, {"service": service, "limit": 6}),
        action("share_note", AgentRole.INVESTIGATOR, {"note": f"Confirmed evidence: {evidence}."}),
        action(
            "submit_root_cause",
            AgentRole.INVESTIGATOR,
            {"root_cause": scenario.root_cause, "confidence": 0.88, "evidence": evidence},
        ),
        action("deploy_fix", AgentRole.REMEDIATOR, {"fix_id": scenario.canonical_fix_id}),
        action(
            "send_update",
            AgentRole.COMMUNICATOR,
            {
                "message": (
                    f"{service} mitigation is deployed. Confirmed cause: "
                    f"{scenario.root_cause}. Monitoring recovery."
                )
            },
        ),
        action("finish_incident", AgentRole.COMMUNICATOR, {"summary": "Closed after confirmed mitigation."}),
    ]


def make_message_row(
    scenario: IncidentScenario,
    behavior: str,
    focus: str,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": f"{scenario.scenario_id}:targeted:{behavior}",
        "split": "sft",
        "scenario_id": scenario.scenario_id,
        "behavior": behavior,
        "messages": [
            {"role": "system", "content": TARGETED_SYSTEM_PROMPT},
            {"role": "user", "content": scenario_prompt(scenario, focus)},
            {"role": "assistant", "content": compact_json(actions)},
        ],
    }


def make_targeted_sft_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        rows.extend(
            [
                make_message_row(
                    scenario,
                    "compact_success",
                    "solve quickly with a naturally terminating JSON list",
                    compact_success_actions(scenario),
                ),
                make_message_row(
                    scenario,
                    "red_herring_rejection",
                    "inspect and explicitly reject a plausible false lead",
                    red_herring_rejection_actions(scenario),
                ),
                make_message_row(
                    scenario,
                    "cascading_origin_trace",
                    "trace downstream symptoms back to the causal origin",
                    cascading_origin_actions(scenario),
                ),
                make_message_row(
                    scenario,
                    "communicator_uncertainty",
                    "communicate uncertainty before mitigation and confirmed facts after",
                    communicator_uncertainty_actions(scenario),
                ),
            ]
        )
    return rows


def make_targeted_pretrain_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        false_cause = misleading_cause(scenario)
        false_service = red_herring_service(scenario)
        rows.extend(
            [
                {
                    "id": f"{scenario.scenario_id}:targeted:red_herring_contrast",
                    "split": "pretrain",
                    "source": "targeted_red_herring_contrast",
                    "text": (
                        f"In {scenario.title}, {false_service} can appear suspicious and "
                        f"may suggest {false_cause}. Treat that as a red herring unless it "
                        f"explains {scenario.affected_service} evidence: {evidence_text(scenario)}. "
                        f"The correct root cause is {scenario.root_cause}."
                    ),
                },
                {
                    "id": f"{scenario.scenario_id}:targeted:cascade_trace",
                    "split": "pretrain",
                    "source": "targeted_cascade_trace",
                    "text": (
                        f"Cascading chain for {scenario.title}: {chain_text(scenario)}. "
                        "Fix the origin, not only the downstream symptom. "
                        f"The safe fix is {scenario.canonical_fix_id}."
                    ),
                },
                {
                    "id": f"{scenario.scenario_id}:targeted:termination_contract",
                    "split": "pretrain",
                    "source": "targeted_json_termination_contract",
                    "text": (
                        "A good Incident Commander answer is a compact JSON array of 6 to 9 "
                        "tool calls. It ends with finish_incident and stops immediately after "
                        "the closing bracket. It does not include markdown, prose, or repeated plans."
                    ),
                },
                {
                    "id": f"{scenario.scenario_id}:targeted:communication_contract",
                    "split": "pretrain",
                    "source": "targeted_communication_contract",
                    "text": (
                        f"For {scenario.affected_service}, the communicator must say "
                        "'investigating' before root cause is confirmed, and only say "
                        f"'{scenario.root_cause}' or '{scenario.canonical_fix_id}' after the "
                        "team has evidence and remediation is deployed."
                    ),
                },
            ]
        )
    return rows


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--replace-main",
        action="store_true",
        help="Overwrite pretrain_corpus.jsonl and sft_trajectories.jsonl with augmented data.",
    )
    parser.add_argument("--eval-seeds", type=int, default=5)
    args = parser.parse_args()

    base_pretrain = make_pretrain_rows()
    base_sft = make_sft_rows()
    targeted_pretrain = make_targeted_pretrain_rows()
    targeted_sft = make_targeted_sft_rows()
    augmented_pretrain = [*base_pretrain, *targeted_pretrain]
    augmented_sft = [*base_sft, *targeted_sft]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "targeted_pretrain_corpus.jsonl", targeted_pretrain)
    write_jsonl(args.out_dir / "targeted_sft_trajectories.jsonl", targeted_sft)
    write_jsonl(args.out_dir / "pretrain_corpus_augmented.jsonl", augmented_pretrain)
    write_jsonl(args.out_dir / "sft_trajectories_augmented.jsonl", augmented_sft)
    write_jsonl(args.out_dir / "eval_scenarios.jsonl", make_eval_rows(args.eval_seeds))
    if args.replace_main:
        write_jsonl(args.out_dir / "pretrain_corpus.jsonl", augmented_pretrain)
        write_jsonl(args.out_dir / "sft_trajectories.jsonl", augmented_sft)

    report = {
        "base_pretrain_rows": len(base_pretrain),
        "targeted_pretrain_rows": len(targeted_pretrain),
        "augmented_pretrain_rows": len(augmented_pretrain),
        "base_sft_rows": len(base_sft),
        "targeted_sft_rows": len(targeted_sft),
        "augmented_sft_rows": len(augmented_sft),
        "replace_main": args.replace_main,
        "targeted_behaviors": [
            "compact_success",
            "red_herring_rejection",
            "cascading_origin_trace",
            "communicator_uncertainty",
        ],
    }
    write_report(args.out_dir / "targeted_data_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
