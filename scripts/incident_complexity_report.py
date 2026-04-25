"""Generate a clean CSV report for incident-complexity evaluation signals."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from incident_commander_env.evaluation import action, evaluate_candidate
from incident_commander_env.models import AgentRole
from incident_commander_env.scenarios import HIDDEN_SCENARIOS, SCENARIOS, IncidentScenario


def _chain_text(scenario: IncidentScenario) -> str:
    return " -> ".join(scenario.causal_chain or (scenario.root_cause,))


def _red_herring_service(scenario: IncidentScenario) -> str:
    if scenario.red_herring_logs:
        return next(iter(scenario.red_herring_logs))
    if scenario.red_herring_metrics:
        return next(iter(scenario.red_herring_metrics))
    if scenario.symptom_services:
        return scenario.symptom_services[0]
    return scenario.affected_service


def ideal_chain_aware_candidate(scenario: IncidentScenario) -> list[dict[str, Any]]:
    evidence = " ".join(scenario.evidence_terms)
    chain = _chain_text(scenario)
    return [
        action("check_metrics", AgentRole.MONITOR, {"service": scenario.affected_service}),
        action(
            "query_logs",
            AgentRole.INVESTIGATOR,
            {"service": scenario.affected_service, "limit": 8},
        ),
        action(
            "query_api",
            AgentRole.INVESTIGATOR,
            {"endpoint": "service_graph", "service": scenario.affected_service},
        ),
        action(
            "share_note",
            AgentRole.INVESTIGATOR,
            {"note": f"Evidence: {evidence}. Causal chain: {chain}."},
        ),
        action(
            "submit_root_cause",
            AgentRole.INVESTIGATOR,
            {
                "root_cause": scenario.root_cause,
                "confidence": 0.92,
                "evidence": f"{evidence}. Chain: {chain}.",
            },
        ),
        action("deploy_fix", AgentRole.REMEDIATOR, {"fix_id": scenario.canonical_fix_id}),
        action(
            "send_update",
            AgentRole.COMMUNICATOR,
            {
                "message": (
                    f"{scenario.affected_service} impact is mitigated. "
                    f"Confirmed root cause: {scenario.root_cause}. "
                    f"Evidence followed the chain {chain}. {scenario.stakeholder_impact}"
                )
            },
        ),
        action(
            "finish_incident",
            AgentRole.COMMUNICATOR,
            {"summary": "Closed after evidence-backed root cause, safe mitigation, and update."},
        ),
    ]


def red_herring_chaser_candidate(scenario: IncidentScenario) -> list[dict[str, Any]]:
    service = _red_herring_service(scenario)
    misleading = (
        scenario.misleading_root_causes[0]
        if scenario.misleading_root_causes
        else "unrelated downstream symptom"
    )
    return [
        action("check_metrics", AgentRole.MONITOR, {"service": service}),
        action("query_logs", AgentRole.INVESTIGATOR, {"service": service, "limit": 6}),
        action(
            "share_note",
            AgentRole.INVESTIGATOR,
            {"note": f"Following suspected red herring: {misleading} on {service}."},
        ),
        action(
            "submit_root_cause",
            AgentRole.INVESTIGATOR,
            {
                "root_cause": misleading,
                "confidence": 0.88,
                "evidence": f"{service} looked suspicious, so this must be the incident.",
            },
        ),
    ]


def premature_communicator_candidate(scenario: IncidentScenario) -> list[dict[str, Any]]:
    evidence = " ".join(scenario.evidence_terms)
    return [
        action("check_metrics", AgentRole.MONITOR, {"service": scenario.affected_service}),
        action(
            "send_update",
            AgentRole.COMMUNICATOR,
            {
                "message": (
                    f"{scenario.affected_service} is fully resolved after remediation. "
                    f"Root cause was {scenario.root_cause}."
                )
            },
        ),
        action(
            "query_logs",
            AgentRole.INVESTIGATOR,
            {"service": scenario.affected_service, "limit": 5},
        ),
        action(
            "submit_root_cause",
            AgentRole.INVESTIGATOR,
            {
                "root_cause": scenario.root_cause,
                "confidence": 0.86,
                "evidence": evidence,
            },
        ),
    ]


def candidate_suite(scenario: IncidentScenario) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    return {
        "ideal_chain_aware": (
            "Should trace origin, ignore red herrings, mitigate safely, and communicate accurately.",
            ideal_chain_aware_candidate(scenario),
        ),
        "red_herring_chaser": (
            "Should be penalized for chasing misleading evidence or false root cause.",
            red_herring_chaser_candidate(scenario),
        ),
        "premature_communicator": (
            "Should expose communication mismatch before evidence and mitigation are established.",
            premature_communicator_candidate(scenario),
        ),
    }


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def build_rows(scenarios: list[IncidentScenario]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hidden_ids = {scenario.scenario_id for scenario in HIDDEN_SCENARIOS}
    for scenario in scenarios:
        for behavior, (notes, candidate) in candidate_suite(scenario).items():
            result = evaluate_candidate(
                candidate,
                label=behavior,
                scenario_id=scenario.scenario_id,
                difficulty=scenario.difficulty,
                include_hidden_scenarios=scenario.scenario_id in hidden_ids,
            )
            rows.append(
                {
                    "scenario_id": result.scenario_id,
                    "difficulty": result.difficulty,
                    "hidden_case": _yes_no(result.hidden_case),
                    "behavior": behavior,
                    "final_score": result.final_score,
                    "base_score": result.base_score,
                    "judge_score": result.judge_score,
                    "security_penalty": result.security_penalty,
                    "accepted": _yes_no(result.accepted),
                    "success": _yes_no(result.success),
                    "resolved": _yes_no(result.resolved),
                    "root_cause_correct": _yes_no(result.root_cause_correct),
                    "red_herring_chased": _yes_no(result.red_herring_chased),
                    "causal_chain_traced": _yes_no(result.causal_chain_traced),
                    "communication_mismatch": _yes_no(
                        result.communication_mismatch_detected
                    ),
                    "secondary_outage": _yes_no(result.secondary_outage),
                    "hallucination_detected": _yes_no(result.hallucination_detected),
                    "evidence_supported": _yes_no(result.evidence_supported),
                    "workflow_complete": _yes_no(result.workflow_complete),
                    "tools_used": " -> ".join(result.tools),
                    "notes": notes,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evals"))
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument("--hidden-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    scenario_source = list(HIDDEN_SCENARIOS) if args.hidden_only else list(SCENARIOS)
    if args.include_hidden and not args.hidden_only:
        scenario_source.extend(HIDDEN_SCENARIOS)
    scenarios = scenario_source[: args.limit] if args.limit else scenario_source

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "hidden" if args.hidden_only else "public_hidden" if args.include_hidden else "public"
    csv_path = args.output_dir / f"incident_complexity_report_{suffix}.csv"
    rows = build_rows(scenarios)

    fieldnames = [
        "scenario_id",
        "difficulty",
        "hidden_case",
        "behavior",
        "final_score",
        "base_score",
        "judge_score",
        "security_penalty",
        "accepted",
        "success",
        "resolved",
        "root_cause_correct",
        "red_herring_chased",
        "causal_chain_traced",
        "communication_mismatch",
        "secondary_outage",
        "hallucination_detected",
        "evidence_supported",
        "workflow_complete",
        "tools_used",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
