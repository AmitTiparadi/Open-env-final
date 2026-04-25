"""Reward helpers for Incident Commander."""

from __future__ import annotations

import re
from typing import Iterable

from incident_commander_env.scenarios import IncidentScenario


ROOT_CAUSE_ALIASES: dict[str, tuple[str, ...]] = {
    "bad_deploy_memory_leak": (
        "bad deploy",
        "memory leak",
        "heap growth",
        "checkout-api:v42",
        "release v42",
        "recommendations feature",
    ),
    "db_connection_pool_exhaustion": (
        "db pool",
        "connection pool",
        "db_pool_timeout",
        "pool exhaustion",
        "pool_max=80",
    ),
    "cache_ttl_misconfiguration": (
        "ttl_seconds=0",
        "ttl misconfiguration",
        "cache stampede",
        "cache miss",
        "singleflight",
    ),
    "expired_tls_certificate": (
        "expired cert",
        "certificate expired",
        "expired tls",
        "tls handshake",
        "cert expiry",
    ),
}

FALSE_ROOT_CAUSE_TERMS = (
    "disk full",
    "network partition",
    "dns outage",
    "database corruption",
    "payment processor outage",
    "ddos",
    "credential leak",
)


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9:_=-]+", " ", value.lower()).strip()


def evidence_terms_for(scenario: IncidentScenario) -> tuple[str, ...]:
    aliases = ROOT_CAUSE_ALIASES.get(scenario.root_cause, ())
    return tuple(dict.fromkeys((*scenario.evidence_terms, *aliases)))


def matches_root_cause(candidate: str, scenario: IncidentScenario) -> bool:
    text = normalize_text(candidate)
    if normalize_text(scenario.root_cause) in text:
        return True
    return any(normalize_text(alias) in text for alias in evidence_terms_for(scenario))


def contains_real_evidence(text: str, scenario: IncidentScenario) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(term) in normalized for term in evidence_terms_for(scenario))


def detects_false_root_cause(text: str, scenario: IncidentScenario) -> bool:
    normalized = normalize_text(text)
    if any(term in normalized for term in FALSE_ROOT_CAUSE_TERMS):
        return True
    other_aliases: list[str] = []
    for root, aliases in ROOT_CAUSE_ALIASES.items():
        if root != scenario.root_cause:
            other_aliases.extend(aliases)
            other_aliases.append(root)
    return any(normalize_text(alias) in normalized for alias in other_aliases)


def score_status_update(
    message: str,
    scenario: IncidentScenario,
    root_cause_correct: bool,
    resolved: bool,
) -> tuple[float, bool, list[str]]:
    """Score a stakeholder update for clarity, accuracy, and restraint."""

    text = normalize_text(message)
    reasons: list[str] = []
    score = 0.0

    if scenario.affected_service in text or scenario.affected_service.replace("-", " ") in text:
        score += 0.04
        reasons.append("names affected service")
    if any(word in text for word in ("impact", "users", "latency", "login", "failing", "slow")):
        score += 0.04
        reasons.append("states user impact")
    if root_cause_correct and matches_root_cause(message, scenario):
        score += 0.05
        reasons.append("uses correct root cause")
    elif "root cause" in text or "caused by" in text:
        score -= 0.05
        reasons.append("asserts root cause before evidence")
    if resolved and any(word in text for word in ("resolved", "mitigated", "rolled back", "rotated", "restored")):
        score += 0.04
        reasons.append("states mitigation status")
    if any(word in text for word in ("investigating", "monitoring", "next update", "eta")):
        score += 0.03
        reasons.append("sets expectation")

    hallucinated = detects_false_root_cause(message, scenario)
    if hallucinated:
        score -= 0.10
        reasons.append("contains unsupported root cause")

    return max(0.0, min(0.2, score)), hallucinated, reasons


def speed_bonus(step_count: int, max_turns: int, resolved: bool) -> float:
    if not resolved:
        return 0.0
    remaining = max(0, max_turns - step_count)
    return round(0.1 * (remaining / max_turns), 4)


def best_evidence_overlap(notes: Iterable[str], scenario: IncidentScenario) -> float:
    evidence = [normalize_text(term) for term in evidence_terms_for(scenario)]
    joined = normalize_text(" ".join(notes))
    if not evidence:
        return 0.0
    hits = sum(1 for term in evidence if term and term in joined)
    return min(0.08, round(0.01 * hits, 4))
