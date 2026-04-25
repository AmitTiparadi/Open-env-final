"""Incident scenario generation and evidence rendering."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class IncidentScenario:
    scenario_id: str
    title: str
    difficulty: str
    affected_service: str
    root_cause: str
    canonical_fix_id: str
    safe_fix_ids: tuple[str, ...]
    dangerous_fix_ids: tuple[str, ...]
    alerts: tuple[str, ...]
    logs: Dict[str, tuple[str, ...]]
    metrics: Dict[str, Dict[str, tuple[float, ...]]]
    evidence_terms: tuple[str, ...]
    red_herrings: tuple[str, ...]
    stakeholder_impact: str


SCENARIOS: tuple[IncidentScenario, ...] = (
    IncidentScenario(
        scenario_id="checkout_bad_deploy_memory_leak",
        title="Checkout latency after release v42",
        difficulty="easy",
        affected_service="checkout-api",
        root_cause="bad_deploy_memory_leak",
        canonical_fix_id="rollback_checkout_api_v42",
        safe_fix_ids=("rollback_checkout_api_v42", "disable_checkout_recommendations_flag"),
        dangerous_fix_ids=("restart_primary_database", "purge_all_sessions"),
        alerts=(
            "P1: checkout-api p95 latency above 2400 ms for 8 minutes",
            "checkout-api container memory at 93 percent and rising",
            "payment-worker queue depth elevated but stable",
        ),
        logs={
            "checkout-api": (
                "10:02 deploy release=checkout-api:v42 commit=9af31d2 actor=deploy-bot",
                "10:04 warn heap_growth_mb=512 route=/cart/quote feature=recommendations",
                "10:06 error oom_risk request_id=c-1842 tenant=retail-east",
                "10:07 warn gc_pause_ms=821 route=/checkout feature=recommendations",
                "10:09 info rollback_candidate release=checkout-api:v41 healthy_baseline=true",
            ),
            "payment-worker": (
                "10:05 warn queue_depth=184 reason=upstream checkout-api latency",
                "10:08 info retries within budget; no payment processor errors",
            ),
        },
        metrics={
            "checkout-api": {
                "latency_p95_ms": (180, 210, 520, 1200, 2600, 3100),
                "memory_percent": (48, 52, 68, 81, 91, 95),
                "error_rate": (0.01, 0.01, 0.03, 0.08, 0.14, 0.17),
            },
            "payment-worker": {
                "queue_depth": (12, 18, 55, 118, 184, 201),
                "error_rate": (0.00, 0.00, 0.01, 0.01, 0.01, 0.01),
            },
        },
        evidence_terms=(
            "checkout-api:v42",
            "heap_growth_mb",
            "memory",
            "gc_pause_ms",
            "recommendations",
            "rollback_checkout_api_v42",
        ),
        red_herrings=("payment-worker queue", "retail-east tenant"),
        stakeholder_impact="Checkout is slow for a subset of retail-east users; payments are delayed but not failing.",
    ),
    IncidentScenario(
        scenario_id="orders_db_connection_pool",
        title="Orders API saturation during inventory sync",
        difficulty="medium",
        affected_service="orders-api",
        root_cause="db_connection_pool_exhaustion",
        canonical_fix_id="increase_orders_pool_and_restart_workers",
        safe_fix_ids=("increase_orders_pool_and_restart_workers",),
        dangerous_fix_ids=("drop_orders_read_replica", "disable_all_inventory_sync"),
        alerts=(
            "P1: orders-api 5xx rate above 8 percent",
            "database connection wait time above 1200 ms",
            "inventory-sync retries climbing",
        ),
        logs={
            "orders-api": (
                "14:21 error db_pool_timeout pool=orders-main waited_ms=1500",
                "14:22 warn active_connections=80 pool_max=80 endpoint=/orders/create",
                "14:23 error request_failed reason=db_pool_timeout trace=o-771",
                "14:25 info config_change inventory-sync batch_size=5000",
            ),
            "inventory-sync": (
                "14:20 info started full catalog reconciliation batch_size=5000",
                "14:24 warn retrying order reservation due upstream orders-api 503",
            ),
        },
        metrics={
            "orders-api": {
                "error_rate": (0.01, 0.02, 0.07, 0.11, 0.13, 0.12),
                "db_wait_ms": (80, 140, 620, 1300, 1700, 1900),
                "cpu_percent": (42, 45, 51, 57, 61, 62),
            },
            "inventory-sync": {
                "retry_count": (0, 2, 18, 55, 89, 144),
                "throughput": (900, 940, 910, 870, 860, 850),
            },
        },
        evidence_terms=(
            "db_pool_timeout",
            "active_connections=80",
            "pool_max=80",
            "db_wait_ms",
            "increase_orders_pool_and_restart_workers",
        ),
        red_herrings=("inventory-sync retries", "catalog reconciliation"),
        stakeholder_impact="New order creation is intermittently failing; inventory sync is a trigger, not the root fault.",
    ),
    IncidentScenario(
        scenario_id="profile_cache_stampede",
        title="Profile service cache stampede",
        difficulty="medium",
        affected_service="profile-service",
        root_cause="cache_ttl_misconfiguration",
        canonical_fix_id="restore_profile_cache_ttl_singleflight",
        safe_fix_ids=("restore_profile_cache_ttl_singleflight", "rate_limit_profile_rebuilds"),
        dangerous_fix_ids=("flush_all_profile_cache", "scale_down_cache_cluster"),
        alerts=(
            "P2: profile-service latency above 900 ms",
            "cache hit rate dropped below 35 percent",
            "user-feed showing elevated dependency latency",
        ),
        logs={
            "profile-service": (
                "09:41 config ttl_seconds=0 source=feature_flag profile-cache-expiry",
                "09:43 warn cache_miss_rate=0.72 keyspace=user_profile",
                "09:44 warn duplicate_rebuilds key=user:483 count=48",
                "09:45 info dependency user-feed slow because profile-service latency",
            ),
            "cache": (
                "09:43 warn evictions normal memory_percent=57",
                "09:45 info cluster healthy nodes=6",
            ),
        },
        metrics={
            "profile-service": {
                "latency_p95_ms": (180, 210, 420, 780, 980, 1240),
                "cache_hit_rate": (0.93, 0.90, 0.58, 0.34, 0.28, 0.25),
                "origin_qps": (400, 420, 1200, 2200, 3100, 3400),
            },
            "cache": {
                "memory_percent": (55, 55, 56, 57, 57, 57),
                "evictions": (12, 11, 13, 12, 12, 13),
            },
        },
        evidence_terms=(
            "ttl_seconds=0",
            "cache_miss_rate",
            "duplicate_rebuilds",
            "cache_hit_rate",
            "restore_profile_cache_ttl_singleflight",
        ),
        red_herrings=("cache evictions", "user-feed latency"),
        stakeholder_impact="Profile reads are slow and user-feed is degraded through dependency latency.",
    ),
    IncidentScenario(
        scenario_id="auth_tls_cert_expiry",
        title="Auth service certificate expiry",
        difficulty="hard",
        affected_service="auth-service",
        root_cause="expired_tls_certificate",
        canonical_fix_id="rotate_auth_service_certificate",
        safe_fix_ids=("rotate_auth_service_certificate",),
        dangerous_fix_ids=("disable_tls_verification_globally", "restart_oauth_database"),
        alerts=(
            "P1: login success rate below 70 percent",
            "edge-gateway upstream TLS failures to auth-service",
            "support tickets mention intermittent login loops",
        ),
        logs={
            "edge-gateway": (
                "18:02 error upstream_tls_handshake_failed service=auth-service verify=certificate_expired",
                "18:03 warn retrying upstream auth-service status=525",
                "18:05 info canary route unaffected service=marketing-site",
            ),
            "auth-service": (
                "18:01 info cpu_percent=34 memory_percent=41",
                "18:04 warn no application exceptions in auth handlers",
                "18:06 info cert_subject=auth.internal expires_at=2026-04-25T12:00:00Z",
            ),
        },
        metrics={
            "auth-service": {
                "login_success_rate": (0.98, 0.96, 0.82, 0.69, 0.63, 0.61),
                "cpu_percent": (30, 33, 34, 34, 35, 35),
                "memory_percent": (39, 40, 41, 41, 42, 42),
            },
            "edge-gateway": {
                "tls_failures": (0, 0, 45, 220, 380, 420),
                "upstream_525": (0, 1, 31, 140, 250, 281),
            },
        },
        evidence_terms=(
            "upstream_tls_handshake_failed",
            "certificate_expired",
            "expires_at",
            "tls_failures",
            "rotate_auth_service_certificate",
        ),
        red_herrings=("login loops", "auth database"),
        stakeholder_impact="Some users cannot log in because edge traffic cannot establish TLS to auth-service.",
    ),
)


def get_scenario(scenario_id: str) -> IncidentScenario:
    for scenario in SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(f"Unknown scenario_id: {scenario_id}")


def generate_scenario(
    seed: Optional[int] = None,
    difficulty: str = "easy",
    preferred_root_cause: Optional[str] = None,
) -> IncidentScenario:
    """Pick a deterministic scenario for reset or curriculum sampling."""

    rng = Random(seed)
    candidates = [
        s
        for s in SCENARIOS
        if s.difficulty == difficulty
        or difficulty == "mixed"
        or (difficulty == "easy" and s.difficulty in {"easy", "medium"})
    ]
    if preferred_root_cause:
        preferred = [s for s in candidates if s.root_cause == preferred_root_cause]
        if preferred:
            candidates = preferred
    if not candidates:
        candidates = list(SCENARIOS)
    return candidates[rng.randrange(len(candidates))]


def render_logs(
    scenario: IncidentScenario,
    service: str,
    query: str = "",
    limit: int = 5,
) -> list[str]:
    lines = list(scenario.logs.get(service, ()))
    if query:
        terms = [term.strip().lower() for term in query.split() if term.strip()]
        filtered = [
            line
            for line in lines
            if any(term in line.lower() for term in terms)
        ]
        lines = filtered or lines
    return lines[: max(1, limit)]


def render_metrics(
    scenario: IncidentScenario,
    service: str,
    metric: str = "",
) -> dict[str, list[float]]:
    metrics = scenario.metrics.get(service, {})
    if metric:
        return {metric: list(metrics.get(metric, ()))}
    return {name: list(values) for name, values in metrics.items()}


def scenario_ids() -> list[str]:
    return [scenario.scenario_id for scenario in SCENARIOS]


def all_root_causes() -> Iterable[str]:
    return {scenario.root_cause for scenario in SCENARIOS}
