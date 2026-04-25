"""Incident scenario generation and evidence rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    causal_chain: tuple[str, ...] = ()
    origin_service: str = ""
    symptom_services: tuple[str, ...] = ()
    red_herring_logs: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    red_herring_metrics: Dict[str, Dict[str, tuple[float, ...]]] = field(default_factory=dict)
    misleading_root_causes: tuple[str, ...] = ()


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
        causal_chain=(
            "checkout-api:v42 recommendations rollout",
            "heap growth and long GC pauses",
            "checkout-api latency and OOM risk",
            "payment-worker queue backs up as a downstream symptom",
        ),
        origin_service="checkout-api",
        symptom_services=("payment-worker",),
        red_herring_logs={
            "payment-worker": (
                "10:06 warn processor_latency_ms=510 provider=paygate-west possible_external_issue=true",
                "10:07 error transient_payment_decline code=issuer_unavailable sample_size=3",
            ),
            "orders-api": (
                "10:06 warn db_pool_wait_ms=380 endpoint=/orders/create below_slo_threshold=true",
            ),
        },
        red_herring_metrics={
            "payment-worker": {
                "processor_latency_ms": (120, 130, 180, 300, 510, 490),
            },
            "orders-api": {
                "db_wait_ms": (60, 65, 80, 130, 250, 380),
            },
        },
        misleading_root_causes=("payment processor outage", "orders db pool exhaustion"),
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
        causal_chain=(
            "inventory-sync batch size increase",
            "orders-api saturates orders-main connection pool",
            "db wait time and 5xx rise on order creation",
            "inventory-sync retries increase as a downstream symptom",
        ),
        origin_service="orders-api",
        symptom_services=("inventory-sync",),
        red_herring_logs={
            "checkout-api": (
                "14:22 warn heap_growth_mb=220 route=/cart/quote unrelated_to_orders=true",
                "14:24 info release=checkout-api:v42 canary_healthy=true",
            ),
            "inventory-sync": (
                "14:25 warn full_catalog_reconciliation still_running suspected_batch_issue=true",
            ),
        },
        red_herring_metrics={
            "checkout-api": {
                "memory_percent": (45, 46, 50, 55, 59, 61),
            },
            "inventory-sync": {
                "cpu_percent": (52, 55, 60, 67, 69, 70),
            },
        },
        misleading_root_causes=("checkout memory leak", "inventory sync outage"),
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
        causal_chain=(
            "profile cache TTL feature flag sets ttl_seconds=0",
            "profile-service cache hit rate collapses",
            "origin QPS and duplicate rebuilds spike",
            "user-feed latency rises through dependency calls",
        ),
        origin_service="profile-service",
        symptom_services=("user-feed", "cache"),
        red_herring_logs={
            "user-feed": (
                "09:44 error feed_render_timeout component=ranking possible_root=true",
                "09:45 warn ranking_model_latency_ms=780 but dependency_wait_ms=1210",
            ),
            "cache": (
                "09:44 warn cache_eviction_spike keyspace=session count=900 unrelated_keyspace=true",
            ),
        },
        red_herring_metrics={
            "user-feed": {
                "ranking_latency_ms": (110, 120, 180, 360, 780, 760),
            },
            "cache": {
                "session_evictions": (20, 22, 60, 260, 900, 880),
            },
        },
        misleading_root_causes=("user-feed ranking regression", "cache cluster memory pressure"),
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
        causal_chain=(
            "auth-service certificate expires",
            "edge-gateway TLS handshakes to auth-service fail",
            "login success rate drops",
            "support sees intermittent login loops",
        ),
        origin_service="auth-service",
        symptom_services=("edge-gateway",),
        red_herring_logs={
            "oauth-database": (
                "18:03 warn slow_query_ms=640 table=sessions below_incident_threshold=true",
                "18:04 info connection_pool healthy active=32 max=120",
            ),
            "auth-service": (
                "18:05 warn password_reset_errors count=4 possible_user_report_noise=true",
            ),
        },
        red_herring_metrics={
            "oauth-database": {
                "query_p95_ms": (80, 90, 120, 250, 640, 610),
            },
        },
        misleading_root_causes=("oauth database timeout", "password reset regression"),
    ),
)


HIDDEN_SCENARIOS: tuple[IncidentScenario, ...] = (
    IncidentScenario(
        scenario_id="hidden_search_ranker_cache_poison",
        title="Search results collapse after ranker canary",
        difficulty="hard",
        affected_service="search-api",
        root_cause="ranking_model_feature_rollout_poisoned_cache",
        canonical_fix_id="rollback_search_ranker_v73_purge_poisoned_cache",
        safe_fix_ids=(
            "rollback_search_ranker_v73_purge_poisoned_cache",
            "disable_ranker_embedding_v12_canary",
        ),
        dangerous_fix_ids=("reindex_entire_catalog", "flush_all_edge_cache"),
        alerts=(
            "P1: search-api zero-result rate above 42 percent for mobile users",
            "catalog-db replica lag elevated but below paging threshold",
            "ads-ranking timeout rate rising after search retries increased",
        ),
        logs={
            "search-api": (
                "11:31 deploy canary ranker_v73 cohort=mobile-us pct=20",
                "11:34 warn zero_results tenant=mobile-us query_class=head embedding_schema=v12",
                "11:36 error cache_poison key=ranker:v73:mobile-us reason=nan_embedding_vector",
                "11:39 info fallback_ranker_v72 healthy_cache=true cohort=web-us",
            ),
            "catalog-db": (
                "11:35 warn replica_lag_ms=210 source=bulk_price_import",
                "11:37 info primary_qps stable no_lock_waits=true",
            ),
            "ads-ranking": (
                "11:36 warn timeout_ms=1200 reason=upstream search-api retries",
            ),
        },
        metrics={
            "search-api": {
                "zero_result_rate": (0.03, 0.04, 0.11, 0.24, 0.42, 0.47),
                "ranker_cache_hit_rate": (0.82, 0.84, 0.86, 0.88, 0.89, 0.88),
                "latency_p95_ms": (210, 230, 360, 780, 1180, 1410),
            },
            "catalog-db": {
                "replica_lag_ms": (40, 55, 120, 190, 210, 205),
                "lock_wait_ms": (2, 3, 2, 4, 3, 3),
            },
            "ads-ranking": {
                "timeout_rate": (0.01, 0.01, 0.03, 0.08, 0.12, 0.13),
            },
        },
        evidence_terms=(
            "ranker_v73",
            "embedding_schema=v12",
            "cache_poison",
            "nan_embedding_vector",
            "zero_result_rate",
            "rollback_search_ranker_v73_purge_poisoned_cache",
        ),
        red_herrings=("catalog-db replica lag", "ads-ranking timeouts"),
        stakeholder_impact="Mobile shoppers see empty search results; ads are slow because search retries increased.",
        causal_chain=(
            "ranker_v73 mobile canary uses embedding_schema=v12",
            "nan embedding vectors poison ranker cache",
            "search-api zero-result rate rises for mobile users",
            "ads-ranking timeouts rise because search retries increase",
        ),
        origin_service="search-api",
        symptom_services=("ads-ranking", "catalog-db"),
        red_herring_logs={
            "catalog-db": (
                "11:36 error bulk_price_import retry_count=8 possible_catalog_staleness=true",
                "11:38 warn index_refresh_delay_ms=900 below_alert_threshold=true",
            ),
            "ads-ranking": (
                "11:37 error model_timeout model=ads_ranker_v18 canary=false",
            ),
        },
        red_herring_metrics={
            "catalog-db": {
                "index_refresh_delay_ms": (90, 120, 260, 510, 900, 850),
            },
            "ads-ranking": {
                "model_latency_ms": (120, 130, 260, 700, 1100, 1050),
            },
        },
        misleading_root_causes=("catalog index lag", "ads ranker timeout"),
    ),
    IncidentScenario(
        scenario_id="hidden_billing_idempotency_drift",
        title="Duplicate billing attempts after config rollout",
        difficulty="hard",
        affected_service="billing-worker",
        root_cause="billing_idempotency_key_salt_mismatch",
        canonical_fix_id="rollback_billing_idempotency_salt_and_replay_dedupe",
        safe_fix_ids=("rollback_billing_idempotency_salt_and_replay_dedupe",),
        dangerous_fix_ids=("truncate_billing_ledger", "disable_payment_authorization"),
        alerts=(
            "P1: duplicate authorization attempts above 6 percent",
            "payments-api error rate stable despite billing retries",
            "ledger-writer latency elevated after dedupe queue growth",
        ),
        logs={
            "billing-worker": (
                "07:12 config idempotency_salt=billing-v2 source=regional_override",
                "07:14 warn idempotency_miss_rate=0.31 previous_salt=billing-v1",
                "07:15 error duplicate_charge_guard triggered order=o-4421 auth=a-991",
                "07:17 info rollback_candidate salt=billing-v1 replay_dedupe_window=45m",
            ),
            "payments-api": (
                "07:15 info processor_status=healthy auth_latency_ms=140",
                "07:16 warn retry_from=billing-worker duplicate_guard=true",
            ),
            "ledger-writer": (
                "07:16 warn queue_depth=2400 reason=dedupe_events_backlog",
            ),
        },
        metrics={
            "billing-worker": {
                "duplicate_auth_rate": (0.001, 0.002, 0.018, 0.041, 0.063, 0.071),
                "idempotency_miss_rate": (0.01, 0.01, 0.09, 0.22, 0.31, 0.35),
                "retry_count": (20, 25, 180, 720, 1300, 1600),
            },
            "payments-api": {
                "processor_error_rate": (0.002, 0.002, 0.003, 0.002, 0.003, 0.002),
            },
            "ledger-writer": {
                "queue_depth": (90, 110, 420, 1200, 2400, 3100),
            },
        },
        evidence_terms=(
            "idempotency_salt=billing-v2",
            "previous_salt=billing-v1",
            "idempotency_miss_rate",
            "duplicate_charge_guard",
            "rollback_billing_idempotency_salt_and_replay_dedupe",
        ),
        red_herrings=("payments-api", "ledger-writer latency"),
        stakeholder_impact="Some customers may see duplicate authorization attempts; payment processor health is normal.",
        causal_chain=(
            "regional override changes billing idempotency salt",
            "billing-worker misses previous idempotency keys",
            "duplicate authorization attempts trigger guards",
            "ledger-writer queue grows from dedupe events backlog",
        ),
        origin_service="billing-worker",
        symptom_services=("payments-api", "ledger-writer"),
        red_herring_logs={
            "payments-api": (
                "07:14 warn processor_decline_rate=0.022 possible_vendor_issue=true",
                "07:16 error auth_timeout sample=5 provider=paygate-east",
            ),
            "ledger-writer": (
                "07:17 warn disk_queue_flush_ms=540 possible_storage_issue=true",
            ),
        },
        red_herring_metrics={
            "payments-api": {
                "processor_decline_rate": (0.004, 0.005, 0.011, 0.018, 0.022, 0.021),
            },
            "ledger-writer": {
                "disk_flush_ms": (40, 45, 90, 220, 540, 520),
            },
        },
        misleading_root_causes=("payment processor outage", "ledger storage latency"),
    ),
    IncidentScenario(
        scenario_id="hidden_notification_poison_pill",
        title="Notification backlog from retrying poison message",
        difficulty="hard",
        affected_service="notification-worker",
        root_cause="notification_poison_pill_retry_loop",
        canonical_fix_id="quarantine_poison_message_and_cap_retries",
        safe_fix_ids=("quarantine_poison_message_and_cap_retries",),
        dangerous_fix_ids=("purge_notification_queue", "scale_down_notification_workers"),
        alerts=(
            "P2: notification delivery delay above 18 minutes",
            "email-provider 429s visible but under vendor limit",
            "worker restarts correlated with a single queue shard",
        ),
        logs={
            "notification-worker": (
                "16:02 error deserialize_failed message_id=n-88421 schema=promo_v9 field=template_vars",
                "16:03 warn retry_loop message_id=n-88421 retry_count=127 shard=queue-3",
                "16:05 error worker_crash reason=poison_pill shard=queue-3",
                "16:06 info dlq_candidate message_id=n-88421 preserve_queue=true",
            ),
            "email-provider": (
                "16:04 warn rate_limit_remaining=74 retry_after_ms=200",
                "16:05 info provider_status=healthy accepted_qps=normal",
            ),
        },
        metrics={
            "notification-worker": {
                "queue_lag_minutes": (1, 2, 6, 12, 18, 24),
                "worker_restarts": (0, 0, 4, 12, 27, 39),
                "dlq_depth": (4, 4, 5, 5, 5, 5),
            },
            "email-provider": {
                "http_429_rate": (0.00, 0.01, 0.02, 0.02, 0.02, 0.02),
            },
        },
        evidence_terms=(
            "message_id=n-88421",
            "deserialize_failed",
            "retry_loop",
            "poison_pill",
            "retry_count=127",
            "quarantine_poison_message_and_cap_retries",
        ),
        red_herrings=("email-provider 429s", "vendor limit"),
        stakeholder_impact="User notifications are delayed; the provider is healthy and one poison message is blocking progress.",
        causal_chain=(
            "promo_v9 message cannot deserialize",
            "notification-worker retries the same poison message",
            "queue shard 3 workers crash repeatedly",
            "notification delivery lag grows while provider remains healthy",
        ),
        origin_service="notification-worker",
        symptom_services=("email-provider",),
        red_herring_logs={
            "email-provider": (
                "16:03 error smtp_421 temporary_failure region=us-east sample=2",
                "16:04 warn vendor_rate_limit_header present but remaining_capacity=74",
            ),
            "template-service": (
                "16:02 warn template_cache_miss_rate=0.18 possible_template_issue=true",
            ),
        },
        red_herring_metrics={
            "template-service": {
                "cache_miss_rate": (0.02, 0.03, 0.07, 0.12, 0.18, 0.17),
            },
        },
        misleading_root_causes=("email provider rate limit", "template cache miss"),
    ),
    IncidentScenario(
        scenario_id="hidden_edge_limiter_cardinality",
        title="Edge throttling after rate-limit key change",
        difficulty="hard",
        affected_service="edge-gateway",
        root_cause="edge_rate_limiter_key_cardinality_explosion",
        canonical_fix_id="rollback_edge_limiter_request_id_key",
        safe_fix_ids=("rollback_edge_limiter_request_id_key", "pin_rate_limiter_key_to_user_tenant"),
        dangerous_fix_ids=("disable_rate_limiting_globally", "flush_all_redis_keys"),
        alerts=(
            "P1: edge-gateway 429 responses above 22 percent",
            "redis CPU saturated in rate-limit cluster",
            "auth-service login failures are rising as a downstream symptom",
        ),
        logs={
            "edge-gateway": (
                "20:42 config ratelimit_key=user_id+tenant_id+request_id source=canary_rl_19",
                "20:44 warn key_cardinality=8.7M window=60s expected=120k",
                "20:45 error limiter_backend_timeout redis_cluster=rl-us-east",
                "20:47 info rollback_candidate config=canary_rl_18 key=user_id+tenant_id",
            ),
            "redis-rate-limit": (
                "20:44 warn cpu_percent=96 evictions=0 memory_percent=63",
                "20:46 warn hot_shards=all reason=key_cardinality_explosion",
            ),
            "auth-service": (
                "20:45 warn login_failed reason=edge_429 upstream_auth_healthy=true",
            ),
        },
        metrics={
            "edge-gateway": {
                "http_429_rate": (0.01, 0.02, 0.08, 0.17, 0.22, 0.29),
                "limiter_timeout_rate": (0.00, 0.00, 0.06, 0.14, 0.21, 0.27),
                "request_rate": (9200, 9300, 9250, 9180, 9100, 9050),
            },
            "redis-rate-limit": {
                "cpu_percent": (44, 48, 72, 89, 96, 98),
                "key_count_millions": (0.12, 0.14, 1.9, 4.8, 8.7, 10.4),
            },
            "auth-service": {
                "login_success_rate": (0.97, 0.96, 0.90, 0.84, 0.78, 0.72),
            },
        },
        evidence_terms=(
            "ratelimit_key=user_id+tenant_id+request_id",
            "key_cardinality=8.7M",
            "key_cardinality_explosion",
            "limiter_backend_timeout",
            "rollback_edge_limiter_request_id_key",
        ),
        red_herrings=("auth-service login failures", "redis memory"),
        stakeholder_impact="Users are incorrectly throttled at the edge; auth failures are downstream of edge 429s.",
        causal_chain=(
            "edge rate-limit config adds request_id to key",
            "rate-limit key cardinality explodes",
            "redis rate-limit backend saturates CPU",
            "edge 429s rise and auth logins fail downstream",
        ),
        origin_service="edge-gateway",
        symptom_services=("redis-rate-limit", "auth-service"),
        red_herring_logs={
            "auth-service": (
                "20:46 error token_refresh_failed sample=11 possible_auth_bug=true",
                "20:47 info oauth_database healthy=true",
            ),
            "redis-rate-limit": (
                "20:45 warn memory_fragmentation_ratio=1.9 possible_memory_issue=true",
            ),
        },
        red_herring_metrics={
            "auth-service": {
                "token_refresh_error_rate": (0.001, 0.002, 0.01, 0.025, 0.04, 0.038),
            },
            "redis-rate-limit": {
                "memory_fragmentation_ratio": (1.1, 1.2, 1.4, 1.7, 1.9, 1.9),
            },
        },
        misleading_root_causes=("auth token refresh regression", "redis memory fragmentation"),
    ),
)


def evaluation_scenarios(include_hidden: bool = False) -> tuple[IncidentScenario, ...]:
    """Return scenarios available to evaluator code.

    Hidden scenarios are deliberately excluded unless evaluator/test code opts in.
    Training data and normal environment metadata use the public set only.
    """

    if include_hidden:
        return (*SCENARIOS, *HIDDEN_SCENARIOS)
    return SCENARIOS


def get_scenario(scenario_id: str, include_hidden: bool = False) -> IncidentScenario:
    for scenario in evaluation_scenarios(include_hidden=include_hidden):
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(f"Unknown scenario_id: {scenario_id}")


def generate_scenario(
    seed: Optional[int] = None,
    difficulty: str = "easy",
    preferred_root_cause: Optional[str] = None,
    include_hidden: bool = False,
) -> IncidentScenario:
    """Pick a deterministic scenario for reset or curriculum sampling."""

    rng = Random(seed)
    source = evaluation_scenarios(include_hidden=include_hidden)
    candidates = [
        s
        for s in source
        if s.difficulty == difficulty
        or difficulty == "mixed"
        or (difficulty == "easy" and s.difficulty in {"easy", "medium"})
    ]
    if preferred_root_cause:
        preferred = [s for s in candidates if s.root_cause == preferred_root_cause]
        if preferred:
            candidates = preferred
    if not candidates:
        candidates = list(source)
    return candidates[rng.randrange(len(candidates))]


def render_logs(
    scenario: IncidentScenario,
    service: str,
    query: str = "",
    limit: int = 5,
) -> list[str]:
    lines = list(scenario.logs.get(service, ()))
    lines.extend(scenario.red_herring_logs.get(service, ()))
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
    metrics = {
        **scenario.metrics.get(service, {}),
        **scenario.red_herring_metrics.get(service, {}),
    }
    if metric:
        return {metric: list(metrics.get(metric, ()))}
    return {name: list(values) for name, values in metrics.items()}


def scenario_ids(include_hidden: bool = False) -> list[str]:
    return [
        scenario.scenario_id
        for scenario in evaluation_scenarios(include_hidden=include_hidden)
    ]


def hidden_scenario_ids() -> list[str]:
    return [scenario.scenario_id for scenario in HIDDEN_SCENARIOS]


def is_hidden_scenario(scenario_id: str | None) -> bool:
    return bool(scenario_id and scenario_id in set(hidden_scenario_ids()))


def all_root_causes(include_hidden: bool = False) -> Iterable[str]:
    return {
        scenario.root_cause
        for scenario in evaluation_scenarios(include_hidden=include_hidden)
    }
