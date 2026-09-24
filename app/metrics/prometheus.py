"""
Prometheus custom metrics for the Compliance Research Assistant.
Guaranteed idempotent against re-import / Streamlit reload.
"""
from prometheus_client import REGISTRY, Counter, Gauge, Histogram


def _get_or_create(metric_cls, name, documentation, labelnames=(), **kwargs):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    try:
        return metric_cls(name, documentation, labelnames=labelnames, **kwargs)
    except Exception:
        return REGISTRY._names_to_collectors.get(name)


# ─── Cache metrics ────────────────────────────────────────────────────────────
cache_hits_total = _get_or_create(
    Counter,
    "cache_hits_total",
    "Total number of cache hits",
    ["namespace"],
)
cache_misses_total = _get_or_create(
    Counter,
    "cache_misses_total",
    "Total number of cache misses",
    ["namespace"],
)

# ─── Run metrics ──────────────────────────────────────────────────────────────
run_latency_seconds = _get_or_create(
    Histogram,
    "run_latency_seconds",
    "End-to-end run latency in seconds",
    ["status"],
    buckets=[5, 10, 30, 60, 120, 300, 600],
)

# ─── Agent metrics ────────────────────────────────────────────────────────────
agent_step_duration_seconds = _get_or_create(
    Histogram,
    "agent_step_duration_seconds",
    "Time spent in each agent step",
    ["agent"],
    buckets=[1, 5, 10, 30, 60, 120],
)

retry_total = _get_or_create(
    Counter,
    "retry_total",
    "Total number of retries per agent/tool",
    ["agent"],
)

# ─── Cost / token metrics ─────────────────────────────────────────────────────
token_cost_usd_total = _get_or_create(
    Counter,
    "token_cost_usd_total",
    "Cumulative estimated LLM cost in USD",
    ["agent"],
)
llm_tokens_total = _get_or_create(
    Counter,
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["agent", "direction"],
)

# ─── Verifier metrics ─────────────────────────────────────────────────────────
verifier_flag_rate = _get_or_create(
    Gauge,
    "verifier_flag_rate",
    "Rolling average fraction of claims flagged by the Verifier",
)
verifier_flags_total = _get_or_create(
    Counter,
    "verifier_flags_total",
    "Total claims flagged (unsupported) by the Verifier",
)
verifier_claims_total = _get_or_create(
    Counter,
    "verifier_claims_total",
    "Total claims checked by the Verifier",
)

# ─── EDGAR / News tool metrics ────────────────────────────────────────────────
edgar_request_total = _get_or_create(
    Counter,
    "edgar_request_total",
    "Total EDGAR API requests",
    ["status"],
)
news_request_total = _get_or_create(
    Counter,
    "news_request_total",
    "Total NewsAPI requests",
    ["status"],
)
