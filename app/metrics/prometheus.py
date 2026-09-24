"""
Prometheus custom metrics for the Compliance Research Assistant.
"""
from prometheus_client import Counter, Histogram, Gauge

# ─── Cache metrics ────────────────────────────────────────────────────────────
cache_hits_total = Counter(
    "cache_hits_total",
    "Total number of cache hits",
    ["namespace"],
)
cache_misses_total = Counter(
    "cache_misses_total",
    "Total number of cache misses",
    ["namespace"],
)

# ─── Run metrics ──────────────────────────────────────────────────────────────
run_latency_seconds = Histogram(
    "run_latency_seconds",
    "End-to-end run latency in seconds",
    ["status"],
    buckets=[5, 10, 30, 60, 120, 300, 600],
)

# ─── Agent metrics ────────────────────────────────────────────────────────────
agent_step_duration_seconds = Histogram(
    "agent_step_duration_seconds",
    "Time spent in each agent step",
    ["agent"],
    buckets=[1, 5, 10, 30, 60, 120],
)

retry_total = Counter(
    "retry_total",
    "Total number of retries per agent/tool",
    ["agent"],
)

# ─── Cost / token metrics ─────────────────────────────────────────────────────
token_cost_usd_total = Counter(
    "token_cost_usd_total",
    "Cumulative estimated LLM cost in USD",
    ["agent"],
)
llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["agent", "direction"],  # direction: input | output
)

# ─── Verifier metrics ─────────────────────────────────────────────────────────
verifier_flag_rate = Gauge(
    "verifier_flag_rate",
    "Rolling average fraction of claims flagged by the Verifier",
)
verifier_flags_total = Counter(
    "verifier_flags_total",
    "Total claims flagged (unsupported) by the Verifier",
)
verifier_claims_total = Counter(
    "verifier_claims_total",
    "Total claims checked by the Verifier",
)

# ─── EDGAR / News tool metrics ────────────────────────────────────────────────
edgar_request_total = Counter(
    "edgar_request_total",
    "Total EDGAR API requests",
    ["status"],  # success | rate_limited | error
)
news_request_total = Counter(
    "news_request_total",
    "Total NewsAPI requests",
    ["status"],
)
