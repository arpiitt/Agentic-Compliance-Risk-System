"""
LangGraph TypedDict state schema shared across all agents.
"""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph import add_messages


# ─── Sub-types ────────────────────────────────────────────────────────────────

class FilingDoc(TypedDict):
    """A single SEC filing document fragment."""
    doc_id: str
    accession_number: str
    form_type: str
    filing_date: str
    company_name: str
    cik: str
    text: str           # Truncated relevant text
    url: str
    source: str         # "edgar"


class NewsItem(TypedDict):
    """A single news article."""
    doc_id: str
    title: str
    url: str
    published_at: str
    source_name: str
    description: str
    content: str        # Truncated


class Citation(TypedDict):
    """A verifiable citation linking a claim to a source passage."""
    source_doc_id: str
    source_type: str    # "edgar" | "news" | "policy"
    passage: str        # Exact excerpt from the source
    url: str
    title: str


class RiskFactor(TypedDict):
    """A single evaluated risk factor."""
    name: str
    level: str          # Low | Medium | High
    rationale: str
    citations: list[Citation]


class RiskReport(TypedDict):
    """The structured risk report produced by Risk-Scorer."""
    entity_name: str
    ticker: str
    overall_level: str  # Low | Medium | High
    risk_score: float   # 0.0 – 1.0
    factors: list[RiskFactor]
    summary: str
    confidence: float   # Set by Verifier


class PolicyMatch(TypedDict):
    """A retrieved policy clause matched against findings."""
    policy_doc_id: str
    doc_type: str
    regulation: str
    clause_text: str
    relevance_score: float
    matched_filing_doc_id: str
    url: str


class ToolCall(TypedDict):
    """Structured record of a single tool invocation."""
    tool_name: str
    args: dict
    result_summary: str
    latency_ms: int
    cached: bool
    error: str | None


# ─── Main State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Identity
    run_id: str
    entity_name: str
    ticker: str

    # Researcher outputs
    filings: list[FilingDoc]
    news_items: list[NewsItem]
    researcher_tool_calls: list[ToolCall]

    # Compliance-Checker outputs
    policy_matches: list[PolicyMatch]
    compliance_tool_calls: list[ToolCall]

    # Risk-Scorer outputs
    draft_report: RiskReport | None

    # Verifier outputs
    flagged_claims: list[str]
    final_report: RiskReport | None
    verifier_loop_count: int

    # Control flow
    should_retry: bool
    errors: list[str]

    # Budget tracking
    token_budget_remaining: int
    cost_budget_remaining: float
    total_tokens_used: int
    total_cost_usd: float

    # Idempotency
    last_completed_step: str   # "researcher" | "compliance_checker" | "risk_scorer" | "verifier"
