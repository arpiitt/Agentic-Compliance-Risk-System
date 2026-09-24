"""
Compliance-Checker Agent — matches findings against policy knowledge base
using hybrid retrieval (semantic + BM25 sparse) with metadata filtering.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

from app.agents.state import AgentState, FilingDoc, NewsItem, PolicyMatch, ToolCall
from app.config import get_settings
from app.metrics.prometheus import agent_step_duration_seconds
from app.retrieval.qdrant_client import hybrid_search

logger = logging.getLogger(__name__)
settings = get_settings()

AGENT_NAME = "compliance_checker"

# Regulation categories to filter on for US equity compliance
REGULATION_FILTERS = [
    "SEC", "SOX", "FINRA", "Dodd-Frank", "AML", "KYC", "BSA", "Reg-FD", "Reg-S-K",
]


def _build_query_from_filing(filing: FilingDoc) -> str:
    """Construct a retrieval query from a filing document."""
    form = filing.get("form_type", "")
    text_snippet = filing.get("text", "")[:500]
    return f"{form} filing compliance risk: {text_snippet}"


def _build_query_from_news(item: NewsItem) -> str:
    """Construct a retrieval query from a news item."""
    return f"compliance risk: {item.get('title', '')} {item.get('description', '')}"


async def _retrieve_for_document(
    query: str,
    source_doc_id: str,
    top_k: int = 4,
) -> list[PolicyMatch]:
    """Run hybrid search for a single document and return PolicyMatch objects."""
    try:
        results = await hybrid_search(
            query=query,
            top_k=top_k,
            regulation_filter=REGULATION_FILTERS,
        )
    except Exception as exc:
        logger.warning("Retrieval failed for doc %s: %s", source_doc_id, exc)
        return []

    matches: list[PolicyMatch] = []
    for r in results:
        payload = r.get("payload", {})
        matches.append({
            "policy_doc_id": r["doc_id"],
            "doc_type": payload.get("doc_type", "unknown"),
            "regulation": payload.get("regulation", "unknown"),
            "clause_text": payload.get("chunk_text", "")[:800],
            "relevance_score": float(r.get("score", 0.0)),
            "matched_filing_doc_id": source_doc_id,
            "url": payload.get("url", ""),
        })
    return matches


async def compliance_checker_node(state: AgentState) -> dict:
    """
    LangGraph node: Compliance-Checker Agent.
    - Builds queries from filings and news items
    - Runs parallel hybrid retrieval against policy KB
    - Deduplicates and ranks by relevance
    """
    t0 = time.perf_counter()
    tool_calls: list[ToolCall] = []
    errors: list[str] = list(state.get("errors", []))

    # Budget guard
    if state.get("token_budget_remaining", settings.token_budget) <= 0:
        return {"last_completed_step": AGENT_NAME, "errors": errors + ["Budget exhausted"]}

    filings: list[FilingDoc] = state.get("filings", [])
    news_items: list[NewsItem] = state.get("news_items", [])

    logger.info(
        "[compliance_checker] Processing %d filings + %d news items",
        len(filings), len(news_items)
    )

    # Build retrieval tasks
    tasks = []
    # For filings: limit to 8 most relevant (avoid token explosion)
    for filing in filings[:8]:
        query = _build_query_from_filing(filing)
        tasks.append(_retrieve_for_document(query, filing["doc_id"], top_k=4))

    # For news: limit to 5 items
    for news_item in news_items[:5]:
        query = _build_query_from_news(news_item)
        tasks.append(_retrieve_for_document(query, news_item["doc_id"], top_k=3))

    # Execute all retrievals in parallel
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    policy_matches: list[PolicyMatch] = []
    seen_ids: set[str] = set()

    for result in all_results:
        if isinstance(result, Exception):
            errors.append(f"Retrieval error: {result}")
            continue
        for match in result:
            # Deduplicate by policy_doc_id + matched_filing_doc_id
            dedup_key = f"{match['policy_doc_id']}::{match['matched_filing_doc_id']}"
            if dedup_key not in seen_ids and match["relevance_score"] > 0.1:
                seen_ids.add(dedup_key)
                policy_matches.append(match)

    # Sort by relevance score descending
    policy_matches.sort(key=lambda m: m["relevance_score"], reverse=True)

    tool_calls.append({
        "tool_name": "hybrid_search",
        "args": {
            "num_queries": len(tasks),
            "regulation_filter": REGULATION_FILTERS,
        },
        "result_summary": f"Retrieved {len(policy_matches)} unique policy matches",
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "cached": False,
        "error": None,
    })

    latency_ms = int((time.perf_counter() - t0) * 1000)
    agent_step_duration_seconds.labels(agent=AGENT_NAME).observe(latency_ms / 1000)

    logger.info(
        "[compliance_checker] Done: %d policy matches in %dms",
        len(policy_matches), latency_ms
    )

    return {
        "policy_matches": policy_matches,
        "compliance_tool_calls": tool_calls,
        "errors": errors,
        "last_completed_step": AGENT_NAME,
    }
