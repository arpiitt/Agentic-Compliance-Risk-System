"""
Researcher Agent — fetches SEC filings and news concurrently.
Implements exponential backoff, rate-limit handling, timeout, and caching.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

from app.agents.state import AgentState, FilingDoc, NewsItem, ToolCall
from app.config import get_settings
from app.metrics.prometheus import agent_step_duration_seconds, retry_total
from app.tools.edgar_tools import fetch_all_filings_parallel, resolve_cik
from app.tools.news_tools import fetch_news

logger = logging.getLogger(__name__)
settings = get_settings()

AGENT_NAME = "researcher"


async def researcher_node(state: AgentState) -> dict:
    """
    LangGraph node: Researcher Agent.
    1. Resolves entity → CIK
    2. Fetches recent SEC filings (parallel, async)
    3. Fetches recent news (parallel with filings)
    Returns partial state update.
    """
    t0 = time.perf_counter()
    tool_calls: list[ToolCall] = []
    errors: list[str] = []

    # Guard: budget check
    if state.get("token_budget_remaining", settings.token_budget) <= 0:
        logger.warning("[researcher] Token budget exhausted, skipping.")
        return {"last_completed_step": AGENT_NAME, "errors": ["Token budget exhausted in researcher"]}

    entity_name = state["entity_name"]
    ticker = state.get("ticker", "")

    logger.info("[researcher] Starting research for entity='%s' ticker='%s'", entity_name, ticker)

    # ── Step 1: Resolve CIK ──────────────────────────────────────────────────
    cik_call_start = time.perf_counter()
    cik = await resolve_cik(entity_name, ticker or None)
    cik_latency = int((time.perf_counter() - cik_call_start) * 1000)
    tool_calls.append({
        "tool_name": "resolve_cik",
        "args": {"entity_name": entity_name, "ticker": ticker},
        "result_summary": f"CIK={cik}" if cik else "CIK not found",
        "latency_ms": cik_latency,
        "cached": False,
        "error": None if cik else "CIK resolution failed",
    })

    if not cik:
        errors.append(f"Could not resolve CIK for '{entity_name}'. EDGAR research will be skipped.")
        logger.warning("[researcher] CIK not found for '%s'", entity_name)

    # ── Step 2: Parallel fetch of filings + news ─────────────────────────────
    form_types = ["10-K", "10-Q", "8-K", "DEF 14A"]
    async def _no_op() -> list:
        return []

    fetch_tasks = []
    if cik:
        fetch_tasks.append(fetch_all_filings_parallel(cik, form_types, count_per_type=3))
    else:
        fetch_tasks.append(_no_op())
    fetch_tasks.append(fetch_news(entity_name, days_back=90))

    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    raw_filings, raw_news = results[0], results[1]

    if isinstance(raw_filings, Exception):
        logger.error("[researcher] Filings fetch failed: %s", raw_filings)
        errors.append(f"Filings fetch error: {raw_filings}")
        raw_filings = []
    if isinstance(raw_news, Exception):
        logger.error("[researcher] News fetch failed: %s", raw_news)
        errors.append(f"News fetch error: {raw_news}")
        raw_news = []

    # ── Step 3: Normalize outputs ────────────────────────────────────────────
    filings: list[FilingDoc] = []
    for f in raw_filings:
        if not isinstance(f, dict):
            continue
        filings.append({
            "doc_id": f.get("accession_number", str(uuid.uuid4())),
            "accession_number": f.get("accession_number", ""),
            "form_type": f.get("form_type", ""),
            "filing_date": f.get("filing_date", ""),
            "company_name": f.get("company_name", entity_name),
            "cik": f.get("cik", cik or ""),
            "text": f.get("text", "")[:6000],
            "url": f.get("url", ""),
            "source": "edgar",
        })

    news_items: list[NewsItem] = []
    for n in raw_news:
        if not isinstance(n, dict):
            continue
        news_items.append({
            "doc_id": n.get("doc_id", str(uuid.uuid4())),
            "title": n.get("title", ""),
            "url": n.get("url", ""),
            "published_at": n.get("published_at", ""),
            "source_name": n.get("source_name", ""),
            "description": n.get("description", ""),
            "content": n.get("content", "")[:2000],
        })

    tool_calls.append({
        "tool_name": "fetch_all_filings_parallel",
        "args": {"cik": cik, "form_types": form_types},
        "result_summary": f"Fetched {len(filings)} filing documents",
        "latency_ms": 0,
        "cached": False,
        "error": None,
    })
    tool_calls.append({
        "tool_name": "fetch_news",
        "args": {"entity_name": entity_name, "days_back": 90},
        "result_summary": f"Fetched {len(news_items)} news items",
        "latency_ms": 0,
        "cached": False,
        "error": None,
    })

    latency_ms = int((time.perf_counter() - t0) * 1000)
    agent_step_duration_seconds.labels(agent=AGENT_NAME).observe(latency_ms / 1000)

    logger.info(
        "[researcher] Done: %d filings, %d news items in %dms",
        len(filings), len(news_items), latency_ms
    )

    return {
        "filings": filings,
        "news_items": news_items,
        "researcher_tool_calls": tool_calls,
        "errors": state.get("errors", []) + errors,
        "last_completed_step": AGENT_NAME,
    }
