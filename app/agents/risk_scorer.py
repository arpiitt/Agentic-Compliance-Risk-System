"""
Risk-Scoring Agent — synthesizes findings into a structured risk report.
Uses Gemini (new google.genai SDK) with JSON response mode.
Every factor MUST cite a retrieved source.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

import google.genai as genai
from google.genai import types as genai_types

from app.agents.state import (
    AgentState,
    FilingDoc,
    NewsItem,
    PolicyMatch,
    RiskFactor,
    RiskReport,
    ToolCall,
)
from app.cache.redis_cache import NS_LLM, get_cached, set_cached
from app.config import get_settings
from app.metrics.prometheus import (
    agent_step_duration_seconds,
    llm_tokens_total,
    token_cost_usd_total,
)

logger = logging.getLogger(__name__)
settings = get_settings()

AGENT_NAME = "risk_scorer"

# Gemini Flash pricing (per 1M tokens)
GEMINI_FLASH_INPUT_COST_PER_1M = 0.075
GEMINI_FLASH_OUTPUT_COST_PER_1M = 0.30


def _get_client() -> genai.Client:
    return genai.Client(api_key=settings.google_api_key)


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * GEMINI_FLASH_INPUT_COST_PER_1M
        + output_tokens / 1_000_000 * GEMINI_FLASH_OUTPUT_COST_PER_1M
    )


def _build_context_summary(
    filings: list[FilingDoc],
    news_items: list[NewsItem],
    policy_matches: list[PolicyMatch],
) -> str:
    parts = []

    if filings:
        parts.append("## SEC Filings (most recent)")
        for f in filings[:6]:
            parts.append(
                f"[EDGAR:{f['doc_id']}] {f['form_type']} ({f['filing_date']}) — {f['company_name']}\n"
                f"URL: {f['url']}\nExcerpt: {f['text'][:1500]}\n"
            )

    if news_items:
        parts.append("\n## Recent News")
        for n in news_items[:5]:
            parts.append(
                f"[NEWS:{n['doc_id']}] {n['title']} ({n['published_at']}) — {n['source_name']}\n"
                f"URL: {n['url']}\n{n['description']}\n{n['content'][:500]}\n"
            )

    if policy_matches:
        parts.append("\n## Relevant Policy Clauses (from Knowledge Base)")
        for p in policy_matches[:10]:
            parts.append(
                f"[POLICY:{p['policy_doc_id']}] {p['regulation']} ({p['doc_type']}) "
                f"Score={p['relevance_score']:.2f}\n{p['clause_text']}\nURL: {p['url']}\n"
            )

    return "\n".join(parts)


RISK_SCORER_SYSTEM_PROMPT = """You are a senior compliance risk analyst specializing in US public equities.

Your task: analyze the provided SEC filings, news, and policy matches to produce a structured JSON risk report.

CRITICAL RULES:
1. Every risk factor MUST cite at least one specific source using its exact doc_id (format: EDGAR:xxx, NEWS:xxx, or POLICY:xxx).
2. No ungrounded claims — if you cannot cite it, do not include it.
3. Risk levels: Low, Medium, High.
4. Overall score: 0.0 (minimal risk) to 1.0 (critical risk).
5. Be concise but specific. Include exact quotes from source passages when possible.

Output ONLY valid JSON matching this exact schema:
{
  "entity_name": "<string>",
  "ticker": "<string>",
  "overall_level": "<Low|Medium|High>",
  "risk_score": <float 0.0-1.0>,
  "summary": "<string, 2-3 sentences>",
  "factors": [
    {
      "name": "<risk factor name>",
      "level": "<Low|Medium|High>",
      "rationale": "<detailed explanation>",
      "citations": [
        {
          "source_doc_id": "<EDGAR:xxx|NEWS:xxx|POLICY:xxx>",
          "source_type": "<edgar|news|policy>",
          "passage": "<exact excerpt from source, max 300 chars>",
          "url": "<url>",
          "title": "<document title or form type>"
        }
      ]
    }
  ],
  "confidence": 0.5
}
"""


def _generate_content_with_fallback(
    client: genai.Client,
    prompt: str,
    config: genai_types.GenerateContentConfig,
    initial_model: str,
) -> Any:
    candidates = [initial_model]
    for m in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"]:
        if m not in candidates:
            candidates.append(m)

    last_error = None
    for m in candidates:
        try:
            return client.models.generate_content(
                model=m,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            last_error = exc
            exc_str = str(exc).lower()
            if "404" in exc_str or "not_found" in exc_str or "no longer available" in exc_str or "not found" in exc_str:
                logger.warning("[risk_scorer] Gemini model '%s' unavailable (404), attempting fallback...", m)
                continue
            raise exc
    if last_error:
        raise last_error


async def risk_scorer_node(state: AgentState) -> dict:
    """
    LangGraph node: Risk-Scoring Agent.
    Calls Gemini with all retrieved context to produce a structured risk report.
    Results cached by content hash.
    """
    t0 = time.perf_counter()
    tool_calls: list[ToolCall] = []
    errors: list[str] = list(state.get("errors", []))

    budget = state.get("token_budget_remaining", settings.token_budget)
    cost_remaining = state.get("cost_budget_remaining", settings.cost_budget_usd)
    if budget <= 0 or cost_remaining <= 0:
        logger.warning("[risk_scorer] Budget exhausted.")
        return {
            "errors": errors + ["Budget exhausted before risk scoring"],
            "last_completed_step": AGENT_NAME,
        }

    entity_name = state["entity_name"]
    ticker = state.get("ticker", "")
    filings = state.get("filings", [])
    news_items = state.get("news_items", [])
    policy_matches = state.get("policy_matches", [])
    verifier_loop_count = state.get("verifier_loop_count", 0)
    flagged_claims = state.get("flagged_claims", [])

    context = _build_context_summary(filings, news_items, policy_matches)

    retry_note = ""
    if verifier_loop_count > 0 and flagged_claims:
        retry_note = (
            f"\n\n## VERIFIER FEEDBACK (Retry {verifier_loop_count})\n"
            "The following claims were flagged as unsupported. Revise or remove them:\n"
            + "\n".join(f"- {c}" for c in flagged_claims[:10])
        )

    user_prompt = (
        f"Entity: {entity_name} (Ticker: {ticker})\n\n"
        f"{context}{retry_note}\n\n"
        "Produce the risk report JSON now."
    )

    active_model = os.getenv("GEMINI_MODEL", settings.gemini_model)
    cache_params = {
        "op": "risk_scorer",
        "model": active_model,
        "system": RISK_SCORER_SYSTEM_PROMPT[:200],
        "prompt_hash": hashlib.sha256(user_prompt.encode()).hexdigest(),
    }
    cached = await get_cached(NS_LLM, cache_params)
    draft_report: RiskReport | None = None
    input_tokens = 0
    output_tokens = 0
    cost = 0.0

    if cached:
        logger.info("[risk_scorer] Cache HIT for run")
        draft_report = cached
    else:
        try:
            client = _get_client()
            full_prompt = f"{RISK_SCORER_SYSTEM_PROMPT}\n\n{user_prompt}"
            response = _generate_content_with_fallback(
                client=client,
                prompt=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
                initial_model=active_model,
            )
            raw_text = response.text

            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
            cost = _estimate_cost(input_tokens, output_tokens)

            llm_tokens_total.labels(agent=AGENT_NAME, direction="input").inc(input_tokens)
            llm_tokens_total.labels(agent=AGENT_NAME, direction="output").inc(output_tokens)
            token_cost_usd_total.labels(agent=AGENT_NAME).inc(cost)

            report_json: dict = json.loads(raw_text)
            report_json.setdefault("entity_name", entity_name)
            report_json.setdefault("ticker", ticker)
            report_json.setdefault("confidence", 0.5)
            report_json.setdefault("factors", [])

            draft_report = report_json  # type: ignore[assignment]
            await set_cached(NS_LLM, cache_params, draft_report, ttl=settings.llm_cache_ttl)

            tool_calls.append({
                "tool_name": "gemini.generate_content",
                "args": {"model": settings.gemini_model, "prompt_len": len(user_prompt)},
                "result_summary": (
                    f"Generated risk report: {report_json.get('overall_level')} risk, "
                    f"{len(report_json.get('factors', []))} factors"
                ),
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "cached": False,
                "error": None,
            })

        except json.JSONDecodeError as exc:
            error_msg = f"Risk scorer JSON parse error: {exc}"
            logger.error("[risk_scorer] %s", error_msg)
            errors.append(error_msg)
        except Exception as exc:
            error_msg = f"Risk scorer LLM call failed: {exc}"
            logger.error("[risk_scorer] %s", error_msg)
            errors.append(error_msg)

    new_budget = max(0, budget - input_tokens - output_tokens)
    new_cost = max(0.0, cost_remaining - cost)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    agent_step_duration_seconds.labels(agent=AGENT_NAME).observe(latency_ms / 1000)

    logger.info("[risk_scorer] Done in %dms, report=%s", latency_ms, draft_report is not None)

    return {
        "draft_report": draft_report,
        "errors": errors,
        "last_completed_step": AGENT_NAME,
        "token_budget_remaining": new_budget,
        "cost_budget_remaining": new_cost,
        "total_tokens_used": state.get("total_tokens_used", 0) + input_tokens + output_tokens,
        "total_cost_usd": state.get("total_cost_usd", 0.0) + cost,
    }
