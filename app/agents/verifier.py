"""
Verifier Agent — re-checks every claim in the draft report against its cited sources.
Uses google.genai SDK (new API).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import google.genai as genai
from google.genai import types as genai_types

from app.agents.state import AgentState, FilingDoc, NewsItem, PolicyMatch, RiskReport, ToolCall
from app.cache.redis_cache import NS_LLM, get_cached, set_cached
from app.config import get_settings
from app.metrics.prometheus import (
    agent_step_duration_seconds,
    llm_tokens_total,
    token_cost_usd_total,
    verifier_claims_total,
    verifier_flag_rate,
    verifier_flags_total,
)

logger = logging.getLogger(__name__)
settings = get_settings()

AGENT_NAME = "verifier"

VERIFIER_SYSTEM_PROMPT = """You are a strict compliance verification auditor.

Your job: given a draft risk report and the source documents it cites, verify every factual claim.

For EACH risk factor:
1. Locate the cited source passage in the provided source documents.
2. Confirm the claim is SUPPORTED by that exact passage.
3. Flag the claim as UNSUPPORTED if:
   - The cited document/passage does not exist in the sources
   - The passage does not actually support the claim made
   - The claim is a generalization not found in the text

Output ONLY valid JSON:
{
  "verified_factors": [
    {
      "name": "<factor name>",
      "supported": true,
      "flagged_reason": null,
      "verified_citations": ["<source_doc_id>"]
    }
  ],
  "flagged_claims": [],
  "confidence": 0.85,
  "hallucination_count": 0,
  "total_claims_checked": 1
}

Be conservative: only flag claims that clearly lack support. Do not penalize reasonable inferences clearly evidenced by the text.
"""


def _get_client() -> genai.Client:
    return genai.Client(api_key=settings.google_api_key)


def _build_source_map(
    filings: list[FilingDoc],
    news_items: list[NewsItem],
    policy_matches: list[PolicyMatch],
) -> str:
    parts = []
    for f in filings[:6]:
        parts.append(f"[EDGAR:{f['doc_id']}] {f['form_type']} {f['filing_date']}: {f['text'][:1000]}")
    for n in news_items[:5]:
        parts.append(f"[NEWS:{n['doc_id']}] {n['title']}: {n['description']} {n['content'][:500]}")
    for p in policy_matches[:8]:
        parts.append(f"[POLICY:{p['policy_doc_id']}] {p['regulation']}: {p['clause_text'][:600]}")
    return "\n\n".join(parts)


def _generate_content_with_fallback(
    client: genai.Client,
    prompt: str,
    config: genai_types.GenerateContentConfig,
    initial_model: str,
) -> Any:
    candidates = [initial_model]
    fallback_list = [
        "gemini-1.5-flash",
        "models/gemini-1.5-flash",
        "gemini-1.5-flash-002",
        "gemini-2.5-flash",
        "models/gemini-2.5-flash",
        "gemini-1.5-pro",
        "models/gemini-1.5-pro",
        "gemini-3.5-flash-lite",
        "models/gemini-3.5-flash-lite",
        "gemini-2.0-flash",
        "models/gemini-2.0-flash",
    ]
    for m in fallback_list:
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
                logger.warning("[verifier] Gemini model '%s' unavailable (404), attempting fallback...", m)
                continue
            raise exc
    if last_error:
        raise last_error


async def verifier_node(state: AgentState) -> dict:
    """
    LangGraph node: Verifier Agent.
    """
    t0 = time.perf_counter()
    errors: list[str] = list(state.get("errors", []))

    budget = state.get("token_budget_remaining", settings.token_budget)
    cost_remaining = state.get("cost_budget_remaining", settings.cost_budget_usd)
    if budget <= 0 or cost_remaining <= 0:
        draft = state.get("draft_report")
        if draft:
            draft["confidence"] = 0.1
        return {
            "final_report": draft,
            "flagged_claims": ["Budget exhausted — verification skipped"],
            "errors": errors + ["Budget exhausted in verifier"],
            "should_retry": False,
            "last_completed_step": AGENT_NAME,
        }

    draft_report: RiskReport | None = state.get("draft_report")
    if not draft_report:
        logger.warning("[verifier] No draft report to verify.")
        return {
            "errors": errors + ["No draft report available for verification"],
            "should_retry": False,
            "last_completed_step": AGENT_NAME,
        }

    filings = state.get("filings", [])
    news_items = state.get("news_items", [])
    policy_matches = state.get("policy_matches", [])
    verifier_loop_count = state.get("verifier_loop_count", 0)

    source_map = _build_source_map(filings, news_items, policy_matches)
    draft_json = json.dumps(draft_report, indent=2)

    user_prompt = (
        f"## Draft Risk Report\n{draft_json}\n\n"
        f"## Source Documents\n{source_map}\n\n"
        "Verify every claim. Output verification JSON now."
    )

    active_model = os.getenv("GEMINI_MODEL", settings.gemini_model)
    cache_params = {
        "op": "verifier",
        "model": active_model,
        "prompt_hash": hashlib.sha256(user_prompt.encode()).hexdigest(),
        "loop": verifier_loop_count,
    }
    cached_result = await get_cached(NS_LLM, cache_params)
    verification: dict | None = None
    input_tokens = 0
    output_tokens = 0
    cost = 0.0

    if cached_result:
        logger.info("[verifier] Cache HIT")
        verification = cached_result
    else:
        try:
            client = _get_client()
            full_prompt = f"{VERIFIER_SYSTEM_PROMPT}\n\n{user_prompt}"
            response = _generate_content_with_fallback(
                client=client,
                prompt=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
                initial_model=active_model,
            )
            verification = json.loads(response.text)

            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
            cost = (input_tokens / 1_000_000 * 0.075) + (output_tokens / 1_000_000 * 0.30)

            llm_tokens_total.labels(agent=AGENT_NAME, direction="input").inc(input_tokens)
            llm_tokens_total.labels(agent=AGENT_NAME, direction="output").inc(output_tokens)
            token_cost_usd_total.labels(agent=AGENT_NAME).inc(cost)
            budget = max(0, budget - input_tokens - output_tokens)
            cost_remaining = max(0.0, cost_remaining - cost)

            await set_cached(NS_LLM, cache_params, verification, ttl=settings.llm_cache_ttl)

        except Exception as exc:
            error_msg = f"Verifier LLM call failed: {exc}"
            logger.error("[verifier] %s", error_msg)
            errors.append(error_msg)
            draft_report["confidence"] = 0.3
            return {
                "final_report": draft_report,
                "flagged_claims": [],
                "should_retry": False,
                "errors": errors,
                "last_completed_step": AGENT_NAME,
                "token_budget_remaining": budget,
                "cost_budget_remaining": cost_remaining,
            }

    # Process verification result
    flagged_claims: list[str] = verification.get("flagged_claims", [])
    confidence: float = float(verification.get("confidence", 0.5))
    hallucination_count: int = int(verification.get("hallucination_count", 0))
    total_claims: int = int(verification.get("total_claims_checked", 1))

    verifier_flags_total.inc(hallucination_count)
    verifier_claims_total.inc(total_claims)
    if total_claims > 0:
        verifier_flag_rate.set(hallucination_count / total_claims)

    verified_factors_info: list[dict] = verification.get("verified_factors", [])
    supported_names = {
        f["name"] for f in verified_factors_info if f.get("supported", True)
    }

    original_factors = draft_report.get("factors", [])
    clean_factors = [f for f in original_factors if f.get("name", "") in supported_names]
    if not clean_factors and original_factors:
        clean_factors = original_factors

    final_report: RiskReport = {
        **draft_report,
        "factors": clean_factors,
        "confidence": confidence,
    }

    flag_rate = hallucination_count / max(total_claims, 1)
    should_retry = (
        flag_rate > 0.3
        and verifier_loop_count < settings.max_verifier_loops
        and budget > 5000
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    agent_step_duration_seconds.labels(agent=AGENT_NAME).observe(latency_ms / 1000)

    logger.info(
        "[verifier] Done: confidence=%.2f hallucination_rate=%.2f should_retry=%s loop=%d",
        confidence, flag_rate, should_retry, verifier_loop_count,
    )

    return {
        "final_report": final_report if not should_retry else None,
        "draft_report": draft_report,
        "flagged_claims": flagged_claims,
        "should_retry": should_retry,
        "verifier_loop_count": verifier_loop_count + 1,
        "errors": errors,
        "last_completed_step": AGENT_NAME,
        "token_budget_remaining": budget,
        "cost_budget_remaining": cost_remaining,
        "total_tokens_used": state.get("total_tokens_used", 0) + input_tokens + output_tokens,
        "total_cost_usd": state.get("total_cost_usd", 0.0) + cost,
    }
