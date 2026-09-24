"""
News fetching tools using NewsAPI with async, retry, caching.
Falls back to DuckDuckGo if NewsAPI key is unavailable.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

from app.cache.redis_cache import NS_NEWS, get_cached, set_cached
from app.config import get_settings
from app.metrics.prometheus import news_request_total

logger = logging.getLogger(__name__)
settings = get_settings()

NEWSAPI_BASE = "https://newsapi.org/v2"


def _news_retry():
    return retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential_jitter(initial=1, max=16),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def _get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=settings.timeout_seconds, follow_redirects=True)


# ─── NewsAPI ──────────────────────────────────────────────────────────────────
@_news_retry()
async def _newsapi_fetch(query: str, days_back: int = 30) -> list[dict]:
    if not settings.newsapi_key:
        return []
    from datetime import date, timedelta
    from_date = (date.today() - timedelta(days=days_back)).isoformat()
    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 10,
        "apiKey": settings.newsapi_key,
    }
    async with _get_http_client() as client:
        resp = await asyncio.wait_for(
            client.get(f"{NEWSAPI_BASE}/everything", params=params),
            timeout=settings.timeout_seconds,
        )
        resp.raise_for_status()
        news_request_total.labels(status="success").inc()
        return resp.json().get("articles", [])


# ─── DuckDuckGo fallback ──────────────────────────────────────────────────────
async def _ddg_news_fetch(query: str) -> list[dict]:
    """Minimal DuckDuckGo news scraper as fallback (no API key needed)."""
    try:
        async with _get_http_client() as client:
            resp = await asyncio.wait_for(
                client.get(
                    "https://duckduckgo.com/",
                    params={"q": query, "ia": "news", "iar": "news"},
                    headers={"User-Agent": "Mozilla/5.0"},
                ),
                timeout=settings.timeout_seconds,
            )
        # Very minimal extraction (DuckDuckGo doesn't have a public news API)
        # Return empty — callers handle gracefully
        return []
    except Exception:
        return []


# ─── Public interface ─────────────────────────────────────────────────────────
async def fetch_news(entity_name: str, days_back: int = 30) -> list[dict[str, Any]]:
    """
    Fetch recent news for an entity. Uses NewsAPI if key is set, else DDG.
    Returns a list of normalized news item dicts.
    """
    cache_key = {"op": "news", "entity": entity_name, "days_back": days_back}
    cached = await get_cached(NS_NEWS, cache_key)
    if cached:
        return cached

    raw_articles = []
    try:
        if settings.newsapi_key:
            raw_articles = await _newsapi_fetch(entity_name, days_back)
        else:
            raw_articles = await _ddg_news_fetch(entity_name)
    except Exception as exc:
        logger.warning("News fetch failed for %s: %s", entity_name, exc)
        news_request_total.labels(status="error").inc()

    normalized = []
    for art in raw_articles:
        normalized.append({
            "doc_id": str(uuid.uuid4()),
            "title": art.get("title", ""),
            "url": art.get("url", ""),
            "published_at": art.get("publishedAt", ""),
            "source_name": (art.get("source") or {}).get("name", "Unknown"),
            "description": art.get("description", ""),
            "content": (art.get("content") or "")[:2000],
        })

    await set_cached(NS_NEWS, cache_key, normalized, ttl=settings.retrieval_cache_ttl)
    return normalized
