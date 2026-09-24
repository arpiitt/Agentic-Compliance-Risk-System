"""
Async Redis cache for retrieval results and LLM responses.
Keys use SHA-256 hashes of (params) for idempotency and collision-safety.
"""
import hashlib
import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis

from app.config import get_settings
from app.metrics.prometheus import cache_hits_total, cache_misses_total

logger = logging.getLogger(__name__)
settings = get_settings()

# ─── Namespaces ────────────────────────────────────────────────────────────────
NS_RETRIEVAL = "retrieval"
NS_LLM = "llm"
NS_EDGAR = "edgar"
NS_NEWS = "news"

_redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = await aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


async def close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


def make_cache_key(namespace: str, params: dict) -> str:
    """Generate a deterministic SHA-256 cache key."""
    serialized = json.dumps(params, sort_keys=True, default=str)
    hash_val = hashlib.sha256(serialized.encode()).hexdigest()
    return f"{namespace}::{hash_val}"


async def get_cached(namespace: str, params: dict) -> Optional[Any]:
    """Try to get a value from cache. Returns None on miss."""
    try:
        client = await get_redis()
        key = make_cache_key(namespace, params)
        raw = await client.get(key)
        if raw is not None:
            cache_hits_total.labels(namespace=namespace).inc()
            logger.debug("Cache HIT  ns=%s key=%s", namespace, key[:16])
            return json.loads(raw)
        cache_misses_total.labels(namespace=namespace).inc()
        logger.debug("Cache MISS ns=%s key=%s", namespace, key[:16])
        return None
    except Exception as exc:
        logger.warning("Redis get failed (falling through): %s", exc)
        return None


async def set_cached(namespace: str, params: dict, value: Any, ttl: int) -> None:
    """Store a value in cache with the given TTL in seconds."""
    try:
        client = await get_redis()
        key = make_cache_key(namespace, params)
        serialized = json.dumps(value, default=str)
        await client.setex(key, ttl, serialized)
        logger.debug("Cache SET  ns=%s key=%s ttl=%s", namespace, key[:16], ttl)
    except Exception as exc:
        logger.warning("Redis set failed (non-fatal): %s", exc)


async def delete_cached(namespace: str, params: dict) -> None:
    """Evict a specific cache entry."""
    try:
        client = await get_redis()
        key = make_cache_key(namespace, params)
        await client.delete(key)
    except Exception as exc:
        logger.warning("Redis delete failed: %s", exc)
