"""Redis cache layer tests."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, patch

from app.cache.redis_cache import make_cache_key, NS_EDGAR, NS_LLM


def test_make_cache_key_deterministic():
    """Same params should always produce the same key."""
    params = {"entity": "AAPL", "form_type": "10-K", "count": 5}
    key1 = make_cache_key(NS_EDGAR, params)
    key2 = make_cache_key(NS_EDGAR, params)
    assert key1 == key2
    assert key1.startswith(f"{NS_EDGAR}::")


def test_make_cache_key_different_namespace():
    """Same params with different namespace should produce different keys."""
    params = {"prompt": "test"}
    key_edgar = make_cache_key(NS_EDGAR, params)
    key_llm = make_cache_key(NS_LLM, params)
    assert key_edgar != key_llm


def test_make_cache_key_order_independent():
    """Dict key order should not affect the cache key."""
    params1 = {"a": 1, "b": 2}
    params2 = {"b": 2, "a": 1}
    assert make_cache_key(NS_EDGAR, params1) == make_cache_key(NS_EDGAR, params2)


@pytest.mark.asyncio
async def test_get_cached_returns_none_on_miss():
    """Cache miss should return None without raising."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    with patch("app.cache.redis_cache.get_redis", new_callable=AsyncMock, return_value=mock_redis):
        from app.cache.redis_cache import get_cached
        result = await get_cached(NS_EDGAR, {"op": "test"})
    assert result is None


@pytest.mark.asyncio
async def test_get_cached_returns_value_on_hit():
    """Cache hit should return the stored value."""
    mock_redis = AsyncMock()
    cached_value = {"result": "apple filing"}
    mock_redis.get.return_value = json.dumps(cached_value)

    with patch("app.cache.redis_cache.get_redis", new_callable=AsyncMock, return_value=mock_redis):
        from app.cache.redis_cache import get_cached
        result = await get_cached(NS_EDGAR, {"op": "test"})
    assert result == cached_value


@pytest.mark.asyncio
async def test_set_cached_calls_setex():
    """set_cached should call Redis SETEX with correct TTL."""
    mock_redis = AsyncMock()
    mock_redis.setex.return_value = True

    with patch("app.cache.redis_cache.get_redis", new_callable=AsyncMock, return_value=mock_redis):
        from app.cache.redis_cache import set_cached
        await set_cached(NS_LLM, {"prompt": "test"}, {"response": "risk report"}, ttl=3600)

    mock_redis.setex.assert_called_once()
    call_args = mock_redis.setex.call_args
    assert call_args[0][1] == 3600  # TTL


@pytest.mark.asyncio
async def test_cache_falls_through_on_redis_error():
    """Cache should not raise when Redis is unavailable."""
    with patch("app.cache.redis_cache.get_redis", side_effect=Exception("Redis down")):
        from app.cache.redis_cache import get_cached, set_cached
        # Should not raise
        result = await get_cached(NS_EDGAR, {"op": "test"})
        assert result is None

        await set_cached(NS_EDGAR, {"op": "test"}, {"data": "value"}, ttl=60)
        # Should not raise
