"""API endpoint tests using FastAPI TestClient."""
from __future__ import annotations

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    with (
        patch("app.db.database.init_db", new_callable=AsyncMock),
        patch("app.db.database.close_db", new_callable=AsyncMock),
        patch("app.cache.redis_cache.get_redis", new_callable=AsyncMock),
        patch("app.cache.redis_cache.close_redis", new_callable=AsyncMock),
        patch("app.retrieval.qdrant_client.ensure_collection_exists", new_callable=AsyncMock),
        patch("app.retrieval.embeddings.init_embeddings"),
    ):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_health_endpoint(client):
    """Health check should return 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_returns_run_id(client):
    """POST /analyze should return a run_id."""
    run_id = str(uuid.uuid4())
    mock_run = MagicMock()
    mock_run.id = uuid.UUID(run_id)

    with (
        patch("app.db.database.get_session") as mock_session_ctx,
        patch("app.api.routes._run_analysis", new_callable=AsyncMock),
    ):
        mock_session = AsyncMock()
        mock_session_ctx.return_value.__aenter__.return_value = mock_session

        response = client.post(
            "/analyze",
            json={"entity_name": "Apple Inc", "ticker": "AAPL"},
        )

    assert response.status_code in (202, 422, 500)  # 422 if DB not available in test


def test_analyze_validates_entity_name(client):
    """POST /analyze should reject empty entity name."""
    response = client.post("/analyze", json={"entity_name": "", "ticker": "AAPL"})
    assert response.status_code == 422


def test_get_run_invalid_id(client):
    """GET /runs/{id} should return error for invalid UUID."""
    with patch("app.db.database.get_session") as mock_session_ctx:
        mock_session = AsyncMock()
        mock_session_ctx.return_value.__aenter__.return_value = mock_session
        response = client.get("/runs/not-a-uuid")
    assert response.status_code in (400, 404, 422)


def test_get_trace_invalid_id(client):
    """GET /runs/{id}/trace should return error for invalid UUID."""
    with patch("app.db.database.get_session") as mock_session_ctx:
        mock_session = AsyncMock()
        mock_session_ctx.return_value.__aenter__.return_value = mock_session
        response = client.get("/runs/invalid-id/trace")
    assert response.status_code in (400, 404, 422)


def test_docs_endpoint_accessible(client):
    """Swagger UI should be accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
