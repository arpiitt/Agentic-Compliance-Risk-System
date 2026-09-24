"""
Application configuration using pydantic-settings.
All values are read from environment variables / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── LLM ──────────────────────────────────────────────────────────────────
    google_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "models/text-embedding-004"

    # ─── News ─────────────────────────────────────────────────────────────────
    newsapi_key: str = ""

    # ─── SEC EDGAR ────────────────────────────────────────────────────────────
    sec_user_agent: str = "ComplianceBot admin@example.com"

    # ─── Database ─────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://compliance:compliance@localhost:5432/compliance_db"
    database_sync_url: str = "postgresql://compliance:compliance@localhost:5432/compliance_db"

    # ─── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ─── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "policy_kb"

    # ─── Guardrails ───────────────────────────────────────────────────────────
    max_retries: int = 3
    token_budget: int = 150_000
    cost_budget_usd: float = 2.00
    timeout_seconds: int = 30
    max_verifier_loops: int = 2

    # ─── Cache TTLs (seconds) ─────────────────────────────────────────────────
    retrieval_cache_ttl: int = 21_600   # 6 hours
    llm_cache_ttl: int = 86_400         # 24 hours

    # ─── Rate Limiting ────────────────────────────────────────────────────────
    rate_limit_analyze: str = "10/minute"
    rate_limit_get: str = "60/minute"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
