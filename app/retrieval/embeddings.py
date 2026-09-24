"""
Embedding utilities using Google Gemini text-embedding-004 (new google.genai SDK).
Also provides a BM25-style sparse vector builder for Qdrant hybrid search.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional

import google.genai as genai
from google.genai import types as genai_types

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_genai_client: Optional[genai.Client] = None

EMBEDDING_DIM = 768  # Gemini text-embedding-004 dimension


def init_embeddings() -> None:
    """Configure the Gemini SDK (call once at startup)."""
    global _genai_client
    _genai_client = genai.Client(api_key=settings.google_api_key)


def _get_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=settings.google_api_key)
    return _genai_client


# Once a working embedding model is found, cache it to avoid re-probing every call.
_working_embedding_model: str | None = None
# Set to True after all candidates are exhausted so future calls skip API probing entirely.
_embedding_unavailable: bool = False


async def embed_text(text: str) -> list[float]:
    """
    Generate a dense embedding vector using Gemini text-embedding-004.
    Falls back to alternative embedding model names or zero vector on error.
    Caches the first working model so subsequent calls are immediate.
    Once all models fail, returns zero vectors without any further API calls.
    """
    global _working_embedding_model, _embedding_unavailable

    if _embedding_unavailable:
        return [0.0] * EMBEDDING_DIM

    try:
        client = _get_client()

        if _working_embedding_model:
            try:
                result = client.models.embed_content(
                    model=_working_embedding_model,
                    contents=text,
                )
                return result.embeddings[0].values
            except Exception as exc:
                if "404" in str(exc) or "NOT_FOUND" in str(exc) or "not found" in str(exc).lower():
                    logger.warning(
                        "Cached embedding model '%s' returned 404; re-probing candidates.",
                        _working_embedding_model,
                    )
                    _working_embedding_model = None
                else:
                    raise exc

        seen: set[str] = set()
        candidates: list[str] = []
        for m in [
            settings.gemini_embedding_model,
            "models/text-embedding-004",
            "text-embedding-004",
            "models/embedding-001",
            "embedding-001",
            "models/text-embedding-005",
            "text-multilingual-embedding-002",
        ]:
            if m not in seen:
                seen.add(m)
                candidates.append(m)

        for m in candidates:
            try:
                result = client.models.embed_content(
                    model=m,
                    contents=text,
                )
                _working_embedding_model = m
                logger.info("Embedding model resolved to '%s'.", m)
                return result.embeddings[0].values
            except Exception as exc:
                if "404" in str(exc) or "NOT_FOUND" in str(exc) or "not found" in str(exc).lower():
                    logger.debug("Embedding model '%s' returned 404, trying next.", m)
                    continue
                raise exc

        logger.warning(
            "All embedding model candidates returned 404. Returning zero vectors for this session. "
            "LLM risk scoring is unaffected; only Qdrant policy retrieval will be bypassed."
        )
        _embedding_unavailable = True
        return [0.0] * EMBEDDING_DIM
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        return [0.0] * EMBEDDING_DIM


async def embed_query(text: str) -> list[float]:
    """Embed a query using Gemini text-embedding-004."""
    return await embed_text(text)


# ─── Sparse (BM25-style) encoding ────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


# Global vocab (built during seeding, loaded from a simple file at runtime)
_vocab: dict[str, int] = {}


def build_vocab(documents: list[str]) -> dict[str, int]:
    """Build a term→index vocabulary from a corpus."""
    global _vocab
    all_tokens: set[str] = set()
    for doc in documents:
        all_tokens.update(_tokenize(doc))
    _vocab = {term: idx for idx, term in enumerate(sorted(all_tokens))}
    return _vocab


def set_vocab(vocab: dict[str, int]) -> None:
    global _vocab
    _vocab = vocab


def get_vocab() -> dict[str, int]:
    return _vocab


def encode_sparse(text: str) -> tuple[list[int], list[float]]:
    """
    Return (indices, values) for a sparse TF vector using the global vocab.
    Safe to call even if vocab is empty (returns empty vectors).
    """
    if not _vocab:
        return [], []
    tokens = _tokenize(text)
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    indices, values = [], []
    for term, cnt in counts.items():
        if term in _vocab:
            indices.append(_vocab[term])
            values.append(cnt / total)
    return indices, values
