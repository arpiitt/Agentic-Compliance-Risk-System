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


async def embed_text(text: str) -> list[float]:
    """
    Generate a dense embedding vector using Gemini text-embedding-004.
    Falls back to a zero vector on error (non-fatal).
    """
    try:
        client = _get_client()
        result = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=text,
        )
        return result.embeddings[0].values
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
