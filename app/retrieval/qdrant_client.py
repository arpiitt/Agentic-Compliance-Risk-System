"""
Qdrant hybrid retrieval client.
Implements semantic (dense) + BM25-style (sparse) search with RRF fusion
and payload-based metadata filtering.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient, models

from app.cache.redis_cache import NS_RETRIEVAL, get_cached, set_cached
from app.config import get_settings
from app.retrieval.embeddings import EMBEDDING_DIM, embed_query, encode_sparse

logger = logging.getLogger(__name__)
settings = get_settings()

COLLECTION = settings.qdrant_collection
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_NAME = "dense"

_qdrant_client: Optional[AsyncQdrantClient] = None


def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.timeout_seconds,
        )
    return _qdrant_client


async def ensure_collection_exists() -> None:
    """Create the policy_kb collection if it doesn't exist yet."""
    client = get_qdrant_client()
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if COLLECTION not in names:
        await client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        # Create payload indexes for fast metadata filtering
        for field, schema in [
            ("doc_type", models.PayloadSchemaType.KEYWORD),
            ("regulation", models.PayloadSchemaType.KEYWORD),
            ("year", models.PayloadSchemaType.INTEGER),
        ]:
            await client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        logger.info("Created Qdrant collection '%s'", COLLECTION)


async def upsert_documents(points: list[dict]) -> None:
    """
    Bulk upsert documents into Qdrant.
    Each point: {id, dense_vector, sparse_indices, sparse_values, payload}
    """
    client = get_qdrant_client()
    qdrant_points = []
    for p in points:
        qdrant_points.append(
            models.PointStruct(
                id=p["id"],
                vector={
                    DENSE_VECTOR_NAME: p["dense_vector"],
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=p["sparse_indices"],
                        values=p["sparse_values"],
                    ),
                },
                payload=p["payload"],
            )
        )
    await client.upsert(collection_name=COLLECTION, points=qdrant_points)


async def hybrid_search(
    query: str,
    top_k: int = 8,
    doc_type_filter: list[str] | None = None,
    regulation_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Perform hybrid (dense + sparse) search with optional metadata filters.
    Returns list of {doc_id, score, payload} dicts.
    """
    # Build cache key
    cache_params = {
        "op": "hybrid_search",
        "query": query,
        "top_k": top_k,
        "doc_type": sorted(doc_type_filter or []),
        "regulation": sorted(regulation_filter or []),
    }
    cached = await get_cached(NS_RETRIEVAL, cache_params)
    if cached:
        return cached

    client = get_qdrant_client()

    # Build filter
    must_conditions: list[models.Condition] = []
    if doc_type_filter:
        must_conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchAny(any=doc_type_filter),
            )
        )
    if regulation_filter:
        must_conditions.append(
            models.FieldCondition(
                key="regulation",
                match=models.MatchAny(any=regulation_filter),
            )
        )
    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    # Compute vectors
    dense_vec = await embed_query(query)
    sparse_indices, sparse_values = encode_sparse(query)

    # Prefetch both dense and sparse, then fuse with RRF
    prefetch = [
        models.Prefetch(
            query=dense_vec,
            using=DENSE_VECTOR_NAME,
            limit=top_k * 3,
            filter=query_filter,
        ),
    ]
    if sparse_indices:
        prefetch.append(
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using=SPARSE_VECTOR_NAME,
                limit=top_k * 3,
                filter=query_filter,
            )
        )

    try:
        results = await client.query_points(
            collection_name=COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        output = [
            {
                "doc_id": str(pt.id),
                "score": pt.score,
                "payload": pt.payload or {},
            }
            for pt in results.points
        ]
    except Exception as exc:
        logger.error("Qdrant hybrid search failed: %s", exc)
        output = []

    await set_cached(NS_RETRIEVAL, cache_params, output, ttl=settings.retrieval_cache_ttl)
    return output
