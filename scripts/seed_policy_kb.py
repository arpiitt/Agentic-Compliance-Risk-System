"""
Script to ingest curated policy documents into Qdrant.
Chunks documents, embeds with Gemini, builds sparse BM25 vectors,
and upserts to the policy_kb collection.

Run: python scripts/seed_policy_kb.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai

from app.config import get_settings
from app.retrieval.embeddings import (
    build_vocab,
    encode_sparse,
    init_embeddings,
    embed_text,
)
from app.retrieval.qdrant_client import ensure_collection_exists, upsert_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

DATA_DIR = Path(__file__).parent.parent / "data" / "policy_docs"
CHUNK_SIZE = 512  # tokens approx (chars / 4)
CHUNK_OVERLAP = 64
BATCH_SIZE = 20


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


async def seed() -> None:
    init_embeddings()
    await ensure_collection_exists()

    # Load all policy documents
    doc_files = list(DATA_DIR.glob("*.json"))
    if not doc_files:
        logger.error("No policy documents found in %s", DATA_DIR)
        return

    logger.info("Loading %d policy document files...", len(doc_files))
    all_docs: list[dict] = []
    for f in doc_files:
        with open(f) as fh:
            docs = json.load(fh)
            if isinstance(docs, list):
                all_docs.extend(docs)
            else:
                all_docs.append(docs)

    logger.info("Loaded %d raw policy documents", len(all_docs))

    # Build vocabulary from all text (for sparse encoding)
    all_texts = [d.get("text", "") for d in all_docs]
    vocab = build_vocab(all_texts)
    logger.info("Built vocabulary of %d terms", len(vocab))

    # Chunk + embed
    points: list[dict] = []
    total_chunks = 0

    for doc in all_docs:
        doc_id = doc.get("doc_id", str(uuid.uuid4()))
        title = doc.get("title", "")
        text = doc.get("text", "")
        payload_base = {
            "doc_id": doc_id,
            "title": title,
            "doc_type": doc.get("doc_type", "regulation"),
            "regulation": doc.get("regulation", "SEC"),
            "year": doc.get("year", 2024),
            "url": doc.get("url", ""),
        }

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}::{i}"))
            dense_vec = await embed_text(chunk)
            sparse_idx, sparse_vals = encode_sparse(chunk)
            points.append({
                "id": point_id,
                "dense_vector": dense_vec,
                "sparse_indices": sparse_idx,
                "sparse_values": sparse_vals,
                "payload": {
                    **payload_base,
                    "chunk_index": i,
                    "chunk_text": chunk,
                },
            })
            total_chunks += 1

        # Batch upsert
        if len(points) >= BATCH_SIZE:
            await upsert_documents(points)
            logger.info("Upserted %d chunks...", total_chunks)
            points = []

    if points:
        await upsert_documents(points)

    logger.info(" Seeding complete: %d chunks from %d documents", total_chunks, len(all_docs))


if __name__ == "__main__":
    asyncio.run(seed())
