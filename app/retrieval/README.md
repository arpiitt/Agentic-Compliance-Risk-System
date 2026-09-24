# Vector Store & Retrieval Architecture

This module implements hybrid semantic and metadata retrieval over regulatory policy knowledge bases.

---

## Technical Overview

1. **Embedding Generation (`embeddings.py`)**:
   - Uses Google Gemini `text-embedding-004` (768-dimensional dense vector space).
   - Normalizes vector outputs for cosine similarity matching.

2. **Qdrant Vector Database (`qdrant_client.py`)**:
   - Stores policy chunks with structured payload metadata (`regulatory_body`, `section_id`, `category`, `title`).
   - Supports hybrid filtering: dense vector similarity combined with exact payload filters (e.g. `regulatory_body="SEC"`).

3. **In-Memory Fallback**:
   - When Qdrant is offline, the retrieval client falls back to local JSON datasets (`data/policy_docs/sec_regulations.json` and `finra_regulations.json`) using keyword/TF-IDF scoring.
