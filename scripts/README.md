# Utility Scripts & Evaluation Framework

This directory contains standalone execution scripts for knowledge base initialization and quantitative model evaluation.

---

## Scripts Overview

### 1. Policy Knowledge Base Seeder (`seed_policy_kb.py`)
Reads regulatory JSON documents from `data/policy_docs/`, generates embeddings via Gemini `text-embedding-004`, and indexes vector points into Qdrant.

```bash
python3 scripts/seed_policy_kb.py
```

### 2. Evaluation Suite (`run_evaluation.py`)
Executes automated benchmark evaluation against the golden test dataset (`app/evaluation/golden_test_set.json`). Calculates:
- **Citation Precision Rate**
- **Verifier Flag Accuracy**
- **Ungrounded Claim Removal Rate**
- **Execution Latency Benchmarks**

```bash
python3 scripts/run_evaluation.py
```
