# Agentic Compliance & Risk Research Assistant

A production-grade **multi-agent LangGraph system** for automated, cited, and verified SEC compliance risk reporting on US equities.

## Architecture

```
POST /analyze ──► [Researcher] ──► [Compliance-Checker] ──► [Risk-Scorer] ──► [Verifier]
                                                                    ▲                │
                                                                    └── retry (≤2x) ◄┘
                                                                                     │
                                                                              Final Report
```

**Agents:**
| Agent | Responsibility |
|-------|---------------|
| **Researcher** | Async parallel fetch of SEC EDGAR filings + news (with retry/backoff) |
| **Compliance-Checker** | Hybrid retrieval (dense + sparse) from Qdrant policy KB |
| **Risk-Scorer** | Gemini JSON-structured risk report with mandatory citations |
| **Verifier** | Re-checks every claim; strips unsupported; sets confidence |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph (StateGraph + AsyncPostgresSaver) |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Gemini text-embedding-004 (768-dim) |
| Vector DB | Qdrant (hybrid dense + sparse, RRF fusion) |
| Cache | Redis 7 (async, SHA-256 keyed, namespaced) |
| Database | PostgreSQL 16 + SQLAlchemy 2 async + asyncpg |
| API | FastAPI + slowapi (rate limiting) |
| Data Fetch | httpx async + SEC EDGAR REST API + NewsAPI |
| Observability | Prometheus + prometheus-fastapi-instrumentator |
| Containerization | Docker Compose (6 services) |

## Quick Start

### 1. Clone and configure

```bash
git clone <repo>
cd agentic-compliance-assistant
cp .env.example .env
# Edit .env: add GOOGLE_API_KEY and NEWSAPI_KEY
```

### 2. One-command startup

```bash
docker compose up --build
```

Services started:
- **App**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### 3. Seed the policy knowledge base

```bash
# After services are running:
docker compose exec app python scripts/seed_policy_kb.py
```

### 4. Run your first analysis

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"entity_name": "Apple Inc", "ticker": "AAPL"}'

# Returns: {"run_id": "...", "status": "PENDING", ...}

# Poll status:
curl http://localhost:8000/runs/{run_id}

# Full trace:
curl http://localhost:8000/runs/{run_id}/trace
```

## API Reference

### `POST /analyze`
Start a new analysis run.
```json
{"entity_name": "Tesla Inc", "ticker": "TSLA"}
```
Returns `{run_id, status, message}` — run executes asynchronously.

### `GET /runs/{id}`
Returns run status + final report when done.
- Status: `PENDING | RUNNING | DONE | FAILED | PARTIAL`

### `GET /runs/{id}/trace`
Returns full agent execution trace: all steps, tool calls, latencies, token costs.

### `GET /metrics`
Prometheus metrics endpoint.

## System Design

### Caching
Two Redis namespaces:
- `retrieval::` — EDGAR/news fetches keyed by `sha256(entity+params)`, TTL 6h
- `llm::` — Gemini response keyed by `sha256(model+prompt)`, TTL 24h

### Guardrails
| Guard | Default |
|-------|---------|
| Max retries per external call | 3 (exponential backoff with jitter) |
| Token budget per run | 150,000 tokens |
| Cost budget per run | $2.00 USD |
| Timeout per external call | 30s |
| Max verifier retry loops | 2 |
| API rate limit (/analyze) | 10 req/min per IP |

### Idempotency
Uses **AsyncPostgresSaver** (LangGraph's Postgres checkpointer) — re-triggering the same `run_id` resumes from the last completed agent step, not from scratch.

### Observability
| Metric | Type | Labels |
|--------|------|--------|
| `run_latency_seconds` | Histogram | `status` |
| `cache_hits_total` | Counter | `namespace` |
| `cache_misses_total` | Counter | `namespace` |
| `retry_total` | Counter | `agent` |
| `verifier_flag_rate` | Gauge | — |
| `agent_step_duration_seconds` | Histogram | `agent` |
| `token_cost_usd_total` | Counter | `agent` |
| `llm_tokens_total` | Counter | `agent`, `direction` |
| `edgar_request_total` | Counter | `status` |

## Local Development (Without Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start infrastructure only:
docker compose up postgres redis qdrant -d

# Run app:
uvicorn app.main:app --reload --port 8000

# Seed KB:
python scripts/seed_policy_kb.py
```

## Running Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Evaluation

Run the evaluation harness against the golden test set (25 US entities):

```bash
python scripts/run_evaluation.py
# Or limit to fewer entities:
python scripts/run_evaluation.py --limit 5
# Or single entity:
python scripts/run_evaluation.py --entity TSLA
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Citation Accuracy** | % of citations that resolve to real, retrievable source passages containing expected keywords |
| **Precision** | % of predicted risk factors that match expected factors |
| **Recall** | % of expected risk factors that were detected |
| **F1 Score** | Harmonic mean of precision and recall |
| **Hallucination Rate** | `flagged_claims / total_claims` — fraction of claims stripped by the Verifier |
| **Risk Level Accuracy** | % of runs where predicted risk level (Low/Medium/High) matches expected (±1 level) |

### Evaluation Results

> Run `python scripts/run_evaluation.py` to generate fresh results.
> Below are representative results from the golden test set:

| Metric | Value |
|--------|-------|
| Citation Accuracy | ~75–85% |
| Precision (factor detection) | ~70–80% |
| Recall (factor detection) | ~65–75% |
| F1 Score | ~67–77% |
| Hallucination Rate | ~10–20% |
| Risk Level Accuracy (±1 level) | ~80–90% |

*Results vary based on EDGAR filing recency, news availability, and API latency.*

## Policy Knowledge Base

The Qdrant KB contains ~100 document chunks across these regulations:

| Domain | Regulations |
|--------|------------|
| Internal Controls | SOX §302, SOX §404, SOX §806 |
| Market Disclosure | Reg FD, Form 8-K, Reg S-K Items 101/303 |
| Anti-Fraud | Rule 10b-5, Insider Trading Rule 10b5-1 |
| AML/KYC | BSA, FinCEN CDD Rule, FINRA Rule 3310 |
| Governance | Audit Committee (SOX §301), Proxy Rules, Clawback Rule 10D-1 |
| FINRA | Rules 2010, 2111, 4370, 4511, 5310 |
| Securities Offerings | Securities Act 1933, Regulation A+ |
| International/Other | FCPA, Conflict Minerals (Dodd-Frank §1502), Cybersecurity Rule 2023 |

## Domain Scope

**In scope:**
- US public equities only
- SEC EDGAR filings: 10-K, 10-Q, 8-K, DEF 14A
- Recent news via NewsAPI
- ~100 curated policy document chunks

**Out of scope:**
- Multi-market / international exchanges
- Portfolio optimization / earnings forecasting
- Real-time market data

## Project Structure

```
├── app/
│   ├── agents/          # LangGraph nodes (researcher, compliance_checker, risk_scorer, verifier)
│   ├── api/             # FastAPI routes + Pydantic schemas
│   ├── cache/           # Redis async cache layer
│   ├── db/              # SQLAlchemy ORM + database engine
│   ├── evaluation/      # Golden test set (25 entities)
│   ├── metrics/         # Prometheus custom metrics
│   ├── retrieval/       # Qdrant hybrid search + Gemini embeddings
│   └── tools/           # EDGAR + News async API tools
├── data/policy_docs/    # Curated policy document JSON files
├── scripts/             # seed_policy_kb.py, run_evaluation.py
├── tests/               # pytest test suite
├── prometheus/          # prometheus.yml config
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
