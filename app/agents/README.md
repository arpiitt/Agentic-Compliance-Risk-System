# Multi-Agent StateGraph Architecture

This directory houses the core multi-agent intelligence system built on **LangGraph**. The workflow orchestrates four specialized agents in a cyclic state graph with verification loops and token budgeting.

---

## Agent Specifications

```
  [START]
     │
     ▼
[Researcher] ──► [Compliance-Checker] ──► [Risk-Scorer] ──► [Verifier]
                                                ▲                │
                                                └── retry (≤2x) ◄┘ (If ungrounded claims exist)
                                                                 │
                                                          [APPROVED / END]
```

### 1. Researcher Agent (`researcher.py`)
- **Role**: Data Collection & Fact Extraction.
- **Actions**: Executes asynchronous parallel calls to fetch recent SEC EDGAR filings (10-K, 10-Q, 8-K) and financial news articles.
- **Resilience**: Implements exponential backoff with jitter retries and Redis caching (`retrieval::` namespace).

### 2. Compliance-Checker Agent (`compliance_checker.py`)
- **Role**: Regulatory Policy Matching.
- **Actions**: Queries vector-indexed policy databases (SEC & FINRA rulebooks) using Qdrant hybrid search (dense embeddings + sparse payload filtering).
- **Fallback**: Performs in-memory semantic filtering over local policy JSON files when Qdrant is unavailable.

### 3. Risk-Scorer Agent (`risk_scorer.py`)
- **Role**: Structured Synthesis & Factor Attribution.
- **Actions**: Invokes **Google Gemini 2.0 Flash** with structured Pydantic schemas to generate an overall risk level (`Low`, `Medium`, `High`), quantitative score (0-100), executive summary, and risk factor breakdown.
- **Mandatory Constraint**: Every identified risk factor must include passage-level source citations.

### 4. Verifier Agent (`verifier.py`)
- **Role**: Self-Reflexive Audit & Anti-Hallucination Guardrail.
- **Actions**: Re-examines every draft claim against cited source passages. Strips ungrounded assertions, sets a confidence score (`Low`, `Medium`, `High`), and determines whether a re-scoring loop is needed.
- **Guardrail Rule**: Maximum 2 verifier retry loops per run to guarantee bounded cost and latency.

---

## State Schema (`state.py`)

The agents pass a shared mutable `AgentState` payload across graph nodes:

```python
class AgentState(TypedDict):
    entity_name: str
    ticker: Optional[str]
    raw_filings: List[Dict[str, Any]]
    raw_news: List[Dict[str, Any]]
    matched_policies: List[Dict[str, Any]]
    draft_report: Dict[str, Any]
    verifier_flags: List[str]
    verifier_loop_count: int
    should_retry: bool
    execution_trace: List[Dict[str, Any]]
    token_budget_remaining: float
```
