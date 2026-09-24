"""Tests for agent nodes with mocked LLM and API calls."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.state import AgentState


def make_state(**kwargs) -> AgentState:
    defaults: AgentState = {
        "run_id": "test-run-123",
        "entity_name": "Apple Inc",
        "ticker": "AAPL",
        "filings": [],
        "news_items": [],
        "researcher_tool_calls": [],
        "policy_matches": [],
        "compliance_tool_calls": [],
        "draft_report": None,
        "flagged_claims": [],
        "final_report": None,
        "verifier_loop_count": 0,
        "should_retry": False,
        "errors": [],
        "token_budget_remaining": 100000,
        "cost_budget_remaining": 2.0,
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        "last_completed_step": "",
    }
    defaults.update(kwargs)
    return defaults


# ─── Researcher Agent Tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_researcher_resolves_cik():
    """Researcher should attempt CIK resolution."""
    state = make_state()
    with (
        patch("app.agents.researcher.resolve_cik", new_callable=AsyncMock, return_value="0000320193"),
        patch("app.agents.researcher.fetch_all_filings_parallel", new_callable=AsyncMock, return_value=[
            {
                "accession_number": "0000320193-23-000001",
                "form_type": "10-K",
                "filing_date": "2023-10-30",
                "company_name": "Apple Inc",
                "cik": "0000320193",
                "text": "Apple Inc annual report. Risk factors include...",
                "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/",
            }
        ]),
        patch("app.agents.researcher.fetch_news", new_callable=AsyncMock, return_value=[
            {
                "doc_id": "news-001",
                "title": "Apple Q4 Results",
                "url": "https://example.com/apple-q4",
                "published_at": "2023-11-01",
                "source_name": "Reuters",
                "description": "Apple reports record revenue",
                "content": "Apple Inc reported...",
            }
        ]),
    ):
        from app.agents.researcher import researcher_node
        result = await researcher_node(state)

    assert "filings" in result
    assert "news_items" in result
    assert len(result["filings"]) > 0
    assert result["last_completed_step"] == "researcher"


@pytest.mark.asyncio
async def test_researcher_handles_cik_not_found():
    """Researcher should handle CIK resolution failure gracefully."""
    state = make_state(entity_name="Nonexistent Corp XYZ")
    with (
        patch("app.agents.researcher.resolve_cik", new_callable=AsyncMock, return_value=None),
        patch("app.agents.researcher.fetch_all_filings_parallel", new_callable=AsyncMock, return_value=[]),
        patch("app.agents.researcher.fetch_news", new_callable=AsyncMock, return_value=[]),
    ):
        from app.agents.researcher import researcher_node
        result = await researcher_node(state)

    assert result["last_completed_step"] == "researcher"
    assert any("CIK" in e for e in result.get("errors", []))


@pytest.mark.asyncio
async def test_researcher_budget_guard():
    """Researcher should skip if token budget is exhausted."""
    state = make_state(token_budget_remaining=0)
    from app.agents.researcher import researcher_node
    result = await researcher_node(state)
    assert result.get("last_completed_step") == "researcher"


# ─── Compliance Checker Tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compliance_checker_calls_hybrid_search():
    """Compliance checker should call hybrid_search for each document."""
    filing = {
        "doc_id": "acc-001",
        "accession_number": "acc-001",
        "form_type": "10-K",
        "filing_date": "2023-10-30",
        "company_name": "Apple Inc",
        "cik": "0000320193",
        "text": "Risk factors: supply chain concentration in Asia...",
        "url": "https://sec.gov",
        "source": "edgar",
    }
    state = make_state(filings=[filing])

    mock_search_results = [
        {
            "doc_id": "policy-001",
            "score": 0.85,
            "payload": {
                "doc_type": "regulation",
                "regulation": "SEC",
                "chunk_text": "Material risks must be disclosed...",
                "url": "https://sec.gov/reg-sk",
            },
        }
    ]

    with patch("app.agents.compliance_checker.hybrid_search", new_callable=AsyncMock, return_value=mock_search_results):
        from app.agents.compliance_checker import compliance_checker_node
        result = await compliance_checker_node(state)

    assert "policy_matches" in result
    assert result["last_completed_step"] == "compliance_checker"


# ─── Risk Scorer Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_risk_scorer_parses_report():
    """Risk scorer should parse Gemini JSON response correctly."""
    filing = {
        "doc_id": "acc-001",
        "accession_number": "acc-001",
        "form_type": "10-K",
        "filing_date": "2023-10-30",
        "company_name": "Apple Inc",
        "cik": "0000320193",
        "text": "Risk factors: supply chain concentration...",
        "url": "https://sec.gov",
        "source": "edgar",
    }
    policy_match = {
        "policy_doc_id": "reg-sk-303",
        "doc_type": "regulation",
        "regulation": "SEC",
        "clause_text": "Companies must disclose material risks...",
        "relevance_score": 0.82,
        "matched_filing_doc_id": "acc-001",
        "url": "https://sec.gov",
    }
    state = make_state(filings=[filing], policy_matches=[policy_match])

    mock_report_json = {
        "entity_name": "Apple Inc",
        "ticker": "AAPL",
        "overall_level": "Low",
        "risk_score": 0.25,
        "summary": "Apple shows low overall compliance risk.",
        "factors": [
            {
                "name": "Supply Chain Concentration",
                "level": "Medium",
                "rationale": "Significant manufacturing in Taiwan and China.",
                "citations": [
                    {
                        "source_doc_id": "EDGAR:acc-001",
                        "source_type": "edgar",
                        "passage": "supply chain concentration in Asia",
                        "url": "https://sec.gov",
                        "title": "10-K 2023",
                    }
                ],
            }
        ],
        "confidence": 0.5,
    }

    mock_response = MagicMock()
    mock_response.text = __import__("json").dumps(mock_report_json)
    mock_response.usage_metadata = None

    with (
        patch("app.cache.redis_cache.get_cached", new_callable=AsyncMock, return_value=None),
        patch("app.cache.redis_cache.set_cached", new_callable=AsyncMock),
        patch("google.generativeai.GenerativeModel") as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model
        patch("google.generativeai.configure")

        from app.agents.risk_scorer import risk_scorer_node
        result = await risk_scorer_node(state)

    assert "draft_report" in result
    if result["draft_report"]:
        assert result["draft_report"]["overall_level"] in ("Low", "Medium", "High")


# ─── Verifier Agent Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verifier_accepts_clean_report():
    """Verifier should accept a report with no flagged claims and set should_retry=False."""
    draft_report = {
        "entity_name": "Apple Inc",
        "ticker": "AAPL",
        "overall_level": "Low",
        "risk_score": 0.25,
        "summary": "Low risk.",
        "factors": [
            {
                "name": "Supply Chain Risk",
                "level": "Medium",
                "rationale": "Manufacturing in Taiwan per 10-K.",
                "citations": [
                    {
                        "source_doc_id": "EDGAR:acc-001",
                        "source_type": "edgar",
                        "passage": "supply chain in Taiwan",
                        "url": "https://sec.gov",
                        "title": "10-K 2023",
                    }
                ],
            }
        ],
        "confidence": 0.5,
    }
    filing = {
        "doc_id": "acc-001",
        "accession_number": "acc-001",
        "form_type": "10-K",
        "filing_date": "2023-10-30",
        "company_name": "Apple Inc",
        "cik": "0000320193",
        "text": "supply chain in Taiwan concentration risk...",
        "url": "https://sec.gov",
        "source": "edgar",
    }
    state = make_state(draft_report=draft_report, filings=[filing])

    mock_verification = {
        "verified_factors": [
            {"name": "Supply Chain Risk", "supported": True, "flagged_reason": None, "verified_citations": ["EDGAR:acc-001"]}
        ],
        "flagged_claims": [],
        "confidence": 0.9,
        "hallucination_count": 0,
        "total_claims_checked": 1,
    }

    mock_response = MagicMock()
    mock_response.text = __import__("json").dumps(mock_verification)
    mock_response.usage_metadata = None

    with (
        patch("app.cache.redis_cache.get_cached", new_callable=AsyncMock, return_value=None),
        patch("app.cache.redis_cache.set_cached", new_callable=AsyncMock),
        patch("google.generativeai.GenerativeModel") as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        from app.agents.verifier import verifier_node
        result = await verifier_node(state)

    assert result.get("should_retry") is False
    assert result.get("final_report") is not None
    assert result["final_report"]["confidence"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_verifier_triggers_retry_on_high_flag_rate():
    """Verifier should set should_retry=True when hallucination rate > 0.3."""
    draft_report = {
        "entity_name": "Tesla Inc",
        "ticker": "TSLA",
        "overall_level": "High",
        "risk_score": 0.8,
        "summary": "High risk.",
        "factors": [
            {
                "name": "Governance Risk",
                "level": "High",
                "rationale": "CEO conduct issues per news.",
                "citations": [{"source_doc_id": "NEWS:news-001", "source_type": "news", "passage": "...", "url": "", "title": ""}],
            }
        ],
        "confidence": 0.5,
    }
    state = make_state(draft_report=draft_report, entity_name="Tesla Inc", ticker="TSLA")

    mock_verification = {
        "verified_factors": [
            {"name": "Governance Risk", "supported": False, "flagged_reason": "Citation passage not found in sources", "verified_citations": []}
        ],
        "flagged_claims": ["CEO conduct issues are not supported by the provided sources."],
        "confidence": 0.2,
        "hallucination_count": 1,
        "total_claims_checked": 1,
    }

    mock_response = MagicMock()
    mock_response.text = __import__("json").dumps(mock_verification)
    mock_response.usage_metadata = None

    with (
        patch("app.cache.redis_cache.get_cached", new_callable=AsyncMock, return_value=None),
        patch("app.cache.redis_cache.set_cached", new_callable=AsyncMock),
        patch("google.generativeai.GenerativeModel") as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        from app.agents.verifier import verifier_node
        result = await verifier_node(state)

    assert result.get("should_retry") is True
    assert result.get("verifier_loop_count") == 1
