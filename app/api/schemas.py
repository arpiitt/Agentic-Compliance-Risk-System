"""
Pydantic request/response schemas for the FastAPI layer.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Request Schemas ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    entity_name: str = Field(..., min_length=1, max_length=255, description="Company name")
    ticker: Optional[str] = Field(None, max_length=20, description="Ticker symbol (e.g. AAPL)")

    model_config = {"json_schema_extra": {"example": {"entity_name": "Apple Inc", "ticker": "AAPL"}}}


# ─── Response Schemas ─────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    run_id: str
    status: str
    message: str


class CitationOut(BaseModel):
    source_doc_id: str
    source_type: str
    passage: str
    url: str
    title: str


class RiskFactorOut(BaseModel):
    name: str
    level: str
    rationale: str
    citations: list[CitationOut]


class RiskReportOut(BaseModel):
    entity_name: str
    ticker: str
    overall_level: str
    risk_score: float
    summary: str
    factors: list[RiskFactorOut]
    confidence: float


class RunStatusResponse(BaseModel):
    run_id: str
    entity_name: str
    ticker: Optional[str]
    status: str
    current_step: Optional[str]
    cost_usd: float
    tokens_used: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    report: Optional[RiskReportOut] = None


class ToolCallOut(BaseModel):
    tool_name: str
    args: dict[str, Any]
    result_summary: str
    latency_ms: int
    cached: bool
    error: Optional[str]


class TraceStepOut(BaseModel):
    agent_name: str
    step_index: int
    inputs: Any
    outputs: Any
    tool_calls: list[ToolCallOut]
    latency_ms: Optional[int]
    tokens_input: int
    tokens_output: int
    cost_usd: float
    error: Optional[str]
    created_at: datetime


class RunTraceResponse(BaseModel):
    run_id: str
    entity_name: str
    steps: list[TraceStepOut]


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
