"""
FastAPI route handlers — clean, no slowapi dependency.
Rate limiting is handled at the infrastructure level (nginx / docker compose).

Endpoints:
  POST /analyze          — start a new analysis run
  GET  /runs/{id}        — status + final report
  GET  /runs/{id}/trace  — full agent trace
  GET  /health           — liveness check
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.agents.graph import create_compiled_graph
from app.agents.state import AgentState
from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CitationOut,
    HealthResponse,
    RiskFactorOut,
    RiskReportOut,
    RunStatusResponse,
    RunTraceResponse,
    ToolCallOut,
    TraceStepOut,
)
from app.config import get_settings
from app.metrics.prometheus import run_latency_seconds

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _render_markdown(report: dict) -> str:
    lines = [
        f"# Risk Report: {report.get('entity_name', '')} ({report.get('ticker', '')})",
        f"**Overall Risk**: {report.get('overall_level', 'N/A')} "
        f"(score: {report.get('risk_score', 0):.2f})",
        f"**Confidence**: {report.get('confidence', 0):.0%}",
        "",
        f"## Summary\n{report.get('summary', '')}",
        "",
        "## Risk Factors",
    ]
    for factor in report.get("factors", []):
        lines.append(f"\n### {factor.get('name', '')} — {factor.get('level', '')}")
        lines.append(factor.get("rationale", ""))
        for cit in factor.get("citations", []):
            lines.append(
                f"\n> **Source**: [{cit.get('title', '')}]({cit.get('url', '')})\n"
                f"> {cit.get('passage', '')[:200]}"
            )
    return "\n".join(lines)


async def _persist_run(run_id: str, entity_name: str, ticker: str, status: str, **kwargs):
    """Best-effort DB persistence — silently skips if Postgres unavailable."""
    try:
        from app.db.database import get_session
        from app.db.models import Run
        async with get_session() as session:
            run = await session.get(Run, uuid.UUID(run_id))
            if run:
                for k, v in kwargs.items():
                    setattr(run, k, v)
                run.status = status
                run.updated_at = datetime.now(timezone.utc)
    except Exception as exc:
        logger.debug("DB persist skipped (no Postgres): %s", exc)


async def _persist_report(run_id: str, entity_name: str, ticker: str, final_report: dict, state_values: dict):
    """Best-effort report persistence."""
    try:
        from app.db.database import get_session
        from app.db.models import Report
        async with get_session() as session:
            report = Report(
                run_id=uuid.UUID(run_id),
                entity_name=entity_name,
                risk_level=final_report.get("overall_level", "Unknown"),
                risk_score=final_report.get("risk_score", 0.0),
                factors=final_report.get("factors", []),
                confidence=final_report.get("confidence", 0.5),
                hallucination_count=len(state_values.get("flagged_claims", [])),
                total_claims=sum(
                    len(f.get("citations", [])) for f in final_report.get("factors", [])
                ),
                report_md=_render_markdown(final_report),
            )
            session.add(report)
    except Exception as exc:
        logger.debug("Report persist skipped (no Postgres): %s", exc)


# ─── In-memory run store (fallback when Postgres unavailable) ─────────────────
_run_store: dict[str, dict] = {}


# ─── Background task ──────────────────────────────────────────────────────────

async def _run_analysis(run_id: str, entity_name: str, ticker: str) -> None:
    """Execute the full LangGraph pipeline."""
    t0 = time.perf_counter()

    _run_store[run_id] = {
        "run_id": run_id,
        "entity_name": entity_name,
        "ticker": ticker,
        "status": "RUNNING",
        "current_step": "researcher",
        "cost_usd": 0.0,
        "tokens_used": 0,
        "error_message": None,
        "report": None,
        "traces": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    initial_state: AgentState = {
        "run_id": run_id,
        "entity_name": entity_name,
        "ticker": ticker or "",
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
        "token_budget_remaining": settings.token_budget,
        "cost_budget_remaining": settings.cost_budget_usd,
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        "last_completed_step": "",
    }

    try:
        graph = await create_compiled_graph()
        config = {"configurable": {"thread_id": run_id}}
        step_index = 0

        async for step_output in graph.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_output in step_output.items():
                if not isinstance(node_output, dict):
                    continue
                _run_store[run_id]["current_step"] = node_name
                _run_store[run_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

                # Store trace step
                _run_store[run_id]["traces"].append({
                    "agent_name": node_name,
                    "step_index": step_index,
                    "inputs": {},
                    "outputs": {
                        k: v for k, v in node_output.items()
                        if k not in ("filings", "news_items", "policy_matches")
                    },
                    "tool_calls": (
                        node_output.get("researcher_tool_calls", [])
                        or node_output.get("compliance_tool_calls", [])
                        or []
                    ),
                    "latency_ms": None,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "cost_usd": node_output.get("total_cost_usd", 0.0),
                    "error": "; ".join(node_output.get("errors", [])) or None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
                step_index += 1

        # Fetch final state
        final_snapshot = await graph.aget_state(config)
        state_values: dict = final_snapshot.values or {}

        final_report = state_values.get("final_report")
        errors = state_values.get("errors", [])
        status = "DONE" if final_report else ("PARTIAL" if errors else "FAILED")

        _run_store[run_id].update({
            "status": status,
            "current_step": "done",
            "cost_usd": state_values.get("total_cost_usd", 0.0),
            "tokens_used": state_values.get("total_tokens_used", 0),
            "error_message": "; ".join(errors[:3]) if errors else None,
            "report": final_report,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        elapsed = time.perf_counter() - t0
        run_latency_seconds.labels(status=status).observe(elapsed)
        logger.info("Run %s completed: status=%s cost=$%.4f in %.1fs",
                    run_id, status, state_values.get("total_cost_usd", 0.0), elapsed)

    except Exception as exc:
        logger.exception("Fatal error in run %s: %s", run_id, exc)
        _run_store[run_id].update({
            "status": "FAILED",
            "error_message": str(exc)[:500],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        run_latency_seconds.labels(status="FAILED").observe(time.perf_counter() - t0)


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse, status_code=202)
async def start_analysis(
    body: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> AnalyzeResponse:
    """Start a new compliance analysis run. Returns run_id immediately."""
    run_id = str(uuid.uuid4())

    # Kick off async pipeline
    background_tasks.add_task(_run_analysis, run_id, body.entity_name, body.ticker or "")

    logger.info("Enqueued run %s for entity='%s' ticker='%s'",
                run_id, body.entity_name, body.ticker)
    return AnalyzeResponse(
        run_id=run_id,
        status="PENDING",
        message="Analysis started. Poll GET /runs/{run_id} for status and results.",
    )


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str) -> RunStatusResponse:
    """Get the current status and final report (if complete) for a run."""
    # Try in-memory store first
    record = _run_store.get(run_id)

    if not record:
        # Try Postgres
        try:
            from app.db.database import get_session
            from app.db.models import Run
            async with get_session() as session:
                result = await session.execute(
                    select(Run).where(Run.id == uuid.UUID(run_id)).options(selectinload(Run.report))
                )
                run = result.scalar_one_or_none()
                if run:
                    record = {
                        "run_id": str(run.id),
                        "entity_name": run.entity_name,
                        "ticker": run.ticker,
                        "status": run.status,
                        "current_step": run.current_step,
                        "cost_usd": run.cost_usd or 0.0,
                        "tokens_used": run.tokens_used or 0,
                        "error_message": run.error_message,
                        "report": run.report.factors if run.report else None,
                        "created_at": run.created_at.isoformat(),
                        "updated_at": run.updated_at.isoformat(),
                    }
        except Exception:
            pass

    if not record:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    # Build report output
    report_out: RiskReportOut | None = None
    raw_report = record.get("report")
    if raw_report and isinstance(raw_report, dict):
        factors_out = []
        for f in raw_report.get("factors", []):
            citations_out = [
                CitationOut(
                    source_doc_id=c.get("source_doc_id", ""),
                    source_type=c.get("source_type", ""),
                    passage=c.get("passage", ""),
                    url=c.get("url", ""),
                    title=c.get("title", ""),
                )
                for c in f.get("citations", [])
            ]
            factors_out.append(
                RiskFactorOut(
                    name=f.get("name", ""),
                    level=f.get("level", ""),
                    rationale=f.get("rationale", ""),
                    citations=citations_out,
                )
            )
        report_out = RiskReportOut(
            entity_name=raw_report.get("entity_name", record.get("entity_name", "")),
            ticker=raw_report.get("ticker", record.get("ticker", "")),
            overall_level=raw_report.get("overall_level", "Unknown"),
            risk_score=raw_report.get("risk_score", 0.0),
            summary=raw_report.get("summary", ""),
            factors=factors_out,
            confidence=raw_report.get("confidence", 0.5),
        )

    created = record.get("created_at", datetime.now(timezone.utc).isoformat())
    updated = record.get("updated_at", datetime.now(timezone.utc).isoformat())

    return RunStatusResponse(
        run_id=run_id,
        entity_name=record.get("entity_name", ""),
        ticker=record.get("ticker"),
        status=record.get("status", "UNKNOWN"),
        current_step=record.get("current_step"),
        cost_usd=record.get("cost_usd", 0.0),
        tokens_used=record.get("tokens_used", 0),
        error_message=record.get("error_message"),
        created_at=datetime.fromisoformat(created) if isinstance(created, str) else created,
        updated_at=datetime.fromisoformat(updated) if isinstance(updated, str) else updated,
        report=report_out,
    )


@router.get("/runs/{run_id}/trace", response_model=RunTraceResponse)
async def get_run_trace(run_id: str) -> RunTraceResponse:
    """Return the full agent execution trace."""
    record = _run_store.get(run_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    steps = []
    for t in record.get("traces", []):
        raw_tool_calls = t.get("tool_calls") or []
        tool_calls_out = [
            ToolCallOut(
                tool_name=tc.get("tool_name", ""),
                args=tc.get("args", {}),
                result_summary=tc.get("result_summary", ""),
                latency_ms=tc.get("latency_ms", 0),
                cached=tc.get("cached", False),
                error=tc.get("error"),
            )
            for tc in raw_tool_calls
        ]
        created = t.get("created_at", datetime.now(timezone.utc).isoformat())
        steps.append(
            TraceStepOut(
                agent_name=t.get("agent_name", ""),
                step_index=t.get("step_index", 0),
                inputs=t.get("inputs", {}),
                outputs=t.get("outputs", {}),
                tool_calls=tool_calls_out,
                latency_ms=t.get("latency_ms"),
                tokens_input=t.get("tokens_input", 0),
                tokens_output=t.get("tokens_output", 0),
                cost_usd=t.get("cost_usd", 0.0),
                error=t.get("error"),
                created_at=datetime.fromisoformat(created) if isinstance(created, str) else created,
            )
        )

    return RunTraceResponse(
        run_id=run_id,
        entity_name=record.get("entity_name", ""),
        steps=steps,
    )


@router.get("/runs", response_model=list[RunStatusResponse])
async def list_runs() -> list[RunStatusResponse]:
    """List all recent runs (in-memory store)."""
    results = []
    for run_id, record in list(_run_store.items())[-50:]:
        created = record.get("created_at", datetime.now(timezone.utc).isoformat())
        updated = record.get("updated_at", datetime.now(timezone.utc).isoformat())
        results.append(RunStatusResponse(
            run_id=run_id,
            entity_name=record.get("entity_name", ""),
            ticker=record.get("ticker"),
            status=record.get("status", "UNKNOWN"),
            current_step=record.get("current_step"),
            cost_usd=record.get("cost_usd", 0.0),
            tokens_used=record.get("tokens_used", 0),
            error_message=record.get("error_message"),
            created_at=datetime.fromisoformat(created) if isinstance(created, str) else created,
            updated_at=datetime.fromisoformat(updated) if isinstance(updated, str) else updated,
            report=None,
        ))
    return list(reversed(results))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")
