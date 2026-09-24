"""
LangGraph StateGraph definition wiring all four agents.
Flow: Researcher → Compliance-Checker → Risk-Scorer → Verifier
      → (loop back to Risk-Scorer if flagged, max 2 retries) → END

Uses MemorySaver for local dev (no Postgres needed).
PostgreSQL AsyncPostgresSaver is used when DATABASE_URL is fully configured.
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.agents.compliance_checker import compliance_checker_node
from app.agents.researcher import researcher_node
from app.agents.risk_scorer import risk_scorer_node
from app.agents.state import AgentState
from app.agents.verifier import verifier_node
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _route_after_verifier(state: AgentState) -> Literal["risk_scorer", "__end__"]:
    """
    Conditional edge: after Verifier, decide whether to retry Risk-Scorer or END.
    """
    should_retry = state.get("should_retry", False)
    loop_count = state.get("verifier_loop_count", 0)
    budget = state.get("token_budget_remaining", settings.token_budget)

    if should_retry and loop_count <= settings.max_verifier_loops and budget > 0:
        logger.info(
            "[graph] Verifier flagged issues — routing back to risk_scorer (loop %d/%d)",
            loop_count, settings.max_verifier_loops,
        )
        return "risk_scorer"
    return "__end__"


def build_graph() -> StateGraph:
    """Construct the LangGraph state graph."""
    graph = StateGraph(AgentState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("compliance_checker", compliance_checker_node)
    graph.add_node("risk_scorer", risk_scorer_node)
    graph.add_node("verifier", verifier_node)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "compliance_checker")
    graph.add_edge("compliance_checker", "risk_scorer")
    graph.add_edge("risk_scorer", "verifier")

    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {
            "risk_scorer": "risk_scorer",
            "__end__": END,
        },
    )

    return graph


def create_compiled_graph_sync(use_memory: bool = True):
    """
    Compile graph synchronously with MemorySaver checkpointer.
    Used for Streamlit UI and evaluation runs.
    """
    graph = build_graph()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


async def create_compiled_graph(checkpointer=None):
    """
    Compile the graph with an optional checkpointer.
    Falls back to MemorySaver if no checkpointer provided.
    """
    graph = build_graph()
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile(checkpointer=MemorySaver())


async def get_postgres_checkpointer():
    """
    Initialize Postgres-backed checkpointer for full persistence.
    Only used in Docker / production deployments.
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_sync_url)
        await checkpointer.setup()
        return checkpointer
    except Exception as exc:
        logger.warning("Postgres checkpointer unavailable, using MemorySaver: %s", exc)
        return MemorySaver()
