"""
SQLAlchemy ORM models for runs, traces, and reports.
"""
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    """Top-level record for a single analysis run."""
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_name: Mapped[str] = mapped_column(String(255), nullable=False)
    ticker: Mapped[str | None] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING"
    )  # PENDING | RUNNING | DONE | FAILED | PARTIAL
    current_step: Mapped[str | None] = mapped_column(String(50))
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    traces: Mapped[list["Trace"]] = relationship(
        "Trace", back_populates="run", cascade="all, delete-orphan"
    )
    report: Mapped["Report | None"] = relationship(
        "Report", back_populates="run", uselist=False, cascade="all, delete-orphan"
    )


class Trace(Base):
    """Per-agent-step execution trace."""
    __tablename__ = "traces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    inputs: Mapped[Any] = mapped_column(JSON)
    outputs: Mapped[Any] = mapped_column(JSON)
    tool_calls: Mapped[Any] = mapped_column(JSON, default=list)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    tokens_input: Mapped[int] = mapped_column(Integer, default=0)
    tokens_output: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    run: Mapped["Run"] = relationship("Run", back_populates="traces")


class Report(Base):
    """Final risk report output for a run."""
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    entity_name: Mapped[str] = mapped_column(String(255), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10))  # Low | Medium | High
    risk_score: Mapped[float | None] = mapped_column(Float)
    factors: Mapped[Any] = mapped_column(JSON)  # list of RiskFactor dicts
    confidence: Mapped[float | None] = mapped_column(Float)
    hallucination_count: Mapped[int] = mapped_column(Integer, default=0)
    total_claims: Mapped[int] = mapped_column(Integer, default=0)
    report_md: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    run: Mapped["Run"] = relationship("Run", back_populates="report")
