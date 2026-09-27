"""Pydantic models shared across the pipeline."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class LLMOutput(BaseModel):
    """Raw structured output requested from the LLM."""

    answerable: bool
    answer: str
    cited_section_id: Optional[str] = None
    evidence_quote: Optional[str] = None


class AgentResult(BaseModel):
    """The final structured result returned to the caller. The reason for
    an escalation is not part of this public schema — it's captured in the
    trace log (Trace.reason) for audit purposes instead."""

    query: str
    category: str
    answer: str
    source: Optional[str] = None
    confidence: Literal["high", "medium", "low"]
    action: Literal["respond", "escalate"]


class Trace(BaseModel):
    """One record appended to logs/traces.jsonl for every request."""

    request_id: str
    timestamp: str
    masked_query: str
    retrieved: list[dict]
    checks: dict
    llm_latency_ms: Optional[float]
    total_latency_ms: float
    confidence: str
    action: str
    reason: Optional[str]
