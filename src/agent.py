"""Orchestrates guard -> retrieve -> llm -> verify into one AgentResult.

Fails closed on any error (returns an escalate result, never raises) and
appends one trace line to logs/traces.jsonl per request.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.config import Config, get_config
from src.guard import GuardResult, run_guard
from src.llm import LLMClient
from src.retriever import Retriever
from src.schemas import AgentResult, Trace
from src.verify import check_citation, check_numbers, decide_action, score_confidence

_retriever: Optional[Retriever] = None


def _get_retriever(config: Config) -> Retriever:
    """Build the retriever once per process and reuse it after that."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(config.policies_dir, config.retrieval)
    return _retriever


def _write_trace(config: Config, trace: Trace) -> None:
    """Append one JSON line to logs/traces.jsonl. Never raises."""
    try:
        config.logs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.logs_path, "a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")
    except OSError:
        pass


def _escalate(
    query: str,
    reason: str,
    config: Config,
    request_id: str,
    started_at: float,
    masked_query: str = "",
    retrieved: Optional[list[dict]] = None,
    checks: Optional[dict] = None,
    llm_latency_ms: Optional[float] = None,
    category: str = "unclear",
    source: Optional[str] = None,
) -> AgentResult:
    """Build a fail-closed escalate result and log its trace."""
    total_latency_ms = (time.perf_counter() - started_at) * 1000
    trace = Trace(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        masked_query=masked_query,
        retrieved=[{"section_id": r["section_id"], "score": r["score"]} for r in (retrieved or [])],
        checks=checks or {},
        llm_latency_ms=llm_latency_ms,
        total_latency_ms=total_latency_ms,
        confidence="low",
        action="escalate",
        reason=reason,
    )
    _write_trace(config, trace)
    return AgentResult(
        query=query,
        category=category,
        answer="This needs a human agent to look into your account.",
        source=source,
        confidence="low",
        action="escalate",
    )


def _run(
    question: str,
    request_id: str,
    started_at: float,
    config: Config,
    llm_client: Optional[LLMClient],
) -> AgentResult:
    """The actual pipeline. Raises on unexpected errors; answer() catches."""
    guard_result: GuardResult = run_guard(question, config.guard)
    if not guard_result.valid:
        return _escalate(
            question, guard_result.rejection_reason or "invalid_input", config, request_id, started_at
        )

    masked_query = guard_result.masked_question
    checks: dict = {
        "injection_detected": guard_result.injection_detected,
        "sensitive_intent_detected": guard_result.sensitive_intent_detected,
    }

    try:
        retriever = _get_retriever(config)
        retrieved = retriever.retrieve(masked_query)
    except Exception as exc:
        return _escalate(
            question, f"retrieval_error: {exc}", config, request_id, started_at,
            masked_query=masked_query, checks=checks,
        )

    if not retrieved:
        return _escalate(
            question, "no_sections_retrieved", config, request_id, started_at,
            masked_query=masked_query, checks=checks,
        )

    llm_started = time.perf_counter()
    client = llm_client or LLMClient(config.connection, config.llm)
    try:
        llm_output = client.generate(masked_query, retrieved)
    except Exception as exc:
        checks["llm_json_retried"] = getattr(client, "last_used_retry", False)
        return _escalate(
            question, f"llm_error: {exc}", config, request_id, started_at,
            masked_query=masked_query, retrieved=retrieved, checks=checks,
        )
    llm_latency_ms = (time.perf_counter() - llm_started) * 1000
    checks["llm_json_retried"] = getattr(client, "last_used_retry", False)

    citation_ok = check_citation(llm_output, retrieved, config.verify.citation_fuzzy_match_threshold)
    numbers_ok = check_numbers(llm_output, retrieved)
    top_score = retrieved[0]["score"]
    confidence = score_confidence(
        llm_output.answerable, citation_ok, numbers_ok, top_score, config.verify
    )
    action = decide_action(
        confidence, guard_result.sensitive_intent_detected, guard_result.injection_detected
    )

    checks.update(
        {
            "answerable": llm_output.answerable,
            "citation_ok": citation_ok,
            "numbers_ok": numbers_ok,
        }
    )

    reason: Optional[str] = None
    if action == "escalate":
        if guard_result.sensitive_intent_detected:
            reason = "sensitive_intent"
        elif guard_result.injection_detected:
            reason = "injection_detected"
        elif not llm_output.answerable:
            reason = "not_answerable"
        else:
            reason = "low_confidence"

    category = retrieved[0]["category"]
    source = llm_output.cited_section_id or retrieved[0]["section_id"]

    result = AgentResult(
        query=question,
        category=category,
        answer=llm_output.answer,
        source=source,
        confidence=confidence,
        action=action,
    )

    total_latency_ms = (time.perf_counter() - started_at) * 1000
    trace = Trace(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        masked_query=masked_query,
        retrieved=[{"section_id": r["section_id"], "score": r["score"]} for r in retrieved],
        checks=checks,
        llm_latency_ms=llm_latency_ms,
        total_latency_ms=total_latency_ms,
        confidence=confidence,
        action=action,
        reason=reason,
    )
    _write_trace(config, trace)
    return result


def answer(question: str, llm_client: Optional[LLMClient] = None) -> AgentResult:
    """Run the full pipeline for one question. Never raises."""
    started_at = time.perf_counter()
    request_id = str(uuid.uuid4())

    try:
        config = get_config()
    except Exception:
        return AgentResult(
            query=question,
            category="unclear",
            answer="This needs a human agent to look into your account.",
            source=None,
            confidence="low",
            action="escalate",
        )

    try:
        return _run(question, request_id, started_at, config, llm_client)
    except Exception as exc:
        return _escalate(question, f"unexpected_error: {exc}", config, request_id, started_at)
