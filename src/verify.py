"""Checkable, non-LLM verification: citation grounding, number grounding,
and confidence scoring from those signals — never the LLM's self-rating."""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from src.config import VerifyConfig
from src.schemas import LLMOutput

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")


def check_citation(output: LLMOutput, retrieved: list[dict], threshold: int) -> bool:
    """The cited section id must be among the retrieved sections, and the
    evidence quote must fuzzy-match that section's text."""
    if not output.cited_section_id or not output.evidence_quote:
        return False
    section = next(
        (s for s in retrieved if s["section_id"] == output.cited_section_id), None
    )
    if section is None:
        return False
    return fuzz.partial_ratio(output.evidence_quote, section["text"]) >= threshold


def check_numbers(output: LLMOutput, retrieved: list[dict]) -> bool:
    """Every number or percentage in the answer must appear in the cited
    section's text (or, if uncited, in any retrieved section)."""
    numbers = NUMBER_PATTERN.findall(output.answer)
    if not numbers:
        return True

    if output.cited_section_id:
        sources = [s["text"] for s in retrieved if s["section_id"] == output.cited_section_id]
    else:
        sources = [s["text"] for s in retrieved]
    if not sources:
        return False

    haystack = " ".join(sources)
    return all(number in haystack for number in numbers)


def score_confidence(
    answerable: bool,
    citation_ok: bool,
    numbers_ok: bool,
    top_score: float,
    config: VerifyConfig,
) -> str:
    """Confidence derived only from checkable signals."""
    if answerable and citation_ok and numbers_ok:
        if top_score >= config.confidence_high_score_threshold:
            return "high"
        if top_score >= config.confidence_low_score_threshold:
            return "medium"
    return "low"


def decide_action(confidence: str, sensitive_intent: bool, injection_detected: bool) -> str:
    """Escalate on low confidence, or on any flagged sensitive/injection
    intent regardless of how confident the underlying answer is."""
    if confidence == "low" or sensitive_intent or injection_detected:
        return "escalate"
    return "respond"
