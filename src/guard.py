"""Input validation, PII masking, and injection / sensitive-intent checks.

None of these functions raise on bad input — a rejected question is reported
via GuardResult.valid=False so the caller can fail closed with an escalate
result instead of crashing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.config import GuardConfig

# Order matters: more specific / longer patterns run first so that generic
# fallbacks (e.g. the loan-account digit run) don't swallow their matches.
PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
MOBILE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
LOAN_ACCOUNT_PATTERN = re.compile(r"\b\d{9,18}\b")


def mask_pii(text: str) -> str:
    """Replace PAN, Aadhaar, Indian mobile numbers, emails, and loan account
    numbers with placeholder tokens before the text reaches the LLM or logs."""
    text = PAN_PATTERN.sub("[PAN]", text)
    text = AADHAAR_PATTERN.sub("[AADHAAR]", text)
    text = MOBILE_PATTERN.sub("[MOBILE]", text)
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = LOAN_ACCOUNT_PATTERN.sub("[LOAN_ACCOUNT]", text)
    return text


def check_injection(text: str, phrases: list[str]) -> bool:
    """Return True if the text contains a known prompt-injection phrase."""
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def check_sensitive_intent(text: str, phrases: list[str]) -> bool:
    """Return True if the text suggests an account-specific request that
    must be escalated regardless of how good a general answer would be."""
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


@dataclass
class GuardResult:
    """Outcome of running all guard checks on one raw question."""

    valid: bool
    masked_question: str
    injection_detected: bool
    sensitive_intent_detected: bool
    rejection_reason: Optional[str] = None


def run_guard(question: str, config: GuardConfig) -> GuardResult:
    """Validate length, mask PII, and flag injection / sensitive intent."""
    stripped = question.strip()
    if not stripped:
        return GuardResult(False, "", False, False, "empty_input")
    if len(stripped) > config.max_question_length:
        return GuardResult(False, "", False, False, "input_too_long")

    masked = mask_pii(stripped)
    injection = check_injection(stripped, config.injection_phrases)
    sensitive = check_sensitive_intent(stripped, config.sensitive_intent_phrases)
    return GuardResult(True, masked, injection, sensitive, None)
