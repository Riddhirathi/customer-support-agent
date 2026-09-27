"""Tests for citation grounding, number grounding, and confidence/action
scoring. No API key needed — these are pure functions over fixed data."""
from src.config import get_config
from src.schemas import LLMOutput
from src.verify import check_citation, check_numbers, decide_action, score_confidence

RETRIEVED = [
    {
        "section_id": "emi_payments#bounce-charges",
        "category": "emi_payments",
        "text": (
            "If an EMI payment fails or a NACH mandate is dishonoured for any reason, a bounce "
            "charge of Rs. 500 per instance is levied, plus applicable GST at 18%."
        ),
        "score": 0.9,
    },
    {
        "section_id": "emi_payments#overview",
        "category": "emi_payments",
        "text": "This document explains EMI due dates and payment modes.",
        "score": 0.4,
    },
]


def test_citation_check_passes_with_matching_evidence_quote() -> None:
    output = LLMOutput(
        answerable=True,
        answer="Rs. 500 plus 18% GST applies per bounced EMI.",
        cited_section_id="emi_payments#bounce-charges",
        evidence_quote="a bounce charge of Rs. 500 per instance is levied",
    )
    assert check_citation(output, RETRIEVED, threshold=70)


def test_citation_check_fails_for_section_not_in_retrieved() -> None:
    output = LLMOutput(
        answerable=True,
        answer="...",
        cited_section_id="kyc_account_updates#address-change",
        evidence_quote="some text from a different document",
    )
    assert not check_citation(output, RETRIEVED, threshold=70)


def test_citation_check_fails_for_unrelated_evidence_quote() -> None:
    output = LLMOutput(
        answerable=True,
        answer="...",
        cited_section_id="emi_payments#bounce-charges",
        evidence_quote="The sun sets in the west every single day without fail.",
    )
    assert not check_citation(output, RETRIEVED, threshold=70)


def test_citation_check_fails_when_not_answerable() -> None:
    output = LLMOutput(answerable=False, answer="Not covered by policy.")
    assert not check_citation(output, RETRIEVED, threshold=70)


def test_number_check_catches_invented_fee() -> None:
    output = LLMOutput(
        answerable=True,
        answer="A bounce charge of Rs. 999 applies.",
        cited_section_id="emi_payments#bounce-charges",
        evidence_quote="bounce charge",
    )
    assert not check_numbers(output, RETRIEVED)


def test_number_check_passes_for_numbers_present_in_cited_section() -> None:
    output = LLMOutput(
        answerable=True,
        answer="A bounce charge of Rs. 500 plus 18% GST applies.",
        cited_section_id="emi_payments#bounce-charges",
        evidence_quote="bounce charge",
    )
    assert check_numbers(output, RETRIEVED)


def test_number_check_passes_when_answer_has_no_numbers() -> None:
    output = LLMOutput(
        answerable=True,
        answer="A bounce charge applies to failed payments.",
        cited_section_id="emi_payments#bounce-charges",
        evidence_quote="bounce charge",
    )
    assert check_numbers(output, RETRIEVED)


def test_confidence_high_when_all_checks_pass_and_score_is_high() -> None:
    config = get_config().verify
    assert score_confidence(True, True, True, 0.9, config) == "high"


def test_confidence_medium_when_score_is_mid_range() -> None:
    config = get_config().verify
    mid_score = (config.confidence_low_score_threshold + config.confidence_high_score_threshold) / 2
    assert score_confidence(True, True, True, mid_score, config) == "medium"


def test_confidence_low_when_not_answerable() -> None:
    config = get_config().verify
    assert score_confidence(False, True, True, 0.9, config) == "low"


def test_confidence_low_when_citation_fails() -> None:
    config = get_config().verify
    assert score_confidence(True, False, True, 0.9, config) == "low"


def test_confidence_low_when_numbers_check_fails() -> None:
    config = get_config().verify
    assert score_confidence(True, True, False, 0.9, config) == "low"


def test_action_escalates_on_low_confidence() -> None:
    assert decide_action("low", sensitive_intent=False, injection_detected=False) == "escalate"


def test_action_escalates_on_sensitive_intent_even_with_high_confidence() -> None:
    assert decide_action("high", sensitive_intent=True, injection_detected=False) == "escalate"


def test_action_escalates_on_injection_even_with_high_confidence() -> None:
    assert decide_action("high", sensitive_intent=False, injection_detected=True) == "escalate"


def test_action_responds_on_high_confidence_with_no_flags() -> None:
    assert decide_action("high", sensitive_intent=False, injection_detected=False) == "respond"


def test_action_responds_on_medium_confidence_with_no_flags() -> None:
    assert decide_action("medium", sensitive_intent=False, injection_detected=False) == "respond"
