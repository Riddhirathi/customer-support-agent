"""End-to-end agent tests using a fake LLM client — no API key needed.
Retrieval and guard checks run for real; only the LLM call is faked."""
import json

from src.agent import answer
from src.config import get_config
from src.llm import LLMError
from src.schemas import LLMOutput


class FakeLLMClient:
    """Stands in for LLMClient: returns a fixed LLMOutput or raises."""

    def __init__(self, output: LLMOutput | None = None, exception: Exception | None = None) -> None:
        self.output = output
        self.exception = exception

    def generate(self, question: str, sections: list[dict]) -> LLMOutput:
        if self.exception is not None:
            raise self.exception
        assert self.output is not None
        return self.output


def _last_trace_reason() -> str | None:
    """AgentResult doesn't carry the escalation reason (it's not part of
    the public schema) — read it back from the trace log instead, which is
    where the "clear reason" hard constraint is actually satisfied."""
    lines = get_config().logs_path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])["reason"]


def test_agent_responds_with_grounded_high_confidence_answer() -> None:
    fake = FakeLLMClient(
        output=LLMOutput(
            answerable=True,
            answer="A bounce charge of Rs. 500 plus 18% GST applies per failed EMI attempt.",
            cited_section_id="emi_payments#bounce-charges",
            evidence_quote="a bounce charge of Rs. 500 per instance is levied",
        )
    )
    result = answer("What charges apply if my EMI bounces?", llm_client=fake)
    assert result.action == "respond"
    assert result.confidence == "high"
    assert result.category == "emi_payments"
    assert not hasattr(result, "reason")


def test_agent_escalates_on_empty_input() -> None:
    result = answer("   ")
    assert result.action == "escalate"
    assert _last_trace_reason() == "empty_input"


def test_agent_escalates_on_sensitive_intent_even_with_good_answer() -> None:
    fake = FakeLLMClient(
        output=LLMOutput(
            answerable=True,
            answer="Rs. 500 plus 18% GST applies per failed EMI attempt.",
            cited_section_id="emi_payments#bounce-charges",
            evidence_quote="a bounce charge of Rs. 500 per instance is levied",
        )
    )
    result = answer("What charges apply if my account's EMI bounces?", llm_client=fake)
    assert result.action == "escalate"
    assert _last_trace_reason() == "sensitive_intent"


def test_agent_escalates_when_llm_says_not_answerable() -> None:
    fake = FakeLLMClient(
        output=LLMOutput(answerable=False, answer="This is not covered by the available policy sections.")
    )
    result = answer("What is the meaning of life?", llm_client=fake)
    assert result.action == "escalate"
    assert _last_trace_reason() == "not_answerable"


def test_agent_escalates_on_invalid_llm_json() -> None:
    fake = FakeLLMClient(exception=LLMError("invalid LLM JSON after retries: Expecting value"))
    result = answer("What charges apply if my EMI bounces?", llm_client=fake)
    assert result.action == "escalate"
    assert result.confidence == "low"
    assert "llm_error" in _last_trace_reason()


def test_agent_never_raises_on_unexpected_exception() -> None:
    fake = FakeLLMClient(exception=RuntimeError("unexpected provider failure"))
    result = answer("What charges apply if my EMI bounces?", llm_client=fake)
    assert result.action == "escalate"


def test_agent_escalates_when_answer_invents_a_number() -> None:
    fake = FakeLLMClient(
        output=LLMOutput(
            answerable=True,
            answer="A bounce charge of Rs. 999 applies.",
            cited_section_id="emi_payments#bounce-charges",
            evidence_quote="a bounce charge of Rs. 500 per instance is levied",
        )
    )
    result = answer("What charges apply if my EMI bounces?", llm_client=fake)
    assert result.action == "escalate"
    assert result.confidence == "low"
