"""OpenAI-compatible LLM client: builds the grounding prompt, requests JSON
output, and retries once on invalid JSON before raising LLMError."""
from __future__ import annotations

import json

import openai
from openai import OpenAI
from pydantic import ValidationError

from src.config import LLMConnection, LLMThresholds
from src.schemas import LLMOutput

SYSTEM_PROMPT_TEMPLATE = """You are a policy assistant for IIFL Finance customer support.
Answer ONLY using the policy sections provided below. Do not use outside knowledge.

Rules:
- Cite exactly one section id you used in "cited_section_id", copied exactly as given below.
- Include a short verbatim quote from that section in "evidence_quote" (at most 30 words), \
copied exactly as it appears in the section text.
- If the sections do not contain enough information to answer, set "answerable" to false, \
"cited_section_id" and "evidence_quote" to null, and write a brief "answer" saying a human \
agent is needed.
- Never invent numbers, fees, percentages, or timelines that are not present in the sections.
- Keep "answer" under {max_words} words, in plain language.
- Respond with a single JSON object only, with exactly these keys: "answerable" (bool), \
"answer" (string), "cited_section_id" (string or null), "evidence_quote" (string or null). \
No other text.

Policy sections:
{sections}
"""

RETRY_NOTE = (
    "\n\n(Your previous response was not valid JSON matching the required schema. "
    "Respond with ONLY the JSON object.)"
)


class LLMError(Exception):
    """Raised when the LLM cannot produce a valid, parseable answer."""


def _format_sections(sections: list[dict]) -> str:
    """Render retrieved sections as a labelled block for the prompt."""
    blocks = [f"### {s['section_id']}\n{s['text']}" for s in sections]
    return "\n\n".join(blocks)


def _parse_output(raw: str) -> LLMOutput:
    """Parse and validate the model's raw JSON response."""
    data = json.loads(raw)
    return LLMOutput(**data)


def _is_ollama(base_url: str) -> bool:
    """Ollama's default port/host, used to gate an Ollama-only request field."""
    return "11434" in base_url or "ollama" in base_url.lower()


class LLMClient:
    """Thin wrapper around the OpenAI-compatible chat completions API."""

    def __init__(self, connection: LLMConnection, thresholds: LLMThresholds) -> None:
        self.thresholds = thresholds
        self.base_url = connection.base_url
        self.client = OpenAI(
            base_url=connection.base_url,
            api_key=connection.api_key,
            timeout=thresholds.request_timeout_seconds,
        )
        self.model = connection.model
        self.last_used_retry = False

    def _call(self, question: str, sections: list[dict], retry_note: str = "") -> str:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            max_words=self.thresholds.max_answer_words,
            sections=_format_sections(sections),
        )
        extra_body = {}
        if _is_ollama(self.base_url):
            # "options.num_ctx" is Ollama-specific and rejected outright by other
            # OpenAI-compatible servers (e.g. Groq returns 400 "options unsupported").
            # It widens the context window so a reasoning model's chain-of-thought
            # doesn't consume the whole budget before emitting the JSON answer.
            extra_body = {"options": {"num_ctx": self.thresholds.context_window_tokens}}

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question + retry_note},
            ],
            extra_body=extra_body,
        )
        return response.choices[0].message.content or ""

    def generate(self, question: str, sections: list[dict]) -> LLMOutput:
        """Ask the LLM to answer the question grounded in the given sections.

        Retries once (per config) on invalid JSON; raises LLMError if the
        request itself fails or it still can't parse after the retry.
        Sets `self.last_used_retry` so callers (e.g. the eval harness) can
        track the JSON retry rate without changing this method's signature.
        """
        self.last_used_retry = False
        try:
            raw = self._call(question, sections)
        except openai.APIError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            return _parse_output(raw)
        except (json.JSONDecodeError, ValidationError) as first_error:
            for _ in range(self.thresholds.max_json_retries):
                try:
                    raw = self._call(question, sections, RETRY_NOTE)
                except openai.APIError as exc:
                    raise LLMError(f"LLM retry request failed: {exc}") from exc
                try:
                    output = _parse_output(raw)
                    self.last_used_retry = True
                    return output
                except (json.JSONDecodeError, ValidationError):
                    continue
            raise LLMError(f"invalid LLM JSON after retries: {first_error}") from first_error
