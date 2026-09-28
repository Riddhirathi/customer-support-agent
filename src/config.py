"""Loads configuration from config.yaml (thresholds) and environment
variables (LLM provider connection, via .env)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")


class RetrievalConfig(BaseModel):
    """Thresholds and settings for the hybrid retriever."""

    top_k: int
    bm25_weight: float
    embedding_weight: float
    embedding_model: str
    pdf_strategy: str


class GuardConfig(BaseModel):
    """Thresholds and phrase lists for input guards."""

    max_question_length: int
    injection_phrases: list[str]
    sensitive_intent_phrases: list[str]


class LLMThresholds(BaseModel):
    """Non-secret LLM behaviour settings."""

    max_answer_words: int
    request_timeout_seconds: int
    max_json_retries: int
    context_window_tokens: int


class VerifyConfig(BaseModel):
    """Thresholds used to turn checks into a confidence level."""

    citation_fuzzy_match_threshold: int
    confidence_high_score_threshold: float
    confidence_low_score_threshold: float


class LLMConnection(BaseModel):
    """Provider connection details, sourced only from env vars."""

    base_url: str
    api_key: str
    model: str


class Config(BaseModel):
    """All configuration needed to run the agent."""

    retrieval: RetrievalConfig
    guard: GuardConfig
    llm: LLMThresholds
    verify: VerifyConfig
    connection: LLMConnection
    policies_dir: Path
    logs_path: Path


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load config.yaml and env vars once, cached for the process lifetime."""
    with open(REPO_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    connection = LLMConnection(
        base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=os.environ.get("LLM_API_KEY", ""),
        model=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
    )

    return Config(
        retrieval=RetrievalConfig(**raw["retrieval"]),
        guard=GuardConfig(**raw["guard"]),
        llm=LLMThresholds(**raw["llm"]),
        verify=VerifyConfig(**raw["verify"]),
        connection=connection,
        policies_dir=REPO_ROOT / "data" / "policies",
        logs_path=REPO_ROOT / "logs" / "traces.jsonl",
    )
