"""Tests that the hybrid retriever returns the right policy section for a
few representative queries. Loads the real embedding model once per module
(no API key needed — this is a local model, not the LLM provider)."""
import pytest

from src.config import get_config
from src.retriever import Retriever


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    config = get_config()
    return Retriever(config.policies_dir, config.retrieval)


def test_bounce_charge_query_hits_bounce_section(retriever: Retriever) -> None:
    results = retriever.retrieve("What happens if my EMI payment bounces?")
    assert results[0]["section_id"] == "emi_payments#bounce-charges"


def test_foreclosure_query_hits_foreclosure_section(retriever: Retriever) -> None:
    results = retriever.retrieve("What is the foreclosure charge for a fixed rate loan?")
    assert results[0]["section_id"] == "prepayment_foreclosure#foreclosure-charges"


def test_address_change_query_hits_address_section(retriever: Retriever) -> None:
    # hit@3, not hit@1: "change" appears three times in the generic
    # how-to-submit-a-request section (in unrelated contexts - mobile
    # number, name, approved changes) which inflates its BM25 score above
    # address-change's, even though address-change is the clear top
    # embedding match. This mirrors the eval harness's own hit@3 metric.
    results = retriever.retrieve("How do I change my registered address?", top_k=3)
    section_ids = [r["section_id"] for r in results]
    assert "kyc_account_updates#address-change" in section_ids


def test_retrieve_respects_top_k(retriever: Retriever) -> None:
    results = retriever.retrieve("EMI due date", top_k=2)
    assert len(results) == 2


def test_retrieve_scores_are_sorted_descending(retriever: Retriever) -> None:
    results = retriever.retrieve("What documents are needed for KYC re-verification?")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
