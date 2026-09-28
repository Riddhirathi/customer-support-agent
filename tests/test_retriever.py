"""Tests for the policy PDF loader (unstructured -> sections) and for the
hybrid retriever's ranking. Loads the real embedding model once per module
(no API key needed — this is a local model, not the LLM provider)."""
import os
import shutil
from pathlib import Path

import pytest

from src.config import get_config

# src.retriever is imported before unstructured on purpose: it opts out of unstructured's
# telemetry, which has to be set before unstructured is first imported.
from src.retriever import Retriever, _group_into_sections, load_sections
from unstructured.documents.elements import Footer, Header, ListItem, NarrativeText, PageBreak, Title

EXPECTED_SECTION_IDS = {
    "emi_payments#due-dates",
    "emi_payments#accepted-payment-modes",
    "emi_payments#bounce-charges",
    "emi_payments#late-payment-penalty",
    "emi_payments#auto-debit-nach-setup-and-changes",
    "emi_payments#statements-and-receipts",
    "kyc_account_updates#address-change",
    "kyc_account_updates#mobile-number-change",
    "kyc_account_updates#email-update",
    "kyc_account_updates#documents-accepted-for-kyc-re-verification",
    "kyc_account_updates#periodic-kyc-re-verification",
    "kyc_account_updates#how-to-submit-a-request",
    "prepayment_foreclosure#lock-in-period",
    "prepayment_foreclosure#part-prepayment-charges",
    "prepayment_foreclosure#foreclosure-charges",
    "prepayment_foreclosure#foreclosure-process",
    "prepayment_foreclosure#documents-required",
}


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    config = get_config()
    return Retriever(config.policies_dir, config.retrieval)


def _blank_pdf_bytes() -> bytes:
    """A minimal valid one-page PDF with no text at all, like an image-only scan."""
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 4\n0000000000 65535 f \n"
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n" + str(xref_at).encode() + b"\n%%EOF\n"
    return out


# ---------- Grouping unstructured elements into sections (synthetic elements) ----------


def test_title_starts_a_section_and_following_elements_become_its_body() -> None:
    sections = _group_into_sections(
        "emi",
        [
            Title("Bounce Charges"),
            NarrativeText("A charge of Rs. 500 applies."),
            ListItem("plus 18% GST"),
            Title("Late Payment Penalty"),
            NarrativeText("2% per month."),
        ],
    )
    assert [s.section_id for s in sections] == ["emi#bounce-charges", "emi#late-payment-penalty"]
    assert sections[0].text == "A charge of Rs. 500 applies.\nplus 18% GST"
    assert sections[0].heading == "Bounce Charges"
    assert sections[0].category == "emi"


def test_document_title_without_body_makes_no_section() -> None:
    sections = _group_into_sections(
        "emi", [Title("EMI Payments Policy"), Title("Due Dates"), NarrativeText("Fixed at disbursement.")]
    )
    assert [s.section_id for s in sections] == ["emi#due-dates"]


def test_title_tag_on_a_wrapped_bullet_tail_does_not_split_the_section() -> None:
    # unstructured's fast strategy tags fragments like these as Title.
    sections = _group_into_sections(
        "emi",
        [
            Title("Accepted Payment Modes"),
            ListItem("NACH auto-debit from the registered bank account, the"),
            Title("default and preferred mode."),
            ListItem("UPI payment via any UPI app."),
            Title("thereafter."),
        ],
    )
    assert [s.section_id for s in sections] == ["emi#accepted-payment-modes"]
    assert "default and preferred mode." in sections[0].text
    assert "thereafter." in sections[0].text


def test_bullet_line_tagged_as_title_does_not_start_a_section() -> None:
    sections = _group_into_sections(
        "kyc",
        [
            Title("Documents Required"),
            Title("(cid:127) Original loan agreement copy (if applicable)"),
            Title("• PAN card and address proof"),
            Title("- Salary slips"),
        ],
    )
    assert [s.section_id for s in sections] == ["kyc#documents-required"]
    assert "Original loan agreement copy" in sections[0].text
    assert "PAN card and address proof" in sections[0].text


def test_cid_artifacts_and_line_breaks_are_cleaned_from_text() -> None:
    sections = _group_into_sections(
        "emi",
        [Title("Modes"), NarrativeText("(cid:127) NACH mandates are\nset up at   origination.")],
    )
    assert sections[0].text == "NACH mandates are set up at origination."
    assert "cid" not in sections[0].text


def test_page_furniture_is_excluded() -> None:
    sections = _group_into_sections(
        "emi",
        [
            Header("IIFL Finance Confidential"),
            Title("Due Dates"),
            NarrativeText("Fixed at disbursement."),
            Footer("IIFL Finance | Page 1"),
            PageBreak(text=""),
            NarrativeText("Grace period is 3 days."),
        ],
    )
    assert len(sections) == 1
    assert sections[0].text == "Fixed at disbursement.\nGrace period is 3 days."


def test_body_continues_across_pages_under_the_same_heading() -> None:
    sections = _group_into_sections(
        "kyc", [Title("How to Submit"), ListItem("1. Log in."), PageBreak(text=""), ListItem("2. Upload a scan.")]
    )
    assert len(sections) == 1
    assert sections[0].text == "1. Log in.\n2. Upload a scan."


def test_repeated_headings_get_unique_ids() -> None:
    sections = _group_into_sections(
        "doc",
        [Title("Overview"), NarrativeText("First."), Title("Overview"), NarrativeText("Second.")],
    )
    assert [s.section_id for s in sections] == ["doc#overview", "doc#overview-2"]


def test_text_before_the_first_heading_is_kept_as_preamble() -> None:
    sections = _group_into_sections("doc", [NarrativeText("Intro text."), Title("Fees"), NarrativeText("Rs. 5.")])
    assert [s.section_id for s in sections] == ["doc#preamble", "doc#fees"]
    assert sections[0].text == "Intro text."


def test_heading_without_letters_or_digits_still_gets_a_usable_id() -> None:
    sections = _group_into_sections("doc", [Title("???"), NarrativeText("Body.")])
    assert [s.section_id for s in sections] == ["doc#section"]


def test_no_elements_gives_no_sections() -> None:
    assert _group_into_sections("doc", []) == []


# ---------- Loading the real sample PDFs ----------


def test_sample_pdfs_produce_exactly_the_expected_sections(retriever: Retriever) -> None:
    assert {s.section_id for s in retriever.sections} == EXPECTED_SECTION_IDS
    assert len(retriever.sections) == len(EXPECTED_SECTION_IDS)  # no duplicate ids


def test_filename_becomes_the_category(retriever: Retriever) -> None:
    for section in retriever.sections:
        assert section.section_id.startswith(f"{section.category}#")
    assert {s.category for s in retriever.sections} == {
        "emi_payments",
        "kyc_account_updates",
        "prepayment_foreclosure",
    }


def test_wrapped_bullets_and_page_breaks_lose_no_content(retriever: Retriever) -> None:
    text = {s.section_id: s.text for s in retriever.sections}
    # A hanging-indent bullet whose tail unstructured tags as a Title.
    assert "default and preferred mode." in text["emi_payments#accepted-payment-modes"]
    assert "2%\nthereafter." in text["prepayment_foreclosure#foreclosure-charges"]
    assert "GST at 18%" in text["prepayment_foreclosure#foreclosure-charges"]
    # A section that runs across a page break, including its closing paragraph.
    how_to = text["kyc_account_updates#how-to-submit-a-request"]
    assert "Track the status of any update request" in how_to
    assert "gazette notification" in how_to
    assert "If three consecutive NACH attempts fail" in text["emi_payments#auto-debit-nach-setup-and-changes"]


def test_no_extraction_artifacts_in_any_section(retriever: Retriever) -> None:
    for section in retriever.sections:
        assert "cid:" not in section.text
        assert section.text.strip() == section.text
        assert section.text


def test_new_pdf_dropped_into_the_folder_is_picked_up(tmp_path: Path) -> None:
    policies_dir = get_config().policies_dir
    shutil.copy(policies_dir / "emi_payments.pdf", tmp_path / "home_loan_faq.pdf")
    shutil.copy(policies_dir / "emi_payments.pdf", tmp_path / "UPPERCASE_EXT.PDF")
    (tmp_path / "notes.md").write_text("# not a policy\n## Ignored\nignored", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("also ignored", encoding="utf-8")

    sections = load_sections(tmp_path, get_config().retrieval.pdf_strategy)

    assert {s.category for s in sections} == {"home_loan_faq", "UPPERCASE_EXT"}
    assert "home_loan_faq#bounce-charges" in {s.section_id for s in sections}


def test_unstructured_telemetry_is_opted_out() -> None:
    # Importing src.retriever must switch off unstructured's outbound usage pings.
    assert os.environ.get("SCARF_NO_ANALYTICS")


def test_pdf_with_no_extractable_text_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "scanned_policy.pdf").write_bytes(_blank_pdf_bytes())
    with pytest.raises(ValueError, match="scanned_policy.pdf"):
        load_sections(tmp_path, get_config().retrieval.pdf_strategy)


# ---------- Ranking ----------


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
