"""Loads policy PDFs with `unstructured`, groups their elements into sections,
and scores queries against them using a BM25 + local-embedding hybrid. No
vector database: embeddings are computed once at startup and held in a numpy
array."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import RetrievalConfig

if TYPE_CHECKING:
    from unstructured.documents.elements import Element

# unstructured pings packages.unstructured.io when it is imported and again on every parse
# (library version, OS, Python version, CPU/GPU type, element-type counts; never document
# text). This service parses internal policy documents, so telemetry is opted out by default.
# It has to be set before anything imports unstructured, because the first ping happens at
# import time. Any non-empty value opts out.
os.environ.setdefault("SCARF_NO_ANALYTICS", "true")

# Element categories that are page furniture (running headers/footers, page
# numbers), not policy content. `unstructured` tags these by page position.
_PAGE_FURNITURE = {"Header", "Footer", "PageNumber", "PageBreak"}

# PDF fonts without a Unicode map make pdfminer emit "(cid:127)" for glyphs
# such as bullets. That is extraction noise, never policy text.
_CID_ARTIFACT = re.compile(r"\(cid:\d+\)")

# A line that starts with a bullet glyph (or a bare "(cid:NN)" bullet) is a
# list item, whatever category `unstructured` gave it.
_LIST_MARKER_START = re.compile(r"^\s*(?:\(cid:\d+\)|[•●▪◦‣·*\-])")


def _slugify(heading: str) -> str:
    """Turn a heading like 'Bounce Charges' into a slug like 'bounce-charges'."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25, stripping punctuation so a query
    like 'address?' still matches the indexed token 'address'."""
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Section:
    """One heading-delimited chunk of a policy PDF."""

    section_id: str
    category: str
    heading: str
    text: str


def _clean_text(text: str) -> str:
    """Drop '(cid:NN)' extraction artifacts and collapse all whitespace
    (including the line breaks PDF layout puts inside a paragraph)."""
    return " ".join(_CID_ARTIFACT.sub(" ", text).split())


def _is_section_heading(element: Element) -> bool:
    """True when a Title element is a genuine section heading.

    unstructured's `fast` strategy also tags some body lines as Title, for
    example the wrapped tail of a hanging-indent bullet ("thereafter.") or a
    short bullet item. So a Title only starts a new section if it also looks
    like a heading: not a list line, no sentence-ending punctuation, and not
    starting with a lowercase letter (which is a mid-sentence fragment).
    """
    if element.category != "Title":
        return False
    raw = element.text.strip()
    if _LIST_MARKER_START.match(raw):
        return False
    text = _clean_text(raw)
    if not text or text[-1] in ".,;":
        return False
    return not text[0].islower()


def _unique_id(base: str, used_ids: set[str]) -> str:
    """Make a section id unique within a file: 'x#overview', 'x#overview-2', ..."""
    candidate, n = base, 2
    while candidate in used_ids:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def _group_into_sections(category: str, elements: Sequence[Element]) -> list[Section]:
    """Group a PDF's elements into sections, one per detected heading.

    A heading Title starts a new section and every other element's text is
    appended to the section it falls under, in reading order. A heading with
    no body text (such as the document title) produces no section. Text that
    appears before any heading is kept, under the heading 'Preamble', so
    nothing in a PDF is silently dropped. Section ids look like
    'emi_payments#bounce-charges' (filename + slugified heading).
    """
    sections: list[Section] = []
    used_ids: set[str] = set()
    heading: Optional[str] = None
    lines: list[str] = []

    def flush() -> None:
        if not lines:
            return
        title = heading or "Preamble"
        section_id = _unique_id(f"{category}#{_slugify(title) or 'section'}", used_ids)
        used_ids.add(section_id)
        sections.append(Section(section_id, category, title, "\n".join(lines)))

    for element in elements:
        if element.category in _PAGE_FURNITURE:
            continue
        text = _clean_text(element.text)
        if not text:
            continue
        if _is_section_heading(element):
            flush()
            heading, lines = text, []
        else:
            lines.append(text)
    flush()
    return sections


def _partition_pdf(path: Path, strategy: str) -> list[Element]:
    """Parse one PDF into typed unstructured elements (Title, NarrativeText,
    ListItem, ...). `strategy="fast"` reads the PDF's text layer directly: no
    OCR or layout model, and no system installs needed."""
    # Imported here rather than at module top: unstructured takes several
    # seconds to import and only PDF loading needs it.
    from unstructured.partition.pdf import partition_pdf

    return partition_pdf(filename=str(path), strategy=strategy, languages=["eng"])


def _load_pdf_sections(path: Path, strategy: str) -> list[Section]:
    """Parse and section one policy PDF. The filename (without extension)
    becomes the category, e.g. 'emi_payments.pdf' -> 'emi_payments'."""
    sections = _group_into_sections(path.stem, _partition_pdf(path, strategy))
    if not sections:
        # Fail loudly: a silently empty file would make the agent escalate
        # every question about that policy with no hint why.
        raise ValueError(
            f"No text could be extracted from {path.name}. Is it a scanned "
            f"(image-only) PDF? The '{strategy}' strategy reads the PDF's text layer only."
        )
    return sections


def load_sections(policies_dir: Path, pdf_strategy: str) -> list[Section]:
    """Load and section every .pdf file in the policies directory."""
    sections: list[Section] = []
    for path in sorted(policies_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            sections.extend(_load_pdf_sections(path, pdf_strategy))
    return sections


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise a score array to the 0..1 range."""
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class Retriever:
    """Hybrid BM25 + embedding retriever over policy sections."""

    def __init__(self, policies_dir: Path, config: RetrievalConfig) -> None:
        self.config = config
        self.sections = load_sections(policies_dir, config.pdf_strategy)
        if not self.sections:
            raise ValueError(f"No policy sections found in {policies_dir} (expected .pdf files)")

        tokenized = [_tokenize(s.text) for s in self.sections]
        self.bm25 = BM25Okapi(tokenized)

        self.model = SentenceTransformer(config.embedding_model)
        self.embeddings = np.asarray(
            self.model.encode([s.text for s in self.sections], normalize_embeddings=True)
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Return the top_k sections for a query with hybrid scores."""
        k = top_k or self.config.top_k

        bm25_scores = np.asarray(self.bm25.get_scores(_tokenize(query)))
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        embedding_scores = self.embeddings @ query_embedding

        bm25_norm = _normalise(bm25_scores)
        embedding_norm = _normalise(embedding_scores)
        combined = (
            self.config.bm25_weight * bm25_norm + self.config.embedding_weight * embedding_norm
        )

        order = np.argsort(-combined)[:k]
        results = []
        for i in order:
            section = self.sections[int(i)]
            results.append(
                {
                    "section_id": section.section_id,
                    "category": section.category,
                    "text": section.text,
                    "score": float(combined[i]),
                    "bm25_score": float(bm25_norm[i]),
                    "embedding_score": float(embedding_norm[i]),
                }
            )
        return results
