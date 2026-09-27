"""Loads policy docs, splits them into '##' sections, and scores queries
against them using a BM25 + local-embedding hybrid. No vector database:
embeddings are computed once at startup and held in a numpy array."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import RetrievalConfig


def _slugify(heading: str) -> str:
    """Turn a '## Some Heading' into a slug like 'some-heading'."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25, stripping punctuation so a query
    like 'address?' still matches the indexed token 'address'."""
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Section:
    """One '##'-delimited chunk of a policy document."""

    section_id: str
    category: str
    heading: str
    text: str


def _split_into_sections(path: Path) -> list[Section]:
    """Split a markdown policy file into sections by '##' headings.

    The filename (without extension) becomes the category, e.g. a section
    under 'emi_payments.md' titled 'Bounce Charges' gets the id
    'emi_payments#bounce-charges'. Content before the first '##' heading
    (the H1 title and intro paragraph) is document framing, not a citable
    fact-bearing section, so it is not indexed.
    """
    category = path.stem
    lines = path.read_text(encoding="utf-8").splitlines()

    sections: list[Section] = []
    current_heading: Optional[str] = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_heading is None:
            return
        text = "\n".join(current_lines).strip()
        if text:
            section_id = f"{category}#{_slugify(current_heading)}"
            sections.append(Section(section_id, category, current_heading, text))

    for line in lines:
        if line.startswith("## "):
            flush()
            current_heading = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            continue  # document title, not a section
        elif current_heading is not None:
            current_lines.append(line)
    flush()
    return sections


def load_sections(policies_dir: Path) -> list[Section]:
    """Load and chunk every .md file in the policies directory."""
    sections: list[Section] = []
    for path in sorted(policies_dir.glob("*.md")):
        sections.extend(_split_into_sections(path))
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
        self.sections = load_sections(policies_dir)
        if not self.sections:
            raise ValueError(f"No policy sections found in {policies_dir}")

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
