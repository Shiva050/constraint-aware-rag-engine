from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.ingest.loaders import Document


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    title: str
    text: str
    url: str | None
    metadata: dict[str, Any]


def _clean(text: str) -> str:
    # Light normalization
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_document(
    doc: Document,
    chunk_size_chars: int = 900,
    chunk_overlap_chars: int = 150,
) -> list[Chunk]:
    text = _clean(doc.text)
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(len(text), start + chunk_size_chars)
        chunk_text = text[start:end]

        # Try to end on sentence boundary if possible
        if end < len(text):
            m = re.search(r"(.{0,200}[.!?])\s", chunk_text[::-1])
            # Reverse search is messy; keep simple: extend to next newline if close
            nl = text.find("\n", end)
            if nl != -1 and nl - end < 120:
                end = nl
                chunk_text = text[start:end]

        chunk_text = chunk_text.strip()
        if chunk_text:
            chunk_id = f"{doc.doc_id}::chunk::{idx}"
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_id=chunk_id,
                    title=doc.title,
                    text=chunk_text,
                    url=doc.url,
                    metadata={
                        **(doc.metadata or {}),
                        "source_path": doc.source_path,
                        "chunk_index": idx,
                    },
                )
            )
            idx += 1

        if end >= len(text):
            break
        start = max(0, end - chunk_overlap_chars)

    return chunks