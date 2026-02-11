from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, confloat, conint


class Citation(BaseModel):
    source_id: str
    title: str
    url: Optional[str] = None
    chunk_id: str
    snippet: str


class RetrievalChunk(BaseModel):
    citation: Citation
    text: str
    score: confloat(ge=0) = 0.0


class RetrievalResult(BaseModel):
    query: str
    top_k: conint(ge=1, le=50) = 10
    chunks: list[RetrievalChunk] = Field(default_factory=list)

    min_score_threshold: confloat(ge=0) = 0.2
    used_chunks: conint(ge=0, le=50) = 0
    retrieval_latency_ms: Optional[int] = None
