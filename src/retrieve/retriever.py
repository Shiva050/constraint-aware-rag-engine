from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.index.faiss_index import FaissVectorIndex
from src.schemas.retrieval import Citation, RetrievalChunk, RetrievalResult


class Retriever:
    def __init__(self, model_name: str, index_dir: str):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.index = FaissVectorIndex.load(index_dir)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        return vec[0]

    def retrieve(self, query: str, top_k: int = 10, min_score_threshold: float = 0.2) -> RetrievalResult:
        t0 = time.time()
        qv = self.embed_query(query)
        scores, idxs = self.index.search(qv, top_k)

        chunks: list[RetrievalChunk] = []
        for score, idx in zip(scores.tolist(), idxs.tolist()):
            if idx < 0:
                continue
            if score < min_score_threshold:
                continue
            chunk_id = self.index.id_map[idx]
            c = self.index.chunk_store[chunk_id]
            snippet = c["text"][:240].replace("\n", " ").strip()

            citation = Citation(
                source_id=c["doc_id"],
                title=c["title"],
                url=c.get("url"),
                chunk_id=c["chunk_id"],
                snippet=snippet,
            )
            chunks.append(RetrievalChunk(citation=citation, text=c["text"], score=float(score)))

        latency_ms = int((time.time() - t0) * 1000)
        return RetrievalResult(
            query=query,
            top_k=top_k,
            chunks=chunks,
            min_score_threshold=min_score_threshold,
            used_chunks=len(chunks),
            retrieval_latency_ms=latency_ms,
        )