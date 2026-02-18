"""
Create a new file: `embedder.py` that does:

- input: list of `ChildChunk`
- output: list of records `{chunk_id, embedding, text, meta...}`

use: `sentence-transformers` with `all-MiniLM-L6-v2`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import json
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class EmbeddedChunk:
    chunk_id: str
    parent_id: str
    text: str
    embedding: List[float]
    meta: Dict[str, Any]

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        # returns shape: (n, dim), dtype float32
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)


def embed_children_to_disk(
    *,
    children,                     # list[ChildChunk] from your chunker.py
    out_dir: str = "data/index",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> tuple[str, str]:
    """
    Writes:
      - {out_dir}/child_chunks.jsonl  (text + metadata, one per line)
      - {out_dir}/embeddings.npy      (float32 matrix aligned with jsonl order)
    Returns paths (jsonl_path, npy_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    jsonl_path = os.path.join(out_dir, "child_chunks.jsonl")
    npy_path = os.path.join(out_dir, "embeddings.npy")

    texts = [c.text for c in children]
    embedder = SentenceTransformerEmbedder(model_name=model_name)
    vecs = embedder.embed_texts(texts, batch_size=batch_size)  # (n, dim)

    # Write JSONL aligned with embeddings row order
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in children:
            rec = {
                "chunk_id": c.chunk_id,
                "parent_id": c.parent_id,
                "text": c.text,
                "meta": c.meta,
                "text_sha256": _sha(c.text),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    np.save(npy_path, vecs)
    return jsonl_path, npy_path