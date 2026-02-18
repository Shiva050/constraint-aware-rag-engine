from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

import faiss
import numpy as np

from src.chunking.chunker import Chunk


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class FaissVectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        # Inner product + normalized vectors ≈ cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.id_map: list[str] = []
        self.chunk_store: dict[str, dict[str, Any]] = {}

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

        for c in chunks:
            self.id_map.append(c.chunk_id)
            self.chunk_store[c.chunk_id] = asdict(c)

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        query_vec = query_vec.astype(np.float32)
        faiss.normalize_L2(query_vec)
        scores, idxs = self.index.search(query_vec, top_k)
        return scores[0], idxs[0]

    def save(self, out_dir: str) -> None:
        _ensure_dir(out_dir)
        faiss.write_index(self.index, os.path.join(out_dir, "index.faiss"))
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "count": len(self.id_map)}, f, indent=2)
        with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for cid in self.id_map:
                f.write(json.dumps(self.chunk_store[cid], ensure_ascii=False) + "\n")

    @staticmethod
    def load(in_dir: str) -> "FaissVectorIndex":
        with open(os.path.join(in_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        dim = int(meta["dim"])
        inst = FaissVectorIndex(dim)
        inst.index = faiss.read_index(os.path.join(in_dir, "index.faiss"))

        id_map: list[str] = []
        chunk_store: dict[str, dict[str, Any]] = {}

        with open(os.path.join(in_dir, "chunks.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cid = obj["chunk_id"]
                id_map.append(cid)
                chunk_store[cid] = obj

        inst.id_map = id_map
        inst.chunk_store = chunk_store
        return inst