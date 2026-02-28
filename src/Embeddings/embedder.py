from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer


class STEmbedder:
    """
    Sentence-Transformers embedder.

    - embed_texts: used when indexing documents/chunks
    - embed_query: used when retrieving (same embedding space)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # normalize_embeddings=True improves cosine similarity behavior
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()