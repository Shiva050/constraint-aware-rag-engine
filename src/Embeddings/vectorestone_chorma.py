# vectorstore_chroma.py
from __future__ import annotations
from typing import Any, Dict, Optional
import chromadb


def _primitive_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma expects metadata values to be primitive types.
    Your chunker stores heading_path as List[str], so we convert it.
    """
    out = dict(meta)

    hp = out.get("heading_path")
    out["heading_path_str"] = " > ".join(hp) if isinstance(hp, list) else (hp or "")
    out.pop("heading_path", None)

    cleaned: Dict[str, Any] = {}
    for k, v in out.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned


class ChromaChildIndex:
    """
    Stores ONLY child chunks in Chroma.
    Parent chunks are stored separately (ParentStore) because we fetch them by id.
    """

    def __init__(self, persist_dir: str = ".chroma", collection_name: str = "child_chunks"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection(name=collection_name)

    def upsert_children(self, children, embedder) -> None:
        ids = [c.chunk_id for c in children]
        docs = [c.text for c in children]

        metas = []
        for c in children:
            # ensure chunk_type + parent_id are present in metadata for filtering + assembly
            m = {**c.meta, "chunk_type": c.chunk_type, "parent_id": c.parent_id}
            metas.append(_primitive_meta(m))

        embs = embedder.embed_texts(docs)

        self.col.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs,
        )

    def query(self, query_text: str, embedder, k: int = 10, where: Optional[Dict] = None):
        q = embedder.embed_query(query_text)
        return self.col.query(
            query_embeddings=[q],
            n_results=k,
            where=where or {},
        )