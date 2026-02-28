# retriever.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


class ContextAwareRetriever:
    """
    Retrieval policy:
    1) Retrieve top-k child chunks from Chroma
    2) Group results by parent_id
    3) Fetch parent text (for coherent expansion)
    4) Return context blocks that contain:
         - heading_path
         - parent_text
         - top child snippets under the parent
    """

    def __init__(self, child_index, parent_store):
        self.child_index = child_index
        self.parent_store = parent_store

    def retrieve(
        self,
        query: str,
        embedder,
        k: int = 12,
        where: Optional[Dict] = None,          # default None => all chunk types
        per_parent_children: int = 3,
    ) -> List[Dict[str, Any]]:
        res = self.child_index.query(query, embedder, k=k, where=where)

        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None] * len(ids)])[0]

        seen = set()
        dedup_ids, dedup_docs, dedup_metas, dedup_dists = [], [], [], []

        for cid, txt, m, dist in zip(ids, docs, metas, dists):
            pid = m.get("parent_id") or "UNKNOWN_PARENT"
            key = (pid, m.get("start_char"), m.get("end_char"), txt.strip()[:200])

            if key in seen:
                continue

            seen.add(key)
            dedup_ids.append(cid)
            dedup_docs.append(txt)
            dedup_metas.append(m)
            dedup_dists.append(dist)

        ids, docs, metas, dists = dedup_ids, dedup_docs, dedup_metas, dedup_dists

        # group hits by parent_id
        grouped: Dict[str, List[Tuple[str, str, Dict[str, Any], Any]]] = {}
        for cid, txt, m, dist in zip(ids, docs, metas, dists):
            pid = m.get("parent_id") or "UNKNOWN_PARENT"
            grouped.setdefault(pid, []).append((cid, txt, m, dist))

        # sort within each parent: constraints first, then facts, then others; tie-break by distance
        priority = {"constraint": 0, "fact": 1, "table": 2, "code": 3, "narrative": 4}

        for pid, items in grouped.items():
            items.sort(
                key=lambda x: (
                    priority.get(x[2].get("chunk_type"), 9),
                    x[3] if x[3] is not None else 1e9,
                )
            )

        assembled: List[Dict[str, Any]] = []
        for pid, items in grouped.items():
            if not items:
                continue
            # fetch parent text for context expansion
            parent = self.parent_store.get(pid)
            parent_text = parent["text"] if parent else ""

            # heading_path_str is stored in child metadata
            heading = items[0][2].get("heading_path_str", "")

            assembled.append(
                {
                    "parent_id": pid,
                    "heading_path": heading,
                    "parent_text": parent_text,
                    "children": [
                        {
                            "chunk_id": cid,
                            "chunk_type": m.get("chunk_type"),
                            "text": txt,
                            "distance": dist,
                            "meta": m,
                        }
                        for (cid, txt, m, dist) in items[:per_parent_children]
                    ],
                }
            )
        
        assembled.sort(key=lambda b: min((c["distance"] if c["distance"] is not None else 1e9) for c in b["children"]))
        return assembled