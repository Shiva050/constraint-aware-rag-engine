# parent_store.py
from __future__ import annotations
import json
from typing import Any, Dict, Optional


class ParentStore:
    """
    Minimal persistent KV store for parent chunks.

    Storage format:
      {
        "<parent_id>": {"text": "...", "meta": {...}},
        ...
      }
    """

    def __init__(self, path: str = "parents.json"):
        self.path = path
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._db: Dict[str, Dict[str, Any]] = json.load(f)
        except FileNotFoundError:
            self._db = {}

    def upsert_parents(self, parents) -> None:
        for p in parents:
            self._db[p.parent_id] = {"text": p.text, "meta": p.meta}

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._db, f)

    def get(self, parent_id: str) -> Optional[Dict[str, Any]]:
        return self._db.get(parent_id)