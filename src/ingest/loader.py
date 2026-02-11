from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source_path: str
    url: Optional[str] = None
    metadata: dict[str, Any] | None = None


def load_markdown(path: str) -> Document:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    title = os.path.splitext(os.path.basename(path))[0]
    return Document(
        doc_id=f"doc::{os.path.basename(path)}",
        title=title,
        text=text,
        source_path=path,
        url=None,
        metadata={"format": "markdown"},
    )


def load_json(path: str) -> Document:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Expect either {title, text} or list of sections
    title = obj.get("title") or os.path.splitext(os.path.basename(path))[0]
    if "text" in obj and isinstance(obj["text"], str):
        text = obj["text"]
    elif "sections" in obj and isinstance(obj["sections"], list):
        parts = []
        for s in obj["sections"]:
            h = s.get("heading", "")
            t = s.get("text", "")
            parts.append(f"{h}\n{t}".strip())
        text = "\n\n".join([p for p in parts if p])
    else:
        raise ValueError("JSON format not supported. Use {title,text} or {title,sections:[...] }")

    return Document(
        doc_id=f"doc::{os.path.basename(path)}",
        title=title,
        text=text.strip(),
        source_path=path,
        url=obj.get("url"),
        metadata={"format": "json"},
    )


def load_documents_from_dir(raw_dir: str) -> list[Document]:
    docs: list[Document] = []
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            path = os.path.join(root, fn)
            if fn.lower().endswith(".md"):
                docs.append(load_markdown(path))
            elif fn.lower().endswith(".json"):
                docs.append(load_json(path))
    if not docs:
        raise FileNotFoundError(f"No .md or .json files found under: {raw_dir}")
    return docs
