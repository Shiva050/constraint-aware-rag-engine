from ..chunking.chunker import chunk_document
from .embedder import STEmbedder
from .parent_store import ParentStore
from .vectorestone_chorma import ChromaChildIndex


def index_one_document(doc_id: str, doc_title: str, source: str, text: str):
    # 1) chunk
    parents, children = chunk_document(
        doc_id=doc_id,
        doc_title=doc_title,
        source=source,
        text=text,
    )

    # 2) embedder
    embedder = STEmbedder()

    # 3) store parents
    parent_store = ParentStore("parents.json")
    parent_store.upsert_parents(parents)

    # 4) store children in Chroma
    child_index = ChromaChildIndex(persist_dir=".chroma", collection_name="child_chunks")
    child_index.upsert_children(children, embedder)

    print(f"Indexed doc_id={doc_id}: parents={len(parents)}, children={len(children)}")


if __name__ == "__main__":
    sample = """
# California Road Trip

## Big Sur
Big Sur is known for coastal views and scenic pullouts. Plan to stop at Bixby Bridge and Pfeiffer Beach.

- Recommended sunrise spot: Garrapata State Park
- Drive time from Monterey: ~45 minutes

Important: You must arrive before sunset; the park closes at 8pm.
Reservations required for some campgrounds.

| Place | Note |
|---|---|
| Bixby Bridge | Busy mid-day |
| Pfeiffer Beach | Limited parking |
"""
    index_one_document("doc_001", "Trip Notes", "user_notes", sample)