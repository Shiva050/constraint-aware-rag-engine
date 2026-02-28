# query_demo.py
from ..embeddings.embedder import STEmbedder
from ..embeddings.parent_store import ParentStore
from ..embeddings.vectorestone_chorma import ChromaChildIndex
from .retriever import ContextAwareRetriever


if __name__ == "__main__":
    embedder = STEmbedder()

    parent_store = ParentStore("parents.json")
    child_index = ChromaChildIndex(persist_dir=".chroma", collection_name="child_chunks")

    retriever = ContextAwareRetriever(child_index, parent_store)

    q = "What are the constraints or rules for visiting Big Sur?"
    ctx_blocks = retriever.retrieve(q, embedder, k=10, per_parent_children=3)

    for b in ctx_blocks:
        print("=" * 80)
        print("Heading:", b["heading_path"])
        print("Parent preview:", (b["parent_text"][:200] + "...") if b["parent_text"] else "<no parent>")
        print("\nTop child hits:")
        for c in b["children"]:
            preview = c["text"][:120].replace("\n", " ")
            print(f"- [{c['chunk_type']}] {preview}...")