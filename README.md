# Constraint-Aware Retrieval Engine for AI Agents

```mermaid
flowchart LR
    A[Raw Sources]
    B[Ingestion]
    C[Chunking]
    D[Embeddings]
    E[Vector Index]
    F[Retrieval]
    G[Constraint Engine]
    H[Preference Ranking]
    I[Context Packing]
    J[Generator]
    K[Structured Output]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K

    subgraph Offline_Build[Offline Build (Indexing)]
      B
      C
      D
      E
    end

    subgraph Online_Query[Online Query (Retrieval + Constraints)]
      F
      G
      H
      I
      J
      K
    end
```
    B --> C[Structure-First Chunker\nParent–Child + Type Classification\n(constraint/fact/narrative/table/code)]
    C --> P[(Parent Store\nparents.json / SQLite\nparent_id → text + meta)]
    C --> E[Embeddings (Sentence-Transformers)]
    E --> V[(Vector Index: ChromaDB\nchild_chunks collection)]
  end

  subgraph Online_Query["Online Query (Retrieval + Constraints)"]
    Q[User Query] --> QE[Query Embedding\n(Sentence-Transformers)]
    QE --> R[Retriever\nTop-k child chunks]
    V --> R
    R --> X[Parent Expansion\nFetch by parent_id]
    P --> X

    X --> CF[Constraint Engine\n(Hard Filters)]
    CF --> PR[Preference Ranking\n(Soft Constraints)]
    PR --> CP[Context Packing\n(Token / Char Budget + Dedup)]
    CP --> G[Generator (LLM)]
    G --> O[Structured Output\nCitedAnswer / TravelBrief\n+ citations]
  end

  V -. serves .-> R
  P -. serves .-> X

## Core Design Principles

1) Structure-First Chunking

- Markdown-aware parsing
- Parent–child hierarchy
- Chunk typing:
  - constraint
  - fact
  - narrative
  - table
  - code

Child chunks are optimized for retrieval precision.
Parent chunks are used for coherent context expansion.

2) Deterministic Constraint Filtering

Hard constraints are applied before ranking or generation. Invalid candidates are removed prior to prompt construction.

3) Preference-Aware Ranking

Soft constraints influence ranking without excluding valid candidates.

4) Token-Budgeted Context Packing

Context is deduplicated, ordered by constraint priority, and truncated within a strict token/character budget.

5) Citation-Grounded Output

Each generated response can trace back to doc_id, parent_id, and chunk_id, enabling auditing and regression tests.

## Project Structure

```
src/
 ├── ingest/
 │   └── loader.py
 │
 ├── chunking/
 │   └── chunker.py
 │
 ├── embeddings/
 │   ├── embeddings_st.py
 │   └── index_pipeline.py
 │
 ├── retrieve/
 │   ├── retriever.py
 │   └── retrieverdemo.py
 │
 ├── constraints/
 │   └── constraint_engine.py
 │
 docs/
  └── architecture.md
```

## Quickstart
Prerequisites: Python 3.10+ and a POSIX-like shell (macOS/Linux).

1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the demo (example)

```bash
python3 -m src.retrieve.retrieverdemo
```

Notes:
- If you re-index data, the local Chroma index files in `.chroma/` will be updated.
- The project exposes core modules under `src/` such as the retriever and constraint engine: [src/retrieve/retriever.py](src/retrieve/retriever.py), [src/constraints/constraint_engine.py](src/constraints/constraint_engine.py).

## Regenerating the Index
If you update source documents or embeddings, rebuild the index via your ingestion pipeline (see `src/ingest/loader.py` and `src/embeddings/index_pipeline.py`). Rebuilding will update `.chroma/` (binary/DB artifacts for the local index).

## Recent changes (local)
- Commit `00d5fa1` updated the retrieval flow and demo runner and regenerated the local Chroma index files. Files changed include [src/retrieve/retriever.py](src/retrieve/retriever.py), [src/retrieve/retrieverdemo.py](src/retrieve/retrieverdemo.py), and the `.chroma/` index artifacts.

## Files of Interest
- [docs/architecture.md](docs/architecture.md) — detailed architecture and data flow
- [src/ingest/loader.py](src/ingest/loader.py) — ingestion pipeline
- [src/chunking/chunker.py](src/chunking/chunker.py) — chunking logic
- [src/embeddings/index_pipeline.py](src/embeddings/index_pipeline.py) — embedding + indexing
- [src/retrieve/retriever.py](src/retrieve/retriever.py) — retrieval logic
- [src/constraints/constraint_engine.py](src/constraints/constraint_engine.py) — constraint filtering

## Contributing
- Follow the code style in `src/` and add small, testable changes.
- When updating the index, avoid committing large binary index files unless intended for distributable demos.