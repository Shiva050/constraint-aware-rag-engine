# Constraint-Aware Retrieval Engine for AI Agents
# Constraint-Aware Retrieval Engine for AI Agents

This repository implements a constraint-aware Retrieval-Augmented Generation (RAG) engine designed to produce structured, auditable, and constraint-compliant outputs for downstream AI agents.

## Problem
LLMs alone cannot reliably answer domain-specific questions when responses must satisfy explicit constraints (budget, preferences, safety, distance, etc.) and remain grounded in verifiable sources.

## Solution Overview
The engine combines semantic retrieval with a deterministic constraint engine and preference-aware ranking to ensure only eligible evidence is injected into the LLM prompt. Key outcomes:
- Constraint filtering (hard constraints)
- Preference-aware ranking (soft constraints)
- Token-budgeted context packing
- Schema-driven generation with explicit citations

## Key Capabilities
- Semantic retrieval with embeddings and vector search
- Deterministic constraint filtering and preference ranking
- Token-budgeted context packing to control prompt size and cost
- Citation-grounded generation with structured output schemas
- Failure-aware responses and evaluation harness for regressions

## Architecture
See the full architecture writeup in [docs/architecture.md](docs/architecture.md). The high-level components are: ingestion → chunking → embeddings & index → retrieval → constraint engine → ranking → context packing → generator → evaluation.

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

## License
See the repository root for license terms (add license file if needed).
