# Architecture — Constraint-Aware RAG Engine

## 1. Goal
Build a reusable, production-style Retrieval-Augmented Generation (RAG) engine that:
- Retrieves relevant knowledge using embeddings + vector search
- Applies deterministic constraint filtering and preference-aware ranking
- Packs only eligible evidence into an LLM context within a token budget
- Generates structured outputs with citations suitable for downstream AI agents

Primary output is an agent-ready `TravelBrief` (and optionally `CitedAnswer`) that can be used later by a Trip Planner agent.

## 2. Non-Goals (v1)
- Real flight/hotel booking or payments
- Real-time pricing guarantees
- Multi-tenant auth, billing, or enterprise governance
- Large-scale distributed indexing (local index is sufficient for portfolio)

## 3. Key Requirements
### Functional
- Semantic retrieval over a curated knowledge base
- Explicit citations (source, chunk_id, snippet)
- Constraint-aware filtering (hard constraints)
- Preference-aware ranking (soft constraints)
- Structured output schemas (Pydantic models)

### Quality
- Deterministic behavior for hard constraints
- Failure-aware responses (no-results, low-confidence)
- Reproducible ingestion/index builds
- Evaluation harness to prevent regressions

## 4. System Components

### 4.1 Data & Ingestion
**Responsibility:** Convert raw sources into normalized `Document` objects with metadata.
- Inputs: Markdown/HTML/JSON travel guides, curated notes
- Outputs: `Document(id, title, text, metadata)`

### 4.2 Chunking
**Responsibility:** Split documents into retrievable units while preserving metadata.
- Strategy: recursive chunking with overlap (configurable)
- Outputs: `Chunk(doc_id, chunk_id, text, metadata)`

### 4.3 Embeddings + Vector Index
**Responsibility:** Create embeddings and index for similarity search.
- Embedding model: local sentence-transformers / BGE / Nomic
- Index: FAISS (or Chroma) storing vectors + chunk metadata + raw text

### 4.4 Retrieval
**Responsibility:** Retrieve topK candidate chunks for a query.
- Input: `query: str`
- Output: `RetrievalResult(query, chunks=[...], scores=[...])`

### 4.5 Constraint Engine (Hard Constraints)
**Responsibility:** Apply deterministic filters based on `ConstraintSpec`.
Examples:
- Must include only chunks relevant to the destination/region
- Exclude evidence that violates explicit user constraints (e.g., avoid nightlife, avoid long walking)
- Enforce minimum retrieval confidence threshold

Output:
- Filtered `RetrievalResult` (may be empty)

### 4.6 Preference Ranking (Soft Constraints)
**Responsibility:** Re-rank remaining chunks to emphasize user preferences.
Examples:
- Boost chunks about photography spots if `interests` include photography
- Boost walkable neighborhoods if `walking_tolerance` is high
- Penalize content that matches `avoid` keywords

Output:
- Ranked list of chunks used for context packing

### 4.7 Context Packing (Token Budgeting)
**Responsibility:** Select the best-ranked evidence that fits within a token limit.
- Input: ranked chunks + token budget + packing policy
- Output:
  - `context_text` (assembled evidence)
  - `selected_citations`

This prevents prompt bloat and demonstrates cost-aware engineering.

### 4.8 Generator
**Responsibility:** Produce schema-driven output grounded in selected evidence.
- Input: query + `context_text` + `ConstraintSpec`
- Output: `TravelBrief` (or `CitedAnswer`)

Generator rules:
- Must cite evidence for key claims
- Must return “insufficient evidence” when context is weak
- Must follow output schema strictly

### 4.9 Evaluation Harness
**Responsibility:** Catch regressions and measure retrieval/generation quality.
- Scenario file defines:
  - query
  - constraints
  - expected properties (not exact text), e.g.:
    - includes 3 neighborhoods
    - all neighborhoods have citations
    - no nightlife recommendations if avoid nightlife is set
- Metrics:
  - constraint satisfaction rate
  - citation coverage
  - retrieval hit rate (proxy)
  - output schema validity

## 5. End-to-End Data Flow

### 5.1 Offline (Build Index)
1. Load raw sources → `Document`
2. Chunk documents → `Chunk[]`
3. Embed chunks → vectors
4. Persist vector index + metadata store

### 5.2 Online (Answer Query)
Input:
- `query: str`
- `ConstraintSpec`

Steps:
1. Retrieve topK chunks by embedding similarity → `RetrievalResult`
2. Apply hard constraint filters → `RetrievalResult_filtered`
3. Apply preference ranking → `RetrievalResult_ranked`
4. Pack context into token budget → `context_text + citations`
5. Generate structured output → `TravelBrief` / `CitedAnswer`

## 6. Failure Modes & Mitigations

### 6.1 No retrieval hits
- Return: structured “no evidence found” response
- Suggest: user clarifications or expanding sources

### 6.2 Low-confidence retrieval
- Use thresholding on similarity scores
- Downgrade `grounding` to `low`
- Avoid strong claims; request clarification

### 6.3 Constraints filter out everything
- Return: “constraints too strict” + show which constraint caused elimination (high-level)
- Suggest: relaxing constraints or adding sources

### 6.4 Token budget exceeded
- Context packing enforces strict budget
- If still too large:
  - reduce k
  - reduce chunk size
  - increase summarization (optional v2)

### 6.5 Hallucinated claims
- Generation prompt instructs: “Use only provided evidence”
- Require citations for key sections
- Add evaluation checks for citation coverage

## 7. Interfaces (Module Contracts)

### 7.1 Retrieval
- `retrieve(query: str, top_k: int) -> RetrievalResult`

### 7.2 Constraint Filtering
- `apply_constraints(result: RetrievalResult, spec: ConstraintSpec) -> RetrievalResult`

### 7.3 Ranking
- `rank_chunks(result: RetrievalResult, spec: ConstraintSpec) -> RetrievalResult`

### 7.4 Context Packing
- `pack_context(result: RetrievalResult, token_budget: int) -> tuple[str, list[Citation]]`

### 7.5 Generation
- `generate_travel_brief(query: str, spec: ConstraintSpec, context: str, citations: list[Citation]) -> TravelBrief`

### 7.6 Evaluation
- `run_eval(scenarios_path: str) -> EvalReport`

## 8. Iteration Plan (v2+)
- Add reranking using a cross-encoder or LLM-based reranker
- Add query decomposition (multi-query retrieval)
- Add destination-level metadata constraints (geo, seasonality)
- Add summarization compression for long contexts
- Convert into a tool callable by a Trip Planner agent (MCP/tool wrapper)