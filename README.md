# Constraint-Aware Retrieval Engine for AI Agents

## Problem
LLMs alone cannot reliably answer domain-specific questions when responses must satisfy explicit constraints (budget, preferences, safety, distance, etc.) and remain grounded in verifiable sources.

## Solution
This project implements a constraint-aware Retrieval-Augmented Generation (RAG) engine that:
- Retrieves semantically relevant knowledge using vector search
- Applies deterministic constraint filtering and ranking
- Injects only eligible evidence into the LLM context
- Produces structured, auditable outputs with citations

The system is designed as a reusable knowledge module for downstream AI agents.

## Key Capabilities
- Semantic retrieval with embeddings and vector search
- Constraint-aware filtering and ranking
- Token-budgeted context packing
- Citation-grounded generation
- Failure-aware responses and confidence scoring
- Evaluation harness for regression testing

## Architecture
Architecture: docs/architecture.md

## Example Use Cases
- Destination knowledge retrieval under user constraints
- Agent-readable travel briefs
- Enterprise knowledge systems with policy constraints

## Why This Matters
This mirrors how production AI systems are built inside large organizations — emphasizing reliability, control, and composability over prompt-only approaches.
