# chunker.py
"""
Context-aware + constraint-aware chunker for a trip-planner / constraint-aware RAG system.

Key ideas implemented:
- Structure-first chunking (headings > blocks)
- Chunk types: constraint / fact / narrative / table / code
- Parent-child indexing:
  - child chunks (~220–450 tokens) are embedded + retrieved
  - parent chunks (~800–1500 tokens) are used for context expansion at answer-time
- Light overlap for non-constraint chunks (sentence-based)
- Rich metadata (heading_path, source, section ids, offsets, etc.)

Dependencies: standard library only.
Token estimation is heuristic (no tokenizer dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable
import re
import hashlib


# -----------------------------
# Data models
# -----------------------------

@dataclass(frozen=True)
class ChunkConfig:
    # Child chunk sizing (retrieval units)
    child_target_tokens: int = 360
    child_max_tokens: int = 550
    child_min_tokens: int = 140

    # Parent chunk sizing (context expansion units)
    parent_target_tokens: int = 1200
    parent_max_tokens: int = 1600
    parent_min_tokens: int = 500

    # Overlap (applied only to fact/narrative; not to constraint/table/code)
    overlap_sentences: int = 2

    # Heuristics
    max_heading_depth: int = 3
    allow_parent_across_h2: bool = False  # usually False for travel docs

@dataclass
class Block:
    """A parsed unit inside a section."""
    kind: str  # "paragraph" | "list" | "table" | "code"
    text: str
    start_char: int
    end_char: int

@dataclass
class Section:
    """A document section with a heading path and blocks."""
    heading_path: List[str]  # ["H1 title", "H2 title", ...]
    blocks: List[Block]
    start_char: int
    end_char: int

@dataclass
class ParentChunk:
    parent_id: str
    text: str
    meta: Dict 

@dataclass
class ChildChunk:
    chunk_id: str
    parent_id: str
    text: str
    chunk_type: str  # constraint | fact | narrative | table | code
    meta: Dict

# -----------------------------
# Utilities
# -----------------------------

_WORD_RE = re.compile(r"\w+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")    


def estimate_tokens(text: str) -> int:
    """
    Heuristic token estimator.
    Empirically: ~0.75 words per token for English prose (varies).
    We use: tokens ~= words / 0.75  => words * 1.33
    """
    words = len(_WORD_RE.findall(text))
    return int(words * 1.33) + 1


def split_sentences(text: str) -> List[str]:
    # Basic sentence splitting; good enough for overlap boundaries
    t = " ".join(text.split())
    if not t:
        return []
    return re.split(_SENT_RE, t)    


def stable_id(*parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:16]

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

_CONSTRAINT_PATTERNS = [
    r"\bmust\b",
    r"\brequired\b",
    r"\bneed to\b",
    r"\bshould not\b",
    r"\bmust not\b",
    r"\bnever\b",
    r"\bavoid\b",
    r"\bprohibited\b",
    r"\bnot allowed\b",
    r"\bclosed\b",
    r"\breservation\b",
    r"\bpermit\b",
    r"\bvisa\b",
    r"\bID\b",
    r"\bminimum\b",
    r"\bmaximum\b",
    r"\bcap\b",
    r"\bno (pets|drones|campfires|fires)\b",
    r"\bhours?\b",
    r"\bopen\b",
    r"\bseason(al)?\b",
    r"\bweather\b",
    r"\broad\b.*\bclosed\b",
]

_FACT_PATTERNS = [
    r"\b(?:\d{1,2}:\d{2})\b",                    # time like 08:30
    r"\b(?:am|pm)\b",                            # am/pm
    r"\b(?:mile|miles|mi|km|kilometer)\b",
    r"\b(?:hour|hours|hr|hrs)\b",
    r"\b(?:\$|usd|eur|₹)\b",
    r"\b(?:temperature|°F|°C)\b",
    r"\b(?:latitude|longitude)\b",
    r"\b(?:drive|driving)\s+time\b",
    r"\b(?:distance)\b",
    r"\b(?:address)\b",
]

_constraint_re = re.compile("|".join(_CONSTRAINT_PATTERNS), re.IGNORECASE)
_fact_re = re.compile("|".join(_FACT_PATTERNS), re.IGNORECASE)

def classify_text(text: str, block_kind: str) -> str:
    """
    Returns chunk_type:
      - table, code are preserved
      - constraint: if strong modal/rule signals
      - fact: if numeric/time/cost/distance signals
      - narrative: default
    """
    if block_kind == "table":
        return "table"
    if block_kind == "code":
        return "code"

    t = text.strip()
    if not t:
        return "narrative"

    if _constraint_re.search(t):
        return "constraint"
    if _fact_re.search(t):
        return "fact"
    return "narrative"
