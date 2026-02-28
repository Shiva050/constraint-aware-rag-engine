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
from .patterns import _CONSTRAINT_PATTERNS, _FACT_PATTERNS
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


# -----------------------------
# Parsing: markdown-ish headings + blocks
# -----------------------------

_MD_H_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_CODE_FENCE_RE = re.compile(r"^```")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")  # naive markdown table row


def parse_markdownish(text: str, cfg: ChunkConfig) -> List[Section]:
    """
    Parses markdown-ish text into sections using # headings.
    Blocks:
      - code fences ```...```
      - tables (markdown pipe tables)
      - lists (lines starting with -, *, 1.)
      - paragraphs

    If no headings exist, produces one section with empty heading_path.
    """
    lines = text.splitlines(keepends=True)
    n = len(lines)
    idx = 0
    char_pos = 0

    sections: List[Section] = []
    heading_stack: List[Tuple[int, str]] = []  # (level, title)

    current_blocks: List[Block] = []
    section_start = 0
    section_heading_path: List[str] = []

    def flush_section(end_char: int):
        nonlocal current_blocks, section_start, section_heading_path
        if not current_blocks and not section_heading_path and not sections:
            # allow empty blocks only if we still want a single section later
            return
        if not current_blocks and sections:
            # avoid empty sections
            return
        sections.append(
            Section(
                heading_path=list(section_heading_path),
                blocks=current_blocks,
                start_char=section_start,
                end_char=end_char,
            )
        )
        current_blocks = []
        section_start = end_char

    def current_heading_path() -> List[str]:
        # include up to max depth
        trimmed = heading_stack[: cfg.max_heading_depth]
        return [t for _, t in trimmed]

    # helper to collect paragraph-like until blank line or structure boundary
    def collect_paragraph(start_i: int, start_char_pos: int) -> Tuple[str, int, int, int]:
        i = start_i
        buf = []
        local_char = start_char_pos
        start_char = start_char_pos
        while i < n:
            line = lines[i]
            # boundary conditions
            if _MD_H_RE.match(line) or _CODE_FENCE_RE.match(line) or line.strip() == "":
                break
            if line.lstrip().startswith(("-", "*")) or re.match(r"^\s*\d+\.\s+", line):
                # list starts
                break
            if _TABLE_ROW_RE.match(line):
                # table starts
                break
            buf.append(line)
            local_char += len(line)
            i += 1
        para = normalize_ws("".join(buf))
        return para, i, start_char, local_char

    def collect_list(start_i: int, start_char_pos: int) -> Tuple[str, int, int, int]:
        i = start_i
        buf = []
        local_char = start_char_pos
        start_char = start_char_pos
        while i < n:
            line = lines[i]
            if line.strip() == "":
                break
            if _MD_H_RE.match(line) or _CODE_FENCE_RE.match(line) or _TABLE_ROW_RE.match(line):
                break
            if not (line.lstrip().startswith(("-", "*")) or re.match(r"^\s*\d+\.\s+", line)):
                break
            buf.append(line)
            local_char += len(line)
            i += 1
        text_list = normalize_ws("".join(buf))
        return text_list, i, start_char, local_char

    def collect_table(start_i: int, start_char_pos: int) -> Tuple[str, int, int, int]:
        i = start_i
        buf = []
        local_char = start_char_pos
        start_char = start_char_pos
        while i < n:
            line = lines[i]
            if not _TABLE_ROW_RE.match(line) or line.strip() == "":
                break
            buf.append(line)
            local_char += len(line)
            i += 1
        table_text = "".join(buf).strip("\n")
        return table_text, i, start_char, local_char

    def collect_code(start_i: int, start_char_pos: int) -> Tuple[str, int, int, int]:
        i = start_i
        buf = []
        local_char = start_char_pos
        start_char = start_char_pos
        # include opening fence
        while i < n:
            line = lines[i]
            buf.append(line)
            local_char += len(line)
            i += 1
            if _CODE_FENCE_RE.match(line) and len(buf) > 1:
                # closing fence found (naive: any ``` after first)
                break
        code_text = "".join(buf).strip("\n")
        return code_text, i, start_char, local_char

    # If no headings, we still create one section; but we’ll parse blocks normally.
    section_heading_path = current_heading_path()

    while idx < n:
        line = lines[idx]
        m = _MD_H_RE.match(line)
        if m:
            # flush prior section
            flush_section(char_pos)
            level = len(m.group(1))
            title = normalize_ws(m.group(2))

            # update heading stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            section_heading_path = current_heading_path()
            idx += 1
            char_pos += len(line)
            continue

        if _CODE_FENCE_RE.match(line):
            code_text, new_i, start_c, end_c = collect_code(idx, char_pos)
            current_blocks.append(Block(kind="code", text=code_text, start_char=start_c, end_char=end_c))
            idx = new_i
            char_pos = end_c
            continue

        if _TABLE_ROW_RE.match(line):
            table_text, new_i, start_c, end_c = collect_table(idx, char_pos)
            current_blocks.append(Block(kind="table", text=table_text, start_char=start_c, end_char=end_c))
            idx = new_i
            char_pos = end_c
            continue

        if line.lstrip().startswith(("-", "*")) or re.match(r"^\s*\d+\.\s+", line):
            list_text, new_i, start_c, end_c = collect_list(idx, char_pos)
            current_blocks.append(Block(kind="list", text=list_text, start_char=start_c, end_char=end_c))
            idx = new_i
            char_pos = end_c
            continue

        if line.strip() == "":
            idx += 1
            char_pos += len(line)
            continue

        # paragraph
        para_text, new_i, start_c, end_c = collect_paragraph(idx, char_pos)
        if para_text:
            current_blocks.append(Block(kind="paragraph", text=para_text, start_char=start_c, end_char=end_c))
        idx = new_i
        char_pos = end_c

    # flush last
    flush_section(char_pos)

    # If headings never existed and we skipped flush logic early, ensure one section exists
    if not sections:
        sections = [
            Section(
                heading_path=[],
                blocks=current_blocks,
                start_char=0,
                end_char=len(text),
            )
        ]
    return sections

# -----------------------------
# Chunk building
# -----------------------------

def build_parent_chunks(
    doc_id: str,
    doc_title: str,
    source: str,
    sections: List[Section],
    cfg: ChunkConfig,
) -> List[ParentChunk]:
    """
    Build parent chunks by grouping blocks within a section (usually H2 scope).
    """
    parents: List[ParentChunk] = []

    for si, sec in enumerate(sections):
        # Build parent text progressively
        buf: List[str] = []
        buf_start = sec.start_char
        cur_start = buf_start

        def flush_parent(text: str, start_char: int, end_char: int):
            t = normalize_ws(text)
            if not t:
                return
            pid = stable_id(doc_id, "P", str(si), str(start_char), str(end_char), t[:64])
            parents.append(
                ParentChunk(
                    parent_id=pid,
                    text=t,
                    meta={
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "source": source,
                        "heading_path": sec.heading_path,
                        "section_index": si,
                        "start_char": start_char,
                        "end_char": end_char,
                        "est_tokens": estimate_tokens(t),
                    },
                )
            )

        cur_tokens = 0
        cur_end = sec.start_char

        for bi, b in enumerate(sec.blocks):
            piece = b.text.strip()
            if not piece:
                continue

            piece_tokens = estimate_tokens(piece)
            # If adding this block would exceed parent_max_tokens, flush and start new.
            if buf and (cur_tokens + piece_tokens) > cfg.parent_max_tokens:
                flush_parent("\n\n".join(buf), cur_start, cur_end)
                buf = []
                cur_tokens = 0
                cur_start = b.start_char

            buf.append(piece)
            cur_tokens += piece_tokens
            cur_end = b.end_char

            # If we reached target, flush (but only if above min)
            if cur_tokens >= cfg.parent_target_tokens and cur_tokens >= cfg.parent_min_tokens:
                flush_parent("\n\n".join(buf), cur_start, cur_end)
                buf = []
                cur_tokens = 0
                cur_start = cur_end

        # Flush remainder
        if buf:
            # Ensure minimum size; if too small and there's a previous parent in same section, merge
            t = "\n\n".join(buf)
            if estimate_tokens(t) < cfg.parent_min_tokens and parents and parents[-1].meta.get("section_index") == si:
                # merge into previous
                prev = parents.pop()
                merged_text = normalize_ws(prev.text + "\n\n" + t)
                merged = ParentChunk(
                    parent_id=prev.parent_id,
                    text=merged_text,
                    meta={**prev.meta, "end_char": cur_end, "est_tokens": estimate_tokens(merged_text)},
                )
                parents.append(merged)
            else:
                flush_parent(t, cur_start, cur_end)

    return parents


def _pack_child_units(
    units: List[Tuple[str, str, Dict]],
    cfg: ChunkConfig,
) -> List[Tuple[str, str, Dict]]:
    """
    Pack (chunk_type, text, meta) semantic units into child-sized chunks.
    Units are assumed to share the same parent/section context already.
    """
    out: List[Tuple[str, str, Dict]] = []

    cur_type: Optional[str] = None
    cur_buf: List[str] = []
    cur_meta: Optional[Dict] = None
    cur_tokens = 0

    def flush():
        nonlocal cur_type, cur_buf, cur_meta, cur_tokens
        if not cur_buf or not cur_meta or not cur_type:
            cur_type = None
            cur_buf = []
            cur_meta = None
            cur_tokens = 0
            return
        text = normalize_ws("\n\n".join(cur_buf))
        # enforce min size for non-constraint by merging forward if needed handled outside; here just emit
        out.append((cur_type, text, dict(cur_meta)))
        cur_type = None
        cur_buf = []
        cur_meta = None
        cur_tokens = 0

    for ttype, txt, meta in units:
        txt = txt.strip()
        if not txt:
            continue
        tks = estimate_tokens(txt)

        # Constraints should be atomic: flush before and emit as single chunk if possible
        if ttype == "constraint":
            flush()
            out.append((ttype, normalize_ws(txt), dict(meta)))
            continue

        # tables/code also should remain atomic
        if ttype in ("table", "code"):
            flush()
            out.append((ttype, txt, dict(meta)))
            continue

        # For fact/narrative, pack until max
        if cur_type is None:
            cur_type = ttype
            cur_meta = meta
            cur_buf = [txt]
            cur_tokens = tks
            continue

        # If type changes, flush to keep chunk type coherent
        if cur_type != ttype:
            flush()
            cur_type = ttype
            cur_meta = meta
            cur_buf = [txt]
            cur_tokens = tks
            continue

        # If adding would exceed max, flush and start new
        if (cur_tokens + tks) > cfg.child_max_tokens:
            flush()
            cur_type = ttype
            cur_meta = meta
            cur_buf = [txt]
            cur_tokens = tks
            continue

        cur_buf.append(txt)
        cur_tokens += tks

        # If reached target, flush
        if cur_tokens >= cfg.child_target_tokens and cur_tokens >= cfg.child_min_tokens:
            flush()

    flush()
    return out


def _apply_overlap_if_needed(
    chunks: List[Tuple[str, str, Dict]],
    cfg: ChunkConfig,
) -> List[Tuple[str, str, Dict]]:
    """
    Adds sentence-based overlap for fact/narrative chunks only.
    Does not overlap constraint/table/code.
    """
    if cfg.overlap_sentences <= 0:
        return chunks

    out: List[Tuple[str, str, Dict]] = []
    prev_tail: List[str] = []

    for ttype, txt, meta in chunks:
        if ttype in ("constraint", "table", "code"):
            # reset overlap chain at hard boundaries (reduces duplication)
            prev_tail = []
            out.append((ttype, txt, meta))
            continue

        sents = split_sentences(txt)
        if prev_tail:
            # prepend previous tail sentences
            merged = normalize_ws(" ".join(prev_tail + sents))
        else:
            merged = txt

        out.append((ttype, merged, meta))

        # compute tail for next overlap
        tail_count = clamp(cfg.overlap_sentences, 0, len(sents))
        prev_tail = sents[-tail_count:] if tail_count > 0 else []

    return out


def build_child_chunks(
    doc_id: str,
    doc_title: str,
    source: str,
    sections: List[Section],
    parents: List[ParentChunk],
    cfg: ChunkConfig,
) -> List[ChildChunk]:
    """
    Create child chunks from blocks, classified into constraint/fact/narrative/table/code.
    Each child chunk is linked to a parent chunk by section membership and char offsets.
    """
    # Index parents by section_index and ranges
    parents_by_section: Dict[int, List[ParentChunk]] = {}
    for p in parents:
        si = int(p.meta["section_index"])
        parents_by_section.setdefault(si, []).append(p)
    # Ensure sorted by start_char for lookup
    for si in parents_by_section:
        parents_by_section[si].sort(key=lambda x: x.meta["start_char"])

    children: List[ChildChunk] = []

    for si, sec in enumerate(sections):
        # Build semantic units from blocks.
        units: List[Tuple[str, str, Dict]] = []
        for bi, b in enumerate(sec.blocks):
            raw = b.text.strip()
            if not raw:
                continue
            ctype = classify_text(raw, b.kind)
            unit_meta = {
                "doc_id": doc_id,
                "doc_title": doc_title,
                "source": source,
                "heading_path": sec.heading_path,
                "section_index": si,
                "block_index": bi,
                "block_kind": b.kind,
                "start_char": b.start_char,
                "end_char": b.end_char,
            }

            # For paragraphs, optionally split into smaller semantic units by sentence groups
            # but keep constraints atomic.
            if b.kind == "paragraph" and ctype in ("fact", "narrative"):
                sents = split_sentences(raw)
                if len(sents) <= 3:
                    units.append((ctype, raw, unit_meta))
                else:
                    # group sentences into units aiming ~200-300 tokens each
                    buf: List[str] = []
                    buf_tokens = 0
                    for s in sents:
                        st = s.strip()
                        if not st:
                            continue
                        st_tokens = estimate_tokens(st)
                        if buf and (buf_tokens + st_tokens) > 320:
                            units.append((ctype, " ".join(buf), unit_meta))
                            buf = []
                            buf_tokens = 0
                        buf.append(st)
                        buf_tokens += st_tokens
                    if buf:
                        units.append((ctype, " ".join(buf), unit_meta))
            else:
                units.append((ctype, raw, unit_meta))

        # Pack units into child chunks
        packed = _pack_child_units(units, cfg)
        packed = _apply_overlap_if_needed(packed, cfg)

        # Assign parent_id for each child based on its char range
        section_parents = parents_by_section.get(si, [])
        for ci, (ctype, txt, meta) in enumerate(packed):
            start_c = meta["start_char"]
            end_c = meta["end_char"]

            parent_id = _find_parent_for_range(section_parents, start_c, end_c)
            if parent_id is None:
                # Fallback: use closest parent in section
                parent_id = section_parents[0].parent_id if section_parents else stable_id(doc_id, "P", str(si))

            c_id = stable_id(doc_id, "C", str(si), str(ci), ctype, txt[:64])
            full_meta = dict(meta)
            full_meta.update(
                {
                    "chunk_type": ctype,
                    "parent_id": parent_id,
                    "est_tokens": estimate_tokens(txt),
                }
            )
            children.append(
                ChildChunk(
                    chunk_id=c_id,
                    parent_id=parent_id,
                    text=txt,
                    chunk_type=ctype,
                    meta=full_meta,
                )
            )

    return children


def _find_parent_for_range(parents: List[ParentChunk], start_char: int, end_char: int) -> Optional[str]:
    """
    Find the parent that fully covers the range; otherwise nearest covering.
    """
    if not parents:
        return None

    # exact cover
    for p in parents:
        ps = int(p.meta["start_char"])
        pe = int(p.meta["end_char"])
        if start_char >= ps and end_char <= pe:
            return p.parent_id

    # nearest by start distance
    best = None
    best_dist = 10**18
    for p in parents:
        ps = int(p.meta["start_char"])
        pe = int(p.meta["end_char"])
        # distance to interval
        dist = 0
        if end_char < ps:
            dist = ps - end_char
        elif start_char > pe:
            dist = start_char - pe
        else:
            dist = 0
        if dist < best_dist:
            best_dist = dist
            best = p.parent_id
    return best

# -----------------------------
# Public API
# -----------------------------

def chunk_document(
    *,
    doc_id: str,
    doc_title: str,
    source: str,
    text: str,
    cfg: Optional[ChunkConfig] = None,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    """
    End-to-end chunking for one document.
    Returns (parents, children).
    """
    cfg = cfg or ChunkConfig()
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    sections = parse_markdownish(cleaned, cfg)
    parents = build_parent_chunks(doc_id, doc_title, source, sections, cfg)
    children = build_child_chunks(doc_id, doc_title, source, sections, parents, cfg)
    return parents, children


def serialize_chunks(
    parents: List[ParentChunk],
    children: List[ChildChunk],
) -> Dict[str, List[Dict]]:
    """
    Converts chunks to JSON-serializable dicts.
    """
    return {
        "parents": [{"parent_id": p.parent_id, "text": p.text, "meta": p.meta} for p in parents],
        "children": [{"chunk_id": c.chunk_id, "parent_id": c.parent_id, "text": c.text, "chunk_type": c.chunk_type, "meta": c.meta} for c in children],
    }

# -----------------------------
# Example usage (manual testing)
# -----------------------------
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

    parents, children = chunk_document(
        doc_id="doc_001",
        doc_title="Trip Notes",
        source="user_notes",
        text=sample,
    )

    print(f"Parents: {len(parents)}")
    print(f"Children: {len(children)}")
    for c in children:
        hp = " > ".join(c.meta.get("heading_path", []))
        preview = c.text[:90] + ("..." if len(c.text) > 90 else "")
        print(f"{c.chunk_type:10} | {hp:30} | {preview}")