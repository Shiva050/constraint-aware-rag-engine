from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.schemas.constraint_spec import ConstraintSpec
from src.schemas.retrieval import RetrievalResult, RetrievalChunk


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    t = _norm(text)
    for n in needles:
        n2 = _norm(n)
        if n2 and n2 in t:
            return True
    return False


@dataclass
class ConstraintReport:
    removed_by_destination: int = 0
    removed_by_avoid: int = 0
    removed_by_low_score: int = 0


class ConstraintEngine:
    """
    Hard constraints first, then soft preference ranking.
    """

    def apply_hard_constraints(self, result: RetrievalResult, spec: ConstraintSpec) -> tuple[RetrievalResult, ConstraintReport]:
        report = ConstraintReport()
        kept: list[RetrievalChunk] = []

        dest = spec.destination.strip()
        avoid = [a for a in spec.preferences.avoid if a.strip()]

        for ch in result.chunks:
            # similarity threshold already applied in retriever; keep a second guard
            if ch.score < spec.min_similarity:
                report.removed_by_low_score += 1
                continue

            if spec.destination_required and dest:
                # Ensure destination is mentioned either in title or chunk text
                hay = f"{ch.citation.title}\n{ch.text}"
                if _norm(dest) not in _norm(hay):
                    report.removed_by_destination += 1
                    continue

            if avoid:
                # Avoid keywords: deterministic removal if chunk strongly matches avoid
                if _contains_any(ch.text, avoid) or _contains_any(ch.citation.title, avoid):
                    report.removed_by_avoid += 1
                    continue

            kept.append(ch)

        filtered = result.model_copy()
        filtered.chunks = kept
        filtered.used_chunks = len(kept)
        return filtered, report

    def rank_soft_preferences(self, result: RetrievalResult, spec: ConstraintSpec) -> RetrievalResult:
        interests = [i for i in spec.preferences.interests if i.strip()]
        walking = spec.mobility.walking_tolerance

        def soft_boost(ch: RetrievalChunk) -> float:
            boost = 0.0
            txt = _norm(ch.text)

            # interests boosts
            for i in interests:
                if _norm(i) in txt:
                    boost += 0.04

            # mobility heuristic
            if walking == "high":
                if any(k in txt for k in ["walkable", "walking", "on foot", "stroll"]):
                    boost += 0.03
            if spec.mobility.prefers_public_transit:
                if any(k in txt for k in ["subway", "metro", "train", "transit", "station"]):
                    boost += 0.02

            return boost

        ranked = sorted(result.chunks, key=lambda c: (c.score + soft_boost(c)), reverse=True)
        out = result.model_copy()
        out.chunks = ranked
        out.used_chunks = len(ranked)
        return out