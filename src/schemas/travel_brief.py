from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, confloat

from .retrieval import Citation


class NeighborhoodRecommendation(BaseModel):
    name: str
    why: str
    best_for: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class DayTripRecommendation(BaseModel):
    name: str
    duration: str
    why: str
    logistics: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class TransitStrategy(BaseModel):
    summary: str
    recommended_passes: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class CitedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounding: Literal["high", "medium", "low"] = "medium"
    confidence: confloat(ge=0, le=1) = 0.7


class TravelBrief(BaseModel):
    destination: str
    season_notes: list[str] = Field(default_factory=list)

    neighborhoods: list[NeighborhoodRecommendation] = Field(default_factory=list)
    must_do: list[str] = Field(default_factory=list)
    food_focus: list[str] = Field(default_factory=list)
    day_trips: list[DayTripRecommendation] = Field(default_factory=list)

    transit: Optional[TransitStrategy] = None
    safety_and_gotchas: list[str] = Field(default_factory=list)

    grounding: Literal["high", "medium", "low"] = "medium"
    overall_confidence: confloat(ge=0, le=1) = 0.7
    notes: list[str] = Field(default_factory=list)
