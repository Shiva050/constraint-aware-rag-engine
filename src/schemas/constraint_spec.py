from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, conint, confloat


class BudgetSpec(BaseModel):
    currency: str = Field(default="USD")
    max_total: Optional[confloat(ge=0)] = None
    max_per_night: Optional[confloat(ge=0)] = None


class DatesSpec(BaseModel):
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")


class TravelerSpec(BaseModel):
    party_size: conint(ge=1, le=12) = 1
    traveler_type: Literal["solo", "couple", "family", "friends", "business"] = "solo"


class MobilitySpec(BaseModel):
    walking_tolerance: Literal["low", "medium", "high"] = "medium"
    prefers_public_transit: bool = True


class PreferenceSpec(BaseModel):
    interests: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    style: Literal["budget", "mid", "premium"] = "mid"


class ConstraintSpec(BaseModel):
    destination: str
    dates: DatesSpec
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    traveler: TravelerSpec = Field(default_factory=TravelerSpec)
    mobility: MobilitySpec = Field(default_factory=MobilitySpec)
    preferences: PreferenceSpec = Field(default_factory=PreferenceSpec)

    max_daily_activities: conint(ge=1, le=12) = 6
    require_citations: bool = True

    # Retrieval constraints
    min_similarity: confloat(ge=0) = 0.20
    destination_required: bool = True
