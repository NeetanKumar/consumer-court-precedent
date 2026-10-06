"""Output schema for a precedent comparison answer.

Every statistic carries the tids that contributed to it (`contributing_tids`)
— this is the "every claim linked to its source judgment" requirement from
the plan, enforced structurally rather than by convention: there is no way
to construct a ComponentStats without naming which judgments back it.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ComponentStats(BaseModel):
    field_name: str
    sample_size: int  # total comparable judgments retrieved (the query's overall N)
    coverage: int  # of those, how many had a non-null value for this specific field
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    contributing_tids: list[int] = Field(default_factory=list)

    @property
    def coverage_fraction(self) -> float:
        return self.coverage / self.sample_size if self.sample_size else 0.0


class Citation(BaseModel):
    tid: int
    outcome: str
    amount_claimed: Optional[float]
    relief_components: dict
    fact_summary: str
    relevance_score: float


class PrecedentAnswer(BaseModel):
    query: str
    filters_applied: dict
    sample_size: int
    refused: bool
    refusal_reason: Optional[str] = None
    outcome_distribution: dict[str, int] = Field(default_factory=dict)
    # Of the judgments where extraction could tell, how many actually directed
    # money to be paid. None = not yet extracted (pre-award_made records).
    award_made_count: Optional[int] = None
    award_known_count: Optional[int] = None
    amount_claimed_stats: Optional[ComponentStats] = None
    relief_component_stats: dict[str, ComponentStats] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
