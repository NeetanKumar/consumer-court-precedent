"""The structured record every judgment is reduced to.

This schema is the point of the whole project (see package docstring in
__init__.py): a compensation-range answer needs these typed fields
aggregated across judgments, not semantic similarity over raw text.

`confidence` is the model's own self-assessment, used by the tiered
extractor to decide whether to escalate from Haiku to Sonnet — see
extractor.py.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ReliefComponents(BaseModel):
    refund: Optional[float] = Field(None, description="Refund amount awarded, in INR.")
    interest_rate: Optional[float] = Field(None, description="Annual interest rate awarded, as a percentage (e.g. 9.0 for 9%).")
    mental_agony_compensation: Optional[float] = Field(None, description="Compensation for mental agony/harassment, in INR.")
    litigation_cost: Optional[float] = Field(None, description="Litigation/legal cost awarded, in INR.")


class Judgment(BaseModel):
    case_type: str = Field(description="Short label for the dispute, e.g. 'builder possession delay'.")
    fact_summary: str = Field(description="2-4 sentence summary of the facts of the case.")
    dispute_duration_months: Optional[int] = Field(
        None, description="Duration of the delay/dispute in months, if stated or computable."
    )
    amount_claimed: Optional[float] = Field(None, description="Total amount originally claimed by the complainant, in INR.")
    relief_components: ReliefComponents
    outcome: Literal["allowed", "dismissed", "partly_allowed"] = Field(
        description="Whether the complaint was allowed, dismissed, or partly allowed."
    )
    forum_level: str = Field(description="Deciding forum, e.g. 'NCDRC', 'State Commission'.")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Your confidence in this extraction. Use 'low' if the judgment text is "
        "ambiguous, heavily truncated, or you had to guess at any required field."
    )
