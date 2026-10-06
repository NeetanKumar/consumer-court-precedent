"""Deterministic follow-up chips — built from the answer's own data, so
they cost nothing and never offer a question the data cannot answer."""
from __future__ import annotations

from ccpf.answer.schema import PrecedentAnswer

MAX_CHIPS = 3


def follow_up_suggestions(answer: PrecedentAnswer) -> list[str]:
    if answer.refused:
        return [
            "Builder delayed possession by 3 years and refused to refund the booking amount",
            "Developer cancelled my booking and won't return the advance payment",
        ]
    chips: list[str] = []
    stats = answer.relief_component_stats
    if (s := stats.get("interest_rate")) and s.median is not None:
        chips.append("What interest rate was usually awarded?")
    if (s := stats.get("mental_agony_compensation")) and s.median is not None:
        chips.append("How much was awarded for mental agony?")
    if (s := stats.get("refund")) and s.coverage_fraction < 0.5:
        chips.append("Why is the refund data limited?")
    if answer.citations:
        chips.append("Tell me about the closest case")
    if answer.outcome_distribution.get("dismissed"):
        chips.append("Why were some complaints dismissed?")
    return chips[:MAX_CHIPS]
