"""The deterministic reflection used when no generated one can be trusted.

The narrator is the only part of the results reply a model writes, and the
output guard cuts any sentence quoting a figure the turn did not declare. So
when generation invents a number — or fails outright — we do not retry forever
and we do not ship a silent gap: we speak this template instead. It lives in
the application layer, not in the Gemini adapter, because it is product copy
about a domain object, and because `HandleAssessmentTurn` must be able to
complete a run with a narrative even with no LLM in the container at all.

Every number it speaks is a trait score at one decimal place, which is exactly
what `results_numbers` declares, so it can never be cut.
"""

from __future__ import annotations

from sarjy.contexts.assessment.domain.scoring import ScoreReport, TraitScore

MIDPOINT = 3.0

NOT_ENOUGH = (
    "There weren't enough answers this time to describe a pattern. "
    "Nothing is lost — the answers you did give are still yours. "
    "We can pick the questionnaire up again whenever you like. "
    "Thanks for giving it a go."
)


def _distinctive(traits: list[TraitScore]) -> list[TraitScore]:
    """Scored traits, furthest from the midpoint first — the ones worth saying."""
    scored = [t for t in traits if t.score is not None]
    return sorted(scored, key=lambda t: (-abs((t.score or 0.0) - MIDPOINT), t.code))


def template_narrative(report: ScoreReport) -> str:
    """A warm, four-sentence reflection built only from the report's own numbers."""
    top = _distinctive(report.traits)
    if not top:
        return NOT_ENOUGH
    first = top[0]
    parts = [
        f"Your most distinctive trait here is {first.name} at {first.score:.1f}, "
        f"which sits in the {first.band} range."
    ]
    if len(top) > 1:
        second = top[1]
        parts.append(
            f"{second.name} also stands out at {second.score:.1f}, in the {second.band} range."
        )
    else:
        parts.append("The other traits didn't have enough answers to score this time.")
    parts.append("The rest of your scores sit closer to the middle, which is very common.")
    parts.append("These are tendencies rather than labels, and what you do with them is yours.")
    return " ".join(parts)
