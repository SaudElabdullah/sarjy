"""A deterministic `NarratorPort` for configurations with no Gemini behind them.

Returns `template_narrative(report)` — the same fallback `GeminiNarrator`
itself degrades to when generation fails or invents a number outside the
report (PRD G-6) — so an in-memory container can finish a run end to end,
narrative included, without ever making a network call.
"""

from __future__ import annotations

from sarjy.contexts.assessment.application.template_narrative import template_narrative
from sarjy.contexts.assessment.domain.scoring import ScoreReport


class OfflineNarrator:
    async def narrate(self, report: ScoreReport) -> str:
        return template_narrative(report)
