"""Gemini-backed `NarratorPort` — the one generated sentence-run in a results
reply, and the one that has to be checked before it is spoken.

The scores are computed deterministically in the domain; the narrative only
reflects on them. A model that rounds 3.6 to "almost 4", or invents a sixth
figure, has changed someone's results — and the output guard would cut that
sentence anyway, leaving a hole in the middle of the reply. So every number in
the generated text is matched against the report before the text is accepted:
one regeneration, then the deterministic `template_narrative`, which is built
from the report's own numbers and therefore always passes.

Note on the check's direction: it requires that at least one real score is
quoted and that *every* number present is one the report declares. Demanding
all five scores appear would contradict the prompt (which asks for the two or
three most notable traits) and would send every well-formed narrative to the
template.

C1: the check reads number WORDS as well as digits, because the output guard
does. It used to match runs of digits only, so "Openness at 5.0 is one of the traits
that stands out" was accepted here and then cut by the guard over the "one" —
a hole in the middle of someone's results that nothing downstream could
repair. `numbers_ok` now runs the guard's own extractor
(`sarjy.shared.numbers.extract_numbers`) and the guard's own tolerance rule,
so anything this function accepts is speakable by construction.
"""

from __future__ import annotations

import re
from pathlib import Path

from sarjy.contexts.assessment.application.template_narrative import template_narrative
from sarjy.contexts.assessment.domain.scoring import ScoreReport, TraitScore
from sarjy.contexts.conversation.application.ports import (
    LLMMessage,
    LLMPort,
    LLMRequest,
    LLMText,
)
from sarjy.observability.logging import get_logger
from sarjy.shared.numbers import extract_numbers

_PROMPT = (Path(__file__).parent / "prompts" / "narrative.md").read_text(encoding="utf-8")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_MAX_TOKENS = 400
_THINKING_BUDGET = 1024
_ATTEMPTS = 2
# Must equal `ungrounded_numbers`' default tolerance: this is the guard's rule
# being applied early, and a looser value here would accept text the guard cuts.
GUARD_TOLERANCE = 1.0

log = get_logger(__name__)


def allowed_numbers(report: ScoreReport) -> set[str]:
    """Every numeric string the narrative may contain, exactly as written.

    Each scored trait contributes its one-decimal score and the integer that
    score rounds to — the same pair `results_numbers` declares to the output
    guard, so an accepted narrative can never be cut for a figure. The top of
    the scale ("out of five") comes from the report, which got it from the
    instrument: hardcoding a 5 here would silently mis-check the first
    instrument that does not use a five-point scale.
    """
    out = {f"{report.scale_top:g}"}
    for t in report.traits:
        if t.score is not None:
            out.add(f"{t.score:.1f}")
            out.add(str(round(t.score)))
    return out


def allowed_values(report: ScoreReport) -> list[float]:
    """The same floats `results_numbers` will arm the output guard with.

    Kept in step with `allowed_numbers` above — one is the exact spellings a
    digit may take, the other is the values the guard compares against.
    """
    out: list[float] = [report.scale_top]
    for t in report.traits:
        if t.score is not None:
            out.extend((float(t.score), float(round(t.score))))
    return list(dict.fromkeys(out))


def numbers_ok(text: str, report: ScoreReport) -> bool:
    """Is `text` safe to speak as this report's narrative?

    Three conditions, all required:

    1. It quotes at least one real trait score — a reflection that names no
       figure is not a reflection on a set of scores.
    2. Every digit string in it is one the report declares, spelled exactly:
       "4.5" or "4" for a 4.5, never "4.7". The guard's +/-1.0 tolerance would
       wave that through; we will not, because a rounded score is a changed
       result even when it is speakable.
    3. Every quantity the *guard's* extractor finds — number words included —
       is within `GUARD_TOLERANCE` of a declared value. This is the guard's
       own verdict, computed here, so an accepted narrative cannot be cut.
    """
    allowed = allowed_numbers(report)
    found = set(_NUM.findall(text))
    scores = {f"{t.score:.1f}" for t in report.traits if t.score is not None}
    if scores and not (found & scores):
        return False  # a reflection that quotes no score is not a reflection
    if not found <= allowed:
        return False
    values = allowed_values(report)
    return all(any(abs(n - a) <= GUARD_TOLERANCE for a in values) for n in extract_numbers(text))


class GeminiNarrator:
    def __init__(self, llm: LLMPort) -> None:
        self.llm = llm

    async def narrate(self, report: ScoreReport) -> str:
        req = LLMRequest(
            system=_PROMPT,
            messages=[LLMMessage(role="user", text=self._scores_block(report))],
            tools=[],
            temperature=0.7,
            max_output_tokens=_MAX_TOKENS,
            thinking_budget=_THINKING_BUDGET,
        )
        for attempt in range(_ATTEMPTS):
            try:
                text = "".join(
                    [ev.text async for ev in self.llm.stream(req) if isinstance(ev, LLMText)]
                ).strip()
            except Exception:
                log.warning("narrator_failed", exc_info=True)
                break
            if text and numbers_ok(text, report):
                return text
            # The rejected text itself is not logged: it is a description of
            # someone's personality results.
            log.info("narrator_rejected", attempt=attempt)
        return template_narrative(report)

    @staticmethod
    def _scores_block(report: ScoreReport) -> str:
        def line(t: TraitScore) -> str:
            if t.score is None:
                return f"{t.name}: not scored"
            return f"{t.name}: {t.score:.1f} ({t.band})"

        return "\n".join(line(t) for t in report.traits)
