"""Deterministic scoring for an instrument — the numbers the user is told.

A trait score is the mean of its answered items (reversed items counted as
`6 - v`), rounded to one decimal place. The rounding is Python's `round`,
which is round-half-even (banker's rounding): a mean of 2.25 becomes 2.2, not
2.3, and 2.35 becomes 2.4. That matters in two places beyond this module, so
it is written down here rather than discovered. `band_for` rounds to one
decimal before comparing against the band edges, so the boundary a score falls
on is decided by the same rule. And `results_numbers` / the narrator's
`allowed_numbers` both declare `round(score)` — half-even again — as a figure
the reply may speak, so a narrative may say "4.5, so about 4" for a 4.5 but
not "about 3"; the top of the scale is declared separately (every line says
"out of five"), so it is not evidence of a rounding.

Nothing here is approximate on purpose: two runs with the same answers produce
the same five numbers, and the eval suite checks exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sarjy.contexts.assessment.domain.instrument import Instrument

MIN_ANSWERED_PER_TRAIT = 3
SCALE_MAX_PLUS_ONE = 6  # reversal: 6 - v


@dataclass(frozen=True, slots=True)
class TraitScore:
    code: str
    name: str
    score: float | None
    band: str | None
    answered: int


@dataclass(frozen=True, slots=True)
class ScoreReport:
    traits: list[TraitScore]
    answered: int
    skipped: int
    # The top of the instrument's response scale — what "out of five" means in
    # this report. Carried on the report because the narrator is handed nothing
    # but the report (`NarratorPort.narrate`) and still has to know which
    # figures the reply will be entitled to speak. Not part of `as_dict`: that
    # is the persisted results shape and the web client reads it.
    scale_top: float

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {t.code: t.score for t in self.traits}
        d["bands"] = {t.code: t.band for t in self.traits}
        d["answered"] = self.answered
        d["skipped"] = self.skipped
        return d

    def trait(self, code: str) -> TraitScore:
        return next(t for t in self.traits if t.code == code)


def band_for(instrument: Instrument, value: float) -> str:
    # Bands are contiguous at 1 dp; a value between two bands (e.g. 2.45) rounds first.
    v = round(value, 1)
    for name, (lo, hi) in instrument.bands.items():
        if lo <= v <= hi:
            return name
    # fallback: nearest band by lower bound
    return min(instrument.bands.items(), key=lambda kv: abs(kv[1][0] - v))[0]


def score(instrument: Instrument, answers: dict[int, int | None]) -> ScoreReport:
    traits: list[TraitScore] = []
    answered_total = sum(1 for v in answers.values() if v is not None)
    skipped_total = sum(1 for v in answers.values() if v is None)
    for code, name in instrument.traits.items():
        vals: list[int] = []
        for item in instrument.items_for_trait(code):
            v = answers.get(item.no)
            if v is None:
                continue
            vals.append(SCALE_MAX_PLUS_ONE - v if item.reverse else v)
        if len(vals) < MIN_ANSWERED_PER_TRAIT:
            traits.append(TraitScore(code, name, None, None, len(vals)))
            continue
        mean = round(sum(vals) / len(vals), 1)
        traits.append(TraitScore(code, name, mean, band_for(instrument, mean), len(vals)))
    return ScoreReport(
        traits=traits,
        answered=answered_total,
        skipped=skipped_total,
        scale_top=float(len(instrument.scale_labels)),
    )
