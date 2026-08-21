from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from sarjy.contexts.assessment.application.handle_turn import results_numbers
from sarjy.contexts.assessment.application.template_narrative import (
    NOT_ENOUGH,
    template_narrative,
)
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.scoring import ScoreReport, TraitScore, score
from sarjy.contexts.assessment.infrastructure.gemini_narrator import (
    GeminiNarrator,
    allowed_numbers,
    numbers_ok,
)
from sarjy.contexts.conversation.application.ports import (
    GuardContext,
    LLMEvent,
    LLMFinished,
    LLMRequest,
    LLMText,
    LLMUnavailable,
)
from sarjy.contexts.guardrails.application.output_guard import OutputGuard
from tests.unit.guardrails.test_input_guard import MemEvents

INS = Instrument.from_definition(
    json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text())
)
ANSWERS: dict[int, int | None] = dict.fromkeys(range(1, 21), 3)
ANSWERS.update({1: 5, 6: 1, 11: 4, 16: 2})  # E = (5 + 5 + 4 + 4) / 4 = 4.5
REPORT = score(INS, ANSWERS)


class TextLLM:
    """Streams one canned text per call, in order."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls = 0
        self.req: LLMRequest | None = None

    async def generate_json(self, req: LLMRequest, schema: type[Any]) -> Any:
        raise NotImplementedError

    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        self.calls += 1
        self.req = req
        if not self.texts:
            raise LLMUnavailable("no more canned replies")
        yield LLMText(self.texts.pop(0))
        yield LLMFinished("stop")


def test_report_fixture_is_the_shape_the_tests_assume() -> None:
    assert REPORT.trait("E").score == 4.5
    assert {f"{t.score:.1f}" for t in REPORT.traits if t.score is not None} == {"4.5", "3.0"}


async def test_a_narrative_that_quotes_real_scores_is_spoken_as_written() -> None:
    text = (
        "Your Extraversion of 4.5 stands out. You enjoy company. "
        "The others sit near 3.0. Keep being you."
    )
    llm = TextLLM([text])
    out = await GeminiNarrator(llm).narrate(REPORT)  # type: ignore[arg-type]

    assert out == text and llm.calls == 1
    assert llm.req is not None
    assert llm.req.temperature == 0.7
    assert llm.req.thinking_budget == 1024 and llm.req.max_output_tokens == 400
    # The model is given the scored figures, not the raw answers.
    assert "Extraversion: 4.5 (high)" in (llm.req.messages[-1].text or "")


async def test_an_invented_number_costs_one_regeneration() -> None:
    good = "Extraversion at 4.5 is the standout. Others sit at 3.0. That is common. Nice work."
    llm = TextLLM(["Your Extraversion of 4.7 stands out.", good])
    out = await GeminiNarrator(llm).narrate(REPORT)  # type: ignore[arg-type]
    assert out == good and llm.calls == 2


async def test_two_bad_narratives_fall_back_to_the_template() -> None:
    llm = TextLLM(["Your Extraversion of 4.7 stands out.", "Extraversion is 4.9, wow."])
    out = await GeminiNarrator(llm).narrate(REPORT)  # type: ignore[arg-type]
    assert llm.calls == 2 and out == template_narrative(REPORT) and "4.5" in out


async def test_a_failing_llm_falls_back_without_retrying_forever() -> None:
    llm = TextLLM([])  # every call raises
    out = await GeminiNarrator(llm).narrate(REPORT)  # type: ignore[arg-type]
    assert out == template_narrative(REPORT) and llm.calls == 1


async def test_a_narrative_with_no_score_at_all_is_rejected() -> None:
    llm = TextLLM(["You seem outgoing and steady.", "You seem outgoing and steady."])
    out = await GeminiNarrator(llm).narrate(REPORT)  # type: ignore[arg-type]
    assert out == template_narrative(REPORT)


def test_numbers_ok_accepts_the_integer_rounding_the_reply_also_declares() -> None:
    # `results_numbers` declares both 4.5 and round(4.5); saying either is safe.
    assert numbers_ok("Extraversion is 4.5, so about 4 out of 5.", REPORT)
    assert not numbers_ok("Extraversion is 4.5 and Openness is 2.9.", REPORT)


# --- the deterministic template -------------------------------------------


def test_template_speaks_the_most_distinctive_trait_with_its_exact_score() -> None:
    out = template_narrative(REPORT)
    assert "Extraversion" in out and "4.5" in out
    assert out.count(".") >= 4  # four sentences
    for word in ("disorder", "problem", "bad", "diagnos"):
        assert word not in out.lower()


def test_template_numbers_are_all_declared_by_the_reply() -> None:
    assert numbers_ok(template_narrative(REPORT), REPORT)


def test_template_handles_a_report_with_nothing_scored() -> None:
    empty = ScoreReport(
        traits=[TraitScore(c, c, None, None, 0) for c in "OCEAN"],
        answered=0,
        skipped=0,
        scale_top=5.0,
    )
    assert template_narrative(empty) == NOT_ENOUGH


def test_template_handles_a_single_scored_trait() -> None:
    one = ScoreReport(
        traits=[
            TraitScore("E", "Extraversion", 4.5, "high", 4),
            TraitScore("O", "Openness", None, None, 1),
        ],
        answered=5,
        skipped=0,
        scale_top=5.0,
    )
    out = template_narrative(one)
    assert "Extraversion" in out and "4.5" in out and "didn't have enough answers" in out


# --- C1: the narrator's check and the output guard agree ------------------

ALL_FIVES = ScoreReport(
    traits=[TraitScore(c, c, 5.0, "high", 4) for c in "OCEAN"],
    answered=20,
    skipped=0,
    scale_top=5.0,
)


def test_a_number_word_the_guard_would_cut_is_rejected() -> None:
    # `results_numbers` for this report is exactly (5.0,) — the scale top and
    # every trait score coincide — so the "one" in "one of the traits" is a
    # figure the reply never declares, and the guard cuts the sentence.
    assert results_numbers(INS, ALL_FIVES) == (5.0,)
    assert not numbers_ok("Openness at 5.0 is one of the traits that stands out", ALL_FIVES)
    # The digits alone were fine: only the spoken word made it unspeakable.
    assert numbers_ok("Openness sits at 5.0, and so does Extraversion.", ALL_FIVES)


def test_the_template_narrative_is_accepted_for_that_same_report() -> None:
    assert numbers_ok(template_narrative(ALL_FIVES), ALL_FIVES)


def test_a_word_number_close_enough_to_a_declared_score_is_allowed() -> None:
    # The guard's tolerance is +/-1.0 and 3.0 is declared, so "two" is grounded
    # for it — the narrator must not be stricter than the guard about words,
    # only about the digits of a score (see test above on 4.7).
    assert numbers_ok("Openness is 3.0 while Extraversion is 4.5, and two traits lead.", REPORT)


NARRATIVES = [
    "Your Extraversion of 4.5 stands out. You enjoy company. The others sit near 3.0. Nice.",
    template_narrative(REPORT),
    "Extraversion at 4.5 is one of the traits that stands out. The rest sit at 3.0.",
    "Extraversion is 4.5 and the rest are 3.0, steadier than twenty other people.",
    "Your Extraversion of 4.7 stands out. The rest sit at 3.0.",
    "Openness is 3.0 and Extraversion 4.5. Two of the five traits lead here.",
]


@pytest.mark.parametrize("text", NARRATIVES)
def test_anything_numbers_ok_accepts_survives_the_real_output_guard(text: str) -> None:
    """The cross-boundary property C1 exists to hold: accepted => speakable.

    The results reply arms the guard with `results_numbers(...)`, then speaks
    the narrative one sentence at a time. If `numbers_ok` says yes, no sentence
    may come back cut — otherwise the user hears a hole in their results.
    """
    if not numbers_ok(text, REPORT):
        return
    guard = OutputGuard(MemEvents(), "enforce")  # type: ignore[arg-type]
    ctx = GuardContext(
        system_prompt="You are Sarjy, a voice assistant.",
        tool_numbers=list(results_numbers(INS, REPORT)),
        facts=[],
    )
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        assert guard.check_sentence(sentence, ctx).action == "pass", sentence


def test_the_narrative_sample_set_covers_both_verdicts() -> None:
    # Otherwise the parametrised test above could pass by rejecting everything.
    verdicts = [numbers_ok(t, REPORT) for t in NARRATIVES]
    assert any(verdicts) and not all(verdicts)


# --- R3: the top of the scale comes from the instrument --------------------

THREE_POINT = Instrument.from_definition(
    {
        **json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text()),
        "scale": {"min": 1, "max": 3, "labels": ["Inaccurate", "Neither", "Accurate"]},
    }
)


def test_the_report_carries_the_instrument_s_scale_top() -> None:
    assert REPORT.scale_top == 5.0
    assert score(THREE_POINT, dict.fromkeys(range(1, 21), 2)).scale_top == 3.0


def test_a_narrative_is_checked_against_that_scale_not_a_hardcoded_five() -> None:
    report3 = score(THREE_POINT, dict.fromkeys(range(1, 21), 2))
    allowed = allowed_numbers(report3)
    assert "3" in allowed and "5" not in allowed
    # "out of 5" is a figure a three-point instrument's reply never declares,
    # so the guard would cut the sentence — and the narrator must say so first.
    assert not numbers_ok("Openness is 3.5 out of 5.", report3)
    assert numbers_ok("Openness is 3.5 out of 3.", report3)
