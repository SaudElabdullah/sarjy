from __future__ import annotations

from typing import Any

from sarjy.contexts.assessment.infrastructure.gemini_interpreter import (
    GeminiAnswerInterpreter,
    GeminiInterpreter,
    InterpretationOut,
)
from sarjy.contexts.conversation.application.ports import LLMRequest, LLMUnavailable


class FakeLLM:
    """Records the request and returns a canned `InterpretationOut`."""

    def __init__(self, out: InterpretationOut | Exception) -> None:
        self.out = out
        self.req: LLMRequest | None = None
        self.calls = 0

    async def generate_json(self, req: LLMRequest, schema: type[Any]) -> Any:
        self.calls += 1
        self.req = req
        assert schema is InterpretationOut
        if isinstance(self.out, Exception):
            raise self.out
        return self.out

    def stream(self, req: LLMRequest) -> Any:
        raise NotImplementedError


LABELS = [
    "Very inaccurate",
    "Moderately inaccurate",
    "Neither",
    "Moderately accurate",
    "Very accurate",
]


async def test_maps_a_value_and_pins_the_request_to_temperature_zero() -> None:
    llm = FakeLLM(InterpretationOut(value=4, confidence=0.9, control=None))
    out = await GeminiAnswerInterpreter(llm).interpret("I like order.", LABELS, "yeah mostly")  # type: ignore[arg-type]

    assert (out.value, out.confidence, out.control) == (4, 0.9, None)
    assert llm.req is not None
    assert llm.req.temperature == 0.0 and llm.req.max_output_tokens == 60
    sent = llm.req.messages[-1].text or ""
    assert "I like order." in sent and "yeah mostly" in sent
    # The scale labels travel with the item, so the model maps against the
    # instrument's own wording rather than a memorised 1-5 scale.
    assert "Very accurate" in sent


async def test_control_word_passes_through_with_no_value() -> None:
    llm = FakeLLM(InterpretationOut(value=None, confidence=1.0, control="repeat"))
    out = await GeminiInterpreter(llm).interpret("I like order.", LABELS, "say that again")  # type: ignore[arg-type]
    assert out.control == "repeat" and out.value is None and out.confidence == 1.0


async def test_value_outside_the_scale_is_not_recorded() -> None:
    llm = FakeLLM(InterpretationOut(value=9, confidence=0.9, control=None))
    out = await GeminiAnswerInterpreter(llm).interpret("x", LABELS, "y")  # type: ignore[arg-type]
    assert out.value is None and out.confidence == 0.0 and out.control is None


async def test_confidence_is_clamped_into_range() -> None:
    llm = FakeLLM(InterpretationOut(value=3, confidence=4.2, control=None))
    out = await GeminiAnswerInterpreter(llm).interpret("x", LABELS, "y")  # type: ignore[arg-type]
    assert out.value == 3 and out.confidence == 1.0


async def test_llm_failure_degrades_to_unreadable_so_the_handler_reasks() -> None:
    llm = FakeLLM(LLMUnavailable("gemini down"))
    out = await GeminiAnswerInterpreter(llm).interpret("x", LABELS, "y")  # type: ignore[arg-type]
    assert out.value is None and out.control is None and out.confidence == 0.0
