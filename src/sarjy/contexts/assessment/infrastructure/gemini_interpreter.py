"""Gemini-backed `AnswerInterpreterPort`.

The model's only job here is to turn "yeah, pretty much" into a 4, or "say that
again" into a control word. It is deliberately given no room to improvise:
temperature 0, sixty output tokens, a fixed schema, and a prompt whose examples
cover every control word the handler knows about.

Nothing it returns is trusted on its face. A value outside the 1-5 scale, or an
error of any kind, degrades to "I could not read that" — and `HandleAssessment
Turn` answers that by re-asking the item with the scale hint, which is a far
better turn than recording a guessed answer into someone's results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from sarjy.contexts.assessment.application.ports import Control, Interpretation
from sarjy.contexts.conversation.application.ports import LLMMessage, LLMPort, LLMRequest
from sarjy.observability.logging import get_logger

_PROMPT = (Path(__file__).parent / "prompts" / "interpreter.md").read_text(encoding="utf-8")
_MAX_TOKENS = 60
SCALE_MIN, SCALE_MAX = 1, 5

log = get_logger(__name__)

UNREADABLE = Interpretation(value=None, confidence=0.0, control=None)


class InterpretationOut(BaseModel):
    value: int | None = None
    confidence: float = 0.0
    control: Literal["repeat", "skip", "back", "explain", "pause", "quit", "off_topic"] | None = (
        None
    )


class GeminiAnswerInterpreter:
    def __init__(self, llm: LLMPort) -> None:
        self.llm = llm

    async def interpret(
        self, item_text: str, scale_labels: list[str], user_text: str
    ) -> Interpretation:
        scale = "; ".join(f"{i}={label}" for i, label in enumerate(scale_labels, start=SCALE_MIN))
        req = LLMRequest(
            system=_PROMPT,
            messages=[
                LLMMessage(
                    role="user",
                    text=f"Scale: {scale}\nItem: {item_text}\nUser said: {user_text}",
                )
            ],
            tools=[],
            temperature=0.0,
            max_output_tokens=_MAX_TOKENS,
        )
        try:
            out = await self.llm.generate_json(req, InterpretationOut)
        except Exception:
            # Unavailable, timed out, or unparseable: the handler re-asks.
            log.warning("interpreter_failed", exc_info=True)
            return UNREADABLE
        control: Control | None = out.control
        value = out.value if out.value is not None and SCALE_MIN <= out.value <= SCALE_MAX else None
        if value is None and control is None:
            return UNREADABLE
        return Interpretation(
            value=value, confidence=max(0.0, min(1.0, out.confidence)), control=control
        )


# The class is named for the port it implements; this is the shorter name the
# container and the task brief use for it.
GeminiInterpreter = GeminiAnswerInterpreter
