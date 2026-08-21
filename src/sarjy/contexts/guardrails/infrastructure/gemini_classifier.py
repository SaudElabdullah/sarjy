from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from sarjy.contexts.conversation.application.ports import LLMMessage, LLMPort, LLMRequest
from sarjy.contexts.guardrails.application.ports import Classification
from sarjy.contexts.guardrails.domain.categories import Category

_PROMPT = (Path(__file__).parent / "prompts" / "classifier.md").read_text(encoding="utf-8")


class ClassifierOut(BaseModel):
    category: Category | None = None
    is_injection: bool = False
    severity: int = 0
    confidence: float = 0.0


class GeminiClassifier:
    def __init__(self, llm: LLMPort) -> None:
        self.llm = llm

    async def classify(self, recent_user_turns: list[str]) -> Classification:
        req = LLMRequest(
            system=_PROMPT,
            messages=[LLMMessage(role="user", text="\n---\n".join(recent_user_turns[-4:]))],
            tools=[],
            temperature=0.0,
            max_output_tokens=60,
        )
        out = await self.llm.generate_json(req, ClassifierOut)
        return Classification(out.category, out.is_injection, out.severity, out.confidence)
