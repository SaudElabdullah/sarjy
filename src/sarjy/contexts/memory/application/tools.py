"""`remember`, `forget`, `recall` as conversation `ToolPort`s.

Adapts the memory context's use cases (`RememberFact`, `ForgetFact`,
`RecallFacts`) to the conversation context's `ToolPort` protocol so the tool
router can drive them without depending on memory internals. `name` and
`declaration` are verbatim from PRD §9.3.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sarjy.contexts.conversation.application.ports import Fact, ToolResult
from sarjy.contexts.memory.application.forget import ForgetFact
from sarjy.contexts.memory.application.recall import RecallFacts
from sarjy.contexts.memory.application.remember import RememberFact
from sarjy.contexts.memory.domain.memory import MemoryKind
from sarjy.shared.ids import UserId

_KINDS: tuple[MemoryKind, ...] = ("fact", "preference", "person", "place", "note")


class RememberTool:
    name = "remember"
    # A fact stored for a sentence the user never finished saying is a wrong
    # fact with no confirmation step to take it back (L-3).
    mutating: ClassVar[bool] = True
    declaration: ClassVar[dict[str, Any]] = {
        "name": "remember",
        "description": (
            "Store a personal fact the user explicitly told you about themselves. "
            "Use snake_case keys like favorite_color, home_city, sister_name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
                "kind": {"type": "string", "enum": list(_KINDS)},
            },
            "required": ["key", "value"],
        },
    }

    def __init__(self, remember: RememberFact) -> None:
        self._remember = remember

    async def invoke(
        self, user_id: UserId, args: dict[str, Any], facts: list[Fact] | None = None
    ) -> ToolResult:
        key, value = str(args.get("key") or ""), str(args.get("value") or "")
        if not key or not value:
            return ToolResult(
                ok=False,
                data={"status": "rejected", "reason": "missing key or value"},
                spoken_error="I'm not sure what you'd like me to remember.",
            )
        kind_raw = args.get("kind")
        kind: MemoryKind = kind_raw if kind_raw in _KINDS else "fact"
        out = await self._remember(user_id, key, value, kind)
        if out.status == "rejected":
            return ToolResult(
                ok=False,
                data={"status": "rejected", "key": out.key, "reason": out.reason},
                spoken_error=f"I won't store that — {out.reason}.",
            )
        return ToolResult(ok=True, data={"status": out.status, "key": out.key, "value": out.value})


class ForgetTool:
    name = "forget"
    mutating: ClassVar[bool] = True  # and a deletion is the least undoable of all
    declaration: ClassVar[dict[str, Any]] = {
        "name": "forget",
        "description": "Delete a stored fact when the user asks you to forget it.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    }

    def __init__(self, forget: ForgetFact) -> None:
        self._forget = forget

    async def invoke(
        self, user_id: UserId, args: dict[str, Any], facts: list[Fact] | None = None
    ) -> ToolResult:
        key = str(args.get("key") or "")
        if not key:
            return ToolResult(
                ok=False,
                data={"status": "not_found", "reason": "missing key"},
                spoken_error="Which thing should I forget?",
            )
        out = await self._forget(user_id, key)
        return ToolResult(ok=True, data={"status": out.status, "key": out.key})


class RecallTool:
    name = "recall"
    mutating: ClassVar[bool] = False  # a search over stored facts
    declaration: ClassVar[dict[str, Any]] = {
        "name": "recall",
        "description": (
            "Search stored facts when the user asks what you remember or when the facts block "
            "doesn't contain what you need. Returns [] if nothing is stored — in that case say "
            "you don't have it stored."
        ),
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }

    def __init__(self, recall: RecallFacts) -> None:
        self._recall = recall

    async def invoke(
        self, user_id: UserId, args: dict[str, Any], facts: list[Fact] | None = None
    ) -> ToolResult:
        q = args.get("query")
        # Deliberately NOT `facts`: the turn's snapshot is the small, prompt-sized
        # set already in the system prompt, and `recall` exists precisely for the
        # question that snapshot could not answer. Searching what the model can
        # already see would make the tool a no-op.
        found = await self._recall(user_id, str(q) if q else None)
        return ToolResult(
            ok=True,
            data={
                "facts": [{"key": f.key, "value": f.value, "kind": f.kind} for f in found],
                "count": len(found),
            },
        )
