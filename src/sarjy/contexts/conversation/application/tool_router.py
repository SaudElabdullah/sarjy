from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any

from sarjy.contexts.conversation.application.ports import (
    Fact,
    FunctionCall,
    ToolPort,
    ToolResult,
)
from sarjy.observability.logging import get_logger
from sarjy.shared.ids import UserId

log = get_logger(__name__)
TOOL_TIMEOUT_S = 4.0


class ToolRouter:
    def __init__(self) -> None:
        self._tools: dict[str, ToolPort] = {}

    def register(self, tool: ToolPort) -> None:
        self._tools[tool.name] = tool

    def declarations(self) -> list[dict[str, Any]]:
        return [t.declaration for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def mutates(self, name: str) -> bool:
        """Does calling `name` change anything outside the turn? (L-3.)

        An unknown name is `False`: `invoke` refuses it without reaching a tool,
        so there is nothing to promote a speculative turn for. See
        `ToolPort.mutating`.
        """
        tool = self._tools.get(name)
        return tool is not None and tool.mutating

    async def invoke(
        self, user_id: UserId, call: FunctionCall, *, facts: list[Fact] | None = None
    ) -> ToolResult:
        """Run `call`, handing the tool the turn's fact snapshot if there is one.

        `facts` is keyword-only because it is context, not an argument the model
        chose: everything positional here comes from the function call, and the
        snapshot comes from the turn that is running it. Passed to every tool
        (the router has no business knowing which ones care) and defaulted to
        None so a caller with no snapshot — an eval harness, a unit test — can
        still drive the router, leaving any tool that needs facts to fetch its
        own the way it always did.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                ok=False,
                data={"error": "unknown_tool"},
                spoken_error="I can't do that.",
            )
        t0 = time.perf_counter()
        try:
            res = await asyncio.wait_for(
                tool.invoke(user_id, call.args, facts), timeout=TOOL_TIMEOUT_S
            )
        except TimeoutError:
            res = ToolResult(
                ok=False,
                data={"error": "timeout"},
                spoken_error="That took too long, sorry.",
            )
        except Exception as e:
            log.warning("tool_error", tool=call.name, error=repr(e))
            res = ToolResult(
                ok=False,
                data={"error": "exception", "detail": e.__class__.__name__},
                spoken_error="Something went wrong with that.",
            )
        # Reported on the result rather than stuffed into `data`: `data` is echoed
        # back to the model as the function_response and is often the tool's own dict,
        # so writing into it both leaked an internal field into the prompt and mutated
        # the tool's object.
        return replace(res, latency_ms=int((time.perf_counter() - t0) * 1000))
