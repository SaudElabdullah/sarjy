"""In-memory `SessionRepo` / `MessageRepo` implementations.

These are production code (not test doubles): useful for local dev, demos,
and as a reference for the real Postgres-backed repos landing in a later
phase. Tests import them directly rather than redefining them.
"""

from __future__ import annotations

from typing import Any

from sarjy.contexts.conversation.application.context_loader import (
    LastResultsPort,
    TurnContext,
    last_results_from_row,
)
from sarjy.contexts.conversation.application.ports import (
    ActiveRunPort,
    FactSnapshotPort,
    MessageRepo,
)
from sarjy.contexts.conversation.domain.message import Message
from sarjy.contexts.conversation.domain.session import Session
from sarjy.shared.ids import MessageId, SessionId, UserId


class MemSessions:
    def __init__(self) -> None:
        self.items: dict[SessionId, Session] = {}

    async def get(self, id: SessionId) -> Session | None:
        return self.items.get(id)

    async def latest_for_user(self, user_id: UserId) -> Session | None:
        return None

    async def save(self, s: Session) -> None:
        self.items[s.id] = s


class MemMessages:
    def __init__(self) -> None:
        self.items: list[Message] = []
        self.tool_calls: list[tuple[Any, ...]] = []

    async def history(self, user_id: UserId, session_id: SessionId, limit: int) -> list[Message]:
        # Same predicate the RPC applies (see the v2 migration): spoken turns
        # only, and never a row the input guard blocked — a refused injection
        # replayed into every later prompt is the persistence the attacker was
        # after (I6/R4). Kept in step deliberately: the in-memory repo is what
        # the unit tests exercise `RunTurn` through, so a filter that lives only
        # in SQL is a filter those tests cannot see working.
        return [
            m
            for m in self.items
            if m.session_id == session_id
            and m.user_id == user_id
            and m.role in ("user", "assistant")
            and not (m.guard_decision or "").startswith("block:")
        ][-limit:]

    async def save(self, m: Message) -> None:
        self.items.append(m)

    async def save_tool_call(
        self,
        message_id: MessageId,
        user_id: UserId,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        status: str,
        latency_ms: int,
    ) -> None:
        self.tool_calls.append((message_id, user_id, tool_name, args, result, status, latency_ms))


class InMemoryContextLoader:
    """`ContextLoaderPort` for a container with no database behind it.

    It composes the three ports the single RPC replaced, so the seam `RunTurn`
    depends on is the same one either way and every existing unit test keeps
    exercising the real orchestrator through `Container.use_in_memory_repos()`.
    The reads stay sequential — there is no round trip to save here, and
    gathering them would only make a failure harder to attribute.

    `last_results_port` is optional because most callers building this by hand
    (tests, a local dev container with no assessment wiring) have nothing to
    ground against; when it is absent, `last_results` is simply never populated.
    It is consulted ONLY when no run is open — the same rule the RPC applies in
    SQL, so the two loaders cannot disagree about which of `workflow` /
    `last_results` a turn gets.
    """

    def __init__(
        self,
        facts: FactSnapshotPort,
        messages: MessageRepo,
        active_run: ActiveRunPort,
        last_results_port: LastResultsPort | None = None,
    ) -> None:
        self.facts = facts
        self.messages = messages
        self.active_run = active_run
        self.last_results_port = last_results_port

    async def load(self, user_id: UserId, session_id: SessionId, history_limit: int) -> TurnContext:
        history = await self.messages.history(user_id, session_id, history_limit)
        facts = await self.facts.snapshot(user_id)
        run = await self.active_run.active_run(user_id)
        last = None
        if run is None and self.last_results_port is not None:
            last = last_results_from_row(await self.last_results_port.latest_results(user_id))
        return TurnContext(
            facts=facts, history=history, workflow=run, profile={}, last_results=last
        )
