from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sarjy.contexts.memory.domain.memory import Memory, MemoryChanged
from sarjy.shared.ids import MemoryId, MessageId, UserId


class MemoryRepo(Protocol):
    async def get_by_key(self, user_id: UserId, key: str) -> Memory | None: ...
    async def get_by_id(self, user_id: UserId, id: MemoryId) -> Memory | None: ...
    async def list_live(self, user_id: UserId, limit: int = 60) -> list[Memory]: ...
    async def upsert(self, m: Memory) -> None: ...
    async def soft_delete(self, m: Memory) -> None: ...
    async def append_history(self, ev: MemoryChanged) -> None: ...
    async def delete_history(self, user_id: UserId, memory_id: MemoryId) -> None:
        """Erase every `memories_history` row for one of `user_id`'s memories.

        Called by `ForgetFact` only. Forgetting is a user asking for a fact to
        be gone, and `memories_history` holds the fact itself — every
        `old_value`/`new_value` it ever had, including the one the delete event
        records. Leaving that behind would make "forget my address" a soft
        delete of the memory row and a permanent copy of the address next to
        it, which is not what the user asked for and not what PRD §11 promises.
        """
        ...

    async def upsert_with_history(self, m: Memory, events: list[MemoryChanged]) -> MemoryId:
        """Persist `m` and `events` atomically.

        Raises `sarjy.shared.errors.NotFound` for a *live* write (`m.deleted_at is
        None`) against a row that was soft-deleted by someone else since `m` was
        read, rather than silently resurrecting it.
        """
        ...


class EmbedderPort(Protocol):
    async def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class ScreenVerdict:
    """Result of screening one string (a `RememberFact`/edit key OR value) before it is stored.

    `reason`, when set, identifies what fired (an adapter-specific id such as
    a guardrail rule id) — it is diagnostic, not spoken text. The use case
    calling `screen` is responsible for turning a refusal into a short,
    neutral outcome reason a caller can safely speak back to the user.
    """

    allowed: bool
    reason: str | None = None


class ValueScreenPort(Protocol):
    """Screens a single string — a memory key OR a memory value — against
    content-safety rules.

    Callers (`RememberFact`, `EditFact`) MUST call this once per field and
    MUST NOT concatenate the key and value into one string first: a key
    that normalises to a word like "note"/"remember"/"save"/"store", paired
    with a value shaped like `"that my X is '<payload>'"`, reconstructs the
    guardrail rule engine's memory-set-frame carve-out at concatenation time
    and can smuggle an uncertain-severity payload past screening (Phase 8
    Task 6b fix round 1). Screening each field alone closes that.

    `user_id`/`message_id` are passed through purely so an adapter can
    attribute an audit-trail event to the write that triggered it — they do
    not affect the verdict.

    Deliberately narrow and sync: the memory context must not import the
    guardrails context directly (DDD boundary), so this port is what a
    guardrails-backed adapter (e.g. `RuleEngineValueScreen`) is plugged in
    behind. Sync because the rule engine it is expected to wrap is sync — a
    future async-only screen can still satisfy this by wrapping a coroutine
    in `asyncio.run`/similar at the adapter boundary if ever needed.
    """

    def screen(
        self, text: str, *, user_id: UserId | None = None, message_id: MessageId | None = None
    ) -> ScreenVerdict: ...
