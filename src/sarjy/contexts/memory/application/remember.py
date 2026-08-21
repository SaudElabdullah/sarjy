from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sarjy.contexts.memory.application.ports import MemoryRepo, ValueScreenPort
from sarjy.contexts.memory.application.screening import screen_reason
from sarjy.contexts.memory.domain.key_normalizer import normalize_key
from sarjy.contexts.memory.domain.memory import Memory, MemoryKind
from sarjy.contexts.memory.domain.pii_filter import reject_reason
from sarjy.shared.clock import Clock
from sarjy.shared.errors import NotFound, ValidationError
from sarjy.shared.ids import MemoryId, MessageId, UserId, new_id


@dataclass(frozen=True, slots=True)
class RememberOutcome:
    status: Literal["created", "updated", "rejected"]
    key: str
    value: str | None = None
    reason: str | None = None


class RememberFact:
    def __init__(self, repo: MemoryRepo, clock: Clock, screen: ValueScreenPort) -> None:
        self.repo, self.clock, self.screen = repo, clock, screen

    async def __call__(
        self,
        user_id: UserId,
        key_raw: str,
        value: str,
        kind: MemoryKind = "fact",
        source_message_id: MessageId | None = None,
    ) -> RememberOutcome:
        try:
            key = normalize_key(key_raw)
        except ValidationError:
            return RememberOutcome(
                "rejected", key_raw, reason="I couldn't tell what to file that under"
            )
        reason = reject_reason(value, key=key)
        if reason:
            return RememberOutcome("rejected", key, reason=reason)
        # Screen the key and the value as data being stored — NOT the whole
        # utterance ("remember that my motto is ..."), and NEVER concatenated
        # (see `screen_reason`'s docstring for why concatenation is itself a
        # bypass). We ARE the value being stored, so the low-precision
        # `unc.*` escalation rules must run at full strength rather than
        # standing down for a memory-set frame.
        screen_refusal = screen_reason(
            self.screen, key, value, user_id=user_id, message_id=source_message_id
        )
        if screen_refusal:
            return RememberOutcome("rejected", key, reason=screen_refusal)
        now = self.clock.now()
        existing = await self.repo.get_by_key(user_id, key)
        try:
            if existing:
                existing.update(value, now)
                try:
                    await self.repo.upsert_with_history(existing, existing.pull_events())
                except NotFound:
                    # The row we read was forgotten (soft-deleted) by someone else
                    # between our read and this write — the repo refused to write
                    # (and, crucially, refused to resurrect it). Treat this exactly
                    # like the key being free: create a brand-new live memory.
                    m = Memory.create(
                        new_id(MemoryId), user_id, key, value, kind, now, source_message_id
                    )
                    await self.repo.upsert_with_history(m, m.pull_events())
                    return RememberOutcome("created", key, m.value)
                return RememberOutcome("updated", key, existing.value)
            m = Memory.create(new_id(MemoryId), user_id, key, value, kind, now, source_message_id)
            await self.repo.upsert_with_history(m, m.pull_events())
            return RememberOutcome("created", key, m.value)
        except ValidationError as e:
            return RememberOutcome("rejected", key, reason=str(e))
