from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sarjy.contexts.memory.application.ports import MemoryRepo
from sarjy.contexts.memory.domain.key_normalizer import normalize_key
from sarjy.contexts.memory.domain.memory import Memory
from sarjy.shared.clock import Clock
from sarjy.shared.errors import ValidationError
from sarjy.shared.ids import MemoryId, UserId


@dataclass(frozen=True, slots=True)
class ForgetOutcome:
    status: Literal["forgotten", "not_found"]
    key: str


class ForgetFact:
    def __init__(self, repo: MemoryRepo, clock: Clock) -> None:
        self.repo, self.clock = repo, clock

    async def __call__(self, user_id: UserId, key_raw: str) -> ForgetOutcome:
        try:
            key = normalize_key(key_raw)
        except ValidationError:
            return ForgetOutcome("not_found", key_raw)
        m = await self.repo.get_by_key(user_id, key)
        if m is None:
            return ForgetOutcome("not_found", key)
        await self._forget(m)
        return ForgetOutcome("forgotten", key)

    async def by_id(self, user_id: UserId, id: MemoryId) -> ForgetOutcome:
        m = await self.repo.get_by_id(user_id, id)
        if m is None:
            return ForgetOutcome("not_found", str(id))
        await self._forget(m)
        return ForgetOutcome("forgotten", m.key or str(id))

    async def _forget(self, m: Memory) -> None:
        m.forget(self.clock.now())
        memory_id = await self.repo.upsert_with_history(m, m.pull_events())
        # ...and then erase the audit trail for that memory, the delete event
        # this very call just wrote included (I2). `memories_history` stores
        # `old_value`/`new_value` verbatim — the remembered fact itself — so a
        # forget that only soft-deletes the `memories` row leaves a complete,
        # indefinitely-retained copy of what the user asked to be forgotten one
        # table over. Forgetting is a user-intent erase, not a bookkeeping
        # change: the trail goes with it.
        #
        # Deliberately after the write rather than instead of it: the write is
        # what makes the memory stop being recalled, and it is the operation
        # allowed to fail (NotFound on a concurrent forget). Doing it in this
        # order means a crash between the two leaves the fact un-recalled with
        # a stale trail, which the `retention_memories_history` cron does not
        # collect (the `memories` row still exists, soft-deleted) — a real, if
        # narrow, gap, and the safer of the two orders: the reverse could erase
        # the trail for a write that then never lands.
        await self.repo.delete_history(m.user_id, memory_id)
