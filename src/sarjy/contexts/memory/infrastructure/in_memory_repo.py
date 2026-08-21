from __future__ import annotations

import copy

from sarjy.contexts.memory.domain.memory import Memory, MemoryChanged
from sarjy.shared.ids import MemoryId, UserId


def _copy(m: Memory) -> Memory:
    """Shallow-copy `m` with a fresh, empty pending-events queue.

    Every copy the repo hands out or stores must start with no pending
    events: `_events` is transient aggregate state meant to be pulled once
    by the caller that mutated it, not part of persisted data. Sharing the
    list across copies (plain `copy.copy`) lets pulled-and-cleared events on
    one copy resurrect on another; carrying it over via `copy.deepcopy`
    still leaks stale events forward whenever `upsert`/`soft_delete` is
    called before the caller pulls its events (as the use cases do).
    """
    c = copy.copy(m)
    c._events = []
    return c


class InMemoryMemoryRepo:
    def __init__(self) -> None:
        self.items: dict[MemoryId, Memory] = {}
        self.history: list[MemoryChanged] = []

    async def get_by_key(self, user_id: UserId, key: str) -> Memory | None:
        for m in self.items.values():
            if m.user_id == user_id and m.key == key and m.is_live:
                return _copy(m)
        return None

    async def get_by_id(self, user_id: UserId, id: MemoryId) -> Memory | None:
        m = self.items.get(id)
        return _copy(m) if m and m.user_id == user_id and m.is_live else None

    async def list_live(self, user_id: UserId, limit: int = 60) -> list[Memory]:
        live = [m for m in self.items.values() if m.user_id == user_id and m.is_live]
        live.sort(key=lambda m: m.updated_at, reverse=True)
        return [_copy(m) for m in live[:limit]]

    async def upsert(self, m: Memory) -> None:
        self.items[m.id] = _copy(m)

    async def soft_delete(self, m: Memory) -> None:
        self.items[m.id] = _copy(m)

    async def append_history(self, ev: MemoryChanged) -> None:
        self.history.append(ev)

    async def delete_history(self, user_id: UserId, memory_id: MemoryId) -> None:
        self.history = [
            ev for ev in self.history if not (ev.user_id == user_id and ev.memory_id == memory_id)
        ]

    async def upsert_with_history(self, m: Memory, events: list[MemoryChanged]) -> MemoryId:
        await self.upsert(m)
        for ev in events:
            await self.append_history(ev)
        return m.id
