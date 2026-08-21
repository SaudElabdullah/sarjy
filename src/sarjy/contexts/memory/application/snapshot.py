from __future__ import annotations

from sarjy.contexts.conversation.application.ports import Fact
from sarjy.contexts.memory.application.ports import MemoryRepo
from sarjy.shared.ids import UserId

SNAPSHOT_BYTES = 2048
SNAPSHOT_LIMIT = 60


class FactSnapshot:
    """Implements conversation FactSnapshotPort (PRD M-4)."""

    def __init__(self, repo: MemoryRepo) -> None:
        self.repo = repo

    async def snapshot(self, user_id: UserId) -> list[Fact]:
        mems = await self.repo.list_live(user_id, limit=SNAPSHOT_LIMIT)
        out: list[Fact] = []
        used = 0
        for m in mems:
            if m.kind == "note" or m.key is None:
                continue
            cost = len(m.key) + len(m.value) + 2
            if used + cost > SNAPSHOT_BYTES:
                break
            out.append(Fact(m.key, m.value, m.kind))
            used += cost
        return out
