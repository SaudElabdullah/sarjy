from __future__ import annotations

from sarjy.contexts.conversation.application.ports import Fact
from sarjy.contexts.memory.application.ports import MemoryRepo
from sarjy.shared.ids import UserId

ALL_THRESHOLD = 40
TOP_K = 10


class RecallFacts:
    def __init__(self, repo: MemoryRepo) -> None:
        self.repo = repo

    async def __call__(self, user_id: UserId, query: str | None = None) -> list[Fact]:
        mems = await self.repo.list_live(user_id, limit=200)
        facts = [Fact(m.key or "note", m.value, m.kind) for m in mems]
        q = (query or "").strip().lower()
        if not q:
            return facts if len(facts) <= ALL_THRESHOLD else facts[:ALL_THRESHOLD]
        terms = [t for t in q.replace("_", " ").split() if t]
        hits = [
            f
            for f in facts
            if any(t in f.key.replace("_", " ").lower() or t in f.value.lower() for t in terms)
        ]
        return hits[:TOP_K]
