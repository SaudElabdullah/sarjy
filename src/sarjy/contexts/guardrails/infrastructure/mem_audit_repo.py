"""In-memory `AuditQueuePort` — the no-Postgres counterpart to `PgAuditRepo`.

Used by `AuditWorker` unit tests: a real (if trivial) queue rather than a
one-off fake per test file, the same reasoning `MemGuardEvents` follows for
`GuardEventRepo`. `claim` does not implement `PgAuditRepo`'s row-locking
(there is nothing to lock against in a single-process in-memory list) — it
just returns up to `limit` unprocessed items in insertion order, minus any
that have already failed `MAX_ATTEMPTS` times, which is this port's contract
as far as any test needs.
"""

from __future__ import annotations

from sarjy.contexts.guardrails.application.ports import AuditItem


class MemAuditQueue:
    MAX_ATTEMPTS = 3

    def __init__(self, items: list[AuditItem] | None = None) -> None:
        self._items: dict[int, AuditItem] = {item.id: item for item in (items or [])}
        self._processed: set[int] = set()
        self._attempts: dict[int, int] = {}

    def add(self, item: AuditItem) -> None:
        self._items[item.id] = item

    @property
    def processed_ids(self) -> list[int]:
        return sorted(self._processed)

    def attempts(self, item_id: int) -> int:
        return self._attempts.get(item_id, 0)

    async def claim(self, limit: int) -> list[AuditItem]:
        pending = [
            item
            for iid, item in sorted(self._items.items())
            if iid not in self._processed and self.attempts(iid) < self.MAX_ATTEMPTS
        ]
        return pending[:limit]

    async def mark_processed(self, ids: list[int]) -> None:
        self._processed.update(ids)

    async def mark_failed(self, ids: list[int]) -> None:
        # Mirrors `PgAuditRepo._MARK_FAILED` including the retry cap it feeds:
        # a double that retried forever would let a worker bug that never marks
        # anything processed still look like progress in a test.
        for i in ids:
            self._attempts[i] = self.attempts(i) + 1
