"""Application-layer ports for the guardrails context.

`InputGuard`/`OutputGuard` (this package) depend only on these Protocols —
never on a concrete classifier or event-store adapter — so infrastructure
(Phase 6+) implements them and the application layer stays swappable/testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from sarjy.contexts.guardrails.domain.categories import Category
from sarjy.shared.ids import MessageId, UserId


@dataclass(frozen=True, slots=True)
class Classification:
    category: Category | None
    is_injection: bool
    severity: int
    confidence: float


class ClassifierPort(Protocol):
    async def classify(self, recent_user_turns: list[str]) -> Classification: ...


class GuardEventRepo(Protocol):
    async def record(
        self,
        *,
        user_id: UserId | None,
        message_id: MessageId | None,
        layer: int,
        kind: str,
        action: str,
        severity: int,
        detail: dict[str, Any],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class AuditItem:
    """One `audit_queue` row, joined with the transcript `AuditWorker` needs.

    `user_text` is the user turn immediately preceding the assistant message —
    the same pairing Layer 3 classifies at request time — resolved by the
    `AuditQueuePort` implementation (a join in `PgAuditRepo`) rather than by
    `AuditWorker`, so the worker stays free of SQL.
    """

    id: int
    message_id: MessageId
    user_id: UserId
    user_text: str
    assistant_text: str


class AuditQueuePort(Protocol):
    async def claim(self, limit: int) -> list[AuditItem]: ...
    async def mark_processed(self, ids: list[int]) -> None: ...
    async def mark_failed(self, ids: list[int]) -> None:
        """Record one failed audit attempt against each item, leaving it unprocessed.

        The item stays claimable — a classifier failure is usually transient
        and the audit is worth retrying — but an implementation MUST stop
        handing out an item that has failed too many times, or an item the
        classifier chokes on permanently is re-fetched and re-charged on
        every scheduled run forever (`PgAuditRepo` caps it at three).
        """
        ...
