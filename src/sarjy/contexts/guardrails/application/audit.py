"""Application service: Layer 7 async audit sampling (PRD Layer 7).

`enqueue_audit_sample` — a Postgres trigger, `supabase/migrations/
20260821000800_audit_sample.sql` — queues ~20% of ALLOWED assistant turns
into `audit_queue` at insert time. `AuditWorker` is the consumer: `run_once`
claims a batch, re-classifies each (user, assistant) pair with the same
`ClassifierPort` Layer 3 escalates to, records a `guardrail_events` row either
way (`layer=7`), and marks the item processed.

This is sampling, not gating (I1 does not apply): nothing here is on a live
turn's critical path, so unlike `InputGuard` there is no fail-closed to
implement — a classifier failure just leaves the item in the queue for the
next scheduled run rather than blocking anything. That is also why a failure
on one item must not abort the batch: `AuditQueuePort.claim` already pulled
every item in the batch out of "unclaimed" (the `for update skip locked`
happens once, in the claim query), so an item skipped here without being
marked processed is retried later rather than lost, and the items after it in
the batch still deserve their audit.

Retried, but not forever: each failure is counted via
`AuditQueuePort.mark_failed`, and an item that has failed three times stops
being claimed. Unbounded retry of an item the classifier fails on every time
is a standing cost (one classifier call per item per ten-minute run) for an
audit that is never going to succeed.
"""

from __future__ import annotations

from dataclasses import dataclass

from sarjy.contexts.guardrails.application.ports import (
    AuditQueuePort,
    ClassifierPort,
    GuardEventRepo,
)
from sarjy.observability.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AuditRunResult:
    processed: int
    flagged: int


class AuditWorker:
    def __init__(
        self, queue: AuditQueuePort, classifier: ClassifierPort, events: GuardEventRepo
    ) -> None:
        self.queue = queue
        self.classifier = classifier
        self.events = events

    async def run_once(self, limit: int = 50) -> AuditRunResult:
        items = await self.queue.claim(limit)
        processed_ids: list[int] = []
        failed_ids: list[int] = []
        flagged = 0
        for item in items:
            try:
                c = await self.classifier.classify([item.user_text, item.assistant_text])
            except Exception:
                # Same posture as `InputGuard`'s classifier-failure handling,
                # minus the fail-closed: there is no live turn to block, so the
                # item is not marked processed — `claim` will hand it out again
                # on the next scheduled run instead of the audit being lost.
                # It IS counted as a failed attempt though: leaving the row
                # completely untouched (what this did before) means an item the
                # classifier fails on every time is re-claimed and re-charged
                # every ten minutes forever, at the head of every batch.
                log.exception("audit_classify_failed", message_id=str(item.message_id))
                failed_ids.append(item.id)
                continue

            is_flagged = c.is_injection or c.category is not None
            category = c.category or ("injection" if c.is_injection else "clean")
            await self.events.record(
                user_id=item.user_id,
                message_id=item.message_id,
                layer=7,
                kind=f"audit:{category}",
                action="allow_flagged" if is_flagged else "audit_clean",
                severity=c.severity,
                detail={"confidence": c.confidence},
            )
            if is_flagged:
                flagged += 1
            processed_ids.append(item.id)

        if processed_ids:
            await self.queue.mark_processed(processed_ids)
        if failed_ids:
            await self.queue.mark_failed(failed_ids)
        return AuditRunResult(processed=len(processed_ids), flagged=flagged)
