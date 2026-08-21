"""`EditFact` — the use case behind `PATCH /memory/{id}` (PRD §9.2).

Phase 8 Task 6b fix round 1, Critical 1: the PATCH route used to write
through `MemoryRepo.upsert_with_history` directly after only the PII filter
(`reject_reason`), so a value the guardrail rule engine would refuse on the
`remember` tool's path went unscreened when set through the REST edit
endpoint instead. `EditFact` is now the one place both entry points share —
it applies the exact same PII filter + `ValueScreenPort` screening
`RememberFact` does (via `screen_reason`), so a rule-engine-refusable value
is rejected the same way regardless of which door it comes through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sarjy.contexts.memory.application.ports import MemoryRepo, ValueScreenPort
from sarjy.contexts.memory.application.screening import screen_reason
from sarjy.contexts.memory.domain.memory import Memory
from sarjy.contexts.memory.domain.pii_filter import reject_reason
from sarjy.shared.clock import Clock
from sarjy.shared.errors import NotFound, ValidationError
from sarjy.shared.ids import MemoryId, UserId


@dataclass(frozen=True, slots=True)
class EditOutcome:
    status: Literal["updated", "not_found", "rejected"]
    memory: Memory | None = None
    reason: str | None = None


class EditFact:
    def __init__(self, repo: MemoryRepo, clock: Clock, screen: ValueScreenPort) -> None:
        self.repo, self.clock, self.screen = repo, clock, screen

    async def __call__(self, user_id: UserId, id: MemoryId, value: str) -> EditOutcome:
        m = await self.repo.get_by_id(user_id, id)
        if m is None:
            return EditOutcome("not_found")
        reason = reject_reason(value, key=m.key)
        if reason:
            return EditOutcome("rejected", reason=reason)
        screen_refusal = screen_reason(
            self.screen, m.key or "", value, user_id=user_id, message_id=None
        )
        if screen_refusal:
            return EditOutcome("rejected", reason=screen_refusal)
        try:
            m.update(value, self.clock.now())
        except ValidationError as e:
            return EditOutcome("rejected", reason=str(e))
        try:
            await self.repo.upsert_with_history(m, m.pull_events())
        except NotFound:
            # The row existed at the `get_by_id` above but was forgotten
            # (soft-deleted) by someone else before this write landed — report
            # it the same as if it had never been found, rather than
            # resurrecting it.
            return EditOutcome("not_found")
        return EditOutcome("updated", m)
