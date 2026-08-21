"""Adapter: screens a `RememberFact`/`EditFact` key or value through the Layer-2 rule engine.

Implements `sarjy.contexts.memory.application.ports.ValueScreenPort`. Lives in
`guardrails.infrastructure` (not in `memory`) because the memory context must
not import the guardrails context directly (DDD boundary) — `Container` wires
this adapter in behind the port instead (see `Container.rebuild_memory`).

Only the rule engine (Layer 2) is used, not the Layer-3 classifier: screening
a stored fact happens outside the request/response turn the classifier's
timeout budget is built for, and `RuleEngine` alone is already precise enough
to catch a block-severity payload smuggled into a "remember that ..." value —
that is exactly the gap this adapter closes (Phase 5 ledger deferral,
Phase 8 Task 6b). Any `block` OR `uncertain` verdict is treated as refused:
there is no Layer-3 classifier in this path to resolve an `uncertain` verdict,
and a value ambiguous enough to reach the classifier on a normal turn is not
safe to store verbatim and re-inject into every later prompt.

`honor_memory_set_frame=False` (fix round 1, Important 2): the memory-set
frame carve-out (`rules._MEMORY_SET_FRAME`) exists for `InputGuard` screening
a whole chat utterance, where a quoted `"remember that my X is '...'"` value
really is data, not an instruction. Here the string being screened IS the
value/key being stored — never the framing around it (`ports.ValueScreenPort`
requires the caller to screen the key and the value separately, never
concatenated) — so the carve-out must never fire, or a key that happens to
normalise to "note"/"remember"/"save"/"store" could reconstruct the frame at
screening time and smuggle an uncertain-severity payload past this screen.
"""

from __future__ import annotations

import asyncio

from sarjy.contexts.guardrails.application.ports import GuardEventRepo
from sarjy.contexts.guardrails.domain.decision import GuardDecision
from sarjy.contexts.guardrails.domain.normalize import normalize_variants
from sarjy.contexts.guardrails.domain.rules import RuleEngine
from sarjy.contexts.memory.application.ports import ScreenVerdict
from sarjy.infrastructure_shared.background import BackgroundTasks
from sarjy.observability.logging import get_logger
from sarjy.shared.ids import MessageId, UserId

log = get_logger(__name__)


class RuleEngineValueScreen:
    def __init__(
        self,
        engine: RuleEngine,
        events: GuardEventRepo | None = None,
        bg: BackgroundTasks | None = None,
    ) -> None:
        """`events`/`bg` are optional so a caller with no audit trail to write to
        (a bare unit test constructing this adapter directly) still gets a
        working screen — the guardrail_events write is best-effort
        observability (fix round 1, Minor 4), never load-bearing for the
        verdict itself. `Container.rebuild_memory` always supplies both.
        """
        self._engine = engine
        self._events = events
        self._bg = bg

    def screen(
        self, text: str, *, user_id: UserId | None = None, message_id: MessageId | None = None
    ) -> ScreenVerdict:
        decision = self._engine.evaluate_variants(
            normalize_variants(text), honor_memory_set_frame=False
        )
        if decision.action not in ("block", "uncertain"):
            return ScreenVerdict(allowed=True)
        self._record_refusal(decision, user_id, message_id)
        return ScreenVerdict(allowed=False, reason=decision.rule_id or decision.category)

    def _record_refusal(
        self, decision: GuardDecision, user_id: UserId | None, message_id: MessageId | None
    ) -> None:
        """Fire-and-forget a `guardrail_events` row, mirroring `InputGuard._record`.

        Best-effort: a missing event repo/task queue, or no running loop (sync
        test code), simply skips the write rather than raising from a call
        whose return value (the `ScreenVerdict`) has already been decided.
        """
        if self._events is None or self._bg is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._bg.spawn(self._safe_record(decision, user_id, message_id))

    async def _safe_record(
        self, decision: GuardDecision, user_id: UserId | None, message_id: MessageId | None
    ) -> None:
        assert self._events is not None
        try:
            await self._events.record(
                user_id=user_id,
                message_id=message_id,
                layer=2,
                kind=f"memory_write:{decision.rule_id}",
                action="refuse",
                severity=decision.severity,
                detail={"category": decision.category},
            )
        except Exception:
            # An unrecorded guard event is a gap in the audit trail, not a
            # failed write to memory — the ScreenVerdict was already returned
            # to the caller by the time this runs.
            log.exception("memory_write_guard_event_failed")
