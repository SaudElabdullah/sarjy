"""Application service: Layer 2 (rules) + Layer 3 (classifier) input guard.

`InputGuard.check` implements `InputGuardPort` (see
`sarjy.contexts.conversation.application.ports`). It runs the deterministic
rule engine first — a `block` there never reaches the classifier, and an
`allow` returns immediately without writing an event row (keeps the guard
event table small: only the ambiguous/blocked traffic is logged). An
`uncertain` rule verdict escalates to the Layer-3 classifier under a timeout;
a timeout — or any other classifier failure (malformed JSON, a reply that
doesn't match the schema, the provider being down) — fails CLOSED (blocks)
per G-12 rather than silently letting an unclassified message through.

Event writes never sit on the critical path. A block is a decision the
caller is waiting on — the user is mid-utterance and the refusal is the
next thing they hear — so `_finish` schedules the guardrail_events write
as a background task and returns immediately. That also means an event
store that is down, slow, or throwing cannot turn a working guard into a
hung or failing one: the exception is logged inside the task and the
decision has already gone out. `drain()` (mirroring `OutputGuard.drain`)
lets tests and the `Container` on shutdown wait for those writes to land.

In `shadow` mode nothing is actually blocked — every would-be block is
logged with `action="shadow_block"` and the caller still gets `allow` — so
new rules/classifier behaviour can be observed against real traffic before
being flipped to `enforce`.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from sarjy.contexts.guardrails.application.ports import ClassifierPort, GuardEventRepo
from sarjy.contexts.guardrails.domain.categories import SEVERITY
from sarjy.contexts.guardrails.domain.decision import GuardDecision
from sarjy.contexts.guardrails.domain.normalize import normalize_variants
from sarjy.contexts.guardrails.domain.rules import RuleEngine
from sarjy.observability.logging import get_logger
from sarjy.shared.ids import MessageId, UserId

log = get_logger(__name__)
GuardMode = Literal["enforce", "shadow"]


def _elapsed_ms(started: float | None) -> int:
    return 0 if started is None else int((time.perf_counter() - started) * 1000)


def _spec(speculative: bool) -> dict[str, bool]:
    """The `speculative` stamp, present only when it is true (L-3).

    Absent rather than `false` on an ordinary turn: every row written before
    this existed has no such key, and a query for the flag should not have to
    distinguish "no" from "before we recorded it".
    """
    return {"speculative": True} if speculative else {}


class InputGuard:
    def __init__(
        self,
        rules: RuleEngine,
        classifier: ClassifierPort,
        events: GuardEventRepo,
        mode: GuardMode = "enforce",
        classifier_timeout_s: float = 0.4,
    ) -> None:
        self.rules = rules
        self.clf = classifier
        self.events = events
        self.mode: GuardMode = mode
        self.timeout = classifier_timeout_s
        self._pending: set[asyncio.Task[None]] = set()

    async def check(
        self,
        user_id: UserId,
        text: str,
        recent_user_turns: list[str],
        message_id: MessageId | None = None,
        speculative: bool = False,
    ) -> GuardDecision:
        # Wall clock from the top of the guard, so `latency_ms` on the recorded
        # event covers what the caller actually waited for — rule matching plus,
        # when it happened, the Layer-3 round trip (I4). Without it a slow guard
        # is invisible in the audit trail: the row says a message was blocked,
        # never that deciding so took 400ms of the user's turn.
        started = time.perf_counter()
        # evaluate_variants (NOT evaluate(normalize(text))) — the joined,
        # " | "-separated single-string form has a literal recall hole: a
        # rule's `.{0,N}` gap can bridge the seam between two variants and
        # match a phrase present in neither.
        d = self.rules.evaluate_variants(normalize_variants(text))
        if d.action == "block":
            return await self._finish(
                user_id,
                d,
                kind=f"rule:{d.rule_id}",
                message_id=message_id,
                started=started,
                speculative=speculative,
            )
        if d.action == "allow":
            return d

        # uncertain → escalate to the Layer-3 classifier, last 4 user turns
        # (including this one) as context. `recent_user_turns` MAY already
        # end with `text` (a caller — e.g. RunTurn — that appends the
        # current message to its own history before calling `check`) or MAY
        # NOT (a caller that passes only prior turns); handle both without
        # duplicating `text` in the window handed to the classifier.
        if recent_user_turns and recent_user_turns[-1] == text:
            turns = recent_user_turns[-4:]
        else:
            turns = [*recent_user_turns, text][-4:]
        try:
            c = await asyncio.wait_for(self.clf.classify(turns), timeout=self.timeout)
        except TimeoutError:
            # Fail closed (G-12): an unresponsive classifier must not let an
            # ambiguous message through silently.
            return await self._fail_closed(
                user_id, d, "classifier_timeout", started, message_id, speculative
            )
        except Exception:
            # Same contract as the timeout above, for the other ways a real
            # classifier fails: malformed JSON (json.JSONDecodeError), a reply
            # that doesn't fit the schema (pydantic ValidationError), or the
            # provider being down/erroring (LLMUnavailable, LLMTimeout). A
            # broken classifier is an absent classifier, and an absent
            # classifier must not turn "uncertain" into "allow".
            log.exception("guard_classifier_error")
            return await self._fail_closed(
                user_id, d, "classifier_error", started, message_id, speculative
            )

        if c.is_injection or (c.category is not None and c.confidence >= 0.6):
            cat = "injection" if c.is_injection and not c.category else c.category
            blocked = GuardDecision(
                "block", cat, layer=3, rule_id=d.rule_id, severity=SEVERITY.get(cat or "", 1)
            )
            return await self._finish(
                user_id,
                blocked,
                kind=f"classifier:{cat}",
                detail={"confidence": c.confidence},
                message_id=message_id,
                started=started,
                speculative=speculative,
            )

        self._record(
            user_id=user_id,
            message_id=message_id,
            layer=3,
            kind="classifier:allow",
            action="allow_flagged",
            severity=0,
            detail={
                "rule": d.rule_id,
                "confidence": c.confidence,
                "latency_ms": _elapsed_ms(started),
                **_spec(speculative),
            },
        )
        return GuardDecision("allow", layer=3)

    async def _fail_closed(
        self,
        user_id: UserId,
        d: GuardDecision,
        kind: str,
        started: float,
        message_id: MessageId | None = None,
        speculative: bool = False,
    ) -> GuardDecision:
        """Turn an unusable classifier verdict into a block (G-12).

        The rule engine's own category/severity carry over, so the caller still
        gets a refusal template matched to what the rules suspected rather than
        a bare generic one.
        """
        blocked = GuardDecision(
            "block", d.category, layer=3, rule_id=d.rule_id, severity=d.severity
        )
        return await self._finish(
            user_id,
            blocked,
            kind=kind,
            message_id=message_id,
            started=started,
            speculative=speculative,
        )

    async def _finish(
        self,
        user_id: UserId,
        d: GuardDecision,
        kind: str,
        detail: dict[str, Any] | None = None,
        message_id: MessageId | None = None,
        started: float | None = None,
        speculative: bool = False,
    ) -> GuardDecision:
        action = "block" if self.mode == "enforce" else "shadow_block"
        self._record(
            user_id=user_id,
            message_id=message_id,
            layer=d.layer,
            kind=kind,
            action=action,
            severity=d.severity,
            detail={
                "category": d.category,
                "latency_ms": _elapsed_ms(started),
                **_spec(speculative),
                **(detail or {}),
            },
        )
        if self.mode == "enforce":
            return d
        return GuardDecision("allow", d.category, layer=d.layer, rule_id=d.rule_id)

    def _record(self, **kw: Any) -> None:
        """Write a guard event without making the caller wait for it (I1).

        The decision is already made by the time this runs; holding the turn
        open for a database round trip only adds latency, and holding it open
        for a database that is *down* turns an audit-trail failure into a
        conversation failure. Scheduled, tracked in `_pending`, drained on
        shutdown.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop (sync test code): nothing to schedule on, so drop the
            # write rather than raise from a code path that only logs.
            return
        task = loop.create_task(self._safe_record(**kw))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _safe_record(self, **kw: Any) -> None:
        try:
            await self.events.record(**kw)
        except Exception:
            # An unrecorded guard event is a gap in the audit trail; an
            # exception escaping a background task is an unhandled-task
            # warning and a lost decision. Log and move on.
            log.exception("guard_event_record_failed")

    async def drain(self) -> None:
        """Await every in-flight event write. Mirrors `OutputGuard.drain`.

        Exceptions are collected, not propagated: one failed write must not
        stop the rest from being awaited (they are already caught inside
        `_safe_record`, so this is belt and braces for a task cancelled
        mid-flight).
        """
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)
