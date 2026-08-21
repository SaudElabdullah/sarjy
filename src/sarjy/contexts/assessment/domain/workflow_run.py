from __future__ import annotations

from collections.abc import Container
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sarjy.contexts.assessment.domain.events import (
    AnswerRecorded,
    RunAbandoned,
    RunCompleted,
    RunConfirmed,
    RunPaused,
    RunProposed,
    RunResumed,
)
from sarjy.contexts.assessment.domain.state_machine import Status, next_status
from sarjy.shared.errors import DomainError
from sarjy.shared.events import DomainEvent
from sarjy.shared.ids import RunId, UserId

MAX_SKIPS = 2

__all__ = ["MAX_SKIPS", "IllegalTransition", "Status", "TooManySkips", "WorkflowRun"]


class IllegalTransition(DomainError):  # noqa: N818
    pass


class TooManySkips(DomainError):  # noqa: N818
    pass


@dataclass(slots=True)
class WorkflowRun:
    id: RunId
    user_id: UserId
    definition_id: str
    definition_version: int
    status: Status
    current_item: int
    skips_used: int
    results: dict[str, Any] | None
    narrative: str | None
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    pending_confirmation: dict[str, Any] | None = None
    resume_hint: bool = False
    events: list[DomainEvent] = field(default_factory=list)

    # -- factories ---------------------------------------------------------
    @classmethod
    def propose(
        cls, id: RunId, user_id: UserId, definition_id: str, version: int, now: datetime
    ) -> WorkflowRun:
        r = cls(
            id=id,
            user_id=user_id,
            definition_id=definition_id,
            definition_version=version,
            status=Status.PROPOSED,
            current_item=1,
            skips_used=0,
            results=None,
            narrative=None,
            started_at=now,
            updated_at=now,
        )
        r.events.append(RunProposed(run_id=id))
        return r

    # -- transitions -------------------------------------------------------
    def _go(self, action: str, now: datetime) -> None:
        nxt = next_status(self.status, action)
        if nxt is None:
            raise IllegalTransition(f"{self.status.value} -> {action}")
        self.status = nxt
        self.updated_at = now

    def confirm(self, now: datetime) -> None:
        self._go("confirm", now)
        self.events.append(RunConfirmed(run_id=self.id))

    def record_answer(
        self,
        item_no: int,
        value: int | None,
        raw_text: str,
        confidence: float,
        total_items: int,
        now: datetime,
    ) -> AnswerRecorded:
        if self.status is not Status.ACTIVE:
            raise IllegalTransition(f"{self.status.value} -> answer")
        if item_no != self.current_item:
            raise IllegalTransition(f"expected item {self.current_item}, got {item_no}")
        if value is None:
            if self.skips_used >= MAX_SKIPS:
                raise TooManySkips(f"max {MAX_SKIPS} skips")
            self.skips_used += 1
        elif not 1 <= value <= 5:
            raise IllegalTransition(f"value out of range: {value}")
        self._go("answer", now)
        self.pending_confirmation = None
        self.resume_hint = False
        if self.current_item < total_items + 1:
            self.current_item += 1
        ev = AnswerRecorded(run_id=self.id, item_no=item_no, value=value)
        self.events.append(ev)
        return ev

    def back(self, now: datetime) -> None:
        """Step the cursor back one item.

        Which statuses may do this is the transition table's business, not a
        second rule written here: ACTIVE steps back for an ordinary "go back",
        and SCORING steps back — reopening the run — when the caller finds a
        missing answer it has to ask for again.
        """
        if self.current_item <= 1:
            raise IllegalTransition("cannot go back: already on the first item")
        self._go("back", now)
        self.current_item -= 1
        self.pending_confirmation = None

    def pause(self, now: datetime) -> None:
        self._go("pause", now)
        self.pending_confirmation = None
        self.events.append(RunPaused(run_id=self.id))

    def resume(self, now: datetime) -> None:
        self._go("resume", now)
        self.resume_hint = False
        self.events.append(RunResumed(run_id=self.id))

    def quit(self, now: datetime) -> None:
        self._go("quit", now)
        self.events.append(RunAbandoned(run_id=self.id))

    def missing_answers(self, answered: Container[int], total_items: int) -> list[int]:
        """Item numbers with neither an answer nor a skip recorded, in order.

        The aggregate cannot see the answer rows itself, so the caller passes
        what it has — a mapping keyed by item number, or any container of
        recorded numbers. Exposed as a question as well as enforced by
        `begin_scoring` below, because the caller needs the answer *before* it
        starts the expensive tail of a run (scoring, then narration): asking
        and being told "item seven" is how it steers back to item seven
        instead of doing all that work only to throw it away.
        """
        return [n for n in range(1, total_items + 1) if n not in answered]

    def begin_scoring(self, now: datetime, answered: Container[int], total_items: int) -> None:
        """Move to SCORING, but only once every item has an answer or a skip.

        `score()` treats a missing item exactly like an unanswered one, so a run
        that reaches scoring with a gap produces a lower trait mean (or a
        silently unscored trait) that reads like a real result.
        """
        missing = self.missing_answers(answered, total_items)
        if missing:
            raise IllegalTransition(f"cannot score: items {missing} have no answer")
        self._go("score", now)

    def finish_scoring(self, results: dict[str, Any], narrative: str, now: datetime) -> None:
        self._go("finish", now)
        self.results, self.narrative, self.completed_at = results, narrative, now
        self.events.append(RunCompleted(run_id=self.id))

    # -- helpers -----------------------------------------------------------
    def is_finished_answering(self, total_items: int) -> bool:
        return self.current_item > total_items

    def set_pending(self, item_no: int, value: int, raw_text: str) -> None:
        self.pending_confirmation = {"item_no": item_no, "value": value, "raw_text": raw_text}

    def clear_pending(self) -> None:
        self.pending_confirmation = None
