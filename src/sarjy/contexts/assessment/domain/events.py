from __future__ import annotations

from dataclasses import dataclass

from sarjy.shared.events import DomainEvent
from sarjy.shared.ids import RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class RunProposed(DomainEvent):
    run_id: RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfirmed(DomainEvent):
    run_id: RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerRecorded(DomainEvent):
    run_id: RunId
    item_no: int
    value: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunPaused(DomainEvent):
    run_id: RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class RunResumed(DomainEvent):
    run_id: RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class RunAbandoned(DomainEvent):
    run_id: RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCompleted(DomainEvent):
    run_id: RunId
