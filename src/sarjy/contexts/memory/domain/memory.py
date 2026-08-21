from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from sarjy.contexts.memory.domain.sanitise import sanitise
from sarjy.shared.errors import ValidationError
from sarjy.shared.events import DomainEvent
from sarjy.shared.ids import MemoryId, MessageId, UserId

MemoryKind = Literal["fact", "preference", "person", "place", "note"]
MAX_VALUE_LEN = 200


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryChanged(DomainEvent):
    memory_id: MemoryId
    user_id: UserId
    action: Literal["create", "update", "delete"]
    old_value: str | None
    new_value: str | None


@dataclass(slots=True)
class Memory:
    id: MemoryId
    user_id: UserId
    key: str | None
    value: str
    kind: MemoryKind
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    source_message_id: MessageId | None = None
    _events: list[MemoryChanged] = field(default_factory=list, repr=False, compare=False)

    @staticmethod
    def _clean(value: str) -> str:
        v = sanitise(value, MAX_VALUE_LEN)
        if not v:
            raise ValidationError("memory value is empty")
        if len(value.strip()) > MAX_VALUE_LEN:
            raise ValidationError("memory value exceeds 200 characters")
        return v

    @classmethod
    def create(
        cls,
        id: MemoryId,
        user_id: UserId,
        key: str | None,
        value: str,
        kind: MemoryKind,
        now: datetime,
        source_message_id: MessageId | None = None,
    ) -> Memory:
        v = cls._clean(value)
        m = cls(
            id=id,
            user_id=user_id,
            key=key,
            value=v,
            kind=kind,
            created_at=now,
            updated_at=now,
            source_message_id=source_message_id,
        )
        m._events.append(
            MemoryChanged(
                memory_id=id, user_id=user_id, action="create", old_value=None, new_value=v
            )
        )
        return m

    @property
    def is_live(self) -> bool:
        return self.deleted_at is None

    def update(self, value: str, now: datetime) -> None:
        if not self.is_live:
            raise ValidationError("cannot update a forgotten memory")
        v = self._clean(value)
        old, self.value, self.updated_at = self.value, v, now
        self._events.append(
            MemoryChanged(
                memory_id=self.id, user_id=self.user_id, action="update", old_value=old, new_value=v
            )
        )

    def forget(self, now: datetime) -> None:
        if not self.is_live:
            return
        self.deleted_at = now
        self.updated_at = now
        self._events.append(
            MemoryChanged(
                memory_id=self.id,
                user_id=self.user_id,
                action="delete",
                old_value=self.value,
                new_value=None,
            )
        )

    def pull_events(self) -> list[MemoryChanged]:
        evs, self._events = list(self._events), []
        return evs
