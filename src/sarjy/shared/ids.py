from __future__ import annotations

import uuid
from typing import NewType, TypeVar

UserId = NewType("UserId", uuid.UUID)
SessionId = NewType("SessionId", uuid.UUID)
TurnId = NewType("TurnId", uuid.UUID)
MessageId = NewType("MessageId", uuid.UUID)
RunId = NewType("RunId", uuid.UUID)
MemoryId = NewType("MemoryId", uuid.UUID)

T = TypeVar("T", UserId, SessionId, TurnId, MessageId, RunId, MemoryId)


def new_id(kind: type[T]) -> T:  # noqa: UP047  # kind is a NewType callable
    return kind(uuid.uuid4())


def parse_id(kind: type[T], raw: str) -> T:  # noqa: UP047
    return kind(uuid.UUID(raw))
