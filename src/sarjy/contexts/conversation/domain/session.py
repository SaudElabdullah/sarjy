from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sarjy.shared.ids import SessionId, UserId

SESSION_TTL = timedelta(minutes=30)


@dataclass(slots=True)
class Session:
    id: SessionId
    user_id: UserId
    started_at: datetime
    last_active_at: datetime
    summary: str | None = None

    @classmethod
    def start(cls, id: SessionId, user_id: UserId, now: datetime) -> Session:
        return cls(id=id, user_id=user_id, started_at=now, last_active_at=now)

    def is_expired(self, now: datetime, ttl: timedelta = SESSION_TTL) -> bool:
        return now - self.last_active_at > ttl

    def touch(self, now: datetime) -> None:
        self.last_active_at = now
