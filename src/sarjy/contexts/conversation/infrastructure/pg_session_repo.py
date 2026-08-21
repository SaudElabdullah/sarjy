from __future__ import annotations

from typing import Any

from sarjy.contexts.conversation.domain.session import Session
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import SessionId, UserId


def _row_to_session(r: Any) -> Session:
    return Session(
        SessionId(r["id"]),
        UserId(r["user_id"]),
        r["started_at"],
        r["last_active_at"],
        r["summary"],
    )


class PgSessionRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def get(self, id: SessionId) -> Session | None:
        r = await self.db.fetchrow(
            "select id,user_id,started_at,last_active_at,summary from sessions where id=$1", id
        )
        return _row_to_session(r) if r else None

    async def latest_for_user(self, user_id: UserId) -> Session | None:
        r = await self.db.fetchrow(
            "select id,user_id,started_at,last_active_at,summary from sessions "
            "where user_id=$1 order by last_active_at desc limit 1",
            user_id,
        )
        return _row_to_session(r) if r else None

    async def save(self, s: Session) -> None:
        await self.db.execute(
            """insert into sessions (id,user_id,started_at,last_active_at,summary)
               values ($1,$2,$3,$4,$5)
               on conflict (id) do update set
                 last_active_at=excluded.last_active_at,
                 -- a touch-only save carries summary=NULL; coalesce keeps whatever
                 -- summariser output is already stored instead of erasing it
                 summary=coalesce(excluded.summary, sessions.summary)""",
            s.id,
            s.user_id,
            s.started_at,
            s.last_active_at,
            s.summary,
        )
