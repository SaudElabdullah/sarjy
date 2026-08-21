"""`ContextLoaderPort` over the `load_turn_context` RPC — one round trip (L-7).

The whole adapter is one `fetchval`. Everything interesting about it is in
`context_from_rpc` (the mapping) and in the migration (the query), which is the
point: the round trip is what this class exists to spend exactly once.
"""

from __future__ import annotations

import json

from sarjy.contexts.conversation.application.context_loader import TurnContext, context_from_rpc
from sarjy.contexts.conversation.application.ports import ActiveRunPort
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import SessionId, UserId


class PgContextLoader:
    def __init__(self, db: Database, runs: ActiveRunPort) -> None:
        self.db, self.runs = db, runs

    async def load(self, user_id: UserId, session_id: SessionId, history_limit: int) -> TurnContext:
        raw = await self.db.fetchval(
            "select public.load_turn_context($1,$2,$3)", user_id, session_id, history_limit
        )
        # asyncpg hands jsonb back as text unless a codec is registered on the
        # connection, and the pool here registers none — see `PgRunRepo._j`.
        return context_from_rpc(
            json.loads(raw) if isinstance(raw, str) else (raw or {}),
            user_id,
            session_id,
            self.runs,
        )
