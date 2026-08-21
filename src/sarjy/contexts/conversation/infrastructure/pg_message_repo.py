from __future__ import annotations

import json
from typing import Any

from sarjy.contexts.conversation.domain.message import Message
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import MessageId, SessionId, UserId


class PgMessageRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def history(self, user_id: UserId, session_id: SessionId, limit: int) -> list[Message]:
        rows = await self.db.fetch(
            """select * from (select id,session_id,user_id,role,content,created_at,speech_content,
               client_turn_id,guard_decision
               from messages where session_id=$1 and user_id=$2
               and role in ('user','assistant')
               order by created_at desc, id desc limit $3) h order by created_at, id""",
            session_id,
            user_id,
            limit,
        )
        return [
            Message(
                MessageId(r["id"]),
                SessionId(r["session_id"]),
                UserId(r["user_id"]),
                r["role"],
                r["content"],
                r["created_at"],
                r["speech_content"],
                r["client_turn_id"],
                r["guard_decision"],
            )
            for r in rows
        ]

    async def save(self, m: Message) -> None:
        await self.db.execute(
            """insert into messages
               (id,session_id,user_id,role,content,speech_content,client_turn_id,
                guard_decision,timings,prompt_hash,created_at)
               values ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11)
               on conflict do nothing""",
            m.id,
            m.session_id,
            m.user_id,
            m.role,
            m.content,
            m.speech_content,
            m.client_turn_id,
            m.guard_decision,
            json.dumps(m.timings) if m.timings else None,
            m.prompt_hash,
            m.created_at,
        )

    async def save_tool_call(
        self,
        message_id: MessageId,
        user_id: UserId,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        status: str,
        latency_ms: int,
    ) -> None:
        await self.db.execute(
            """insert into tool_calls (message_id,user_id,tool_name,args,result,status,latency_ms)
               values ($1,$2,$3,$4::jsonb,$5::jsonb,$6,$7)""",
            message_id,
            user_id,
            tool_name,
            json.dumps(args),
            json.dumps(result, default=str),
            status,
            latency_ms,
        )
