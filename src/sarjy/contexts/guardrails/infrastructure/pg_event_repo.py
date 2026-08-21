from __future__ import annotations

import json
from typing import Any

from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import MessageId, UserId


class PgGuardEventRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def record(
        self,
        *,
        user_id: UserId | None,
        message_id: MessageId | None,
        layer: int,
        kind: str,
        action: str,
        severity: int,
        detail: dict[str, Any],
    ) -> None:
        await self.db.execute(
            "insert into guardrail_events "
            "(user_id,message_id,layer,kind,action,severity,detail) "
            "values ($1,$2,$3,$4,$5,$6,$7::jsonb)",
            user_id,
            message_id,
            layer,
            kind,
            action,
            severity,
            json.dumps(detail, default=str),
        )
