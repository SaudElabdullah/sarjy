"""Repos for `telemetry_turns` (PRD §9.4, L-1/L-2) — Postgres-backed and in-memory."""

from __future__ import annotations

import json
from typing import Any, Protocol

from sarjy.infrastructure_shared.db import Database


class TelemetryRepo(Protocol):
    async def save(self, **kw: Any) -> None: ...


class PgTelemetryRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def save(self, **kw: Any) -> None:
        await self.db.execute(
            """insert into telemetry_turns
               (user_id,message_id,ttfa_ms,t_request_ms,t_first_byte_ms,t_first_sentence_ms,
                t_last_audio_ms,server_timings,client_info)
               values ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb)""",
            kw["user_id"],
            kw["message_id"],
            kw["ttfa_ms"],
            kw["t_request_ms"],
            kw["t_first_byte_ms"],
            kw["t_first_sentence_ms"],
            kw["t_last_audio_ms"],
            json.dumps(kw["server_timings"]),
            json.dumps(kw["client_info"]),
        )


class MemTelemetry:
    """In-memory stand-in for `PgTelemetryRepo` — used whenever `connect_db` is false."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def save(self, **kw: Any) -> None:
        self.rows.append(kw)
