"""Repo for the `GET /admin/latency` dashboard (PRD §13) — reads the
`v_latency_daily` / `v_latency_by_browser` / `v_guard_daily` / `v_ocean_funnel`
views (see `supabase/migrations/20260821000600_latency_views.sql`).
"""

from __future__ import annotations

import datetime
from typing import Any, Protocol

import asyncpg

from sarjy.infrastructure_shared.db import Database


class AdminLatencyRepo(Protocol):
    async def latency_daily(self) -> list[dict[str, Any]]: ...
    async def latency_by_browser(self) -> list[dict[str, Any]]: ...
    async def guard_daily(self) -> list[dict[str, Any]]: ...
    async def ocean_funnel(self) -> list[dict[str, Any]]: ...


def _rows(records: list[asyncpg.Record]) -> list[dict[str, Any]]:
    """`asyncpg.Record` -> `dict`, with any `date`/`datetime` column rendered as ISO text."""
    out = []
    for r in records:
        row = dict(r)
        for k, v in row.items():
            if isinstance(v, datetime.date):  # covers datetime.datetime too (it subclasses date)
                row[k] = v.isoformat()
        out.append(row)
    return out


class PgAdminLatencyRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def latency_daily(self) -> list[dict[str, Any]]:
        return _rows(await self.db.fetch("select * from public.v_latency_daily"))

    async def latency_by_browser(self) -> list[dict[str, Any]]:
        return _rows(await self.db.fetch("select * from public.v_latency_by_browser"))

    async def guard_daily(self) -> list[dict[str, Any]]:
        return _rows(await self.db.fetch("select * from public.v_guard_daily"))

    async def ocean_funnel(self) -> list[dict[str, Any]]:
        return _rows(await self.db.fetch("select * from public.v_ocean_funnel"))
