from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg


class Database:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10) -> None:
        self._dsn = dsn
        self._min, self._max = min_size, max_size
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        # statement_cache_size=0 is required behind pgbouncer transaction pooling
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=self._min, max_size=self._max, statement_cache_size=0
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    @property
    def pool(self) -> asyncpg.Pool:
        assert self._pool is not None, "Database.connect() not called"
        return self._pool

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        async with self.pool.acquire() as conn:
            yield conn

    async def fetch(self, q: str, *args: Any) -> list[asyncpg.Record]:
        return await self.pool.fetch(q, *args)  # type: ignore[no-any-return]

    async def fetchrow(self, q: str, *args: Any) -> asyncpg.Record | None:
        return await self.pool.fetchrow(q, *args)

    async def fetchval(self, q: str, *args: Any) -> Any:
        return await self.pool.fetchval(q, *args)

    async def execute(self, q: str, *args: Any) -> str:
        return await self.pool.execute(q, *args)  # type: ignore[no-any-return]
