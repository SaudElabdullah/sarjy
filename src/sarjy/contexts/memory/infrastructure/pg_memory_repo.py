from __future__ import annotations

import dataclasses

import asyncpg

from sarjy.contexts.memory.domain.memory import Memory, MemoryChanged, MemoryKind
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.errors import NotFound
from sarjy.shared.ids import MemoryId, MessageId, UserId

_COLS = "id,user_id,key,value,kind,created_at,updated_at,deleted_at,source_message_id"


def _rowcount(status: str) -> int:
    """Parse asyncpg's execute() status string (e.g. "UPDATE 1") into a row count."""
    return int(status.rsplit(" ", 1)[-1])


def _row_to_memory(r: asyncpg.Record) -> Memory:
    kind: MemoryKind = r["kind"]
    return Memory(
        id=MemoryId(r["id"]),
        user_id=UserId(r["user_id"]),
        key=r["key"],
        value=r["value"],
        kind=kind,
        created_at=r["created_at"],
        updated_at=r["updated_at"],
        deleted_at=r["deleted_at"],
        source_message_id=MessageId(r["source_message_id"]) if r["source_message_id"] else None,
    )


class PgMemoryRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def get_by_key(self, user_id: UserId, key: str) -> Memory | None:
        r = await self.db.fetchrow(
            f"select {_COLS} from memories where user_id=$1 and key=$2 and deleted_at is null",  # noqa: S608
            user_id,
            key,
        )
        return _row_to_memory(r) if r else None

    async def get_by_id(self, user_id: UserId, id: MemoryId) -> Memory | None:
        r = await self.db.fetchrow(
            f"select {_COLS} from memories where user_id=$1 and id=$2 and deleted_at is null",  # noqa: S608
            user_id,
            id,
        )
        return _row_to_memory(r) if r else None

    async def list_live(self, user_id: UserId, limit: int = 60) -> list[Memory]:
        rows = await self.db.fetch(
            f"select {_COLS} from memories where user_id=$1 and deleted_at is null "  # noqa: S608
            "order by updated_at desc limit $2",
            user_id,
            limit,
        )
        return [_row_to_memory(r) for r in rows]

    @staticmethod
    async def _write(conn: asyncpg.Connection, m: Memory) -> MemoryId:
        # `memories_user_key_live` is a *partial* unique index on (user_id, key)
        # (live rows only), so this can't be a plain `insert ... on conflict
        # (user_id, key)`. Locking this memory's own row by id is not enough
        # on its own: two concurrent CREATEs for the same brand-new key each
        # get a no-op lock (neither row exists yet) and would otherwise race
        # to insert, tripping the partial unique index. So for a keyed write
        # we first take a per-(user_id, key) advisory lock — held for the
        # rest of this transaction — to serialise writers on that key, then
        # re-check for a live row under that key. If one now exists under a
        # *different* id (another writer won the race first), fold this
        # write into that row instead of inserting a duplicate, and report
        # its id as the winner so history is recorded against it. Otherwise
        # fall back to the simple lock-by-own-id path (covers updates, and
        # first-writer creates/soft-deletes).
        if m.key is not None:
            await conn.execute(
                "select pg_advisory_xact_lock(hashtext($1 || ':' || $2))",
                str(m.user_id),
                m.key,
            )
            live_by_key = await conn.fetchrow(
                "select id from memories where user_id=$1 and key=$2 and deleted_at is null "
                "for update",
                m.user_id,
                m.key,
            )
            if live_by_key is not None and live_by_key["id"] != m.id:
                winning_id = MemoryId(live_by_key["id"])
                await conn.execute(
                    "update memories set value=$2, kind=$3::memory_kind, updated_at=$4 where id=$1",
                    winning_id,
                    m.value,
                    m.kind,
                    m.updated_at,
                )
                return winning_id

        live = await conn.fetchrow("select id from memories where id=$1 for update", m.id)
        if live is not None:
            # Scoped to user_id (defence in depth: a write should never cross users)
            # and guarded so a *live* write (m.deleted_at is None) can never clear an
            # already-set deleted_at — i.e. never resurrect a row that a concurrent
            # forget() soft-deleted between this write's read and this write. A
            # write that is itself a soft-delete (m.deleted_at is not None) is exempt
            # from that guard so forget() stays idempotent. If the guard excludes the
            # row (or the id/user_id no longer match), 0 rows are affected and that's
            # reported as NotFound rather than silently doing nothing.
            status = await conn.execute(
                "update memories set key=$2, value=$3, kind=$4::memory_kind, "
                "updated_at=$5, deleted_at=$6 where id=$1 and user_id=$7 "
                "and (deleted_at is null or $6::timestamptz is not null)",
                m.id,
                m.key,
                m.value,
                m.kind,
                m.updated_at,
                m.deleted_at,
                m.user_id,
            )
            if _rowcount(status) == 0:
                raise NotFound(f"memory {m.id} not found")
        else:
            await conn.execute(
                "insert into memories "
                "(id,user_id,key,value,kind,created_at,updated_at,deleted_at,source_message_id) "
                "values ($1,$2,$3,$4,$5::memory_kind,$6,$7,$8,$9)",
                m.id,
                m.user_id,
                m.key,
                m.value,
                m.kind,
                m.created_at,
                m.updated_at,
                m.deleted_at,
                m.source_message_id,
            )
        return m.id

    @staticmethod
    async def _write_history(conn: asyncpg.Connection, ev: MemoryChanged) -> None:
        await conn.execute(
            "insert into memories_history (memory_id,user_id,old_value,new_value,action,at) "
            "values ($1,$2,$3,$4,$5,$6)",
            ev.memory_id,
            ev.user_id,
            ev.old_value,
            ev.new_value,
            ev.action,
            ev.occurred_at,
        )

    async def upsert(self, m: Memory) -> None:
        async with self.db.acquire() as conn, conn.transaction():
            await self._write(conn, m)

    async def soft_delete(self, m: Memory) -> None:
        # `updated_at` has no DB-side trigger, so it must be written explicitly
        # on every mutation, soft-delete included.
        await self.db.execute(
            "update memories set deleted_at=$3, updated_at=$4 where id=$1 and user_id=$2",
            m.id,
            m.user_id,
            m.deleted_at,
            m.updated_at,
        )

    async def append_history(self, ev: MemoryChanged) -> None:
        async with self.db.acquire() as conn:
            await self._write_history(conn, ev)

    async def delete_history(self, user_id: UserId, memory_id: MemoryId) -> None:
        # Scoped by user_id as well as memory_id — defence in depth, the same
        # way `_write`'s UPDATE is: a history purge must never cross users even
        # if a caller hands over an id that isn't theirs.
        await self.db.execute(
            "delete from public.memories_history where user_id=$1 and memory_id=$2",
            user_id,
            memory_id,
        )

    async def upsert_with_history(self, m: Memory, events: list[MemoryChanged]) -> MemoryId:
        # Same transaction as the write: a history row with no matching
        # memory mutation (or vice versa) is the partial-failure gap flagged
        # in the Task 2 review.
        async with self.db.acquire() as conn, conn.transaction():
            winning_id = await self._write(conn, m)
            for ev in events:
                if ev.memory_id != winning_id:
                    # `_write` folded this write into another writer's
                    # already-live row for the same key; the caller's event
                    # still carries its own (losing) memory_id, so re-point
                    # the history row at the row that actually got written.
                    ev = dataclasses.replace(ev, memory_id=winning_id)
                await self._write_history(conn, ev)
        return winning_id
