import json
import os
import uuid
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

pytestmark = pytest.mark.integration

DB = os.environ["DATABASE_URL_DIRECT"]


async def _conn() -> asyncpg.Connection:
    return await asyncpg.connect(DB)


async def test_rls_blocks_other_users_memories() -> None:
    c = await _conn()
    u1, u2 = uuid.uuid4(), uuid.uuid4()
    for u in (u1, u2):
        await c.execute("insert into auth.users (id, email) values ($1, $2)", u, f"{u}@x.test")
    await c.execute(
        "insert into public.memories (user_id, key, value) values ($1, 'favorite_color', 'teal')",
        u1,
    )
    # impersonate u2 through RLS
    # set_config(..., true) is transaction-local, and asyncpg autocommits each
    # statement outside an explicit transaction, so the role/claims must be
    # set and used within a single transaction.
    async with c.transaction():
        await c.execute("set role authenticated")
        await c.execute(
            "select set_config('request.jwt.claims', $1, true)",
            f'{{"sub":"{u2}","role":"authenticated"}}',
        )
        rows = await c.fetch("select * from public.memories")
    await c.execute("reset role")
    assert rows == []
    await c.close()


async def test_load_turn_context_shape() -> None:
    c = await _conn()
    u = uuid.uuid4()
    await c.execute("insert into auth.users (id, email) values ($1, $2)", u, f"{u}@x.test")
    sid = await c.fetchval("insert into public.sessions (user_id) values ($1) returning id", u)
    ctx = await c.fetchval("select public.load_turn_context($1,$2,12)", u, sid)
    d = json.loads(ctx)
    # Two keys joined the four originals in v2 (L-7): `session`, so resolving
    # the session costs no read of its own, and `last_results`, the finished run
    # a follow-up question is grounded against.
    assert set(d) == {"memories", "history", "workflow", "profile", "session", "last_results"}
    assert d["session"]["id"] == str(sid)
    assert d["profile"]["user_id"] == str(u)  # trigger created the profile
    await c.close()


async def test_ops_tables_deny_all_for_authenticated() -> None:
    c = await _conn()
    async with c.transaction():
        await c.execute("set role authenticated")
        await c.execute(
            "select set_config('request.jwt.claims', $1, true)",
            f'{{"sub":"{uuid.uuid4()}","role":"authenticated"}}',
        )
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await c.execute("delete from public.rate_limits")
    await c.execute("reset role")
    await c.close()


async def test_load_turn_context_memories_ordered_and_capped() -> None:
    c = await _conn()
    u = uuid.uuid4()
    await c.execute("insert into auth.users (id, email) values ($1, $2)", u, f"{u}@x.test")
    sid = await c.fetchval("insert into public.sessions (user_id) values ($1) returning id", u)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(70):
        await c.execute(
            "insert into public.memories (user_id, key, value, updated_at) "
            "values ($1, $2, 'v', $3)",
            u,
            f"key{i}",
            base + timedelta(minutes=i),
        )
    ctx = await c.fetchval("select public.load_turn_context($1,$2,12)", u, sid)
    d = json.loads(ctx)
    keys = [m["k"] for m in d["memories"]]
    assert len(keys) == 60
    assert "key69" in keys  # newest kept
    assert "key0" not in keys  # oldest dropped
    await c.close()
