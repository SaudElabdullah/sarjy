import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from sarjy.contexts.memory.application.forget import ForgetFact
from sarjy.contexts.memory.application.ports import ScreenVerdict
from sarjy.contexts.memory.application.remember import RememberFact
from sarjy.contexts.memory.domain.memory import Memory
from sarjy.contexts.memory.infrastructure.pg_memory_repo import PgMemoryRepo
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.clock import FakeClock
from sarjy.shared.errors import NotFound
from sarjy.shared.ids import MemoryId, MessageId, UserId

pytestmark = pytest.mark.integration
T0 = datetime(2026, 8, 21, tzinfo=UTC)


class _AllowAllScreen:
    """`ValueScreenPort` double: these tests exercise `PgMemoryRepo`, not
    guardrail screening (Phase 8 T6b fix round 1, Minor 3: `screen` is a
    required `RememberFact` constructor arg now)."""

    def screen(
        self, text: str, *, user_id: UserId | None = None, message_id: MessageId | None = None
    ) -> ScreenVerdict:
        return ScreenVerdict(allowed=True)


@pytest.fixture
async def db():  # type: ignore[no-untyped-def]
    d = Database(os.environ["DATABASE_URL_DIRECT"])
    await d.connect()
    yield d
    await d.close()


@pytest.fixture
async def user(db: Database) -> UserId:
    u = uuid.uuid4()
    await db.execute("insert into auth.users (id,email) values ($1,$2)", u, f"{u}@x.test")
    return UserId(u)


async def test_upsert_get_list_soft_delete_history(db: Database, user: UserId) -> None:
    repo = PgMemoryRepo(db)
    m = Memory.create(MemoryId(uuid.uuid4()), user, "favorite_color", "teal", "fact", T0)
    await repo.upsert(m)
    for ev in m.pull_events():
        await repo.append_history(ev)
    got = await repo.get_by_key(user, "favorite_color")
    assert got is not None and got.value == "teal" and got.id == m.id

    got.update("navy", T0 + timedelta(minutes=1))
    await repo.upsert(got)
    for ev in got.pull_events():
        await repo.append_history(ev)
    assert (await repo.get_by_key(user, "favorite_color")).value == "navy"  # type: ignore[union-attr]
    assert len(await repo.list_live(user)) == 1

    got.forget(T0 + timedelta(minutes=2))
    await repo.soft_delete(got)
    for ev in got.pull_events():
        await repo.append_history(ev)
    assert await repo.get_by_key(user, "favorite_color") is None
    assert await repo.list_live(user) == []
    rows = await db.fetch(
        "select action from memories_history where memory_id=$1 order by id", m.id
    )
    assert [r["action"] for r in rows] == ["create", "update", "delete"]


async def test_remember_after_forget_creates_new_row_without_unique_violation(
    db: Database, user: UserId
) -> None:
    repo = PgMemoryRepo(db)
    remember = RememberFact(repo, FakeClock(T0), screen=_AllowAllScreen())
    assert (await remember(user, "home_city", "Lisbon")).status == "created"
    m = await repo.get_by_key(user, "home_city")
    assert m is not None
    m.forget(T0)
    await repo.soft_delete(m)
    assert (await remember(user, "home_city", "Porto")).status == "created"
    rows = await db.fetch(
        "select value, deleted_at from memories where user_id=$1 and key='home_city' "
        "order by created_at",
        user,
    )
    assert [r["value"] for r in rows] == ["Lisbon", "Porto"] and rows[0]["deleted_at"] is not None


async def test_list_live_orders_newest_first_and_limits(db: Database, user: UserId) -> None:
    repo = PgMemoryRepo(db)
    for i in range(5):
        m = Memory.create(
            MemoryId(uuid.uuid4()), user, f"k{i}", f"v{i}", "fact", T0 + timedelta(minutes=i)
        )
        await repo.upsert(m)
    out = await repo.list_live(user, limit=3)
    assert [m.key for m in out] == ["k4", "k3", "k2"]


async def test_upsert_with_history_persists_atomically(db: Database, user: UserId) -> None:
    repo = PgMemoryRepo(db)
    m = Memory.create(MemoryId(uuid.uuid4()), user, "pet", "cat", "fact", T0)
    await repo.upsert_with_history(m, m.pull_events())
    got = await repo.get_by_key(user, "pet")
    assert got is not None and got.value == "cat"
    rows = await db.fetch(
        "select action, new_value from memories_history where memory_id=$1 order by id", m.id
    )
    assert [(r["action"], r["new_value"]) for r in rows] == [("create", "cat")]


async def test_stale_write_after_concurrent_forget_does_not_resurrect_row(
    db: Database, user: UserId
) -> None:
    # Reproduces the read-then-write interleave flagged in the Phase 3 review:
    # connection 1 reads a live memory, connection 2 forgets it, then connection
    # 1's stale in-memory aggregate (still believing it's live) is written back
    # with a new value. That write must not resurrect the soft-deleted row, and
    # RememberFact on the same key afterwards must create a brand-new live row
    # rather than raising or silently no-op'ing.
    repo = PgMemoryRepo(db)
    clock = FakeClock(T0)
    remember = RememberFact(repo, clock, screen=_AllowAllScreen())
    forget = ForgetFact(repo, clock)

    assert (await remember(user, "favorite_color", "teal")).status == "created"

    # "connection 1": read the live memory, then hang on to the stale aggregate.
    stale = await repo.get_by_key(user, "favorite_color")
    assert stale is not None
    original_id = stale.id

    # "connection 2": forget it (soft-delete) before connection 1 writes back.
    assert (await forget(user, "favorite_color")).status == "forgotten"
    assert await repo.get_by_key(user, "favorite_color") is None

    # connection 1's stale aggregate still has deleted_at=None; mutate and write it.
    stale.update("navy", T0 + timedelta(minutes=1))
    with pytest.raises(NotFound):
        await repo.upsert_with_history(stale, stale.pull_events())

    # The row must still be soft-deleted, not resurrected with "navy".
    assert await repo.get_by_key(user, "favorite_color") is None
    row = await db.fetchrow("select deleted_at, value from memories where id=$1", original_id)
    assert row is not None and row["deleted_at"] is not None and row["value"] == "teal"

    # RememberFact on the same key afterwards must create a NEW live row.
    outcome = await remember(user, "favorite_color", "navy")
    assert outcome.status == "created"
    fresh = await repo.get_by_key(user, "favorite_color")
    assert fresh is not None and fresh.value == "navy" and fresh.id != original_id


async def test_concurrent_upsert_same_new_key_serialises_without_unique_violation(
    db: Database, user: UserId
) -> None:
    repo = PgMemoryRepo(db)
    a = Memory.create(MemoryId(uuid.uuid4()), user, "nickname", "Al", "fact", T0)
    b = Memory.create(MemoryId(uuid.uuid4()), user, "nickname", "Bo", "fact", T0)
    # Two concurrent creates for the same brand-new key: neither id exists yet
    # so a naive lock-by-own-id is a no-op for both, and without the
    # per-(user_id, key) advisory lock in `_write` the second insert would
    # trip the partial unique index `memories_user_key_live`.
    await asyncio.gather(
        repo.upsert_with_history(a, a.pull_events()),
        repo.upsert_with_history(b, b.pull_events()),
    )
    live = await db.fetch(
        "select id from memories where user_id=$1 and key='nickname' and deleted_at is null",
        user,
    )
    assert len(live) == 1
    history = await db.fetch("select memory_id from memories_history where user_id=$1", user)
    assert len(history) == 2
    assert {r["memory_id"] for r in history} == {live[0]["id"]}


async def test_forget_erases_the_memories_history_rows(db: Database, user: UserId) -> None:
    # I2: `memories_history.old_value`/`new_value` hold the remembered fact
    # verbatim. A forget that only soft-deletes the `memories` row leaves that
    # copy behind indefinitely, which is not what "forget my address" means.
    repo = PgMemoryRepo(db)
    remember = RememberFact(repo, FakeClock(T0), screen=_AllowAllScreen())
    forget = ForgetFact(repo, FakeClock(T0 + timedelta(minutes=5)))

    await remember(user, "home_address", "10 Downing Street")
    await remember(user, "home_address", "11 Downing Street")
    rows = await db.fetch(
        "select old_value,new_value from public.memories_history where user_id=$1", user
    )
    assert len(rows) == 2

    assert (await forget(user, "home_address")).status == "forgotten"

    left = await db.fetch(
        "select old_value,new_value from public.memories_history where user_id=$1", user
    )
    assert left == []
    # The memory row itself is still there, soft-deleted (the existing
    # soft-delete contract) — it is only the stored text that is gone.
    row = await db.fetchrow(
        "select deleted_at from public.memories where user_id=$1 and key='home_address'", user
    )
    assert row is not None and row["deleted_at"] is not None


async def test_forget_leaves_other_memories_history_intact(db: Database, user: UserId) -> None:
    repo = PgMemoryRepo(db)
    remember = RememberFact(repo, FakeClock(T0), screen=_AllowAllScreen())
    forget = ForgetFact(repo, FakeClock(T0 + timedelta(minutes=5)))

    await remember(user, "favorite_color", "teal")
    await remember(user, "favorite_food", "pho")
    assert (await forget(user, "favorite_color")).status == "forgotten"

    left = await db.fetch("select new_value from public.memories_history where user_id=$1", user)
    assert [r["new_value"] for r in left] == ["pho"]
