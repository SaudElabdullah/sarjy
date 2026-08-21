import os
import uuid
from datetime import UTC, datetime

import pytest

from sarjy.contexts.conversation.domain.message import Message
from sarjy.contexts.conversation.domain.session import Session
from sarjy.contexts.conversation.infrastructure.pg_message_repo import PgMessageRepo
from sarjy.contexts.conversation.infrastructure.pg_session_repo import PgSessionRepo
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import MessageId, SessionId, UserId

pytestmark = pytest.mark.integration


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


async def test_session_and_message_roundtrip_idempotent(db: Database, user: UserId) -> None:
    now = datetime.now(UTC)
    sessions, messages = PgSessionRepo(db), PgMessageRepo(db)
    s = Session.start(SessionId(uuid.uuid4()), user, now)
    await sessions.save(s)
    assert (await sessions.get(s.id)) is not None
    m = Message(MessageId(uuid.uuid4()), s.id, user, "user", "hi", now, client_turn_id="t1")
    await messages.save(m)
    await messages.save(m)  # second save is a no-op
    assert len(await messages.history(user, s.id, 12)) == 1


async def test_save_tool_call_persists_row(db: Database, user: UserId) -> None:
    now = datetime.now(UTC)
    sessions, messages = PgSessionRepo(db), PgMessageRepo(db)
    s = Session.start(SessionId(uuid.uuid4()), user, now)
    await sessions.save(s)
    m = Message(MessageId(uuid.uuid4()), s.id, user, "user", "what's the weather", now)
    await messages.save(m)

    await messages.save_tool_call(
        m.id, user, "get_weather", {"city": "Riyadh"}, {"temp_c": 41}, "ok", 120
    )

    count = await db.fetchval("select count(*) from tool_calls where message_id=$1", m.id)
    assert count == 1


async def test_history_is_scoped_to_the_owning_user(db: Database, user: UserId) -> None:
    now = datetime.now(UTC)
    sessions, messages = PgSessionRepo(db), PgMessageRepo(db)
    s = Session.start(SessionId(uuid.uuid4()), user, now)
    await sessions.save(s)
    await messages.save(Message(MessageId(uuid.uuid4()), s.id, user, "user", "secret", now))

    assert len(await messages.history(user, s.id, 12)) == 1

    other = UserId(uuid.uuid4())
    await db.execute("insert into auth.users (id,email) values ($1,$2)", other, f"{other}@x.test")
    assert await messages.history(other, s.id, 12) == []


async def test_touch_only_save_does_not_erase_a_stored_summary(db: Database, user: UserId) -> None:
    now = datetime.now(UTC)
    sessions = PgSessionRepo(db)
    s = Session.start(SessionId(uuid.uuid4()), user, now)
    await sessions.save(s)

    s.summary = "They asked about the weather in Lisbon."
    await sessions.save(s)

    # A later turn touches the session without recomputing the summary.
    touched = Session.start(s.id, user, now)
    touched.touch(now)
    await sessions.save(touched)

    stored = await sessions.get(s.id)
    assert stored is not None
    assert stored.summary == "They asked about the weather in Lisbon."
