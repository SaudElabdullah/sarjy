import json
import os
import uuid

import pytest

from sarjy.contexts.guardrails.infrastructure.pg_event_repo import PgGuardEventRepo
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import UserId

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


async def test_record_inserts_a_readable_row(db: Database, user: UserId) -> None:
    repo = PgGuardEventRepo(db)
    await repo.record(
        user_id=user,
        message_id=None,
        layer=1,
        kind="injection",
        action="refuse",
        severity=2,
        detail={"reason": "prompt_leak", "score": 0.9},
    )

    row = await db.fetchrow(
        "select layer, kind, action, severity, detail from guardrail_events where user_id=$1",
        user,
    )
    assert row is not None
    assert row["layer"] == 1
    assert row["kind"] == "injection"
    assert row["action"] == "refuse"
    assert row["severity"] == 2
    assert json.loads(row["detail"])["reason"] == "prompt_leak"
