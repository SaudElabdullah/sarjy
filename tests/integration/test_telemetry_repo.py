import json
import os
import uuid

import pytest

from sarjy.infrastructure_shared.db import Database
from sarjy.observability.telemetry_repo import PgTelemetryRepo
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


async def test_save_inserts_a_readable_row(db: Database, user: UserId) -> None:
    repo = PgTelemetryRepo(db)
    message_id = uuid.uuid4()
    await repo.save(
        user_id=user,
        message_id=message_id,
        ttfa_ms=650,
        t_request_ms=10,
        t_first_byte_ms=400,
        t_first_sentence_ms=500,
        t_last_audio_ms=2000,
        server_timings={"t_gemini_first_token": 320},
        client_info={"ua": "x", "stt": True, "tts": True, "mode": "voice"},
    )
    row = await db.fetchrow(
        "select message_id, ttfa_ms, t_request_ms, t_first_byte_ms, t_first_sentence_ms,"
        " t_last_audio_ms, server_timings, client_info from telemetry_turns where user_id=$1",
        user,
    )
    assert row is not None
    assert row["message_id"] == message_id
    assert row["ttfa_ms"] == 650
    assert row["t_request_ms"] == 10
    assert row["t_first_byte_ms"] == 400
    assert row["t_first_sentence_ms"] == 500
    assert row["t_last_audio_ms"] == 2000
    assert json.loads(row["server_timings"]) == {"t_gemini_first_token": 320}
    assert json.loads(row["client_info"]) == {"ua": "x", "stt": True, "tts": True, "mode": "voice"}
