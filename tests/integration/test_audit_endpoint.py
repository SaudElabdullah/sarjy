"""`POST /internal/audit/run` end to end: the endpoint, `Container.audit_worker`
(`PgAuditRepo` + `PgGuardEventRepo`), and the trigger's target table
`public.audit_queue` (PRD Layer 7, `supabase/migrations/
20260821000800_audit_sample.sql`).

The row is force-enqueued (a direct `insert into audit_queue`) rather than
relying on `enqueue_audit_sample`'s `random() < 0.2` sampling — deterministic
setup for a deterministic assertion. The classifier is swapped for a fake
after startup for the same reason `test_turn_read_count.py` swaps in
`OfflineClassifier`: an integration test must not depend on a live Gemini
call succeeding, but unlike that suite this one needs the call to actually
SUCCEED (to prove the endpoint writes a `guardrail_events` row and marks the
queue item processed), so the fake here returns a benign `Classification`
rather than raising.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from sarjy.config import Settings
from sarjy.contexts.guardrails.application.audit import AuditWorker
from sarjy.contexts.guardrails.application.ports import Classification
from sarjy.main import create_app

pytestmark = pytest.mark.integration

TOKEN = "audit-endpoint-test-token"  # noqa: S105


class _FakeClassifier:
    async def classify(self, recent_user_turns: list[str]) -> Classification:
        return Classification(None, False, 0, 0.1)


async def test_audit_run_processes_a_forced_item_and_writes_a_layer7_event() -> None:
    app = create_app(Settings(internal_token=SecretStr(TOKEN)))
    async with LifespanManager(app):
        c = app.state.container
        # A fake classifier that always succeeds, so the endpoint's happy path
        # is exercised without depending on a real Gemini call landing.
        assert c.audit_queue is not None and c.guard_events is not None
        c.audit_worker = AuditWorker(c.audit_queue, _FakeClassifier(), c.guard_events)

        db = c.db
        u = uuid.uuid4()
        await db.execute("insert into auth.users (id,email) values ($1,$2)", u, f"{u}@x.test")
        session_id = await db.fetchval(
            "insert into public.sessions (user_id) values ($1) returning id", u
        )
        now = datetime.now(UTC)
        user_msg = await db.fetchval(
            "insert into public.messages (session_id,user_id,role,content,created_at) "
            "values ($1,$2,'user','how do I bake bread',$3) returning id",
            session_id,
            u,
            now,
        )
        assistant_msg = await db.fetchval(
            "insert into public.messages (session_id,user_id,role,content,created_at) "
            "values ($1,$2,'assistant','Here is a simple recipe.',$3) returning id",
            session_id,
            u,
            now,
        )
        await db.execute(
            "insert into public.audit_queue (message_id,user_id) values ($1,$2)",
            assistant_msg,
            u,
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            r = await client.post("/internal/audit/run", headers={"X-Internal-Token": TOKEN})

        assert r.status_code == 200
        body = r.json()
        assert body["processed"] >= 1
        assert body["flagged"] >= 0

        event = await db.fetchrow(
            "select layer, kind, action from public.guardrail_events "
            "where message_id = $1 and layer = 7",
            assistant_msg,
        )
        assert event is not None
        assert event["kind"] == "audit:clean"
        assert event["action"] == "audit_clean"

        processed_at = await db.fetchval(
            "select processed_at from public.audit_queue where message_id = $1", assistant_msg
        )
        assert processed_at is not None

        # user_msg is asserted on only to document why it exists (the FK the
        # trigger's own preceding-user-message lookup relies on); nothing here
        # reads it back directly since the fake classifier ignores its input.
        assert user_msg is not None


async def test_audit_run_requires_the_internal_token() -> None:
    app = create_app(Settings(internal_token=SecretStr(TOKEN)))
    async with (
        LifespanManager(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client,
    ):
        r = await client.post("/internal/audit/run", headers={"X-Internal-Token": "wrong"})
    assert r.status_code == 401
