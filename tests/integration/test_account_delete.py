import os
import time
import uuid

import jwt
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from sarjy.config import Settings
from sarjy.main import create_app

pytestmark = pytest.mark.integration
SECRET = os.environ["SUPABASE_JWT_SECRET"]

# Every other integration test's `insert into auth.users (id,email) values (...)` is
# enough for RLS/repo tests that only ever touch the row through Postgres. This one
# also has to survive GoTrue's own admin-user lookup, which needs several columns
# that don't have SQL defaults and 500s ("Database error loading user") if they're
# NULL: instance_id (GoTrue's local, single-tenant default id), aud/role, the
# confirmation/recovery/email-change token columns, and created_at/updated_at.
_INSERT_AUTH_USER = """
    insert into auth.users
        (id, instance_id, email, aud, role, confirmation_token, recovery_token,
         email_change_token_new, email_change, created_at, updated_at)
    values ($1, '00000000-0000-0000-0000-000000000000', $2, 'authenticated',
            'authenticated', '', '', '', '', now(), now())
"""


async def test_delete_account_cascades() -> None:
    app = create_app(Settings())
    u = uuid.uuid4()
    async with LifespanManager(app):
        db = app.state.container.db
        await db.execute(_INSERT_AUTH_USER, u, f"{u}@x.test")
        await db.execute("insert into memories (user_id,key,value) values ($1,'k','v')", u)
        tok = jwt.encode(
            {"sub": str(u), "aud": "authenticated", "exp": int(time.time()) + 60},
            SECRET,
            algorithm="HS256",
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.delete("/account", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 204
        assert await db.fetchval("select count(*) from memories where user_id=$1", u) == 0
        assert await db.fetchval("select count(*) from auth.users where id=$1", u) == 0


async def test_delete_account_requires_auth() -> None:
    app = create_app(Settings())
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.delete("/account")
        assert r.status_code == 401


async def test_export_account_is_not_yet_implemented() -> None:
    app = create_app(Settings())
    u = uuid.uuid4()
    async with LifespanManager(app):
        tok = jwt.encode(
            {"sub": str(u), "aud": "authenticated", "exp": int(time.time()) + 60},
            SECRET,
            algorithm="HS256",
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/account/export", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 501
