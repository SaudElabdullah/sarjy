"""`GET /admin/latency` — internal latency/guard/funnel dashboard (PRD §13)."""

import time
import uuid

import jwt
from fastapi.testclient import TestClient

from sarjy.config import Settings
from sarjy.main import create_app

SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"  # noqa: S105


class StubAdminRepo:
    async def latency_daily(self):  # type: ignore[no-untyped-def]
        return [{"day": "2026-08-21T00:00:00", "mode": "voice", "turns": 3, "ttfa_p50": 650.0}]

    async def latency_by_browser(self):  # type: ignore[no-untyped-def]
        return [{"browser": "chrome", "turns": 3, "ttfa_p50": 650.0, "ttfa_p95": 900.0}]

    async def guard_daily(self):  # type: ignore[no-untyped-def]
        return [
            {
                "day": "2026-08-21T00:00:00",
                "layer": 2,
                "kind": "off_topic",
                "action": "block",
                "n": 1,
            }
        ]

    async def ocean_funnel(self):  # type: ignore[no-untyped-def]
        return [{"day": "2026-08-21T00:00:00", "proposed": 5, "started": 3, "completed": 1}]


def _token(sub: uuid.UUID) -> str:
    return jwt.encode(
        {"sub": str(sub), "aud": "authenticated", "exp": int(time.time()) + 60},
        SECRET,
        algorithm="HS256",
    )


def test_admin_latency_forbidden_for_non_admin() -> None:
    app = create_app(Settings(admin_user_ids=str(uuid.uuid4())), connect_db=False)
    app.state.container.admin_repo = StubAdminRepo()
    tok = _token(uuid.uuid4())  # not in admin_user_ids
    with TestClient(app) as c:
        r = c.get("/admin/latency", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403


def test_admin_latency_ok_for_admin() -> None:
    admin_id = uuid.uuid4()
    app = create_app(Settings(admin_user_ids=str(admin_id)), connect_db=False)
    app.state.container.admin_repo = StubAdminRepo()
    tok = _token(admin_id)
    with TestClient(app) as c:
        r = c.get("/admin/latency", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"latency_daily", "latency_by_browser", "guard_daily", "ocean_funnel"}
    assert body["latency_daily"][0]["ttfa_p50"] == 650.0
    assert body["ocean_funnel"][0]["completed"] == 1


def test_admin_latency_requires_jwt() -> None:
    app = create_app(Settings(admin_user_ids=str(uuid.uuid4())), connect_db=False)
    app.state.container.admin_repo = StubAdminRepo()
    with TestClient(app) as c:
        r = c.get("/admin/latency")
    assert r.status_code == 401
