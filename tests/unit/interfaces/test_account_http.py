"""`DELETE /account` against the Supabase Admin API, without a real Supabase (Phase 8
Task 2 fix round 1): idempotent on 404 (already gone), and a clean 502 — with no
service-role key or response body in the logs — when the Admin API is unreachable."""

from __future__ import annotations

import time
import uuid

import httpx
import jwt
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from sarjy.config import Settings
from sarjy.main import create_app

SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"  # noqa: S105
SERVICE_ROLE_KEY = "distinctive-service-role-secret-do-not-log"


def _tok(uid: uuid.UUID) -> str:
    return jwt.encode(
        {"sub": str(uid), "aud": "authenticated", "exp": int(time.time()) + 60},
        SECRET,
        algorithm="HS256",
    )


def _app():  # type: ignore[no-untyped-def]
    return create_app(
        Settings(  # type: ignore[call-arg]
            supabase_jwt_secret=SecretStr(SECRET),
            supabase_service_role_key=SecretStr(SERVICE_ROLE_KEY),
        ),
        connect_db=False,
    )


@respx.mock
def test_delete_account_is_idempotent_when_the_admin_api_says_already_gone() -> None:
    # A retried request, or a user deleted out-of-band, means the Admin API 404s.
    # That must still look like a successful deletion to the caller, not a 502.
    uid = uuid.uuid4()
    respx.delete(url__regex=r".*/auth/v1/admin/users/.*").mock(
        return_value=httpx.Response(404, json={"msg": "User not found"})
    )
    app = _app()
    with TestClient(app) as c:
        r = c.delete("/account", headers={"Authorization": f"Bearer {_tok(uid)}"})
    assert r.status_code == 204
    assert r.content == b""


@respx.mock
def test_delete_account_maps_admin_api_5xx_to_a_clean_502() -> None:
    uid = uuid.uuid4()
    respx.delete(url__regex=r".*/auth/v1/admin/users/.*").mock(
        return_value=httpx.Response(500, text="internal error, key=" + SERVICE_ROLE_KEY)
    )
    app = _app()
    with TestClient(app) as c:
        r = c.delete("/account", headers={"Authorization": f"Bearer {_tok(uid)}"})
    assert r.status_code == 502
    assert r.json() == {"detail": "deletion failed"}
    assert SERVICE_ROLE_KEY not in r.text


@respx.mock
def test_delete_account_transport_error_maps_to_a_clean_502_and_does_not_log_the_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # ConnectError, timeouts, etc. are httpx.HTTPError subclasses raised by the
    # client itself (never a response), so they can't be read off r.status_code —
    # they have to be caught around the request.
    uid = uuid.uuid4()
    respx.delete(url__regex=r".*/auth/v1/admin/users/.*").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    app = _app()
    with TestClient(app) as c:
        r = c.delete("/account", headers={"Authorization": f"Bearer {_tok(uid)}"})
    assert r.status_code == 502
    assert r.json() == {"detail": "deletion failed"}

    out = capsys.readouterr().out
    assert SERVICE_ROLE_KEY not in out
    assert "account_delete_admin_api_unreachable" in out


def test_delete_account_requires_auth() -> None:
    app = _app()
    with TestClient(app) as c:
        assert c.delete("/account").status_code == 401
