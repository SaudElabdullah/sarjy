"""scripts/_supabase_auth.py: anonymous sign-up against a mocked Supabase Auth
API (Phase 8 Task 5). No network, no real Supabase project."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from scripts._supabase_auth import SignupError, anon_signup


@respx.mock
def test_anon_signup_returns_the_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TURNSTILE_TOKEN", raising=False)
    route = respx.post("https://proj.supabase.co/auth/v1/signup").mock(
        return_value=httpx.Response(200, json={"access_token": "tok-123", "user": {}})
    )
    token = anon_signup("https://proj.supabase.co", "anon-key")
    assert token == "tok-123"  # noqa: S105
    assert route.calls.last.request.headers["apikey"] == "anon-key"
    assert route.calls.last.request.content == b"{}"


@respx.mock
def test_anon_signup_strips_a_trailing_slash_from_the_url() -> None:
    respx.post("https://proj.supabase.co/auth/v1/signup").mock(
        return_value=httpx.Response(200, json={"access_token": "tok-456"})
    )
    assert anon_signup("https://proj.supabase.co/", "anon-key") == "tok-456"


@respx.mock
def test_anon_signup_raises_signuperror_when_no_token_comes_back() -> None:
    respx.post("https://proj.supabase.co/auth/v1/signup").mock(
        return_value=httpx.Response(200, json={"msg": "anonymous sign-ins are disabled"})
    )
    with pytest.raises(SignupError, match="no access_token"):
        anon_signup("https://proj.supabase.co", "anon-key")


@respx.mock
def test_anon_signup_raises_on_http_error() -> None:
    respx.post("https://proj.supabase.co/auth/v1/signup").mock(
        return_value=httpx.Response(500, text="internal error")
    )
    with pytest.raises(httpx.HTTPStatusError):
        anon_signup("https://proj.supabase.co", "anon-key")


@respx.mock
def test_anon_signup_sends_the_captcha_token_when_turnstile_token_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PRD §11: a project with captcha protection on rejects a bare `{}` sign-up.
    # CI supplies Cloudflare's always-pass dummy token via TURNSTILE_TOKEN (see
    # docs/runbook.md) and it must reach GoTrue under `gotrue_meta_security`.
    monkeypatch.setenv("TURNSTILE_TOKEN", "XXXX.DUMMY.TOKEN.XXXX")
    route = respx.post("https://proj.supabase.co/auth/v1/signup").mock(
        return_value=httpx.Response(200, json={"access_token": "tok-789"})
    )
    assert anon_signup("https://proj.supabase.co", "anon-key") == "tok-789"
    sent = json.loads(route.calls.last.request.content)
    assert sent == {"gotrue_meta_security": {"captcha_token": "XXXX.DUMMY.TOKEN.XXXX"}}


@respx.mock
def test_anon_signup_body_is_bare_when_turnstile_token_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A project without captcha must see exactly the body it saw before captcha
    # existed — no empty/None captcha_token that GoTrue would then reject.
    monkeypatch.delenv("TURNSTILE_TOKEN", raising=False)
    route = respx.post("https://proj.supabase.co/auth/v1/signup").mock(
        return_value=httpx.Response(200, json={"access_token": "tok-000"})
    )
    anon_signup("https://proj.supabase.co", "anon-key")
    assert route.calls.last.request.content == b"{}"
