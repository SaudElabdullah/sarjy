"""``tests/evals/run_evals.py --target`` token source (Phase 8 Task 5).

A `--target` run needs a bearer token per row and has two ways to get one,
picked by what secrets the caller actually handed it: local HS256 signing
when `SUPABASE_JWT_SECRET` is set (dev, against a target sharing that
secret), else a real anonymous sign-up against `SUPABASE_URL` /
`SUPABASE_ANON_KEY` — the only credential the manual release flow's evals
step (`make evals-staging`) carries. No network in either branch here: the
local-signing branch never leaves the process, and the sign-up branch is
exercised via `Chat`/respx in
`tests/unit/scripts/test_supabase_auth.py`; this file is about which branch
`_remote_token_fn` picks, not what anon_signup itself does over HTTP.
"""

from __future__ import annotations

import pytest

from tests.evals.run_evals import _remote_token_fn


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "SUPABASE_JWT_SECRET",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "GEMINI_API_KEY",
        "DATABASE_URL",
        "DATABASE_URL_DIRECT",
        "APP_ENV",
    ):
        monkeypatch.delenv(key, raising=False)


def test_signs_locally_when_jwt_secret_is_explicitly_set(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "explicit-secret-at-least-32-characters-long-000"  # noqa: S105
    monkeypatch.setenv("SUPABASE_JWT_SECRET", secret)
    token_fn = _remote_token_fn()
    token = token_fn()
    # A locally-signed token round-trips through the same secret `_token` uses —
    # decoding it (without verifying `exp`, which is 3600s out but not the point
    # here) confirms it was signed with *this* secret, not fetched from anywhere.
    import jwt

    claims = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
    assert claims["aud"] == "authenticated"


def test_falls_back_to_anonymous_signup_when_no_jwt_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls: list[tuple[str, str]] = []

    def fake_anon_signup(url: str, anon_key: str) -> str:
        calls.append((url, anon_key))
        return "signed-up-token"

    import scripts._supabase_auth as supabase_auth

    monkeypatch.setattr(supabase_auth, "anon_signup", fake_anon_signup)

    token_fn = _remote_token_fn()
    assert token_fn() == "signed-up-token"
    assert calls == [("https://proj.supabase.co", "anon-key")]


def test_local_signing_produces_a_fresh_jwt_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every row gets its own user — a fresh `sub` per call, not a cached token."""
    secret = "another-explicit-secret-at-least-32-chars-long-0"  # noqa: S105
    monkeypatch.setenv("SUPABASE_JWT_SECRET", secret)
    token_fn = _remote_token_fn()
    first, second = token_fn(), token_fn()
    assert first != second
    assert first.count(".") == 2  # header.payload.signature
