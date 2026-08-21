"""Shared helper: mint a bearer token via Supabase anonymous sign-up.

Used by `scripts/smoke.py` (post-deploy smoke test, Phase 8 Task 5) and
`tests/evals/run_evals.py` (`--target` remote mode) — anywhere that needs a
fresh authenticated user against a real Supabase project without a JWT secret
to sign HS256 tokens locally (the manual release flow (`make release-*`) has
`SUPABASE_URL` / `SUPABASE_ANON_KEY` as shell environment variables, not the
project's JWT signing secret). `tests/evals/run_memory_eval.py` mints tokens
the same way inline; this module exists so smoke.py and run_evals.py don't
each carry their own copy.
"""

from __future__ import annotations

import os

import httpx


class SignupError(RuntimeError):
    """Anonymous sign-up succeeded at the transport level but returned no usable token."""


def anon_signup(supabase_url: str, anon_key: str, *, timeout: float = 10.0) -> str:
    """POST {supabase_url}/auth/v1/signup with an empty body and return the access token.

    An empty JSON body (`{}`) is GoTrue's anonymous sign-in: no email/password,
    a brand-new `auth.users` row, a real session. Requires "Allow anonymous
    sign-ins" enabled on the target project (Phase 8 Task 1) — this raises
    `httpx.HTTPStatusError` if it isn't.

    If the project has captcha protection turned on (PRD §11 — Turnstile, which
    is what `TURNSTILE_SITE_KEY` gates in the browser client), GoTrue rejects a
    bare `{}` body with a 400. Set `TURNSTILE_TOKEN` in the environment and it is
    sent as GoTrue's `gotrue_meta_security.captcha_token`. The manual release
    flow uses Cloudflare's always-pass test credentials for this (sitekey
    `1x00000000000000000000AA`,
    secret `1x0000000000000000000000000000000AA`, token `XXXX.DUMMY.TOKEN.XXXX`)
    — see docs/runbook.md. Unset, the body is byte-identical to what it was
    before captcha existed, so a project without captcha is unaffected.
    """
    body: dict[str, object] = {}
    captcha = os.environ.get("TURNSTILE_TOKEN")
    if captcha:
        body["gotrue_meta_security"] = {"captcha_token": captcha}
    r = httpx.post(
        f"{supabase_url.rstrip('/')}/auth/v1/signup",
        headers={"apikey": anon_key, "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise SignupError(f"signup response had no access_token: {r.text[:200]}")
    return str(token)
