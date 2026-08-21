import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sarjy.config import Settings
from sarjy.interfaces.http.security import SecurityHeadersMiddleware
from sarjy.main import create_app


def test_headers_present() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.headers["strict-transport-security"].startswith("max-age=")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "microphone=(self)" in r.headers["permissions-policy"]


def test_csp_forbids_inline_scripts_and_third_party_script_src() -> None:
    # supabase-js is vendored locally (static/supabase.js) so script-src never needs
    # a CDN, and must never fall back to 'unsafe-inline' either.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as c:
        r = c.get("/healthz")
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "cdn.jsdelivr.net" not in csp
    assert "unsafe-inline" not in csp.split("style-src")[0]


def test_csp_connect_src_scopes_to_the_configured_supabase_origin() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as c:
        r = c.get("/healthz")
    csp = r.headers["content-security-policy"]
    assert "connect-src 'self' http://localhost:54321 wss://localhost:54321" in csp


def test_x_frame_options_deny() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.headers["x-frame-options"] == "DENY"


@pytest.mark.parametrize("bad_origin", ["not-a-url", "localhost:54321", ""])
def test_middleware_raises_a_clear_value_error_on_a_malformed_supabase_origin(
    bad_origin: str,
) -> None:
    with pytest.raises(ValueError, match="supabase_origin"):
        SecurityHeadersMiddleware(FastAPI(), supabase_origin=bad_origin)


def test_csp_has_no_turnstile_origin_without_a_site_key() -> None:
    # The default shape: no TURNSTILE_SITE_KEY, so no third-party origin is
    # advertised anywhere in the policy and there is no frame-src at all
    # (default-src 'self' already covers frames).
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as c:
        csp = c.get("/healthz").headers["content-security-policy"]
    assert "challenges.cloudflare.com" not in csp
    assert "frame-src" not in csp


def test_csp_allows_turnstile_in_script_frame_and_connect_when_a_site_key_is_set() -> None:
    # PRD §11: the captcha widget needs Cloudflare's api.js (script-src), the
    # challenge iframe it injects (frame-src), and the widget's own XHRs back to
    # Cloudflare (connect-src) — and nothing else widens.
    app = create_app(Settings(turnstile_site_key="0x4AAA"), connect_db=False)
    with TestClient(app) as c:
        csp = c.get("/healthz").headers["content-security-policy"]
    assert "script-src 'self' https://challenges.cloudflare.com" in csp
    assert "frame-src https://challenges.cloudflare.com" in csp
    assert "wss://localhost:54321 https://challenges.cloudflare.com" in csp
    # Still no inline scripts, and framing US is still forbidden.
    assert "unsafe-inline" not in csp.split("style-src")[0]
    assert "frame-ancestors 'none'" in csp


def _app_with_a_route_that_raises() -> FastAPI:
    app = create_app(Settings(), connect_db=False)

    @app.get("/boom")
    async def boom() -> None:  # pragma: no cover - the body is the point
        raise RuntimeError("kaboom")

    return app


def test_an_unhandled_exception_returns_a_json_500_with_the_security_headers() -> None:
    # I3: an unhandled exception used to propagate past every user middleware to
    # Starlette's ServerErrorMiddleware, which sits outside the stack — so the
    # one response most likely to be reflected back at an attacker went out with
    # no CSP, no HSTS and no nosniff.
    with TestClient(_app_with_a_route_that_raises(), raise_server_exceptions=False) as c:
        r = c.get("/boom")
    assert r.status_code == 500
    assert r.json() == {"detail": "internal error"}
    assert r.headers["content-type"].startswith("application/json")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["strict-transport-security"].startswith("max-age=")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "strict-origin-when-cross-origin"


def test_the_500_body_never_leaks_the_exception_message() -> None:
    # The exception text can carry a path, a query fragment or a key; the client
    # has no use for any of it. The traceback goes to the log instead.
    with TestClient(_app_with_a_route_that_raises(), raise_server_exceptions=False) as c:
        r = c.get("/boom")
    assert "kaboom" not in r.text
    assert "RuntimeError" not in r.text
    assert "Traceback" not in r.text


def test_the_unhandled_exception_is_still_logged_with_its_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Converting the exception into a response must not make it disappear: this
    # log line is the only record that the request failed at all. structlog
    # renders JSON to stdout (see observability/logging.py), not through the
    # stdlib logging handlers caplog captures.
    with TestClient(_app_with_a_route_that_raises(), raise_server_exceptions=False) as c:
        c.get("/boom")
    lines = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    logged = [e for e in lines if e.get("event") == "unhandled_exception"]
    assert logged, "the unhandled exception was swallowed without a log line"
    assert logged[0]["level"] == "error"
    assert logged[0]["exc_info"] is True
    assert logged[0]["path"] == "/boom"
    assert logged[0]["method"] == "GET"


def test_a_normal_response_is_untouched_by_the_unhandled_error_middleware() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.status_code == 200
