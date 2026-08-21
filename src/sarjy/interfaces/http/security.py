"""`SecurityHeadersMiddleware` — CSP/HSTS and friends on every response (PRD §12)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from sarjy.observability.logging import get_logger

log = get_logger(__name__)

TURNSTILE_ORIGIN = "https://challenges.cloudflare.com"


class UnhandledErrorMiddleware(BaseHTTPMiddleware):
    """Turn an unhandled exception into a JSON 500 *inside* the middleware stack.

    I3: `SecurityHeadersMiddleware` sets its headers on the response `call_next`
    returns — and an unhandled exception never produces one. It propagates past
    every user middleware to Starlette's `ServerErrorMiddleware`, which sits
    OUTSIDE the whole user stack and renders its own bare response. So a 500 —
    exactly the response most likely to be reflected back to an attacker, and
    the one class of response an app cannot promise to have got right — went
    out with no CSP, no HSTS, no `nosniff`.

    Registering an `Exception` handler with `add_exception_handler` does not fix
    that: Starlette routes the `Exception`/500 key to `ServerErrorMiddleware`,
    the outermost layer, so the handler's response bypasses the user stack the
    same way. It has to be caught *below* the header middleware instead, which
    is what this is — added first in `create_app` so it nests innermost, and the
    JSON 500 it returns is an ordinary response that CORS and the header
    middleware then decorate on the way out.

    The body is a fixed `{"detail": "internal error"}`: an exception message can
    carry a query fragment, a file path or a key, and the client has no use for
    any of it. The detail that matters goes to the log, with the traceback.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            return await call_next(request)
        except Exception:
            # structlog, with the traceback — the same `log.exception` posture
            # the rest of the app uses for a swallowed error. This is the only
            # record that the request failed at all, so it must not be quiet.
            log.exception("unhandled_exception", path=request.url.path, method=request.method)
            return JSONResponse({"detail": "internal error"}, status_code=500)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, app: ASGIApp, supabase_origin: str, turnstile_site_key: str | None = None
    ) -> None:
        super().__init__(app)
        host = urlsplit(supabase_origin).netloc
        if not host:
            raise ValueError(
                f"supabase_origin must be an absolute URL with a host, got {supabase_origin!r}"
            )
        # Turnstile (PRD §11 captcha on anonymous sign-in) is the one script that
        # cannot be vendored: Cloudflare's api.js must be loaded from their origin
        # (it is versioned/rotated server-side, and the widget iframe it injects is
        # served from the same host). So the three directives it needs are added
        # ONLY when a site key is configured — script-src to load api.js, frame-src
        # for the challenge iframe it injects, connect-src for the widget's own
        # XHRs back to Cloudflare. With TURNSTILE_SITE_KEY unset the CSP is
        # byte-identical to what it was before captcha existed: no third-party
        # origin is advertised for a widget the page never renders.
        turnstile = f" {TURNSTILE_ORIGIN}" if turnstile_site_key else ""
        # No `frame-src` directive at all without a key: `default-src 'self'` already
        # covers frames, so the effective policy is unchanged (and `frame-ancestors
        # 'none'` below, which is about who may frame US, is unaffected either way).
        frame_src = f"frame-src {TURNSTILE_ORIGIN}; " if turnstile_site_key else ""
        # supabase-js is vendored locally (static/supabase.js — see static/VENDOR.md),
        # so script-src is 'self' plus (when configured) Turnstile only: no CDN, no
        # 'unsafe-inline'. The inline `window.SARJY = {...}` block lives at
        # GET /config.js for the same reason.
        self.csp = (
            "default-src 'self'; "
            f"connect-src 'self' {supabase_origin} wss://{host}{turnstile}; "
            f"script-src 'self'{turnstile}; "
            f"{frame_src}"
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        resp = await call_next(request)
        resp.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        resp.headers["Content-Security-Policy"] = self.csp
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        resp.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
        resp.headers["X-Frame-Options"] = "DENY"
        return resp
