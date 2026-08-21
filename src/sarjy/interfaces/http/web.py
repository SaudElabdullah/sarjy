from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request, Response
from fastapi.templating import Jinja2Templates

router = APIRouter()
_tpl = Jinja2Templates(directory=str(Path(__file__).parent.parent / "web" / "templates"))


def _local_asset(name: str) -> str:
    """Default `asset()` for index.html: same-origin `/static/<name>`, unhashed.

    `scripts/upload_static.py` renders the same template with a different `asset`
    callable (hashed filename, `{asset_base}/name.<sha8>.ext`) when publishing to
    Supabase Storage. This default keeps `GET /` unchanged for local/dev, where
    FastAPI serves the static files itself (see `StaticFiles` mount in main.py).
    """
    return f"/static/{name}"


@router.get("/")
async def index(request: Request):  # type: ignore[no-untyped-def]
    s = request.app.state.settings
    return _tpl.TemplateResponse(
        request,
        "index.html",
        {
            # Only used for the <link rel="preconnect"> — the client's actual
            # config comes from GET /config.js so the CSP can forbid inline
            # scripts (see security.py).
            "supabase_url": s.supabase_url,
            # Empty here: the page and the API are same-origin in local/dev, so
            # "" + "/config.js" == "/config.js". A Storage-hosted build (see
            # scripts/upload_static.py) renders with an absolute Fly URL instead,
            # since the page is then served from a different origin than the API.
            "api_base": "",
            # Empty unless TURNSTILE_SITE_KEY is set. The template renders the
            # Cloudflare api.js tag and the widget host ONLY when it is non-empty,
            # so an unconfigured deployment gets the exact same bytes it did before
            # captcha existed — matching security.py, which only widens the CSP for
            # challenges.cloudflare.com under the same condition.
            "turnstile_site_key": s.turnstile_site_key or "",
            "asset": _local_asset,
        },
    )


@router.get("/config.js")
async def config_js(request: Request) -> Response:
    s = request.app.state.settings
    # CSP forbids inline <script> (security.py), so what used to be the
    # `window.SARJY = {...}` block in index.html is served here instead. Only the
    # anon key goes out — it's public by design, unlike the service-role key used
    # by account.py, which never reaches this response.
    config = {
        "supabaseUrl": s.supabase_url,
        "anonKey": s.supabase_anon_key,
        # Empty by default (relative fetches — same-origin in local/dev). Set
        # PUBLIC_API_BASE to the Fly app's absolute URL for a Storage-hosted build,
        # so voice.js/memory.js/ocean.js's `${apiBase}` fetches reach it from
        # whatever origin actually served the page.
        "apiBase": s.public_api_base,
        "turnstileSiteKey": s.turnstile_site_key or "",
        # L-3: the client only speculates when the server will honour it.
        # Templated rather than probed, so the feature has exactly one
        # switch (`SPECULATIVE_ENABLED`) and a page served with it off never
        # opens a stream whose writes nothing would ever confirm.
        "speculative": s.speculative_enabled,
    }
    body = f"window.SARJY = {json.dumps(config)};"
    return Response(
        content=body,
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )
