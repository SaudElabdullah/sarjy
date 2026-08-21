#!/usr/bin/env python3
"""Publish the static client to a Supabase Storage bucket (Phase 8 Task 3).

Renders the same Jinja2 template FastAPI serves at `GET /`
(`src/sarjy/interfaces/web/templates/index.html`), but with content-hashed asset
filenames and an absolute `api_base`, then uploads everything — `index.html` plus
every file in `src/sarjy/interfaces/web/static/` — to the public `web` bucket via
the Storage REST API.

Why: publishing the client somewhere other than the Fly app itself (Supabase
Storage, a CDN) means the page and the API are no longer same-origin, so every
asset URL in the page has to be absolute, and `/config.js`'s `apiBase` has to
point at the Fly app instead of "" (relative). See PUBLIC_API_BASE in config.py
and the `asset`/`api_base` template variables in web.py.

Usage:
    uv run python scripts/upload_static.py --env local --create-bucket
    uv run python scripts/upload_static.py --env staging --create-bucket
    uv run python scripts/upload_static.py --env prod --dry-run

Credentials, per --env (never printed):
    local:   SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY — the same vars the app
             itself reads from .env.
    staging: STAGING_SUPABASE_URL / STAGING_SUPABASE_SERVICE_ROLE_KEY, falling
             back to the plain SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY if the
             prefixed ones aren't set.
    prod:    PROD_SUPABASE_URL / PROD_SUPABASE_SERVICE_ROLE_KEY, same fallback.

--api-base overrides the page's API origin; it otherwise defaults per --env (see
DEFAULT_API_BASE below — the Fly app for staging/prod, localhost for local).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from jinja2 import Environment, FileSystemLoader, select_autoescape

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "src" / "sarjy" / "interfaces" / "web" / "static"
TEMPLATE_DIR = REPO_ROOT / "src" / "sarjy" / "interfaces" / "web" / "templates"

BUCKET = "web"
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
NO_CACHE = "no-cache"

# Retry policy for Storage REST calls: up to 3 attempts total, with this backoff
# between attempts (only the first two delays are ever used at 3 attempts — the
# third is here so the schedule reads the same as the spec: 0.5s / 1s / 2s).
# Retries only apply to httpx.HTTPError (connection errors, timeouts, ...) and
# 5xx responses — a 4xx response is never transient, so it's returned/raised
# immediately with no delay.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S: tuple[float, ...] = (0.5, 1.0, 2.0)

DEFAULT_API_BASE = {
    "local": "http://127.0.0.1:8000",
    "staging": "https://sarjy-staging.fly.dev",
    "prod": "https://sarjy-prod.fly.dev",
}

# Local ES-module import specifiers — `from "./x.js"` / `import "./x.js"` — rewritten
# to the hashed filename of the imported module. Bare specifiers ("supabase-js"),
# absolute URLs, and anything not ending in .js/.css are left untouched.
_IMPORT_RE = re.compile(r'((?:from|import)\s+["\'])\./([\w.-]+\.(?:js|css))(["\'])')

_CONTENT_TYPES = {".js": "application/javascript", ".css": "text/css; charset=utf-8"}


@dataclass(frozen=True)
class Asset:
    """One file to upload: its final (possibly hashed) name, bytes, and headers."""

    name: str
    content: bytes
    content_type: str
    cache_control: str


def _env_credentials(env: str) -> tuple[str, str]:
    """Return (SUPABASE_URL, service-role key) for --env. Never logs the key."""
    prefix = env.upper()
    url = os.environ.get(f"{prefix}_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = os.environ.get(f"{prefix}_SUPABASE_SERVICE_ROLE_KEY") or os.environ.get(
        "SUPABASE_SERVICE_ROLE_KEY"
    )
    if not url or not key:
        raise SystemExit(
            f"missing SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY (or {prefix}_-prefixed) "
            f"for --env {env}"
        )
    return url.rstrip("/"), key


def _sha8(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:8]


def _hashed_name(original: str, content: bytes) -> str:
    stem, _, ext = original.rpartition(".")
    return f"{stem}.{_sha8(content)}.{ext}"


def _local_import_targets(js_source: str) -> set[str]:
    return {m.group(2) for m in _IMPORT_RE.finditer(js_source)}


def _rewrite_imports(js_source: str, name_map: dict[str, str]) -> str:
    def _sub(m: re.Match[str]) -> str:
        prefix, fname, quote = m.group(1), m.group(2), m.group(3)
        return f"{prefix}./{name_map.get(fname, fname)}{quote}"

    return _IMPORT_RE.sub(_sub, js_source)


def build_assets() -> tuple[dict[str, str], list[Asset]]:
    """Hash + rewrite every file in static/.

    Returns ({original filename: hashed filename}, [Asset, ...]) for everything
    except index.html (rendered separately — see render_index_html).

    JS files are processed in dependency order (leaves first, e.g. fillers.js
    before voice.js before memory.js/ocean.js) so a module's own hash reflects
    its *rewritten* content — the hashed names of whatever it imports — not its
    pre-rewrite source.
    """
    all_files = sorted(p for p in STATIC_DIR.iterdir() if p.is_file() and p.suffix != ".md")
    js_sources = {p.name: p.read_text(encoding="utf-8") for p in all_files if p.suffix == ".js"}
    deps = {
        name: _local_import_targets(src) & js_sources.keys() for name, src in js_sources.items()
    }

    name_map: dict[str, str] = {}
    assets: list[Asset] = []
    remaining = set(js_sources)
    while remaining:
        ready = {n for n in remaining if deps[n] <= name_map.keys()}
        if not ready:
            raise SystemExit(f"circular import among static JS modules: {sorted(remaining)}")
        for name in sorted(ready):
            content = _rewrite_imports(js_sources[name], name_map).encode("utf-8")
            hashed = _hashed_name(name, content)
            name_map[name] = hashed
            assets.append(Asset(hashed, content, _CONTENT_TYPES[".js"], IMMUTABLE_CACHE))
        remaining -= ready

    for p in all_files:
        if p.suffix != ".css":
            continue
        content = p.read_bytes()
        hashed = _hashed_name(p.name, content)
        name_map[p.name] = hashed
        assets.append(Asset(hashed, content, _CONTENT_TYPES[".css"], IMMUTABLE_CACHE))

    unrecognised = {p.name for p in all_files} - set(name_map)
    for name in sorted(unrecognised):
        print(f"warning: skipping unrecognised static file {name!r}", file=sys.stderr)

    return name_map, assets


def render_index_html(
    name_map: dict[str, str], *, api_base: str, supabase_url: str, turnstile_site_key: str = ""
) -> bytes:
    asset_base = f"{supabase_url}/storage/v1/object/public/{BUCKET}"

    def asset(name: str) -> str:
        return f"{asset_base}/{name_map.get(name, name)}"

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("index.html")
    rendered = template.render(
        supabase_url=supabase_url,
        api_base=api_base,
        asset=asset,
        # Same switch the app's own `GET /` uses (web.py): the Turnstile tag and
        # widget host only appear when a site key is configured. Read from the
        # environment rather than a flag so the deploy workflow passes it the same
        # way it passes it to Fly.
        turnstile_site_key=turnstile_site_key,
    )
    return rendered.encode("utf-8")


def _headers(key: str) -> dict[str, str]:
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def _with_retry(
    do_request: Callable[[], httpx.Response],
    *,
    what: str,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """Run `do_request` (one HTTP call) with up to `_RETRY_ATTEMPTS` tries.

    Retries — with `_RETRY_BACKOFF_S` backoff between attempts — happen only for
    `httpx.HTTPError` (connection errors, timeouts, ...) and 5xx responses; any
    other response (including 4xx) is returned immediately on the first try, no
    delay. Never logs `key`/credentials — `what` is a plain description of the
    call, not the request itself.

    Exhausting all attempts on an `httpx.HTTPError` raises `SystemExit` (there's
    no response to hand back to the caller). Exhausting all attempts on a 5xx
    response instead returns that last response, so the caller's own
    status-code check reports it exactly like a non-retried failure would.
    """
    for attempt in range(_RETRY_ATTEMPTS):
        is_last_attempt = attempt == _RETRY_ATTEMPTS - 1
        try:
            resp = do_request()
        except httpx.HTTPError as exc:
            if is_last_attempt:
                raise SystemExit(f"{what} failed after {_RETRY_ATTEMPTS} attempts: {exc}") from exc
            sleep(_RETRY_BACKOFF_S[attempt])
            continue
        if resp.status_code < 500 or is_last_attempt:
            return resp
        sleep(_RETRY_BACKOFF_S[attempt])
    raise AssertionError("unreachable: loop above always returns or raises")


def ensure_bucket(
    client: httpx.Client, url: str, key: str, *, sleep: Callable[[float], None] = time.sleep
) -> None:
    resp = _with_retry(
        lambda: client.get(f"{url}/storage/v1/bucket/{BUCKET}", headers=_headers(key)),
        what=f"check bucket {BUCKET!r}",
        sleep=sleep,
    )
    if resp.status_code == 200:
        return
    resp = _with_retry(
        lambda: client.post(
            f"{url}/storage/v1/bucket",
            headers=_headers(key),
            json={"id": BUCKET, "name": BUCKET, "public": True},
        ),
        what=f"create bucket {BUCKET!r}",
        sleep=sleep,
    )
    if resp.status_code not in (200, 201):
        raise SystemExit(
            f"failed to create bucket {BUCKET!r}: HTTP {resp.status_code} {resp.text[:200]}"
        )
    print(f"created public bucket {BUCKET!r}")


def upload(
    client: httpx.Client,
    url: str,
    key: str,
    asset: Asset,
    *,
    dry_run: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    dest = f"{BUCKET}/{asset.name}"
    if dry_run:
        print(
            f"[dry-run] would upload {dest} ({len(asset.content)} bytes, {asset.cache_control!r})"
        )
        return
    resp = _with_retry(
        lambda: client.post(
            f"{url}/storage/v1/object/{dest}",
            headers={
                **_headers(key),
                "Content-Type": asset.content_type,
                "x-upsert": "true",
                "Cache-Control": asset.cache_control,
            },
            content=asset.content,
        ),
        what=f"upload of {dest}",
        sleep=sleep,
    )
    if resp.status_code not in (200, 201):
        raise SystemExit(f"upload of {dest} failed: HTTP {resp.status_code} {resp.text[:200]}")
    print(f"uploaded {dest} ({len(asset.content)} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", choices=["local", "staging", "prod"], required=True)
    parser.add_argument(
        "--api-base",
        default=None,
        help="Absolute URL of the API the page should call. Defaults per --env "
        "(see DEFAULT_API_BASE).",
    )
    parser.add_argument(
        "--create-bucket",
        action="store_true",
        help="Create the public 'web' bucket first if it doesn't already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without making any network writes.",
    )
    args = parser.parse_args(argv)

    url, key = _env_credentials(args.env)
    api_base = args.api_base or DEFAULT_API_BASE[args.env]

    name_map, assets = build_assets()
    index_html = render_index_html(
        name_map,
        api_base=api_base,
        supabase_url=url,
        turnstile_site_key=os.environ.get("TURNSTILE_SITE_KEY", ""),
    )
    assets.append(Asset("index.html", index_html, "text/html; charset=utf-8", NO_CACHE))

    with httpx.Client(timeout=30) as client:
        if args.create_bucket:
            if args.dry_run:
                print(f"[dry-run] would ensure bucket {BUCKET!r} exists and is public")
            else:
                ensure_bucket(client, url, key)
        for asset in assets:
            upload(client, url, key, asset, dry_run=args.dry_run)

    public_url = f"{url}/storage/v1/object/public/{BUCKET}/index.html"
    print(f"public URL: {public_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
