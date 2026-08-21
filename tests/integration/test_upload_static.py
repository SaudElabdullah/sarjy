"""`scripts/upload_static.py` against the local Supabase Storage stack.

Not a mock: runs the real uploader (bucket creation, content-hash renaming, import
rewriting, upload) against http://127.0.0.1:54321 and fetches the published page
back over HTTP, the same way a browser would.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "upload_static.py"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]


def _load_upload_static() -> ModuleType:
    # scripts/ isn't part of the installed `sarjy` package, so it's loaded straight
    # from its file path rather than imported by name.
    spec = importlib.util.spec_from_file_location("upload_static", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclasses needs the module registered in sys.modules (it looks itself up
    # there while processing the @dataclass decorator on Asset) before exec_module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


upload_static = _load_upload_static()


def test_upload_static_publishes_a_working_page_to_local_storage() -> None:
    name_map, assets = upload_static.build_assets()
    index_html = upload_static.render_index_html(
        name_map, api_base="http://127.0.0.1:8000", supabase_url=SUPABASE_URL
    )
    assets.append(
        upload_static.Asset(
            "index.html", index_html, "text/html; charset=utf-8", upload_static.NO_CACHE
        )
    )

    with httpx.Client(timeout=30) as client:
        upload_static.ensure_bucket(client, SUPABASE_URL, SERVICE_ROLE_KEY)
        for asset in assets:
            upload_static.upload(client, SUPABASE_URL, SERVICE_ROLE_KEY, asset, dry_run=False)

        r = client.get(f"{SUPABASE_URL}/storage/v1/object/public/web/index.html")
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-cache"
        body = r.text

        # Every module/stylesheet the page references is the content-hashed name,
        # not the bare filename FastAPI's own /static mount would serve.
        for original in ("voice.js", "memory.js", "ocean.js", "app.css", "supabase.js"):
            hashed = name_map[original]
            assert hashed != original
            assert hashed in body
            assert f"/static/{original}" not in body

        # api_base is absolute (the page is no longer same-origin with the API).
        assert '<script src="http://127.0.0.1:8000/config.js"></script>' in body

        # A hashed JS asset is fetchable and carries the immutable cache header —
        # the whole point of content-hashing the filename.
        hashed_voice = name_map["voice.js"]
        r_js = client.get(f"{SUPABASE_URL}/storage/v1/object/public/web/{hashed_voice}")
        assert r_js.status_code == 200
        assert r_js.headers["cache-control"] == "public, max-age=31536000, immutable"
        # The rewritten import inside voice.js points at fillers.js's hashed name too.
        assert name_map["fillers.js"] in r_js.text
