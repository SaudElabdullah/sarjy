"""scripts/upload_static.py: bounded retry for transient Supabase Storage errors,
mocked via respx (no network, no local Supabase needed).

Retries (0.5s / 1s / 2s backoff, 3 attempts total) apply only to `httpx.HTTPError`
(connection errors, timeouts, ...) and 5xx responses; a 4xx response aborts with
`SystemExit` on the very first try, no delay. `sleep` is injected so no test ever
really sleeps.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from scripts.upload_static import Asset, ensure_bucket, upload

URL = "https://proj.supabase.co"
KEY = "sb-service-role-secret"

BUCKET_URL = f"{URL}/storage/v1/bucket/web"
CREATE_URL = f"{URL}/storage/v1/bucket"
OBJECT_URL = f"{URL}/storage/v1/object/web/app.abc12345.css"

ASSET = Asset(
    "app.abc12345.css", b"body{}", "text/css; charset=utf-8", "public, max-age=31536000, immutable"
)


class _RecordingSleep:
    """Fake `sleep` callable: records requested delays instead of sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


# ------------------------------------------------------------------------- upload
@respx.mock
def test_upload_retries_502_then_succeeds() -> None:
    route = respx.post(OBJECT_URL)
    route.side_effect = [httpx.Response(502, text="bad gateway"), httpx.Response(200)]
    sleep = _RecordingSleep()

    with httpx.Client() as client:
        upload(client, URL, KEY, ASSET, dry_run=False, sleep=sleep)

    assert route.call_count == 2
    assert sleep.calls == [0.5]


@respx.mock
def test_upload_4xx_aborts_immediately_without_retry() -> None:
    route = respx.post(OBJECT_URL).mock(return_value=httpx.Response(403, text="forbidden"))
    sleep = _RecordingSleep()

    with httpx.Client() as client, pytest.raises(SystemExit, match="403") as exc_info:
        upload(client, URL, KEY, ASSET, dry_run=False, sleep=sleep)

    assert route.call_count == 1
    assert sleep.calls == []
    assert KEY not in str(exc_info.value)


@respx.mock
def test_upload_exhausts_retries_on_repeated_503() -> None:
    route = respx.post(OBJECT_URL).mock(return_value=httpx.Response(503, text="unavailable"))
    sleep = _RecordingSleep()

    with httpx.Client() as client, pytest.raises(SystemExit, match="503") as exc_info:
        upload(client, URL, KEY, ASSET, dry_run=False, sleep=sleep)

    assert route.call_count == 3
    assert sleep.calls == [0.5, 1.0]
    assert KEY not in str(exc_info.value)


@respx.mock
def test_upload_retries_connection_error_then_succeeds() -> None:
    route = respx.post(OBJECT_URL)
    route.side_effect = [httpx.ConnectError("refused"), httpx.Response(201)]
    sleep = _RecordingSleep()

    with httpx.Client() as client:
        upload(client, URL, KEY, ASSET, dry_run=False, sleep=sleep)

    assert route.call_count == 2
    assert sleep.calls == [0.5]


@respx.mock
def test_upload_never_prints_the_service_role_key(capsys: pytest.CaptureFixture[str]) -> None:
    respx.post(OBJECT_URL).mock(return_value=httpx.Response(200))
    sleep = _RecordingSleep()

    with httpx.Client() as client:
        upload(client, URL, KEY, ASSET, dry_run=False, sleep=sleep)

    assert KEY not in capsys.readouterr().out


@respx.mock
def test_upload_dry_run_makes_no_request() -> None:
    route = respx.post(OBJECT_URL).mock(return_value=httpx.Response(200))

    with httpx.Client() as client:
        upload(client, URL, KEY, ASSET, dry_run=True)

    assert route.call_count == 0


# ------------------------------------------------------------------- ensure_bucket
@respx.mock
def test_ensure_bucket_returns_immediately_when_bucket_already_exists() -> None:
    get_route = respx.get(BUCKET_URL).mock(return_value=httpx.Response(200))
    post_route = respx.post(CREATE_URL)
    sleep = _RecordingSleep()

    with httpx.Client() as client:
        ensure_bucket(client, URL, KEY, sleep=sleep)

    assert get_route.call_count == 1
    assert post_route.call_count == 0
    assert sleep.calls == []


@respx.mock
def test_ensure_bucket_creates_when_missing_and_retries_502_then_succeeds() -> None:
    respx.get(BUCKET_URL).mock(return_value=httpx.Response(404))
    post_route = respx.post(CREATE_URL)
    post_route.side_effect = [httpx.Response(502, text="bad gateway"), httpx.Response(201)]
    sleep = _RecordingSleep()

    with httpx.Client() as client:
        ensure_bucket(client, URL, KEY, sleep=sleep)

    assert post_route.call_count == 2
    assert sleep.calls == [0.5]


@respx.mock
def test_ensure_bucket_create_4xx_aborts_immediately_without_retry() -> None:
    respx.get(BUCKET_URL).mock(return_value=httpx.Response(404))
    post_route = respx.post(CREATE_URL).mock(return_value=httpx.Response(400, text="bad request"))
    sleep = _RecordingSleep()

    with httpx.Client() as client, pytest.raises(SystemExit, match="400"):
        ensure_bucket(client, URL, KEY, sleep=sleep)

    assert post_route.call_count == 1
    assert sleep.calls == []


@respx.mock
def test_ensure_bucket_create_exhausts_retries_on_repeated_503() -> None:
    respx.get(BUCKET_URL).mock(return_value=httpx.Response(404))
    post_route = respx.post(CREATE_URL).mock(return_value=httpx.Response(503, text="unavailable"))
    sleep = _RecordingSleep()

    with httpx.Client() as client, pytest.raises(SystemExit, match="503"):
        ensure_bucket(client, URL, KEY, sleep=sleep)

    assert post_route.call_count == 3
    assert sleep.calls == [0.5, 1.0]
