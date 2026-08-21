"""`POST /internal/audit/run` — shared-token auth and wiring (PRD Layer 7)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from sarjy.config import Settings
from sarjy.contexts.guardrails.application.audit import AuditRunResult
from sarjy.main import create_app


class StubAuditWorker:
    def __init__(self, result: AuditRunResult) -> None:
        self.result = result
        self.calls = 0

    async def run_once(self, limit: int = 50) -> AuditRunResult:
        self.calls += 1
        return self.result


def test_run_audit_503_when_token_unset() -> None:
    # internal_token=None explicitly: local dev's .env sets INTERNAL_TOKEN (so
    # `make run`/integration tests can exercise the endpoint), and pydantic-
    # settings reads that file regardless of the test's own env — a bare
    # `Settings()` here would silently pick it up and this test would assert
    # nothing.
    app = create_app(Settings(internal_token=None), connect_db=False)
    with TestClient(app) as c:
        r = c.post("/internal/audit/run", headers={"X-Internal-Token": "anything"})
    assert r.status_code == 503


def test_run_audit_401_on_mismatch() -> None:
    app = create_app(Settings(internal_token=SecretStr("correct-token")), connect_db=False)
    with TestClient(app) as c:
        r = c.post("/internal/audit/run", headers={"X-Internal-Token": "wrong-token"})
    assert r.status_code == 401


def test_run_audit_401_when_header_missing() -> None:
    app = create_app(Settings(internal_token=SecretStr("correct-token")), connect_db=False)
    with TestClient(app) as c:
        r = c.post("/internal/audit/run")
    assert r.status_code == 401


def test_run_audit_503_when_worker_not_configured() -> None:
    # connect_db=False: Container.rebuild_audit leaves audit_worker as None.
    app = create_app(Settings(internal_token=SecretStr("correct-token")), connect_db=False)
    with TestClient(app) as c:
        r = c.post("/internal/audit/run", headers={"X-Internal-Token": "correct-token"})
    assert r.status_code == 503


async def test_run_audit_401_on_non_ascii_header_byte() -> None:
    # hmac.compare_digest raises TypeError on a non-ASCII `str` operand;
    # `TestClient`'s httpx transport itself rejects a raw non-ASCII header
    # value before it reaches the app, so this needs a raw ASGI request (a
    # (bytes, bytes) header pair bypasses httpx's own str-header encoding) to
    # reproduce what a real client sending a stray high byte in the token
    # header would trigger. Before the `_eq` fix this 500ed instead of 401ing.
    app = create_app(Settings(internal_token=SecretStr("correct-token")), connect_db=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        r = await client.post("/internal/audit/run", headers=[(b"x-internal-token", b"\xe9wrong")])
    assert r.status_code == 401


def test_run_audit_200_returns_processed_and_flagged_counts() -> None:
    app = create_app(Settings(internal_token=SecretStr("correct-token")), connect_db=False)
    stub = StubAuditWorker(AuditRunResult(processed=3, flagged=1))
    app.state.container.audit_worker = stub
    with TestClient(app) as c:
        r = c.post("/internal/audit/run", headers={"X-Internal-Token": "correct-token"})
    assert r.status_code == 200
    assert r.json() == {"processed": 3, "flagged": 1}
    assert stub.calls == 1
