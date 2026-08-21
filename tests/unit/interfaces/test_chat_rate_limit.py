import asyncio
import json
import time
import uuid

import jwt
from fastapi.testclient import TestClient

from sarjy.config import Settings
from sarjy.contexts.conversation.application.ports import LLMFinished, LLMText
from sarjy.contexts.guardrails.infrastructure.pg_rate_limiter import RateLimitResult
from sarjy.main import create_app
from sarjy.shared.ids import UserId
from tests.unit.conversation.test_run_turn import FakeLLM

SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"  # noqa: S105


def _tok() -> str:
    claims = {"sub": str(uuid.uuid4()), "aud": "authenticated", "exp": int(time.time()) + 60}
    return jwt.encode(claims, SECRET, algorithm="HS256")


class StubLimiter:
    def __init__(self, allowed: bool, retry_after_s: int = 42) -> None:
        self.allowed, self.retry_after_s = allowed, retry_after_s
        self.namespaces: list[str] = []

    async def hit(
        self, user_id: UserId, is_anonymous: bool, *, window_ns: str = "chat"
    ) -> RateLimitResult:
        self.namespaces.append(window_ns)
        return RateLimitResult(self.allowed, self.retry_after_s)


def test_chat_returns_429_when_rate_limited() -> None:
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    c.rate_limiter = StubLimiter(allowed=False)
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={"client_turn_id": "t1", "text": "hi"},
            headers={"Authorization": f"Bearer {_tok()}"},
        )
    assert r.status_code == 429
    assert r.headers["retry-after"] == "42"


class SlowLimiter:
    """An allowing limiter that takes measurable time — the round trip a real
    one makes to Postgres, made long enough for an integer millisecond to see."""

    async def hit(
        self, user_id: UserId, is_anonymous: bool, *, window_ns: str = "chat"
    ) -> RateLimitResult:
        await asyncio.sleep(0.02)
        return RateLimitResult(True)


def test_t_auth_measures_the_gate_the_turn_never_sees() -> None:
    # I5: the JWT verify and the rate limiter both finish before `RunTurn`
    # exists, so nothing in the turn could time them and the breakdown simply
    # did not mention them. `_arrival` is declared ahead of `rate_limited` in the
    # route's dependencies, which is what puts both inside the measurement.
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()
    c.llm = FakeLLM([[LLMText("Hello there."), LLMFinished("stop")]])
    c.rebuild_run_turn()
    c.rate_limiter = SlowLimiter()
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={"client_turn_id": "t-auth", "text": "hi"},
            headers={"Authorization": f"Bearer {_tok()}"},
        )
    done = next(b for b in r.text.split("\n\n") if b.startswith("event: done"))
    timings = json.loads(done.split("data: ")[1])["timings"]
    assert timings["t_auth"] >= 20
    assert timings["t_total"] >= 0
