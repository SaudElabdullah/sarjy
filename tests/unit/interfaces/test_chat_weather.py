"""Executable stand-in for the parked weather eval (tests/evals/weather.jsonl):
Task 6 Step 4 parks the real 20/20 Gemini run pending an API key, so this
exercises the same wiring end to end through `/chat` with a scripted LLM and
the mock weather provider instead.
"""

from __future__ import annotations

import time
import uuid

import jwt
from fastapi.testclient import TestClient

from sarjy.config import Settings
from sarjy.container import Container
from sarjy.contexts.conversation.application.ports import (
    FunctionCall,
    LLMFinished,
    LLMFunctionCall,
    LLMText,
)
from sarjy.contexts.weather.infrastructure.mock_provider import MockProvider
from sarjy.main import create_app
from tests.unit.conversation.test_run_turn import FakeLLM

SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"  # noqa: S105


def _tok() -> str:
    claims = {"sub": str(uuid.uuid4()), "aud": "authenticated", "exp": int(time.time()) + 60}
    return jwt.encode(claims, SECRET, algorithm="HS256")


def _app_with_mock_weather(*, temp_c: float = 22.0, fail: bool = False) -> tuple:  # type: ignore[type-arg]
    app = create_app(Settings(weather_provider="mock"), connect_db=False)  # type: ignore[call-arg]
    c: Container = app.state.container
    c.use_in_memory_repos()
    # `use_in_memory_repos` -> `rebuild_weather` won't replace an already-set
    # provider, so overriding it here and rebuilding again pins the mock's
    # exact temperature/failure mode this test needs.
    c.weather_provider = MockProvider(c.clock, temp_c=temp_c, fail=fail)
    c.weather_cache = None
    c.rebuild_weather()
    return app, c


def test_weather_question_forces_get_weather_tool_and_grounds_the_response() -> None:
    app, c = _app_with_mock_weather(temp_c=22.0)
    c.llm = FakeLLM(
        [
            [
                LLMFunctionCall(FunctionCall("get_weather", {"location": "Tokyo"})),
                LLMFinished("stop"),
            ],
            [LLMText("It's 22 degrees in Tokyo."), LLMFinished("stop")],
        ]
    )
    c.rebuild_run_turn()

    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={
                "client_turn_id": "t1",
                "text": "what's the weather in Tokyo",
                "input_mode": "text",
            },
            headers={"Authorization": f"Bearer {_tok()}"},
        )
    assert r.status_code == 200

    # The weather-shaped question forces the model's first hop onto get_weather...
    assert c.llm.requests[0].force_tool == "get_weather"
    # ...and what actually goes back to the model as the function_response is the
    # mock provider's own number, untouched.
    function_response = next(
        m.function_response for m in c.llm.requests[1].messages if m.function_response is not None
    )
    assert function_response.response["temp_c"] == 22


def test_weather_question_for_unknown_place_is_not_found() -> None:
    app, c = _app_with_mock_weather()
    c.llm = FakeLLM(
        [
            [
                LLMFunctionCall(FunctionCall("get_weather", {"location": "Gondor"})),
                LLMFinished("stop"),
            ],
            [LLMText("I couldn't find a place called Gondor."), LLMFinished("stop")],
        ]
    )
    c.rebuild_run_turn()

    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={
                "client_turn_id": "t1",
                "text": "what's the weather in Gondor",
                "input_mode": "text",
            },
            headers={"Authorization": f"Bearer {_tok()}"},
        )
    assert r.status_code == 200

    assert c.llm.requests[0].force_tool == "get_weather"
    function_response = next(
        m.function_response for m in c.llm.requests[1].messages if m.function_response is not None
    )
    assert function_response.response["error"] == "not_found"
