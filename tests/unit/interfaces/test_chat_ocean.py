"""End-to-end `/chat` coverage for the OCEAN assessment workflow (Task 6).

In-memory container (`Container.use_in_memory_repos`), a `FakeLLM` standing
in for Gemini on the two turns the model actually gets asked anything (the
opening `start_workflow` call and the off-topic weather aside mid-run), and a
`ScriptedInterpreter` swapped in for `GeminiAnswerInterpreter` before
`rebuild_assessment()`/`rebuild_run_turn()` — see the "Controller rulings"
brief for Task 6.

The twenty answer turns never reach the model at all: `RunTurn._run` calls
`active_run.handle_turn` before it ever builds a prompt, and once a run is
open `HandleAssessmentTurn.execute` intercepts every turn until it declines
one (a control word it doesn't own, or off-topic text) — see
`sarjy.contexts.conversation.application.run_turn`.
"""

from __future__ import annotations

import json
import time
import uuid

import jwt
from fastapi.testclient import TestClient

from sarjy.config import Settings
from sarjy.container import Container
from sarjy.contexts.assessment.application.ports import Interpretation
from sarjy.contexts.assessment.domain.scoring import score
from sarjy.contexts.conversation.application.ports import (
    FunctionCall,
    LLMFinished,
    LLMFunctionCall,
    LLMText,
)
from sarjy.contexts.weather.infrastructure.mock_provider import MockProvider
from sarjy.main import create_app
from sarjy.shared.ids import UserId
from tests.unit.conversation.test_run_turn import FakeLLM

SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"  # noqa: S105


class ScriptedInterpreter:
    """Digits 1-5 answer the item in front of it; anything else is off-topic.

    Enough to drive `HandleAssessmentTurn` through a full run — and through
    one off-topic aside — without a real interpreter model.
    """

    async def interpret(
        self, item_text: str, scale_labels: list[str], user_text: str
    ) -> Interpretation:
        t = user_text.strip()
        if t.isdigit() and 1 <= int(t) <= 5:
            return Interpretation(value=int(t), confidence=1.0, control=None)
        return Interpretation(value=None, confidence=1.0, control="off_topic")


def _user() -> uuid.UUID:
    return uuid.uuid4()


def _token(user_id: uuid.UUID) -> str:
    claims = {"sub": str(user_id), "aud": "authenticated", "exp": int(time.time()) + 3600}
    return jwt.encode(claims, SECRET, algorithm="HS256")


def _events(body: str) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    out = []
    for block in body.split("\n\n"):
        if not block.strip() or "data: " not in block:
            continue
        name = block.split("\n", 1)[0].removeprefix("event: ").strip()
        data = json.loads(block.split("data: ", 1)[1])
        out.append((name, data))
    return out


def _post(
    client: TestClient, token: str, text: str, session_id: str | None
) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    r = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "client_turn_id": str(uuid.uuid4()),
            "text": text,
            "input_mode": "text",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    return _events(r.text)


def _sentences(events: list[tuple[str, dict]]) -> str:  # type: ignore[type-arg]
    return " ".join(d["text"] for name, d in events if name == "sentence")


def _session_id(events: list[tuple[str, dict]], fallback: str | None) -> str | None:  # type: ignore[type-arg]
    for name, d in events:
        if name == "session":
            return str(d["session_id"])
    return fallback


def _done_workflow(events: list[tuple[str, dict]]) -> dict | None:  # type: ignore[type-arg]
    for name, d in events:
        if name == "done":
            return d.get("workflow")
    raise AssertionError(f"no done event in {events!r}")


def _app(speculative: bool = False) -> tuple[TestClient, Container, uuid.UUID, str]:
    app = create_app(
        Settings(weather_provider="mock", speculative_enabled=speculative),  # type: ignore[call-arg]
        connect_db=False,
    )
    c: Container = app.state.container
    c.use_in_memory_repos()
    c.weather_provider = MockProvider(c.clock, temp_c=22.0)
    c.weather_cache = None
    c.rebuild_weather()
    # Swap the interpreter before rebuilding the assessment use cases/tools and
    # the turn orchestrator that depends on them, per the Task 6 ruling.
    c.interpreter = ScriptedInterpreter()
    c.rebuild_assessment()
    c.llm = FakeLLM(
        [
            # Turn 1 ("give me a personality test"): no run is open yet, so
            # `handle_turn` returns None and the model gets asked — it calls
            # `start_workflow`.
            [
                LLMFunctionCall(FunctionCall("start_workflow", {"workflow_id": "ocean_mini_ipip"})),
                LLMFinished("stop"),
            ],
            # The off-topic weather aside, hop 0: forced onto get_weather (W-8).
            [
                LLMFunctionCall(FunctionCall("get_weather", {"location": "Tokyo"})),
                LLMFinished("stop"),
            ],
            # ...hop 1: the model answers using the tool's own number.
            [LLMText("It's 22 degrees in Tokyo."), LLMFinished("stop")],
        ]
    )
    c.rebuild_run_turn()
    user = _user()
    token = _token(user)
    client = TestClient(app)
    client.__enter__()
    return client, c, user, token


def test_ocean_workflow_end_to_end_through_chat() -> None:
    client, c, user, token = _app()
    try:
        session_id: str | None = None

        # 1. "give me a personality test" -> start_workflow tool call -> intro
        # sentences spoken, workflow proposed.
        ev = _post(client, token, "give me a personality test", session_id)
        session_id = _session_id(ev, session_id)
        text = _sentences(ev)
        assert "Big Five" in text and "Ready?" in text
        wf = _done_workflow(ev)
        assert wf is not None and wf["status"] == "proposed"

        # 2. "yes" -> item one spoken, workflow active.
        ev = _post(client, token, "yes", session_id)
        session_id = _session_id(ev, session_id)
        text = _sentences(ev)
        assert "One:" in text and "How accurate is that for you?" in text
        wf = _done_workflow(ev)
        assert wf is not None and wf["status"] == "active" and wf["item"] == 1

        # 3. Answer items 1-5 with "4".
        for _ in range(5):
            ev = _post(client, token, "4", session_id)
            session_id = _session_id(ev, session_id)
            wf = _done_workflow(ev)
            assert wf is not None and wf["status"] == "active"
        assert wf["item"] == 6

        # 4. Off-topic aside mid-run: a weather question is answered (mocked),
        # and the run stays open on the same item.
        ev = _post(client, token, "what's the weather in Tokyo", session_id)
        session_id = _session_id(ev, session_id)
        tool_starts = [
            d["tool"] for name, d in ev if name == "tool_status" and d.get("state") == "start"
        ]
        assert tool_starts == ["get_weather"]
        text = _sentences(ev)
        assert "22" in text and "Tokyo" in text

        # The *next* prompt block (what the model would see if this turn had
        # fallen through to it) now carries the resume nudge, per
        # `ActiveRunAdapter`/`_BLOCK[Status.ACTIVE]`.
        import asyncio

        snap = asyncio.run(c.active_run.active_run(UserId(user)))
        assert snap is not None
        assert "Ready to continue? We were on item 6." in snap.prompt_block

        # 5. Finish the run: items 6-20 with "4" too (twenty answers total).
        for _ in range(6, 21):
            ev = _post(client, token, "4", session_id)
            session_id = _session_id(ev, session_id)
        wf = _done_workflow(ev)
        assert wf is not None and wf["status"] == "complete"

        # 6. `/workflow/latest` reports exactly what `score()` computes for
        # twenty answers of 4.
        r = client.get("/workflow/latest", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "complete"

        ins = asyncio.run(c.instrument_repo.get("ocean_mini_ipip"))
        expected = score(ins, dict.fromkeys(range(1, 21), 4)).as_dict()
        for trait in "OCEAN":
            assert body["results"][trait] == expected[trait], trait
    finally:
        client.__exit__(None, None, None)


def test_a_speculative_turn_during_a_run_is_never_a_guess() -> None:
    """L-3: an open run disqualifies speculation before anything is buffered.

    Answering an item advances the run — a write with no confirmation step
    behind it — so a turn the assessment engine will take is run and persisted
    like any other, and the confirmation the client sends afterwards finds
    nothing to confirm.
    """
    client, c, _user_id, token = _app(speculative=True)
    try:
        ev = _post(client, token, "give me a personality test", None)
        session_id = _session_id(ev, None)
        ev = _post(client, token, "yes", session_id)
        assert "One:" in _sentences(ev)
        before = len(c.messages.items)

        r = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "client_turn_id": "spec-run",
                "text": "4",
                "input_mode": "text",
                "speculative": True,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        wf = _done_workflow(_events(r.text))
        assert wf is not None and wf["item"] == 2  # the run really did advance
        # ...and the turn that advanced it is in the transcript already.
        assert [m.client_turn_id for m in c.messages.items[before:]] == ["spec-run", "spec-run"]

        conf = client.post(
            "/chat/confirm",
            json={"client_turn_id": "spec-run", "text": "4"},
            headers={"Authorization": f"Bearer {token}"},
        )
        # Nothing was ever parked (an open run disqualifies speculation outright),
        # so the confirmation has nothing to contradict: 202, and no second write.
        assert conf.status_code == 202
        assert len(c.messages.items) == before + 2
    finally:
        client.__exit__(None, None, None)
