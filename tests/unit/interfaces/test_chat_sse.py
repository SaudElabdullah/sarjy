import asyncio
import json
import time
import uuid

import jwt
from fastapi.testclient import TestClient

from sarjy.config import Settings
from sarjy.contexts.conversation.application.ports import (
    FunctionCall,
    LLMFinished,
    LLMFunctionCall,
    LLMText,
)
from sarjy.main import create_app
from sarjy.shared.ids import UserId
from tests.unit.conversation.test_run_turn import FakeLLM

SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"  # noqa: S105


def _tok(sub: str | None = None) -> str:
    claims = {"sub": sub or str(uuid.uuid4()), "aud": "authenticated", "exp": int(time.time()) + 60}
    return jwt.encode(claims, SECRET, algorithm="HS256")


def _spec_app():  # type: ignore[no-untyped-def]
    """An app with speculation on and a scripted one-sentence reply."""
    app = create_app(Settings(speculative_enabled=True), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()
    c.llm = FakeLLM([[LLMText("Hello there."), LLMFinished("stop")]])
    c.rebuild_run_turn()
    return app, c


def test_chat_streams_sse() -> None:
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()  # test helper added in container
    c.llm = FakeLLM([[LLMText("Hello there."), LLMFinished("stop")]])
    c.rebuild_run_turn()
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={"client_turn_id": "t1", "text": "hi"},
            headers={"Authorization": f"Bearer {_tok()}"},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [b for b in r.text.split("\n\n") if b]
    assert events[0].startswith("event: session")
    sentence = next(e for e in events if e.startswith("event: sentence"))
    assert json.loads(sentence.split("data: ")[1])["text"] == "Hello there."
    assert events[-1].startswith("event: done")


def test_chat_requires_auth() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        r = client.post("/chat", json={"client_turn_id": "t", "text": "x"})
    assert r.status_code == 401


def test_speculative_request_is_ignored_when_the_feature_is_off() -> None:
    # speculative_enabled defaults to False, so a client asking for a speculative
    # turn gets an ordinary one — including the assistant row it would otherwise
    # have buffered for a confirmation that never comes.
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()
    c.llm = FakeLLM([[LLMText("Hello there."), LLMFinished("stop")]])
    c.rebuild_run_turn()
    assert c.settings.speculative_enabled is False
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={"client_turn_id": "t1", "text": "hi", "speculative": True},
            headers={"Authorization": f"Bearer {_tok()}"},
        )
    assert r.status_code == 200
    assert [m.role for m in c.messages.items] == ["user", "assistant"]
    assert c.messages.items[1].content == "Hello there."


def test_remember_via_chat_persists_and_appears_in_next_turns_prompt() -> None:
    # Executable stand-in for the parked memory eval (m01/m09-style script): a
    # scripted LLM calls `remember`, we check the fact landed in the repo, then a
    # second /chat turn's prompt must carry it back in the <facts> block.
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()
    c.llm = FakeLLM(
        [
            [
                LLMFunctionCall(
                    FunctionCall("remember", {"key": "favorite_color", "value": "teal"})
                ),
                LLMFinished("stop"),
            ],
            [LLMText("Got it, teal it is."), LLMFinished("stop")],
            [LLMText("Your favorite color is teal."), LLMFinished("stop")],
        ]
    )
    c.rebuild_run_turn()
    uid = uuid.uuid4()
    claims = {"sub": str(uid), "aud": "authenticated", "exp": int(time.time()) + 60}
    token = jwt.encode(claims, SECRET, algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        r1 = client.post(
            "/chat",
            json={
                "client_turn_id": "t1",
                "text": "My favorite color is teal.",
                "input_mode": "text",
            },
            headers=headers,
        )
        assert r1.status_code == 200

        stored = asyncio.run(c.memory_repo.get_by_key(UserId(uid), "favorite_color"))
        assert stored is not None and stored.value == "teal"

        r2 = client.post(
            "/chat",
            json={
                "client_turn_id": "t2",
                "text": "What's my favorite color?",
                "input_mode": "text",
            },
            headers=headers,
        )
        assert r2.status_code == 200

    assert "favorite_color: teal" in c.llm.requests[-1].system


def test_forget_via_chat_removes_fact_from_a_fresh_sessions_prompt() -> None:
    # Executable acceptance test for the "no stale memory after forget" scenario
    # (Phase 3 review item I3 / acceptance script M): remember a fact, confirm a
    # *different, freshly-minted* session still recalls it (memory is per-user,
    # not per-session), forget it via the `forget` tool, then confirm a later
    # turn on that same fresh session no longer carries it.
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()
    c.llm = FakeLLM(
        [
            [
                LLMFunctionCall(
                    FunctionCall("remember", {"key": "favorite_color", "value": "teal"})
                ),
                LLMFinished("stop"),
            ],
            [LLMText("Got it, teal it is."), LLMFinished("stop")],
            [LLMText("Your favorite color is teal."), LLMFinished("stop")],
            [
                LLMFunctionCall(FunctionCall("forget", {"key": "favorite_color"})),
                LLMFinished("stop"),
            ],
            [LLMText("Okay, forgotten."), LLMFinished("stop")],
            [LLMText("I don't have that stored."), LLMFinished("stop")],
        ]
    )
    c.rebuild_run_turn()
    uid = uuid.uuid4()
    claims = {"sub": str(uid), "aud": "authenticated", "exp": int(time.time()) + 60}
    token = jwt.encode(claims, SECRET, algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        # Turn 1: remember teal (no explicit session — server mints one).
        r1 = client.post(
            "/chat",
            json={
                "client_turn_id": "t1",
                "text": "My favorite color is teal.",
                "input_mode": "text",
            },
            headers=headers,
        )
        assert r1.status_code == 200

        # Turn 2: an EXPLICIT, never-before-seen session id. The server can't
        # resume it (it doesn't exist), so this starts a brand-new session —
        # proving the fact comes back via user-scoped memory, not session history.
        fresh_session = str(uuid.uuid4())
        r2 = client.post(
            "/chat",
            json={
                "session_id": fresh_session,
                "client_turn_id": "t2",
                "text": "What's my favorite color?",
                "input_mode": "text",
            },
            headers=headers,
        )
        assert r2.status_code == 200
        assert "favorite_color: teal" in c.llm.requests[-1].system

        # Turn 3: forget it, on the same fresh session.
        r3 = client.post(
            "/chat",
            json={
                "session_id": fresh_session,
                "client_turn_id": "t3",
                "text": "Forget my favorite color.",
                "input_mode": "text",
            },
            headers=headers,
        )
        assert r3.status_code == 200

        # Turn 4: the fact must no longer appear in the prompt.
        r4 = client.post(
            "/chat",
            json={
                "session_id": fresh_session,
                "client_turn_id": "t4",
                "text": "What's my favorite color?",
                "input_mode": "text",
            },
            headers=headers,
        )
        assert r4.status_code == 200

    assert "favorite_color" not in c.llm.requests[-1].system
    assert "(none stored)" in c.llm.requests[-1].system


# ---------------------------------------------------------------------------
# L-3: /chat/confirm.
# ---------------------------------------------------------------------------


def test_a_speculative_turn_is_answered_but_not_written_until_confirmed() -> None:
    app, c = _spec_app()
    headers = {"Authorization": f"Bearer {_tok()}"}
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={"client_turn_id": "spec-1", "text": "hello there friend", "speculative": True},
            headers=headers,
        )
        assert r.status_code == 200
        assert "Hello there." in r.text
        # The answer is on the wire and nothing is in the transcript.
        assert c.messages.items == []

        ok = client.post(
            "/chat/confirm",
            json={"client_turn_id": "spec-1", "text": "Hello there, friend!"},
            headers=headers,
        )
    assert ok.status_code == 204
    assert [m.role for m in c.messages.items] == ["user", "assistant"]
    assert [m.client_turn_id for m in c.messages.items] == ["spec-1", "spec-1"]


def test_a_confirmation_that_does_not_match_is_a_409_and_writes_nothing() -> None:
    app, c = _spec_app()
    headers = {"Authorization": f"Bearer {_tok()}"}
    with TestClient(app) as client:
        client.post(
            "/chat",
            json={"client_turn_id": "spec-2", "text": "hello there friend", "speculative": True},
            headers=headers,
        )
        r = client.post(
            "/chat/confirm",
            json={"client_turn_id": "spec-2", "text": "hello there stranger"},
            headers=headers,
        )
    assert r.status_code == 409
    assert c.messages.items == []


def test_an_unknown_turn_id_is_a_202() -> None:
    # C1: "nothing parked under that id" is not "the guess was wrong". It is far
    # more often a confirmation that overtook its own turn, so the transcript is
    # recorded for the park to settle against and the client is told to sit
    # still (202). An id that is simply never parked — this one — costs an entry
    # that expires unread.
    app, c = _spec_app()
    with TestClient(app) as client:
        r = client.post(
            "/chat/confirm",
            json={"client_turn_id": "never-seen", "text": "hello"},
            headers={"Authorization": f"Bearer {_tok()}"},
        )
    assert r.status_code == 202
    assert c.messages.items == []


def test_another_user_cannot_confirm_someone_elses_turn() -> None:
    # C1: the pending turn is keyed by its owner, so a caller guessing at ids
    # neither writes it nor consumes it out from under the owner.
    app, c = _spec_app()
    mine, theirs = _tok(), _tok()
    with TestClient(app) as client:
        client.post(
            "/chat",
            json={"client_turn_id": "spec-3", "text": "hello there friend", "speculative": True},
            headers={"Authorization": f"Bearer {mine}"},
        )
        stolen = client.post(
            "/chat/confirm",
            json={"client_turn_id": "spec-3", "text": "hello there friend"},
            headers={"Authorization": f"Bearer {theirs}"},
        )
        # 202, not 409: the lookup is scoped by the caller's own id, so from the
        # thief's side there is simply nothing parked. What matters is that the
        # owner's entry is untouched — including by the early confirm, which is
        # recorded under the thief's key and can never be read by the owner's park.
        assert stolen.status_code == 202
        assert c.messages.items == []

        ok = client.post(
            "/chat/confirm",
            json={"client_turn_id": "spec-3", "text": "hello there friend"},
            headers={"Authorization": f"Bearer {mine}"},
        )
    assert ok.status_code == 204
    assert [m.role for m in c.messages.items] == ["user", "assistant"]


def test_confirm_requires_auth() -> None:
    app, _ = _spec_app()
    with TestClient(app) as client:
        r = client.post("/chat/confirm", json={"client_turn_id": "spec-1", "text": "hi"})
    assert r.status_code == 401


def test_a_speculative_remember_promotes_the_turn_and_persists_it_unconfirmed() -> None:
    # L-3's hard edge: a stored fact cannot be un-stored by declining to confirm,
    # so the turn stops being a guess the moment the model reaches for the tool.
    # The whole turn is written on the spot, under the id it was sent with, and
    # the confirmation that follows finds nothing — which is the point.
    app = create_app(Settings(speculative_enabled=True), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()
    c.llm = FakeLLM(
        [
            [
                LLMFunctionCall(
                    FunctionCall("remember", {"key": "favorite_color", "value": "teal"})
                ),
                LLMFinished("stop"),
            ],
            [LLMText("Got it, teal it is."), LLMFinished("stop")],
        ]
    )
    c.rebuild_run_turn()
    uid = uuid.uuid4()
    headers = {"Authorization": f"Bearer {_tok(str(uid))}"}
    text = "My favorite color is teal."

    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={"client_turn_id": "spec-mem", "text": text, "speculative": True},
            headers=headers,
        )
        assert r.status_code == 200

        stored = asyncio.run(c.memory_repo.get_by_key(UserId(uid), "favorite_color"))
        assert stored is not None and stored.value == "teal"
        assert [m.role for m in c.messages.items] == ["user", "assistant"]
        assert [m.client_turn_id for m in c.messages.items] == ["spec-mem", "spec-mem"]
        assert len(c.messages.tool_calls) == 1

        conf = client.post(
            "/chat/confirm",
            json={"client_turn_id": "spec-mem", "text": text},
            headers=headers,
        )
    # 202: the turn was promoted mid-flight and never parked, so there is nothing
    # to contradict and nothing to write. The client does nothing on a 202, which
    # is exactly right — the rows are already there.
    assert conf.status_code == 202
    assert len(c.messages.items) == 2  # and the confirmation wrote nothing a second time


# ---------------------------------------------------------------------------
# I5: what the server-side breakdown accounts for.
# ---------------------------------------------------------------------------


def _done_timings(text: str) -> dict:  # type: ignore[type-arg]
    done = next(b for b in text.split("\n\n") if b.startswith("event: done"))
    return json.loads(done.split("data: ")[1])["timings"]


def test_done_timings_account_for_the_gate_the_tools_and_the_whole_turn() -> None:
    # Three gaps the breakdown used to leave, all of them things a reader would
    # otherwise have to guess at:
    #
    #   t_auth  — the JWT verify and the rate limiter, which happen before
    #             `RunTurn` exists and so showed up nowhere. The client's
    #             `t_request_ms` covered them; nothing on the server did.
    #   t_total — the turn end to end, so the stages can be checked against
    #             something rather than only against each other.
    #   t_tool  — every `t_tool_*` summed. The per-tool keys stay (that is the
    #             question a breakdown answers) but "how much of this turn was
    #             tool time" was a sum the reader had to do, once per turn.
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()
    c.llm = FakeLLM(
        [
            [
                LLMFunctionCall(FunctionCall("get_weather", {"location": "Tokyo"})),
                LLMFinished("stop"),
            ],
            [LLMText("It's mild in Tokyo."), LLMFinished("stop")],
        ]
    )
    c.rebuild_run_turn()
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={"client_turn_id": "t-timings", "text": "weather in tokyo"},
            headers={"Authorization": f"Bearer {_tok()}"},
        )
    timings = _done_timings(r.text)
    assert {"t_auth", "t_total", "t_tool"} <= timings.keys()
    # The per-tool key is still there, and t_tool is its sum.
    assert "t_tool_get_weather" in timings
    assert timings["t_tool"] == timings["t_tool_get_weather"]
    assert all(isinstance(v, int) for v in timings.values())
    # The turn cannot have taken less time than the stages inside it.
    assert timings["t_total"] >= timings["t_tool"]


def test_a_turn_with_no_tool_call_reports_no_t_tool() -> None:
    # Omitted rather than zero: a `t_tool: 0` on every chat turn is a column of
    # noise in a table read for outliers.
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()
    c.llm = FakeLLM([[LLMText("Hello there."), LLMFinished("stop")]])
    c.rebuild_run_turn()
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={"client_turn_id": "t-notool", "text": "hi"},
            headers={"Authorization": f"Bearer {_tok()}"},
        )
    timings = _done_timings(r.text)
    assert "t_tool" not in timings
    assert {"t_auth", "t_total"} <= timings.keys()


def test_the_assistant_row_stores_the_same_timings_the_client_was_told() -> None:
    # The stored breakdown and the one on the wire have to be the same dict, or
    # `telemetry_turns.server_timings` and `messages.timings` disagree about the
    # turn they both describe.
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()
    c.llm = FakeLLM([[LLMText("Hello there."), LLMFinished("stop")]])
    c.rebuild_run_turn()
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={"client_turn_id": "t-stored", "text": "hi"},
            headers={"Authorization": f"Bearer {_tok()}"},
        )
    assert c.messages.items[-1].timings == _done_timings(r.text)


def test_a_flood_of_junk_confirmations_cannot_grow_the_store_unbounded() -> None:
    # `/chat/confirm` is the only way into this worker's memory with no turn
    # behind it: every miss records an early confirm, in case the turn it names
    # is still streaming. 300 from one caller must leave at most the per-user
    # cap — and the entries kept are the newest, which are the ones that could
    # still have a turn to settle.
    app, c = _spec_app()
    headers = {"Authorization": f"Bearer {_tok()}"}
    with TestClient(app) as client:
        for i in range(300):
            r = client.post(
                "/chat/confirm",
                json={"client_turn_id": f"junk-{i}", "text": "hello there"},
                headers=headers,
            )
            assert r.status_code == 202
    assert c.run_turn.speculation.early_size <= 256
    assert c.messages.items == []
