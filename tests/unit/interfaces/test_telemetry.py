"""`POST /telemetry` — client latency marks ingestion (PRD §9.4, L-1/L-2)."""

import time
import uuid

import jwt
from fastapi.testclient import TestClient

from sarjy.config import Settings
from sarjy.contexts.conversation.application.ports import LLMFinished, LLMText
from sarjy.main import create_app
from tests.unit.conversation.test_run_turn import FakeLLM
from tests.unit.interfaces.test_chat_rate_limit import StubLimiter

SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"  # noqa: S105


class MemTelemetry:
    def __init__(self) -> None:
        self.rows: list[dict] = []  # type: ignore[type-arg]

    async def save(self, **kw) -> None:  # type: ignore[no-untyped-def]
        self.rows.append(kw)


_CLIENT = {"ua": "x", "stt": True, "tts": True, "mode": "voice"}


def _token() -> str:
    return jwt.encode(
        {"sub": str(uuid.uuid4()), "aud": "authenticated", "exp": int(time.time()) + 60},
        SECRET,
        algorithm="HS256",
    )


def test_telemetry_derives_ttfa() -> None:
    app = create_app(Settings(), connect_db=False)
    app.state.container.telemetry = MemTelemetry()
    tok = _token()
    with TestClient(app) as c:
        r = c.post(
            "/telemetry",
            headers={"Authorization": f"Bearer {tok}"},
            json={
                "message_id": None,
                "marks": {
                    "speech_end": 1000.0,
                    "request_sent": 1010.0,
                    "first_byte": 1400.0,
                    "first_sentence": 1500.0,
                    "first_audio": 1650.5,
                    "last_audio": 3000.0,
                },
                "server_timings": {"t_gemini_first_token": 320},
                "client_info": {"ua": "x", "stt": True, "tts": True, "mode": "voice"},
            },
        )
    assert r.status_code == 204
    row = app.state.container.telemetry.rows[0]
    assert row["ttfa_ms"] == 650 and row["t_first_sentence_ms"] == 500
    assert row["t_request_ms"] == 10 and row["t_first_byte_ms"] == 400
    assert row["t_last_audio_ms"] == 2000
    assert row["server_timings"] == {"t_gemini_first_token": 320}


def test_telemetry_missing_optional_marks_derive_none() -> None:
    # Only speech_end is required on Marks; every other timestamp — and every
    # derived *_ms field that depends on it — is allowed to come back None
    # (e.g. a turn that never produced audio, or was barged-in before "done").
    app = create_app(Settings(), connect_db=False)
    app.state.container.telemetry = MemTelemetry()
    tok = _token()
    with TestClient(app) as c:
        r = c.post(
            "/telemetry",
            headers={"Authorization": f"Bearer {tok}"},
            json={"message_id": None, "marks": {"speech_end": 1000.0}, "client_info": _CLIENT},
        )
    assert r.status_code == 204
    row = app.state.container.telemetry.rows[0]
    assert row["ttfa_ms"] is None
    assert row["t_request_ms"] is None
    assert row["t_first_byte_ms"] is None
    assert row["t_first_sentence_ms"] is None
    assert row["t_last_audio_ms"] is None
    assert row["server_timings"] == {}
    # Stored as exactly the four declared keys, never the raw body (I2).
    assert row["client_info"] == {"ua": "x", "stt": True, "tts": True, "mode": "voice"}


def test_telemetry_requires_jwt() -> None:
    app = create_app(Settings(), connect_db=False)
    app.state.container.telemetry = MemTelemetry()
    with TestClient(app) as c:
        r = c.post("/telemetry", json={"marks": {"speech_end": 1000.0}})
    assert r.status_code == 401
    assert app.state.container.telemetry.rows == []


def test_telemetry_rejects_non_finite_marks() -> None:
    # performance.now() values are always finite; a NaN/Infinity mark is a malformed
    # client and must 422, not blow up _d()'s subtraction or the insert. httpx's own
    # JSON encoder refuses to emit "NaN" (allow_nan=False), so the raw body is sent
    # directly — Python's `json` module (which Starlette's Request.json() uses to
    # parse it) accepts the "NaN" literal by default, same as most JS engines emitting
    # a non-finite `performance.now()` delta would.
    app = create_app(Settings(), connect_db=False)
    app.state.container.telemetry = MemTelemetry()
    tok = _token()
    with TestClient(app) as c:
        r = c.post(
            "/telemetry",
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
            content=b'{"marks": {"speech_end": NaN}}',
        )
    assert r.status_code == 422
    assert app.state.container.telemetry.rows == []


# ---------------------------------------------------------------------------
# I2/I3: what the endpoint will and will not accept.
# ---------------------------------------------------------------------------


def _post(body: dict) -> tuple[int, MemTelemetry]:  # type: ignore[type-arg]
    app = create_app(Settings(), connect_db=False)
    app.state.container.telemetry = MemTelemetry()
    with TestClient(app) as c:
        r = c.post("/telemetry", headers={"Authorization": f"Bearer {_token()}"}, json=body)
    return r.status_code, app.state.container.telemetry


def test_client_info_is_required_and_typed() -> None:
    # It used to be `dict[str, Any] = {}`: anything, or nothing, straight into a
    # jsonb column. The two views read `ua` and `mode` out of it, so a body that
    # carries neither is telemetry that cannot be grouped by anything.
    status, repo = _post({"marks": {"speech_end": 1000.0}})
    assert status == 422
    assert repo.rows == []


def test_an_oversized_user_agent_is_truncated_not_rejected() -> None:
    # The browser chooses this string, not the developer. 300 chars is already
    # generous for a real UA and the views only substring-match four browser
    # names out of it, so refusing the body would throw away a turn's latency
    # marks to protect a column the truncation already bounds.
    app = create_app(Settings(), connect_db=False)
    app.state.container.telemetry = MemTelemetry()
    with TestClient(app) as c:
        r = c.post(
            "/telemetry",
            headers={"Authorization": f"Bearer {_token()}"},
            json={"marks": {"speech_end": 1.0}, "client_info": {**_CLIENT, "ua": "u" * 5000}},
        )
    assert r.status_code == 204
    assert app.state.container.telemetry.rows[0]["client_info"]["ua"] == "u" * 300


def test_an_unknown_client_info_key_is_rejected() -> None:
    # extra="forbid": the column holds the four declared keys or the request is
    # refused. Silently storing the rest is how a jsonb column becomes a dumping
    # ground nobody can query.
    status, repo = _post(
        {"marks": {"speech_end": 1.0}, "client_info": {**_CLIENT, "payload": "x" * 10_000}}
    )
    assert status == 422
    assert repo.rows == []


def test_an_unknown_mode_is_rejected() -> None:
    status, repo = _post({"marks": {"speech_end": 1.0}, "client_info": {**_CLIENT, "mode": "fax"}})
    assert status == 422
    assert repo.rows == []


def test_marks_outside_a_plausible_page_lifetime_are_rejected() -> None:
    # A mark beyond 1e9 ms (eleven days) is not a `performance.now()` value, and
    # every delta derived from it would be nonsense in a column read as a p95.
    status, repo = _post({"marks": {"speech_end": 1e12}, "client_info": _CLIENT})
    assert status == 422
    assert repo.rows == []
    status, repo = _post(
        {"marks": {"speech_end": 1000.0, "first_audio": 1e12}, "client_info": _CLIENT}
    )
    assert status == 422
    assert repo.rows == []


def test_negative_marks_are_rejected() -> None:
    status, repo = _post({"marks": {"speech_end": -1.0}, "client_info": _CLIENT})
    assert status == 422
    assert repo.rows == []


def test_derived_deltas_stay_inside_an_int_column() -> None:
    # The extremes the bounds still allow: a delta of ±1e9, which fits a
    # four-byte int with room to spare, and is stored as-is rather than clamped.
    status, repo = _post({"marks": {"speech_end": 0.0, "last_audio": 1e9}, "client_info": _CLIENT})
    assert status == 204
    assert repo.rows[0]["t_last_audio_ms"] == 1_000_000_000
    status, repo = _post({"marks": {"speech_end": 1e9, "first_audio": 0.0}, "client_info": _CLIENT})
    assert status == 204
    assert repo.rows[0]["ttfa_ms"] == -1_000_000_000


def test_telemetry_is_rate_limited() -> None:
    # I3: an authenticated, unbounded insert loop otherwise. A dropped telemetry
    # row costs a datapoint; there is no turn waiting on it.
    app = create_app(Settings(), connect_db=False)
    app.state.container.telemetry = MemTelemetry()
    app.state.container.rate_limiter = StubLimiter(allowed=False)
    with TestClient(app) as c:
        r = c.post(
            "/telemetry",
            headers={"Authorization": f"Bearer {_token()}"},
            json={"marks": {"speech_end": 1000.0}, "client_info": _CLIENT},
        )
    assert r.status_code == 429
    assert r.headers["retry-after"] == "42"
    assert app.state.container.telemetry.rows == []


def test_server_timings_keys_must_look_like_stage_names() -> None:
    # The column is queried by key — `v_latency_daily` already casts
    # `server_timings->>'t_gemini_first_token'` — so an arbitrary key is the
    # free-form blob `client_info` used to be, one field over.
    for key in ("gemini", "t_Gemini", "t_", "t_" + "x" * 41, "drop table", ""):
        status, repo = _post(
            {"marks": {"speech_end": 1.0}, "client_info": _CLIENT, "server_timings": {key: 1}}
        )
        assert status == 422, key
        assert repo.rows == []


def test_the_keys_the_app_actually_emits_are_accepted() -> None:
    # The other half of the rule: every stage `Timings` mints has to pass, or
    # the endpoint refuses its own client.
    status, repo = _post(
        {
            "marks": {"speech_end": 1.0},
            "client_info": _CLIENT,
            "server_timings": {
                "t_auth": 3,
                "t_context": 12,
                "t_guard": 4,
                "t_gemini_first_token": 320,
                "t_tool_get_weather": 120,
                "t_tool": 120,
                "t_total": 900,
            },
        }
    )
    assert status == 204
    assert repo.rows[0]["server_timings"]["t_gemini_first_token"] == 320


def test_too_many_server_timings_entries_are_rejected() -> None:
    status, repo = _post(
        {
            "marks": {"speech_end": 1.0},
            "client_info": _CLIENT,
            "server_timings": {f"t_stage_{i}": 1 for i in range(33)},
        }
    )
    assert status == 422
    assert repo.rows == []
    status, repo = _post(
        {
            "marks": {"speech_end": 1.0},
            "client_info": _CLIENT,
            "server_timings": {f"t_stage_{i}": 1 for i in range(32)},
        }
    )
    assert status == 204


def test_server_timing_values_are_clamped_not_rejected() -> None:
    # A key is structure and a value is a measurement: one absurd stage number
    # costs that cell, and refusing the body over it would throw away the five
    # derived latency columns the request exists for.
    status, repo = _post(
        {
            "marks": {"speech_end": 1.0},
            "client_info": _CLIENT,
            "server_timings": {"t_total": 10**12, "t_guard": -(10**12), "t_auth": 7},
        }
    )
    assert status == 204
    row = repo.rows[0]["server_timings"]
    assert row["t_total"] == 2_147_483_647
    assert row["t_guard"] == -2_147_483_648
    assert row["t_auth"] == 7


def test_telemetry_and_chat_spend_separate_rate_limit_budgets() -> None:
    # Adding the limiter to /telemetry on a shared bucket silently halved
    # conversational throughput: a voice turn posts its marks, so every turn
    # cost two hits of an allowance sized for one.
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()
    c.llm = FakeLLM([[LLMText("Hello there."), LLMFinished("stop")]])
    c.rebuild_run_turn()
    limiter = StubLimiter(allowed=True)
    c.rate_limiter = limiter
    tok = _token()
    with TestClient(app) as client:
        client.post(
            "/chat",
            json={"client_turn_id": "t-ns", "text": "hi"},
            headers={"Authorization": f"Bearer {tok}"},
        )
        client.post(
            "/telemetry",
            headers={"Authorization": f"Bearer {tok}"},
            json={"marks": {"speech_end": 1000.0}, "client_info": _CLIENT},
        )
        client.post(
            "/chat/confirm",
            json={"client_turn_id": "t-ns", "text": "hi"},
            headers={"Authorization": f"Bearer {tok}"},
        )
    assert limiter.namespaces == ["chat", "tele", "confirm"]


def test_confirm_is_rate_limited() -> None:
    # It writes a turn's rows and inserts into an in-process store, so leaving it
    # unmetered made it the cheapest way to make this worker do work.
    app = create_app(Settings(speculative_enabled=True), connect_db=False)
    c = app.state.container
    c.use_in_memory_repos()
    c.rate_limiter = StubLimiter(allowed=False)
    with TestClient(app) as client:
        r = client.post(
            "/chat/confirm",
            json={"client_turn_id": "spec-1", "text": "hello"},
            headers={"Authorization": f"Bearer {_token()}"},
        )
    assert r.status_code == 429
    assert r.headers["retry-after"] == "42"
    assert c.run_turn.speculation.early_size == 0  # nothing reached the store
