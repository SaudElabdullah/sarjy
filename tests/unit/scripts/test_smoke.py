"""scripts/smoke.py: SSE parsing and the health -> sign-up -> /chat flow, mocked
via respx (no network, no real Supabase/Gemini) — Phase 8 Task 5."""

from __future__ import annotations

import httpx
import pytest
import respx
from scripts.smoke import SmokeError, check_health, main, parse_sse, stream_chat

CANNED_STREAM = (
    b'event: session\ndata: {"session_id": "11111111-1111-1111-1111-111111111111"}\n\n'
    b"event: sentence\n"
    b'data: {"i": 0, "text": "Hi there!", "speech": "Hi there!", "final": false}\n\n'
    b'event: done\ndata: {"message_id": "m1", "timings": {}}\n\n'
)


# --------------------------------------------------------------------- parse_sse
def test_parse_sse_reads_event_and_data_pairs_in_order() -> None:
    events = parse_sse(CANNED_STREAM.decode())
    assert [name for name, _ in events] == ["session", "sentence", "done"]
    assert events[1][1] == {"i": 0, "text": "Hi there!", "speech": "Hi there!", "final": False}
    assert events[2][1]["message_id"] == "m1"


def test_parse_sse_ignores_blank_input() -> None:
    assert parse_sse("\n\n  \n\n") == []


def test_parse_sse_skips_a_block_with_no_data_line() -> None:
    assert parse_sse("event: ping\n\nevent: done\ndata: {}\n\n") == [("done", {})]


# -------------------------------------------------------------------- check_health
@respx.mock
def test_check_health_ok_on_200() -> None:
    respx.get("https://app.test/healthz").mock(return_value=httpx.Response(200))
    check_health("https://app.test")  # does not raise


@respx.mock
def test_check_health_raises_on_non_200() -> None:
    respx.get("https://app.test/healthz").mock(return_value=httpx.Response(503, text="db down"))
    with pytest.raises(SmokeError, match="503"):
        check_health("https://app.test")


@respx.mock
def test_check_health_raises_when_unreachable() -> None:
    respx.get("https://app.test/healthz").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(SmokeError, match="unreachable"):
        check_health("https://app.test")


# -------------------------------------------------------------------- stream_chat
@respx.mock
def test_stream_chat_times_sentence_and_done() -> None:
    respx.post("https://app.test/chat").mock(
        return_value=httpx.Response(
            200, content=CANNED_STREAM, headers={"content-type": "text/event-stream"}
        )
    )
    sentence_s, done_s = stream_chat("https://app.test", "tok", "say hi")
    assert 0 <= sentence_s <= done_s


@respx.mock
def test_stream_chat_fails_when_sentence_never_arrives() -> None:
    respx.post("https://app.test/chat").mock(
        return_value=httpx.Response(
            200,
            content=b'event: done\ndata: {"message_id": "m1", "timings": {}}\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )
    with pytest.raises(SmokeError, match="sentence"):
        stream_chat("https://app.test", "tok", "say hi")


@respx.mock
def test_stream_chat_fails_when_done_never_arrives() -> None:
    respx.post("https://app.test/chat").mock(
        return_value=httpx.Response(
            200,
            content=b"event: sentence\n"
            b'data: {"i": 0, "text": "hi", "speech": "hi", "final": false}\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )
    with pytest.raises(SmokeError, match="done"):
        stream_chat("https://app.test", "tok", "say hi")


@respx.mock
def test_stream_chat_fails_on_non_200() -> None:
    respx.post("https://app.test/chat").mock(return_value=httpx.Response(401, text="unauthorized"))
    with pytest.raises(SmokeError, match="401"):
        stream_chat("https://app.test", "bad-tok", "say hi")


# -------------------------------------------------------------------------- main
@respx.mock
def test_main_health_only_skips_signup_and_chat(capsys: pytest.CaptureFixture[str]) -> None:
    respx.get("https://app.test/healthz").mock(return_value=httpx.Response(200))
    code = main(["https://app.test", "--health-only"])
    assert code == 0
    assert "healthz" in capsys.readouterr().out


def test_main_requires_supabase_args_without_health_only() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["https://app.test"])
    assert exc.value.code == 2


@respx.mock
def test_main_happy_path_health_signup_and_chat(capsys: pytest.CaptureFixture[str]) -> None:
    respx.get("https://app.test/healthz").mock(return_value=httpx.Response(200))
    respx.post("https://proj.supabase.co/auth/v1/signup").mock(
        return_value=httpx.Response(200, json={"access_token": "tok-123"})
    )
    respx.post("https://app.test/chat").mock(
        return_value=httpx.Response(
            200, content=CANNED_STREAM, headers={"content-type": "text/event-stream"}
        )
    )
    code = main(["https://app.test", "https://proj.supabase.co", "anon-key"])
    assert code == 0
    out = capsys.readouterr().out
    assert "anonymous sign-up" in out
    assert "sentence at" in out


@respx.mock
def test_main_returns_nonzero_and_prints_reason_on_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    respx.get("https://app.test/healthz").mock(return_value=httpx.Response(500))
    code = main(["https://app.test", "https://proj.supabase.co", "anon-key"])
    assert code == 1
    assert "SMOKE FAILED" in capsys.readouterr().err
