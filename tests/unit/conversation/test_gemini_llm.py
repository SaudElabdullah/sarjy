from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel

from sarjy.contexts.conversation.application.ports import (
    LLMFinished,
    LLMFunctionCall,
    LLMMessage,
    LLMRequest,
    LLMText,
    LLMTimeout,
    LLMUnavailable,
)
from sarjy.contexts.conversation.infrastructure.gemini_llm import GeminiLLM


class _FakeAio:
    def __init__(self, chunks):  # type: ignore[no-untyped-def]
        self._chunks = chunks
        self.models = self

    async def generate_content_stream(self, **_):  # type: ignore[no-untyped-def]
        async def gen():  # type: ignore[no-untyped-def]
            for c in self._chunks:
                yield c

        return gen()

    async def generate_content(self, **_):  # type: ignore[no-untyped-def]
        return SimpleNamespace(text='{"value": 4, "confidence": 0.9}')


def _text_chunk(t: str):  # type: ignore[no-untyped-def]
    part = SimpleNamespace(text=t, function_call=None)
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason=None)]
    )


def _fc_chunk(name: str, args: dict):  # type: ignore[no-untyped-def]
    part = SimpleNamespace(text=None, function_call=SimpleNamespace(name=name, args=args, id="c1"))
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason="STOP")]
    )


def _req() -> LLMRequest:
    return LLMRequest(
        system="s",
        messages=[LLMMessage(role="user", text="hi")],
        tools=[],
        temperature=0.6,
        max_output_tokens=300,
    )


async def test_stream_yields_text_then_function_call() -> None:
    llm = GeminiLLM(api_key="k", model="m")
    llm._client = SimpleNamespace(  # type: ignore[assignment]
        aio=_FakeAio(
            [
                _text_chunk("Hel"),
                _text_chunk("lo."),
                _fc_chunk("get_weather", {"location": "Tokyo"}),
            ]
        )
    )
    events = [e async for e in llm.stream(_req())]
    assert events[0] == LLMText("Hel")
    assert events[1] == LLMText("lo.")
    assert isinstance(events[2], LLMFunctionCall)
    assert events[2].call.name == "get_weather"
    assert isinstance(events[-1], LLMFinished)


def test_config_omits_cached_content_when_unset() -> None:
    llm = GeminiLLM(api_key="k", model="m")
    assert llm._config(_req()).cached_content is None


def test_config_passes_cached_content_through_when_set() -> None:
    # Phase 7 Task 6 (L-6) hook: `cached_content` is a Gemini cache *name*
    # (e.g. "cachedContents/abc123") a caller created and is asking be reused —
    # `GeminiLLM` itself never creates or refreshes the cache.
    llm = GeminiLLM(api_key="k", model="m", cached_content="cachedContents/abc123")
    assert llm._config(_req()).cached_content == "cachedContents/abc123"


class _Out(BaseModel):
    value: int
    confidence: float


async def test_generate_json_validates() -> None:
    llm = GeminiLLM(api_key="k", model="m")
    llm._client = SimpleNamespace(aio=_FakeAio([]))  # type: ignore[assignment]
    out = await llm.generate_json(_req(), _Out)
    assert out.value == 4


# -- Fix round 1: no retry after first token, upstream close, timeout mapping ----


class _FailsAfterFirstChunkAio:
    """Yields one text chunk, then a retryable 503 APIError on the 2nd __anext__."""

    def __init__(self):  # type: ignore[no-untyped-def]
        self.models = self
        self.calls = 0

    async def generate_content_stream(self, **_):  # type: ignore[no-untyped-def]
        self.calls += 1

        async def gen():  # type: ignore[no-untyped-def]
            yield _text_chunk("Hello")
            raise genai_errors.APIError(503, {"message": "unavailable", "status": "UNAVAILABLE"})

        return gen()


async def test_stream_does_not_retry_after_first_token() -> None:
    llm = GeminiLLM(api_key="k", model="m")
    fake = _FailsAfterFirstChunkAio()
    llm._client = SimpleNamespace(aio=fake)  # type: ignore[assignment]
    events = []
    with pytest.raises(LLMUnavailable):
        async for e in llm.stream(_req()):
            events.append(e)
    assert events == [LLMText("Hello")]
    assert fake.calls == 1  # no retry once a token was already emitted


class _FailsOnceThenSucceedsAio:
    """Raises a retryable 503 APIError on the *initial* stream-open call, then
    succeeds on the retry (before any token has been emitted)."""

    def __init__(self, chunks):  # type: ignore[no-untyped-def]
        self.models = self
        self.calls = 0
        self._chunks = chunks

    async def generate_content_stream(self, **_):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            raise genai_errors.APIError(503, {"message": "unavailable", "status": "UNAVAILABLE"})

        async def gen():  # type: ignore[no-untyped-def]
            for c in self._chunks:
                yield c

        return gen()


async def test_stream_retries_once_before_first_token() -> None:
    llm = GeminiLLM(api_key="k", model="m")
    fake = _FailsOnceThenSucceedsAio([_text_chunk("Hi")])
    llm._client = SimpleNamespace(aio=fake)  # type: ignore[assignment]
    events = [e async for e in llm.stream(_req())]
    assert fake.calls == 2
    assert events.count(LLMText("Hi")) == 1


class _TrackingIter:
    def __init__(self, chunks):  # type: ignore[no-untyped-def]
        self._chunks = list(chunks)
        self.aclose_called = False

    def __aiter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __anext__(self):  # type: ignore[no-untyped-def]
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self):  # type: ignore[no-untyped-def]
        self.aclose_called = True


class _TrackingAio:
    def __init__(self, chunks):  # type: ignore[no-untyped-def]
        self.models = self
        self.iterator = _TrackingIter(chunks)

    async def generate_content_stream(self, **_):  # type: ignore[no-untyped-def]
        return self.iterator


async def test_stream_closes_upstream_iterator_on_aclose() -> None:
    llm = GeminiLLM(api_key="k", model="m")
    fake = _TrackingAio([_text_chunk("Hel"), _text_chunk("lo.")])
    llm._client = SimpleNamespace(aio=fake)  # type: ignore[assignment]
    gen = llm.stream(_req())
    first = await gen.__anext__()
    assert first == LLMText("Hel")
    await gen.aclose()
    assert fake.iterator.aclose_called is True


class _HangingIter:
    def __aiter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __anext__(self):  # type: ignore[no-untyped-def]
        await asyncio.sleep(10)
        raise AssertionError("should have timed out before this returned")  # pragma: no cover


class _HangingAio:
    def __init__(self):  # type: ignore[no-untyped-def]
        self.models = self

    async def generate_content_stream(self, **_):  # type: ignore[no-untyped-def]
        return _HangingIter()


async def test_stream_raises_first_token_timeout() -> None:
    llm = GeminiLLM(api_key="k", model="m", first_token_timeout_s=0.05)
    llm._client = SimpleNamespace(aio=_HangingAio())  # type: ignore[assignment]
    with pytest.raises(LLMTimeout, match="first_token"):
        async for _e in llm.stream(_req()):
            pass


class _SlowSecondChunkIter:
    def __init__(self):  # type: ignore[no-untyped-def]
        self._n = 0

    def __aiter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __anext__(self):  # type: ignore[no-untyped-def]
        self._n += 1
        if self._n == 1:
            return _text_chunk("Hi")
        await asyncio.sleep(5)
        raise AssertionError("should have timed out before this returned")  # pragma: no cover


class _SlowSecondChunkAio:
    def __init__(self):  # type: ignore[no-untyped-def]
        self.models = self

    async def generate_content_stream(self, **_):  # type: ignore[no-untyped-def]
        return _SlowSecondChunkIter()


async def test_stream_raises_total_timeout_on_later_chunk() -> None:
    llm = GeminiLLM(api_key="k", model="m", first_token_timeout_s=1.0, total_timeout_s=0.1)
    llm._client = SimpleNamespace(aio=_SlowSecondChunkAio())  # type: ignore[assignment]
    events = []
    with pytest.raises(LLMTimeout, match="total"):
        async for e in llm.stream(_req()):
            events.append(e)
    assert events == [LLMText("Hi")]


# -- I4: the first-token budget covers the open *and* the first chunk -----------


class _SlowOpenSlowFirstChunkIter:
    def __aiter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __anext__(self):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.2)
        raise AssertionError("should have timed out before this returned")  # pragma: no cover


class _SlowOpenAio:
    def __init__(self):  # type: ignore[no-untyped-def]
        self.models = self

    async def generate_content_stream(self, **_):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.2)
        return _SlowOpenSlowFirstChunkIter()


async def test_first_token_budget_is_shared_by_open_and_first_chunk() -> None:
    # 0.2 s to open + 0.2 s for the first chunk. With one shared 0.30 s deadline the
    # timeout fires at ~0.30 s, not at 0.2 + 0.3 = 0.5 s (a budget spent twice).
    llm = GeminiLLM(api_key="k", model="m", first_token_timeout_s=0.30, total_timeout_s=25.0)
    llm._client = SimpleNamespace(aio=_SlowOpenAio())  # type: ignore[assignment]
    t0 = asyncio.get_running_loop().time()
    with pytest.raises(LLMTimeout, match="first_token"):
        async for _e in llm.stream(_req()):
            pass
    elapsed = asyncio.get_running_loop().time() - t0
    assert elapsed < 0.45, f"first-token budget was spent twice ({elapsed:.3f}s)"


# -- Re-review residual: parallel calls map to one model + one user Content -----


def test_to_contents_coalesces_a_parallel_call_round() -> None:
    from sarjy.contexts.conversation.application.ports import FunctionCall, FunctionResponse

    msgs = [
        LLMMessage(role="user", text="<user>weather at home</user>"),
        LLMMessage(role="model", function_call=FunctionCall("recall", {"key": "city"}, "c1")),
        LLMMessage(role="model", function_call=FunctionCall("get_weather", {"q": "Tokyo"}, "c2")),
        LLMMessage(role="tool", function_response=FunctionResponse("recall", {"v": "Tokyo"}, "c1")),
        LLMMessage(role="tool", function_response=FunctionResponse("get_weather", {"t": 22}, "c2")),
    ]
    contents = GeminiLLM._to_contents(msgs)

    assert len(contents) == 3
    assert contents[0].role == "user"

    calls = contents[1]
    assert calls.role == "model"
    assert calls.parts is not None and len(calls.parts) == 2
    assert [p.function_call.name for p in calls.parts if p.function_call] == [
        "recall",
        "get_weather",
    ]
    assert [p.function_call.id for p in calls.parts if p.function_call] == ["c1", "c2"]

    responses = contents[2]
    assert responses.role == "user"
    assert responses.parts is not None and len(responses.parts) == 2
    assert [p.function_response.name for p in responses.parts if p.function_response] == [
        "recall",
        "get_weather",
    ]
    assert [p.function_response.id for p in responses.parts if p.function_response] == ["c1", "c2"]


def test_to_contents_keeps_separate_rounds_apart() -> None:
    from sarjy.contexts.conversation.application.ports import FunctionCall, FunctionResponse

    msgs = [
        # The user turn that provoked the calls. It was absent here before R4,
        # which no real history ever is — and `_to_contents` now drops leading
        # model turns, so an isolated call round would be trimmed away.
        LLMMessage(role="user", text="<user>weather in Tokyo and Lisbon?</user>"),
        LLMMessage(role="model", function_call=FunctionCall("a", {}, "c1")),
        LLMMessage(role="tool", function_response=FunctionResponse("a", {}, "c1")),
        LLMMessage(role="model", function_call=FunctionCall("b", {}, "c2")),
        LLMMessage(role="tool", function_response=FunctionResponse("b", {}, "c2")),
    ]
    contents = GeminiLLM._to_contents(msgs)
    # Two sequential single-call rounds must stay four Contents, not be merged.
    assert [c.role for c in contents] == ["user", "model", "user", "model", "user"]
    assert all(c.parts is not None and len(c.parts) == 1 for c in contents)


def test_leading_model_turns_are_dropped() -> None:
    """R4 safety net: a history must open with a user turn.

    Gemini rejects one that opens with `model`, and there is nothing for a
    model turn at the front to be replying to. RunTurn is the one that knows
    which pair belongs together (it drops both halves of a refused turn) — this
    just means no caller can get it wrong twice.
    """
    from sarjy.contexts.conversation.application.ports import LLMMessage
    from sarjy.contexts.conversation.infrastructure.gemini_llm import GeminiLLM

    out = GeminiLLM._to_contents(
        [
            LLMMessage(role="model", text="That's outside what I can help with."),
            LLMMessage(role="model", text="Still me talking."),
            LLMMessage(role="user", text="<user>hello</user>"),
            LLMMessage(role="model", text="Hi there."),
        ]
    )
    assert [c.role for c in out] == ["user", "model"]


def test_a_history_that_already_opens_with_user_is_untouched() -> None:
    from sarjy.contexts.conversation.application.ports import LLMMessage
    from sarjy.contexts.conversation.infrastructure.gemini_llm import GeminiLLM

    out = GeminiLLM._to_contents(
        [
            LLMMessage(role="user", text="<user>hello</user>"),
            LLMMessage(role="model", text="Hi there."),
            LLMMessage(role="user", text="<user>and tomorrow?</user>"),
        ]
    )
    assert [c.role for c in out] == ["user", "model", "user"]


def test_zero_thinking_budget_maps_to_minimal_level() -> None:
    """Gemini 3.x rejects thinking_budget=0; MINIMAL level is the equivalent."""
    from sarjy.contexts.conversation.infrastructure.gemini_llm import _thinking

    assert _thinking(0).thinking_level == "MINIMAL"
    assert _thinking(0).thinking_budget is None
    assert _thinking(256).thinking_budget == 256
