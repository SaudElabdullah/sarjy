"""Gemini adapter implementing `LLMPort` (streaming + JSON generation)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as gt
from pydantic import BaseModel

from sarjy.contexts.conversation.application.ports import (
    FunctionCall,
    LLMEvent,
    LLMFinished,
    LLMFunctionCall,
    LLMMessage,
    LLMRequest,
    LLMText,
    LLMTimeout,
    LLMUnavailable,
)
from sarjy.observability.logging import get_logger

T = TypeVar("T", bound=BaseModel)
log = get_logger(__name__)

# Backward-compat aliases (kept for tests / callers written against the brief).
GeminiUnavailable = LLMUnavailable
GeminiTimeout = LLMTimeout

_RETRYABLE_CODES = (429, 500, 502, 503, 504)


def _thinking(budget: int) -> gt.ThinkingConfig:
    """Map the port's integer budget onto what current models accept.

    Gemini 3.x rejects `thinking_budget=0` with INVALID_ARGUMENT (verified
    against gemini-3.6-flash and gemini-3.5-flash-lite); the way to say "answer
    immediately" there is `thinking_level="MINIMAL"`. A positive budget is still
    expressed as a budget so the narrator's deliberate allowance is honoured.
    """
    if budget <= 0:
        return gt.ThinkingConfig(thinking_level="MINIMAL")
    return gt.ThinkingConfig(thinking_budget=budget)


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        first_token_timeout_s: float = 8.0,
        total_timeout_s: float = 25.0,
        cached_content: str | None = None,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._ftt = first_token_timeout_s
        self._tt = total_timeout_s
        # Explicit context caching hook (Phase 7 Task 6, L-6): a Gemini cache
        # *name* (e.g. "cachedContents/abc123"), created and refreshed by the
        # caller — this class does not create or manage the cache itself. Left
        # unset in `container.py` today; see the comment there for why.
        self._cached_content = cached_content

    # -- mapping -------------------------------------------------------------
    @staticmethod
    def _to_contents(msgs: list[LLMMessage]) -> list[gt.Content]:
        out: list[gt.Content] = []
        prev: str | None = None

        def add(role: str, part: gt.Part, kind: str) -> None:
            """Append `part`, merging it into the previous Content when it belongs there.

            Gemini represents a parallel-call round as one model Content holding every
            function_call, followed by one user Content holding every function_response.
            Emitting a Content per call would instead present that round as a run of
            separate single-call turns, which is not the shape the model produced.
            """
            nonlocal prev
            if kind != "text" and kind == prev and out:
                out[-1] = gt.Content(role=role, parts=[*(out[-1].parts or []), part])
            else:
                out.append(gt.Content(role=role, parts=[part]))
            prev = kind

        for m in msgs:
            if m.function_call is not None:
                # `Part.from_function_call` drops the call id, which is what pairs a
                # response with its call in a parallel round — build the Part directly.
                add(
                    "model",
                    gt.Part(
                        function_call=gt.FunctionCall(
                            name=m.function_call.name,
                            args=m.function_call.args,
                            id=m.function_call.id,
                        )
                    ),
                    "call",
                )
            elif m.function_response is not None:
                add(
                    "user",
                    gt.Part(
                        function_response=gt.FunctionResponse(
                            name=m.function_response.name,
                            response=m.function_response.response,
                            id=m.function_response.id,
                        )
                    ),
                    "response",
                )
            else:
                add(
                    m.role if m.role != "tool" else "user",
                    gt.Part.from_text(text=m.text or ""),
                    "text",
                )
        # A conversation has to open with a user turn: Gemini rejects a history
        # whose first Content is `model`, and there is nothing sensible for a
        # model turn to be replying to. Callers can produce one legitimately —
        # RunTurn drops refused user rows from replay (I6/R4), and the very
        # first turn of a session being one leaves its refusal at the front —
        # so rather than make every caller remember, drop leading model turns
        # here. This is a safety net, not the fix: the caller that built the
        # list is the one that knows which pair belongs together.
        while out and out[0].role == "model":
            out.pop(0)
        return out

    def _config(
        self, req: LLMRequest, json_schema: type[BaseModel] | None = None
    ) -> gt.GenerateContentConfig:
        cfg: dict[str, Any] = {
            "system_instruction": req.system,
            "temperature": req.temperature,
            "max_output_tokens": req.max_output_tokens,
            "thinking_config": _thinking(req.thinking_budget),
        }
        if req.tools:
            cfg["tools"] = [
                gt.Tool(function_declarations=[gt.FunctionDeclaration(**d) for d in req.tools])
            ]
            mode = "ANY" if req.force_tool else "AUTO"
            fcc = gt.FunctionCallingConfig(
                mode=mode, allowed_function_names=[req.force_tool] if req.force_tool else None
            )
            cfg["tool_config"] = gt.ToolConfig(function_calling_config=fcc)
        if json_schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = json_schema
        if self._cached_content:
            cfg["cached_content"] = self._cached_content
        return gt.GenerateContentConfig(**cfg)

    # -- port ----------------------------------------------------------------
    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        attempt = 0
        emitted = False
        while True:
            try:
                async with contextlib.aclosing(self._stream_once(req)) as gen:
                    async for ev in gen:
                        emitted = True
                        yield ev
                return
            except LLMUnavailable:
                # PRD C-9: retry only if no tokens have been streamed to the caller
                # yet — once something has been emitted, retrying would replay it.
                if emitted:
                    raise
                attempt += 1
                if attempt > 1:
                    raise
                await asyncio.sleep(0.2)  # one retry before first token

    async def _stream_once(self, req: LLMRequest) -> AsyncGenerator[LLMEvent, None]:
        # Both budgets start here, before the stream is opened: opening the stream is
        # part of the wait the caller experiences. Sharing one first-token deadline
        # between the open and the first chunk stops the two from each spending a full
        # `_ftt` and pushing time-to-first-token to twice the configured budget.
        loop = asyncio.get_running_loop()
        deadline_first = loop.time() + self._ftt
        deadline = loop.time() + self._tt
        try:
            it = await asyncio.wait_for(
                self._client.aio.models.generate_content_stream(
                    model=self._model,
                    contents=self._to_contents(req.messages),
                    config=self._config(req),
                ),
                timeout=max(0.05, deadline_first - loop.time()),
            )
        except genai_errors.APIError as e:
            if e.code in _RETRYABLE_CODES:
                raise LLMUnavailable(str(e)) from e
            raise
        except TimeoutError as e:
            raise LLMTimeout("first_token") from e

        first = True
        aiter = it.__aiter__()
        try:
            while True:
                remaining = deadline_first if first else deadline
                timeout = max(0.05, remaining - loop.time())
                try:
                    chunk = await asyncio.wait_for(aiter.__anext__(), timeout=timeout)
                except StopAsyncIteration:
                    break
                except TimeoutError as e:
                    raise LLMTimeout("first_token" if first else "total") from e
                except genai_errors.APIError as e:
                    if e.code in _RETRYABLE_CODES:
                        raise LLMUnavailable(str(e)) from e
                    raise
                first = False
                for cand in chunk.candidates or []:
                    for part in (cand.content.parts if cand.content else []) or []:
                        fc = part.function_call
                        if fc is not None:
                            yield LLMFunctionCall(
                                FunctionCall(name=fc.name or "", args=dict(fc.args or {}), id=fc.id)
                            )
                        elif part.text:
                            yield LLMText(part.text)
        finally:
            # Barge-in / cancellation path: make sure the upstream SDK stream is
            # released rather than left open when the consumer stops early.
            if hasattr(aiter, "aclose"):
                await aiter.aclose()
        yield LLMFinished("stop")

    async def generate_json(self, req: LLMRequest, schema: type[T]) -> T:
        try:
            resp = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self._model,
                    contents=self._to_contents(req.messages),
                    config=self._config(req, schema),
                ),
                timeout=self._ftt,
            )
        except genai_errors.APIError as e:
            if e.code in _RETRYABLE_CODES:
                raise LLMUnavailable(str(e)) from e
            raise
        except TimeoutError as e:
            raise LLMTimeout("first_token") from e
        return schema.model_validate_json(resp.text or "{}")
