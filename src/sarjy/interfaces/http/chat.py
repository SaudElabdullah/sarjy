"""`POST /chat` — streams a `RunTurn` as Server-Sent Events, and its confirmation."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sarjy.contexts.conversation.domain.turn import InputMode, TurnInput
from sarjy.interfaces.http.auth import CurrentUserDep
from sarjy.interfaces.http.rate_limit import rate_limited, rate_limited_confirm
from sarjy.interfaces.http.sse import encode
from sarjy.shared.ids import SessionId

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: uuid.UUID | None = None
    client_turn_id: str = Field(min_length=1, max_length=64)
    text: str = Field(max_length=2000)
    speculative: bool = False
    client_ts: int | None = None
    input_mode: InputMode = "voice"


async def _arrival(request: Request) -> None:
    """Stamp the instant this request entered the dependency chain (I5).

    Declared FIRST in the route's `dependencies`, which is what makes it useful:
    FastAPI resolves route-level dependencies in order, and before the
    endpoint's own parameters, so everything between this and the handler body —
    the JWT verify and the rate limiter's round trip — is exactly what `t_auth`
    measures. Nothing else can measure it: by the time `RunTurn` exists, it has
    already happened.
    """
    request.state.t_arrival = time.perf_counter()


@router.post("/chat", dependencies=[Depends(_arrival), Depends(rate_limited)])
async def chat(req: ChatRequest, user: CurrentUserDep, request: Request) -> StreamingResponse:
    container = request.app.state.container
    run_turn = container.run_turn
    # Closes the gap between the client's `request_sent` mark and the first
    # stage the server used to report. `getattr` rather than a bare read so a
    # hand-built app that mounts this router without the dependency degrades to
    # "unmeasured" instead of a 500.
    t_arrival = getattr(request.state, "t_arrival", None)
    request.state.t_auth = 0 if t_arrival is None else int((time.perf_counter() - t_arrival) * 1000)
    inp = TurnInput(
        user_id=user.user_id,
        session_id=SessionId(req.session_id) if req.session_id else None,
        client_turn_id=req.client_turn_id,
        text=req.text,
        # A client asking for a speculative turn cannot switch the feature on: with
        # speculation disabled the turn runs normally (and is persisted normally)
        # rather than buffering writes that nothing will ever confirm.
        speculative=req.speculative and container.settings.speculative_enabled,
        input_mode=req.input_mode,
        t_auth_ms=request.state.t_auth,
    )

    async def gen() -> AsyncIterator[bytes]:
        # aclosing: a client disconnect breaks out mid-iteration, and the turn's
        # generator must be closed then rather than at some later GC, so RunTurn's
        # finally-blocks (and the upstream LLM stream) are released promptly.
        async with aclosing(run_turn(inp)) as events:
            async for ev in events:
                if await request.is_disconnected():
                    break
                yield encode(ev)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


class ConfirmIn(BaseModel):
    client_turn_id: str = Field(min_length=1, max_length=64)
    text: str = Field(max_length=2000)


@router.post("/chat/confirm", status_code=204, dependencies=[Depends(rate_limited_confirm)])
async def confirm(body: ConfirmIn, user: CurrentUserDep, request: Request) -> Response:
    """Confirm a speculative turn: the final transcript really did say that (L-3).

    Three statuses, one per outcome of `RunTurn.confirm` (C1):

    * **204** — the parked turn was found, matched, and its rows are written.
    * **202** — accepted, nothing written *yet*. The usual cause is a
      confirmation that overtook its own turn: the transcript is recorded and
      the turn writes its rows when it reaches the park, moments later. The
      client does nothing on this — the work is the server's now.
    * **409** — a guess IS parked under that id and the final transcript
      contradicts it. The client sends an ordinary turn under a new id.

    202 rather than 204 for the accepted-but-unwritten case because the two are
    genuinely different promises, and a client that cannot tell them apart
    cannot know whether a retry is safe.

    Rate limited in its own `confirm` namespace — not alongside `/chat`. The
    original reasoning (a confirmation writes rows for work `/chat`'s limiter
    already admitted, so refusing it would leave that work spoken and
    unrecorded) still holds, and a separate budget is what preserves it: a
    caller cannot spend their conversation allowance on confirmations, or the
    reverse. What changed is that this endpoint is no longer only a lookup —
    it writes a turn's rows, and since C1 it also inserts into an in-process
    store — so leaving it entirely unmetered made it the cheapest way to make
    this worker do work. The store is bounded independently
    (`SpeculativeTurnCache`); this is the other half of that.
    """
    outcome = await request.app.state.container.run_turn.confirm(
        body.client_turn_id, body.text, user.user_id
    )
    if outcome == "pending":
        return Response(status_code=202)
    return Response(status_code=204 if outcome else 409)
