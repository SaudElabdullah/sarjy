"""`POST /telemetry` — client latency marks ingestion (PRD §9.4, L-1/L-2)."""

from __future__ import annotations

import re
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sarjy.interfaces.http.auth import CurrentUserDep
from sarjy.interfaces.http.rate_limit import rate_limited_telemetry
from sarjy.shared.ids import UserId

router = APIRouter()

# performance.now() is milliseconds since the page loaded. A mark beyond this is
# a tab that has been open for eleven days, or — far more likely — a client
# sending something that is not a timestamp at all. Rejecting the whole body is
# right either way: a bogus mark makes every delta derived from it bogus too,
# and one 422 costs a single turn's telemetry rather than poisoning the p95s
# everything downstream is read off. It is also what keeps the derived deltas
# inside a range `int` columns can hold — see `_d`.
_MAX_MARK_MS = 1e9

# `telemetry_turns`' derived columns are `int` (four bytes). A delta between two
# in-range marks can still be ±1e9, which fits; two marks at opposite ends of
# the range cannot overflow it. The clamp below is belt and braces for that
# arithmetic, not the primary defence — the primary defence is the field bound.
_INT32_MIN, _INT32_MAX = -2_147_483_648, 2_147_483_647

# `server_timings` is echoed back from the DoneEvent this client was sent, so
# every key it can legitimately carry is one `Timings` minted: `t_` plus a stage
# name. Pinning the shape keeps the jsonb column queryable — `v_latency_daily`
# already casts `server_timings->>'t_gemini_first_token'` to int — and keeps it
# from becoming the free-form blob `client_info` used to be.
_TIMING_KEY = re.compile(r"^t_[a-z0-9_]{1,40}$")
# A turn reports a handful of stages plus one key per tool called. 32 is far more
# than MAX_TOOL_HOPS can produce and small enough that no row is a payload.
_MAX_TIMING_KEYS = 32
_UA_MAX = 300


class Marks(BaseModel):
    # allow_inf_nan=False: performance.now() values are always finite, so a NaN/Infinity here
    # is a malformed client — reject with 422 rather than let it reach `_d()` (whose subtraction
    # would happily propagate a NaN into the derived *_ms columns) or crash the insert.
    # ge/le bound them to a plausible page lifetime (I3), which is also what bounds the deltas.
    speech_end: float = Field(allow_inf_nan=False, ge=0, le=_MAX_MARK_MS)
    request_sent: float | None = Field(default=None, allow_inf_nan=False, ge=0, le=_MAX_MARK_MS)
    first_byte: float | None = Field(default=None, allow_inf_nan=False, ge=0, le=_MAX_MARK_MS)
    first_sentence: float | None = Field(default=None, allow_inf_nan=False, ge=0, le=_MAX_MARK_MS)
    first_audio: float | None = Field(default=None, allow_inf_nan=False, ge=0, le=_MAX_MARK_MS)
    last_audio: float | None = Field(default=None, allow_inf_nan=False, ge=0, le=_MAX_MARK_MS)


class ClientInfo(BaseModel):
    """The four things the browser tells us about itself, and nothing else (I2).

    This used to be a free `dict[str, Any]` written straight into a `jsonb`
    column: an unauthenticated-shaped blob of arbitrary size and arbitrary keys,
    stored per turn and read back by `v_latency_by_browser`. A typed model with
    `extra="forbid"` makes the column's contents exactly what the two views
    query — `ua` and `mode` — plus the two capability flags, and makes an
    oversized user agent a 422 rather than a row nobody notices growing.

    300 characters for `ua`: the longest real user-agent strings run to about
    200, and the view only ever substring-matches four browser names out of it.
    It is TRUNCATED rather than rejected — the browser chooses this string, not
    the developer, and refusing the whole body over a field the views read four
    substrings out of would throw away a turn's latency marks to protect a
    column that is already bounded by the truncation.
    """

    model_config = ConfigDict(extra="forbid")

    ua: str
    stt: bool
    tts: bool
    mode: Literal["voice", "text"]

    @field_validator("ua")
    @classmethod
    def _truncate_ua(cls, v: str) -> str:
        return v[:_UA_MAX]


class TelemetryIn(BaseModel):
    message_id: uuid.UUID | None = None
    marks: Marks
    server_timings: dict[str, int] | None = None
    client_info: ClientInfo

    @field_validator("server_timings")
    @classmethod
    def _check_timings(cls, v: dict[str, int] | None) -> dict[str, int] | None:
        """Keys are refused; values are clamped.

        The asymmetry is deliberate and matches `ua` above. A key is structure —
        it decides what the jsonb column can be queried for, and an unexpected
        one is a client sending something this endpoint does not model, so the
        body is refused. A value is a measurement: one absurd stage number costs
        that cell, and rejecting the whole body over it would throw away the
        five derived latency columns that are the reason the request exists.
        """
        if v is None:
            return v
        if len(v) > _MAX_TIMING_KEYS:
            raise ValueError(f"at most {_MAX_TIMING_KEYS} server_timings entries")
        bad = next((k for k in v if not _TIMING_KEY.match(k)), None)
        if bad is not None:
            raise ValueError(f"server_timings key is not a stage name: {bad!r}")
        return {k: max(_INT32_MIN, min(_INT32_MAX, n)) for k, n in v.items()}


def _d(a: float | None, b: float) -> int | None:
    """A derived delta in whole milliseconds, clamped to what an `int` column holds.

    Both operands are already bounded to [0, 1e9] by `Marks`, so the clamp can
    only bite on arithmetic that should not be reachable. It is here so that if
    it ever is, the row is stored slightly wrong rather than the insert failing
    and taking the turn's telemetry with it.
    """
    if a is None:
        return None
    return max(_INT32_MIN, min(_INT32_MAX, round(a - b)))


@router.post("/telemetry", status_code=204, dependencies=[Depends(rate_limited_telemetry)])
async def telemetry(body: TelemetryIn, user: CurrentUserDep, request: Request) -> Response:
    """Store one turn's client-side latency marks.

    Rate limited (I3), but in its OWN bucket namespace. It is an authenticated
    write to a table nothing else bounds — one row per POST, as fast as a client
    cares to send them — and the limiter is the only thing standing between a
    misbehaving client and an unbounded insert loop. Sharing `/chat`'s budget
    would have made every voice turn cost two hits of an allowance sized for
    one, so `tele` gets the same limits on a separate counter. A dropped
    telemetry row costs a datapoint; there is no turn waiting on it.
    """
    m = body.marks
    await request.app.state.container.telemetry.save(
        user_id=UserId(user.user_id),
        message_id=body.message_id,
        ttfa_ms=_d(m.first_audio, m.speech_end),
        t_request_ms=_d(m.request_sent, m.speech_end),
        t_first_byte_ms=_d(m.first_byte, m.speech_end),
        t_first_sentence_ms=_d(m.first_sentence, m.speech_end),
        t_last_audio_ms=_d(m.last_audio, m.speech_end),
        server_timings=body.server_timings or {},
        client_info=body.client_info.model_dump(),
    )
    return Response(status_code=204)
