from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sarjy.shared.ids import SessionId, UserId

InputMode = Literal["voice", "text"]


@dataclass(frozen=True, slots=True)
class TurnInput:
    user_id: UserId
    session_id: SessionId | None
    client_turn_id: str
    text: str
    input_mode: InputMode = "voice"
    speculative: bool = False
    # What the HTTP gate cost before the turn existed: the JWT verify plus the
    # rate limiter's round trip (I5). Measured by the interface layer because
    # only the interface layer is there for it, and carried in so `Timings` can
    # report a `t_total` that covers the whole request rather than the part of
    # it the orchestrator happens to be present for. Zero for a caller with no
    # gate — an eval harness, a test — which is the honest answer for them.
    t_auth_ms: int = 0
