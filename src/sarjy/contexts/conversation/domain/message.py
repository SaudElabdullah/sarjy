from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from sarjy.shared.ids import MessageId, SessionId, UserId

Role = Literal["user", "assistant", "tool", "system_event"]


@dataclass(slots=True)
class Message:
    id: MessageId
    session_id: SessionId
    user_id: UserId
    role: Role
    content: str
    created_at: datetime
    speech_content: str | None = None
    client_turn_id: str | None = None
    guard_decision: str | None = None
    timings: dict[str, int] | None = None
    prompt_hash: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
