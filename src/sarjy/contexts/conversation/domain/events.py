from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from sarjy.shared.ids import MessageId, SessionId
from sarjy.shared.text import Sentence


@dataclass(frozen=True, slots=True)
class SessionEvent:
    session_id: SessionId
    type: Literal["session"] = "session"


@dataclass(frozen=True, slots=True)
class GuardEvent:
    decision: Literal["allow", "block"]
    category: str | None = None
    type: Literal["guard"] = "guard"


@dataclass(frozen=True, slots=True)
class ToolStatusEvent:
    tool: str
    state: Literal["start", "end"]
    ok: bool | None = None
    type: Literal["tool_status"] = "tool_status"


@dataclass(frozen=True, slots=True)
class SentenceEvent:
    sentence: Sentence
    type: Literal["sentence"] = "sentence"


@dataclass(frozen=True, slots=True)
class TokenEvent:
    text: str
    type: Literal["token"] = "token"


@dataclass(frozen=True, slots=True)
class DoneEvent:
    message_id: MessageId
    timings: dict[str, int] = field(default_factory=dict)
    workflow: dict[str, Any] | None = None
    type: Literal["done"] = "done"


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    code: Literal["rate_limited", "gemini_unavailable", "timeout", "invalid_input", "internal"]
    message_spoken: str
    type: Literal["error"] = "error"


TurnEvent = (
    SessionEvent
    | GuardEvent
    | ToolStatusEvent
    | SentenceEvent
    | TokenEvent
    | DoneEvent
    | ErrorEvent
)
