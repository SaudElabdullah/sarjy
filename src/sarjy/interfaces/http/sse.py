"""Server-Sent Events encoder for `TurnEvent` (PRD §9.1)."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from sarjy.contexts.conversation.domain.events import TurnEvent


def _payload(ev: TurnEvent) -> dict[str, Any]:
    d = dataclasses.asdict(ev)
    d.pop("type", None)
    if "sentence" in d:  # flatten SentenceEvent
        d = {
            "i": d["sentence"]["index"],
            "text": d["sentence"]["text"],
            "speech": d["sentence"]["speech"],
            "final": False,
        }
    return d


def encode(ev: TurnEvent) -> bytes:
    return f"event: {ev.type}\ndata: {json.dumps(_payload(ev), default=str)}\n\n".encode()
