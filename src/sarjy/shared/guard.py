"""Shared kernel: the guard decision value object.

`GuardDecision` is used by both the conversation application layer (which
consumes it via `InputGuardPort`) and the guardrails domain (which produces
it). It lives in the shared kernel so the guardrails domain never has to
import the conversation application layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["GuardDecision"]


@dataclass(frozen=True)
class GuardDecision:
    action: Literal["allow", "block", "uncertain"]
    category: str | None = None
    layer: int = 0
    rule_id: str | None = None
    severity: int = 1
