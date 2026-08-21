"""Re-export of the shared `GuardDecision` value object.

The guardrails domain must not depend on the conversation application
layer, so `GuardDecision` lives in the shared kernel
(`sarjy.shared.guard`). This module re-exports it under the guardrails
domain namespace for consumers within this context.
"""

from __future__ import annotations

from sarjy.shared.guard import GuardDecision

__all__ = ["GuardDecision"]
