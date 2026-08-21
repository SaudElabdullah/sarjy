"""Shared key/value screening helper for `RememberFact` and `EditFact`.

Both use cases must screen the key and the value SEPARATELY through the
same `ValueScreenPort` — never concatenated into one string — or a key that
normalises to "note"/"remember"/"save"/"store" paired with a value shaped
like `"that my X is '<payload>'"` reconstructs the guardrail rule engine's
memory-set-frame carve-out at screening time and smuggles an
uncertain-severity payload past the screen meant to catch it (Phase 8
Task 6b fix round 1, Important 2). Factored out here so the two use cases
enforce identically rather than drifting.
"""

from __future__ import annotations

from sarjy.contexts.memory.application.ports import ValueScreenPort
from sarjy.shared.ids import MessageId, UserId

REFUSAL_REASON = "that doesn't look safe to store"


def screen_reason(
    screen: ValueScreenPort,
    key: str,
    value: str,
    *,
    user_id: UserId | None = None,
    message_id: MessageId | None = None,
) -> str | None:
    """Return a short, neutral refusal reason if `key` or `value` is unsafe, else `None`.

    Screens `key` and `value` as two independent calls (see module
    docstring) and stops at the first refusal.
    """
    for candidate in (key, value):
        verdict = screen.screen(candidate, user_id=user_id, message_id=message_id)
        if not verdict.allowed:
            return REFUSAL_REASON
    return None
