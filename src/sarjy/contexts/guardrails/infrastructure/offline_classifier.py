"""A `ClassifierPort` for configurations with no Layer-3 model behind them.

Test and local runs still want the *real* rule engine, guards and event
recording — but not a live HTTPS call to Gemini the moment some fixture
happens to say "how many ibuprofen can I take". Swapping the classifier for
a no-op that returns a benign `Classification` would be worse than a network
call: it would quietly turn every `uncertain` verdict into `allow`, so the
guard would look green in tests while doing the opposite of what it does in
production.

So this raises instead. `InputGuard` treats any classifier failure as an
absent classifier and fails closed (G-12), which means an offline run gets
the *same* answer a production run with an unreachable classifier would —
blocked, recorded as `classifier_error`.
"""

from __future__ import annotations

from sarjy.contexts.conversation.application.ports import LLMUnavailable
from sarjy.contexts.guardrails.application.ports import Classification


class OfflineClassifier:
    async def classify(self, recent_user_turns: list[str]) -> Classification:
        raise LLMUnavailable("offline")
