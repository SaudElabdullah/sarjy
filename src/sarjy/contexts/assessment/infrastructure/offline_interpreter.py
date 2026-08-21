"""A no-op `AnswerInterpreterPort` for configurations with no Gemini behind them.

Mirrors `OfflineClassifier` (guardrails): an in-memory container is by
definition not talking to anything, so this must not guess at a value from a
live HTTPS call. Unlike the classifier — which fails a whole turn closed —
guessing here would silently write a score into someone's results, which is
worse than doing nothing useful. So this always returns "I could not read
that" (`value=None, confidence=0.0, control=None`), the exact `UNREADABLE`
sentinel `GeminiAnswerInterpreter` itself degrades to on a bad/failed call —
`HandleAssessmentTurn` answers that by re-asking the item with the scale
hint, never by recording a guess.

A test or eval that needs an actual answer recorded swaps in a scripted
interpreter instead (see `tests/evals/run_ocean_eval.py`'s
`ScriptedInterpreter`, or `tests/unit/assessment/test_handle_turn.py`'s).
"""

from __future__ import annotations

from sarjy.contexts.assessment.application.ports import Interpretation

UNREADABLE = Interpretation(value=None, confidence=0.0, control=None)


class OfflineInterpreter:
    async def interpret(
        self, item_text: str, scale_labels: list[str], user_text: str
    ) -> Interpretation:
        return UNREADABLE
