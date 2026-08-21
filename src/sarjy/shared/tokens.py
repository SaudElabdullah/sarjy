"""Heuristic token estimator (Phase 7 Task 6, L-6).

`GeminiLLM` has no way to call `client.models.count_tokens` in this
environment — there is no Gemini API key configured for local dev/CI — so
the ~1,200-token budget on `PromptBuilder.static_text` and the ~1,800-token
budget on a fully built prompt are checked against this estimate instead of
the real API. It is a documented approximation, not a billing figure:

- About four characters per token for English text (`ceil(len(text) / 4)`).
- Plus one token for every contiguous run of whitespace. A pure chars/4
  count treats "helloworld" and "hello world" as costing the same, but a
  space is very often its own token boundary in real tokenisers — this adds
  it back without trying to model where word/subword boundaries actually
  fall.

This over-counts short, whitespace-heavy text and under-counts dense text
with few spaces (long numbers, IDs, no-space languages). Treat it as a
prompt-budget guard, not a token-billing estimate — swap in
`client.models.count_tokens` once a Gemini key is available to verify
against the real tokenizer.
"""

from __future__ import annotations

import math
import re

_WHITESPACE_RUN = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Estimate the token count of `text`. See module docstring for the heuristic."""
    if not text:
        return 0
    chars_estimate = math.ceil(len(text) / 4)
    whitespace_runs = len(_WHITESPACE_RUN.findall(text))
    return chars_estimate + whitespace_runs
