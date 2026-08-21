"""Detects verbatim leakage of confidential system-prompt content.

`LeakDetector` does NOT compare against the full system prompt — only
against a caller-supplied `corpus`. The persona line and capability blurb
in the static prompt are things Sarjy is allowed (even expected) to
restate in conversation ("I'm Sarjy, I can check the weather for you"),
so counting overlap with those would false-positive on ordinary replies.
Callers should pass only the confidential portion — the policy,
grounding, instruction-hierarchy, and tool-guidance blocks — such as
`PromptBuilder.confidential_text`. `GuardContext.leak_corpus` carries this
string end to end; `RunTurn` is the one that sets it.

Two checks run off this, at two granularities — see
`OutputGuard.check_sentence`. A per-sentence check at `n=8` catches one
sentence quoting a block verbatim. A rolling check at `n=6` over the last
few sentences catches the shape that one misses: a model reciting the
prompt a clause at a time, where no single clause carries eight
consecutive matching words but six sentences together carry plenty.

Detection compares word n-grams of the candidate sentence against a
precomputed, cached set of n-grams drawn from the corpus. A sentence that
shares more than `threshold` n-grams with the corpus is flagged as a leak
— this catches near-verbatim regurgitation without needing an exact
substring match (so light rephrasing or partial quoting is still caught).
"""

from __future__ import annotations

import re
from functools import lru_cache

_WORD = re.compile(r"[a-z0-9']+")


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = _WORD.findall(text.lower())
    return {tuple(w[i : i + n]) for i in range(max(0, len(w) - n + 1))}


@lru_cache(maxsize=16)
def _corpus_grams(corpus: str, n: int) -> frozenset[tuple[str, ...]]:
    return frozenset(_ngrams(corpus, n))


def leak_overlap(sentence: str, corpus: str, n: int = 8) -> int:
    """Count of word n-grams shared between `sentence` and `corpus`."""
    return len(_ngrams(sentence, n) & _corpus_grams(corpus, n))


class LeakDetector:
    """Flags sentences that leak substantial chunks of a confidential corpus."""

    def __init__(self, corpus: str, n: int = 8, threshold: int = 2) -> None:
        self._corpus = corpus
        self._n = n
        self._threshold = threshold
        # Prime the cache eagerly so is_leak() calls don't pay for it later.
        # Keyed on the corpus text itself, so callers reusing the same
        # (static) corpus across turns reuse this cache entry.
        _corpus_grams(corpus, n)

    def is_leak(self, sentence: str) -> bool:
        return leak_overlap(sentence, self._corpus, self._n) > self._threshold
