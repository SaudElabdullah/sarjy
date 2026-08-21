"""Numeric-grounding check for spoken output.

`ungrounded_numbers` filters the quantities `extract_numbers` finds in a
sentence down to the ones that are not within `tolerance` of any value in
`allowed` — a fabricated number the model invented rather than read off the
tool payload.

The extractor itself lives in `sarjy.shared.numbers`, the shared kernel: the
assessment narrator has to pre-screen its own text against exactly the rule
this guard will apply later, and two near-identical extractors is precisely
how a narrative gets accepted upstream and cut downstream. `extract_numbers`
is re-exported here because this module is where callers have always
imported it from.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from sarjy.shared.numbers import extract_numbers

__all__ = ["extract_numbers", "mentions_results", "ungrounded_numbers"]

# What makes a sentence a claim ABOUT the user's assessment results, and so
# checkable against the scores on file. Deliberately narrow: the scores are
# checked to the decimal, and a check that tight applied to every sentence
# would cut "your train leaves in 15 minutes" the moment a user happened to
# have finished the test. A trait name or the language of scoring is what
# separates "you scored 4.9" from any other sentence with a number in it.
# "out of five" and a bare "results"/"scores" are in here because a follow-up
# answers in the vocabulary of the QUESTION, not of the instrument: "the other
# four are 3.2, 4.1, 2.9 and 3.8, each out of five" names no trait at all, and
# without one of these the tightest check in the guard would have sat out the
# sentence with the most invented numbers in it.
_RESULTS_MENTION = re.compile(
    r"\b(openness|conscientiousness|extraversion|agreeableness|neuroticism"
    r"|scores?|scored|results?|big five|personality|out of five)\b",
    re.I,
)


def mentions_results(sentence: str) -> bool:
    """Is this sentence talking about the user's assessment results?"""
    return _RESULTS_MENTION.search(sentence) is not None


def ungrounded_numbers(
    sentence: str, allowed: Sequence[float], tolerance: float = 1.0
) -> list[float]:
    """Quantities in `sentence` not within `tolerance` of any `allowed` value."""
    return [
        n for n in extract_numbers(sentence) if not any(abs(n - a) <= tolerance for a in allowed)
    ]
