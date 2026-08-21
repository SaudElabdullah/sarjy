"""Hate-slur lexicon, kept separate so it is never surfaced in logs.

This module is intentionally not imported by anything other than
`rules.py`, and its contents are not included in exception messages,
log lines, or the `Rule` objects it feeds (only the compiled pattern is
kept). The list is deliberately short: a handful of unambiguous,
widely-recognised slurs is enough to catch overt hate speech without
false-positiving on reclaimed or ambiguous terms, which is left to the
Layer 3 classifier.
"""

from __future__ import annotations

HATE_SLURS: tuple[str, ...] = (
    "nigger",
    "nigga",
    "faggot",
    "kike",
    "chink",
    "spic",
    "tranny",
    "retard",
    # Not a real slur: a sentinel token so tests can exercise the
    # hate.slurs rule without putting real slurs in the test suite.
    "zzzguardrailtestslur",
)
