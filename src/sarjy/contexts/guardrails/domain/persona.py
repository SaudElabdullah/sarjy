"""Detects the assistant breaking the Sarjy persona in its own output.

Catches self-identification as an underlying model/vendor, claims of
having no restrictions, and explicit denial of the Sarjy name.

The vendor-attribution forms cover the whole verb family — "developed
by", but also "made / built / created / trained by" — because the
paraphrase is what a model actually reaches for once "developed" is
refused, and all five are the same claim. All of them keep the
first-person gate.

Deliberately does NOT match a bare "I'm an AI" / "I'm just an AI" — that
phrase is too common in benign self-description ("I'm an AI assistant, so
I can't taste coffee") to use as a signal on its own, and it produced
false positives in review. Likewise a bare "developed by <vendor>" only
counts when it has a first-person subject ("I was/am developed by
OpenAI") — third-person mentions ("It was developed by OpenAI") are not
the model talking about itself. The "an AI ... developed by <vendor>"
form is still caught in first person ("I am/I'm/As an AI developed by
OpenAI") since that phrasing is unambiguously the model describing
itself, regardless of which of the three subject spellings introduces it.
"""

from __future__ import annotations

import re

_PATTERNS = re.compile(
    r"\bas an ai (language )?model\b"
    r"|\bi am an ai (language )?model\b"
    r"|\bas a large language model\b"
    r"|\bi am (chatgpt|claude|gemini|bard|gpt)\b"
    r"|\bi'?m (dan|chatgpt|claude|gemini|bard|gpt)\b"
    r"|\bi'?ll be (chatgpt|claude|gemini|dan) for\b"
    r"|\b(?:i am|i'?m|as) an ai (?:\w+ )?"
    r"(?:developed|made|built|created|trained) by (openai|anthropic|google)\b"
    r"|\bi (was|am) (developed|made|built|created|trained) by (openai|anthropic|google)\b"
    r"|\bi (have|now have) no (restrictions|rules|limits)\b"
    r"|\bmy name is not sarjy\b"
    r"|\bcall me (dan|chatgpt|claude|gemini|bard|gpt)\b",
    re.I,
)


def is_persona_break(sentence: str) -> bool:
    return bool(_PATTERNS.search(sentence))
