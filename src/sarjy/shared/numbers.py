"""Shared kernel: pulling quantities out of a sentence.

`extract_numbers` finds the numbers a listener would actually hear — digits
(including decimals, negatives, and thousands separators like "1,000") and
spoken number words ("twenty-two", "minus three", "twenty two point five",
"one hundred and four", "a hundred", "two hundred"). Ordinals, both digit
("21st") and spoken ("twenty-first"), times, both digit ("3 pm", "8:30") and
spoken ("eight thirty p.m.", "three o'clock"), and years, both bare
four-digit (19xx/20xx) and spoken ("two thousand and twenty-six", "since
nineteen ninety-eight"), are deliberately ignored — they are not quantities
the caller needs to verify against tool output. A trailing sentence
terminator ("It's 25.") does not swallow the digits before it.

It lives in the shared kernel because two layers have to agree on it exactly.
The guardrails domain (`ungrounded_numbers`) uses it to decide whether a
spoken sentence quotes a figure the tools never produced; the assessment
narrator (`GeminiNarrator.numbers_ok`) uses it to decide whether a generated
narrative is safe to speak. When those two disagreed — the narrator matching
digits only while the guard also read number words — a narrative saying
"Openness at 5.0 is one of the traits that stands out" was accepted by the
narrator and then cut mid-reply by the guard, over the "one" in "one of the
traits". One extractor, one answer.
"""

from __future__ import annotations

import re

__all__ = ["extract_numbers"]

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

# A "hundred" phrase ("a hundred", "one hundred", "one hundred and four") is
# a complete number on its own — the tens/units tail after it is optional.
# Everything else (a tens word with an optional units suffix, or a bare
# units/teens word) requires that tail, since "hundred" is the only word
# that can stand alone.
_HUNDRED_PREFIX = r"(?:a|one|two|three|four|five|six|seven|eight|nine)\s+hundred(?:\s+and)?\s*"
_TAIL = (
    r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?"
    r"|(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen"
    r"|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen))"
)
_WORDNUM = re.compile(
    rf"\b(minus|negative)?\s*(?:({_HUNDRED_PREFIX}(?:{_TAIL})?)|({_TAIL}))"
    rf"(?:\s+point\s+((?:zero|one|two|three|four|five|six|seven|eight|nine)"
    rf"(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine))*))?\b",
    re.I,
)

# --- Spoken forms that are not "quantities" and must be stripped before the
# cardinal-word parser above runs, so it cannot pick a dangling fragment out
# of them — e.g. the "twenty" inside "twenty-first", or the "eight" inside
# "eight thirty p.m." Order matters: these all run before _WORDNUM.

_ORDINAL_WORD = re.compile(
    r"\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[-\s]"
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)\b"
    r"|\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth"
    r"|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth"
    r"|twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth"
    r"|hundredth)\b",
    re.I,
)

_HOUR_WORD = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
# "eight p.m.", "eight thirty p.m.", "eight:thirty pm" (to_speech joins a
# digit "8:30" with a bare colon, no space), "three o'clock".
_SPOKEN_TIME_SUFFIXED = re.compile(
    rf"\b{_HOUR_WORD}(?:[:\s]+(?:oh[:\s]+)?[a-z-]+)?\s+(?:a\.?m\.?|p\.?m\.?|o'clock)\b",
    re.I,
)
# "eight thirty" is ambiguous on its own (could be a plain count of things),
# so only strip the bare hour-minute phrase when a time-of-day frame follows.
_SPOKEN_TIME_CONTEXTUAL = re.compile(
    rf"\b{_HOUR_WORD}[:\s]+(?:thirty|fifteen|forty-five|oh[:\s]+[a-z]+)"
    r"(?=\s+(?:in the (?:morning|afternoon|evening)\b|at night\b|tonight\b))",
    re.I,
)
# "two thousand and twenty-six" (num2words' cardinal reading of 2026).
_SPOKEN_YEAR_THOUSAND = re.compile(r"\btwo thousand(?:\s+and)?(?:\s+[a-z-]+){0,2}\b", re.I)
# "since nineteen ninety-eight" / "in twenty twenty-six" — only stripped when
# led by a year-ish preposition, since "nineteen" or "twenty" alone are
# ordinary quantities. The tail must itself be year-shaped (a tens/teen word,
# optionally suffixed with a unit) — NOT any two words, or a genuine quantity
# right after the preposition gets eaten too ("dropped by twenty five
# degrees" must keep its 25, not be mistaken for a year).
_YEAR_TAIL = (
    r"(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
    r"|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?"
)
_SPOKEN_YEAR_CONTEXTUAL = re.compile(
    rf"\b(?:in|since|until|by|year)\s+(?:nineteen|twenty)\s+(?:hundred|{_YEAR_TAIL})\b",
    re.I,
)

# Strip clock times like "8:30" or "8:30 pm" before digit extraction so
# neither side of the colon is mistaken for a standalone quantity.
_TIME = re.compile(r"\b\d{1,2}:\d{2}\s*(?:[ap]\.?m\.?)?\b", re.I)
_DIGIT = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"(?!\s*(?:am|pm|a\.m\.|p\.m\.|o'clock|st|nd|rd|th)\b)(?!\w)",
    re.I,
)
_YEAR = re.compile(r"(19|20)\d{2}")


def _words_to_num(s: str) -> float:
    cur = 0
    for w in re.split(r"[-\s]+", s.lower()):
        if w in _UNITS:
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            cur = (cur or 1) * 100
    return float(cur)


def extract_numbers(sentence: str) -> list[float]:
    """Pull quantities (digit and spoken-word) out of a sentence.

    Duplicate values are collapsed (first occurrence wins the position).
    """
    out: list[float] = []
    s = _ORDINAL_WORD.sub(" ", sentence)
    s = _SPOKEN_TIME_SUFFIXED.sub(" ", s)
    s = _SPOKEN_TIME_CONTEXTUAL.sub(" ", s)
    s = _SPOKEN_YEAR_CONTEXTUAL.sub(" ", s)
    s = _SPOKEN_YEAR_THOUSAND.sub(" ", s)
    for m in _WORDNUM.finditer(s):
        num_text = m.group(2) or m.group(3)
        val = _words_to_num(num_text)
        if m.group(4):
            val += float("0." + "".join(str(_UNITS[w]) for w in m.group(4).lower().split()))
        if m.group(1):
            val = -val
        out.append(val)
    s = _WORDNUM.sub(" ", s)
    s = re.sub(r"\b(minus|negative)\s+(\d)", r"-\2", s, flags=re.I)
    s = _TIME.sub(" ", s)
    for m in _DIGIT.finditer(s):
        token = m.group(1)
        raw = token.replace(",", "")
        if "," not in token and "." not in raw and _YEAR.fullmatch(raw.lstrip("-")):
            continue
        out.append(float(raw))
    return list(dict.fromkeys(out))
