from __future__ import annotations

import re
from dataclasses import dataclass

from num2words import num2words

_ABBREV = {"st", "mr", "mrs", "ms", "dr", "vs", "etc", "e.g", "i.e", "jr", "sr", "no"}
_TERMINAL = re.compile(r"([.!?]+)(\s+|$)")
_CLAUSE_MIN = 60


@dataclass(frozen=True, slots=True)
class Sentence:
    index: int
    text: str
    speech: str


class SentenceSplitter:
    """Incrementally splits a token stream into speakable sentences."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out: list[str] = []
        while True:
            cut = self._find_cut(self._buf)
            if cut is None:
                break
            sentence, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str | None:
        tail, self._buf = self._buf.strip(), ""
        return tail or None

    def _find_cut(self, s: str) -> int | None:
        for m in _TERMINAL.finditer(s):
            end = m.end(1)
            if m.group(2) == "" and end == len(s):  # punctuation at very end, maybe more coming
                continue
            before = s[: m.start(1)]
            last_word = before.split()[-1].lower().rstrip(".") if before.split() else ""
            if last_word in _ABBREV:
                continue
            if m.group(1) == "." and end < len(s) and s[end].isdigit():
                continue
            return end
        if len(s) >= _CLAUSE_MIN:
            idx = _find_clause_cut(s)
            if idx is not None:
                return idx
        return None


def _find_clause_cut(s: str) -> int | None:
    """Find a ", " clause boundary to cut on.

    Prefers the first boundary at or past index 40 (so the head chunk is a
    reasonable size), falling back to the last boundary at or past index 20.
    A boundary only counts if the character right after the ", " is a
    letter, so commas inside numbers (e.g. "1,000") are never mistaken for
    clause breaks.
    """
    pos = s.find(", ", 40)
    while pos != -1:
        if pos + 2 < len(s) and s[pos + 2].isalpha():
            return pos + 1
        pos = s.find(", ", pos + 1)
    pos = s.rfind(", ")
    while pos >= 20:
        if pos + 2 < len(s) and s[pos + 2].isalpha():
            return pos + 1
        pos = s.rfind(", ", 0, pos)
    return None


_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_MD = re.compile(r"[*_`#>]+")
_ORDINAL = re.compile(r"(\d+)(?:st|nd|rd|th)\b")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BULLET = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)


def to_speech(text: str) -> str:
    t = _LINK.sub(r"\1", text)
    t = _BULLET.sub("", t)
    t = t.replace("\n", " ")
    t = _MD.sub("", t)
    t = (
        t.replace("°C", " degrees Celsius")
        .replace("°F", " degrees Fahrenheit")
        .replace("°", " degrees")
    )
    t = t.replace("%", " percent")

    def rep_ordinal(m: re.Match[str]) -> str:
        words: str = num2words(int(m.group(1)), to="ordinal")
        return words

    t = _ORDINAL.sub(rep_ordinal, t)
    t = _THOUSANDS.sub("", t)

    def rep(m: re.Match[str]) -> str:
        raw = m.group(0)
        neg = raw.startswith("-")
        num = float(raw.lstrip("-")) if "." in raw else int(raw.lstrip("-"))
        words: str = num2words(num)
        return ("minus " if neg else "") + words

    t = _NUM.sub(rep, t)
    return re.sub(r"\s{2,}", " ", t).strip()
