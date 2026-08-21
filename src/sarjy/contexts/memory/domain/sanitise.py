from __future__ import annotations

import re
import unicodedata

from sarjy.shared.invisible import INVISIBLE_RE

_DELIMS = re.compile(r"</?(facts|user|workflow|system|tool|instructions?)>", re.I)
# Invisible characters an injection can hide inside a delimiter (a zero-width space
# between "us" and "er" keeps <user> off the regex) or use to smuggle instructions
# past a human reviewer. Shared with the guard's normaliser and the prompt builder
# (`sarjy.shared.invisible`) — see that module for what each range buys an attacker.
_INVISIBLE = INVISIBLE_RE
# Control whitespace (newlines, tabs, ...) is dropped outright rather than folded to a
# space: it is most often introduced to force a line break mid-word around a stripped
# delimiter, and collapsing it to a space would hand back a token boundary that was
# never really there.
_CONTROL_WS = re.compile(r"[\r\n\t\v\f]+")
_MULTI_SPACE = re.compile(r" {2,}")


def sanitise(value: str, limit: int = 200) -> str:
    # NFKC first so compatibility forms (the fullwidth less-than/greater-than signs
    # U+FF1C and U+FF1E, etc.) fold onto the ASCII delimiters, then strip invisibles
    # so they cannot break a delimiter apart, and only then remove the delimiters.
    v = unicodedata.normalize("NFKC", value)
    v = _INVISIBLE.sub("", v)
    # To a fixed point: one pass over "<</user>/user>" removes the inner "</user>" and
    # splices the surviving halves into a fresh one, so a single sub would hand the
    # model back the very delimiter it was supposed to strip.
    while True:
        stripped = _DELIMS.sub("", v)
        if stripped == v:
            break
        v = stripped
    v = _CONTROL_WS.sub("", v)
    v = _MULTI_SPACE.sub(" ", v)
    return v.strip()[:limit]
