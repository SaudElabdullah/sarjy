"""Shared kernel: the one set of invisible/format characters we strip.

Three places used to carry their own hand-written copy of this class:
`guardrails.domain.normalize` (guard input), `conversation.application.
prompt_builder` (prompt assembly) and `memory.domain.sanitise` (stored
facts). The three lists had drifted — `normalize` was missing the soft
hyphen (U+00AD) and the Unicode tag block (U+E0000..U+E007F), so
"ki<SHY>ll myself" sailed past a `\\b`-anchored self-harm rule while the
*same* string was stripped correctly on its way into the prompt. A guard
and the sanitiser disagreeing about what counts as invisible is a hole by
construction, so the set lives here once and all three import it.

What is in it, and why each range is a smuggling vector:

* U+00AD SOFT HYPHEN — renders as nothing mid-word; splits a token so
  `\\bkill\\b` no longer matches.
* U+200B..U+200F — zero-width space/non-joiner/joiner, LTR/RTL marks.
* U+2028..U+202E — line/paragraph separators and the bidi
  embedding/override controls (text that displays in a different order
  than it is stored).
* U+2060..U+206F — word joiner, the invisible math operators, and the
  deprecated format controls. U+2066..U+2069 (the bidi isolates) are a
  subset of this range and are named separately in the older comments.
* U+FEFF — BOM / zero-width no-break space.
* U+E0000..U+E007F — the Unicode tag block: a full invisible ASCII
  alphabet, which is how a hidden "ignore all previous instructions" is
  carried inside otherwise innocent text.

Stripping (not space-substituting) is the primary form; callers that also
need a space-substituted variant build it themselves — see
`normalize_variants`.
"""

from __future__ import annotations

import re

__all__ = ["INVISIBLE_RE"]

INVISIBLE_RE: re.Pattern[str] = re.compile(
    "["
    "\u00ad"  # soft hyphen
    "\u200b-\u200f"  # zero-width space/ZWNJ/ZWJ, LRM/RLM
    "\u2028-\u202e"  # line/paragraph separators, bidi embedding/override
    "\u2060-\u206f"  # word joiner, invisible operators, bidi isolates, deprecated
    "\ufeff"  # BOM / ZWNBSP
    "\U000e0000-\U000e007f"  # Unicode tag block (invisible ASCII)
    "]"
)
