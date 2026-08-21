"""C1/M3: one invisible-character set, shared by every module that strips one."""

from sarjy.contexts.conversation.application import prompt_builder
from sarjy.contexts.guardrails.domain import normalize as normalize_mod
from sarjy.contexts.memory.domain import sanitise as sanitise_mod
from sarjy.shared.invisible import INVISIBLE_RE

# One representative character per range in the shared set, written as escapes
# so the file stays reviewable — these are invisible by definition.
SAMPLES = [
    "\u00ad",  # soft hyphen
    "\u200b",  # zero-width space
    "\u200f",  # right-to-left mark
    "\u2028",  # line separator
    "\u202e",  # right-to-left override
    "\u2060",  # word joiner
    "\u2066",  # left-to-right isolate
    "\u206f",  # deprecated format control
    "\ufeff",  # BOM / ZWNBSP
    "\U000e0001",  # language tag
    "\U000e007f",  # cancel tag
]


def test_the_three_modules_share_one_compiled_pattern() -> None:
    # Identity, not equality: three private copies had already drifted apart
    # (normalize was missing U+00AD and the tag block), and sharing the *object*
    # makes a future divergence impossible rather than merely unlikely.
    assert normalize_mod._ZW is INVISIBLE_RE
    assert prompt_builder._INVISIBLE is INVISIBLE_RE
    assert sanitise_mod._INVISIBLE is INVISIBLE_RE


def test_every_sample_is_stripped_by_all_three() -> None:
    for ch in SAMPLES:
        assert INVISIBLE_RE.search(ch), f"{ch!r} not in the shared set"
        assert "ab" in normalize_mod.normalize(f"a{ch}b")
        assert prompt_builder.sanitise_value(f"a{ch}b") == "ab"
        assert sanitise_mod.sanitise(f"a{ch}b") == "ab"


def test_ordinary_spacing_is_left_alone() -> None:
    # A no-break space and a normal space are visible separators: stripping them
    # would silently fuse words that really are distinct tokens.
    for ch in (" ", "\u00a0", "\u2007"):
        assert not INVISIBLE_RE.search(ch)
