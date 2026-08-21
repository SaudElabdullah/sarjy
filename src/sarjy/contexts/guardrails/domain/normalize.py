"""Input normalisation for guardrail checks.

Applies Unicode canonicalisation, strips invisible/control characters,
and de-obfuscates common evasion tricks (leetspeak, base64, rot13) by
appending decoded variants to the normalised text so downstream matchers
see both the surface form and the underlying intent.

De-obfuscation runs to a bounded fixed point (depth 2): every variant a
first pass produces is fed back through the same extractors once more.
A single pass only ever peeled one layer, so *stacking* two was a
one-line bypass — base64 of a leetspeak payload, rot13 of a leetspeak
payload, or base64 of base64 all decoded to something the rules never
saw. Depth is capped (and the work per input bounded, see `_MAX_SEEDS`
/ `_MAX_VARIANTS`) because each level multiplies the candidate set and
an unbounded fixed point is a denial-of-service surface, not a feature.
"""

from __future__ import annotations

import base64
import codecs
import re
import unicodedata

from sarjy.shared.invisible import INVISIBLE_RE

# The one shared invisible/format-character set (`sarjy.shared.invisible`),
# aliased under the name this module has always used. It used to be a
# hand-written copy that had drifted from the sanitiser's: it was missing
# the soft hyphen and the Unicode tag block, so "ki<SHY>ll myself" and a
# tag-block-smuggled "ignore" both walked past \b-anchored rules here while
# being stripped correctly on the prompt-assembly side.
_ZW = INVISIBLE_RE
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LEET = str.maketrans(
    {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s"}
)
_B64 = re.compile(r"(?<![A-Za-z0-9+/_-])([A-Za-z0-9+/_-]{16,}={0,2})(?![A-Za-z0-9+/_=-])")
# A run long enough to judge as rot13. Digits are allowed *inside* the run
# (rot13 leaves them untouched) so a leetspeak payload — which is riddled
# with them — is not chopped into fragments too short to test. The stop-word
# gate in `_maybe_rot13`, not the run's alphabet, is what keeps this precise.
_ALPHA_RUN = re.compile(r"[A-Za-z][A-Za-z0-9' ]{19,}")
# Bounds on the depth-2 fixed point: how many first-pass variants get fed back
# in, and the hard cap on the variant list handed to the rule engine. Each
# level multiplies the candidate set, and the engine matches every rule against
# every variant, so this is a cost ceiling as much as a sanity one.
_MAX_SEEDS = 8
_MAX_VARIANTS = 24
_STOP = {"the", "and", "you", "your", "all", "tell", "me", "ignore", "are", "is", "to", "of"}


def _maybe_b64(s: str) -> str | None:
    try:
        raw = base64.b64decode(s, validate=True)
        txt = raw.decode("ascii")
    except Exception:
        try:
            raw = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
            txt = raw.decode("ascii")
        except Exception:
            return None
    return txt if txt.isprintable() and any(c.isalpha() for c in txt) else None


def _maybe_rot13(s: str) -> str | None:
    dec = codecs.decode(s, "rot13")
    words = set(re.findall(r"[a-z]+", dec.lower()))
    original_words = set(re.findall(r"[a-z]+", s.lower()))
    if len(words & _STOP) >= 2 and len(original_words & _STOP) < 2:
        return dec
    return None


def _deobfuscate(t: str) -> list[str]:
    """One pass of every decoder, over one piece of text.

    Order is base64, then rot13, then leetspeak — the same order (and the
    same gates) the single-pass version used. Case matters to base64, so the
    caller passes text that has NOT been lowercased yet; the leet variant
    lowercases on its own.
    """
    out: list[str] = []
    for m in _B64.finditer(t):
        d = _maybe_b64(m.group(1))
        if d:
            out.append(d)
    for m in _ALPHA_RUN.finditer(t):
        d = _maybe_rot13(m.group(0))
        if d:
            out.append(d)
    low = t.lower()
    leet = low.translate(_LEET)
    if leet != low and re.search(r"[a-z]", leet):
        out.append(leet)
    return out


def normalize_variants(text: str) -> list[str]:
    """Normalise `text` into a list of independently-matchable variants.

    Element 0 is the primary normalised form; the rest are de-obfuscated
    "extras" (zero-width-spaced, base64-decoded, rot13-decoded, de-leeted).
    Every element is already lowercased and whitespace-collapsed.

    Variants are returned as a LIST, not a single concatenated string, so a
    matcher never sees a seam between two of them. Concatenating created a
    boundary where the tail of one variant and the head of the next spell out
    a phrase present in neither — and rules with `.{0,N}` gaps happily bridge
    such a seam even when a non-word separator sits in the gap (e.g.
    "...do not ignore" + "your instructions are good..." matches
    `ignore.{0,30}your instructions`). Evaluating each variant on its own
    removes that class of false positive structurally.
    """
    base = unicodedata.normalize("NFKC", text)
    # Zero-width characters are DELETED as the primary form: an attacker
    # who splits a single word mid-token with a ZWSP ("k​ill myself") is
    # relying on the char surviving as a separator; deleting it re-fuses
    # the word so \b-anchored rules still see "kill".
    #
    # But deletion alone lets an attacker join two DIFFERENT words with a
    # ZWSP ("ignore​all") to dodge a rule that requires them as separate
    # \b-bounded tokens ("ignoreall" no longer matches \bignore\b...\ball\b).
    # So, same pattern as the leet/base64/rot13 de-obfuscation below, we
    # additionally compute a space-substituted variant and append it as an
    # extra — never as the primary text — so both attack shapes are caught
    # without either bypassing the other.
    t = _ZW.sub("", base)
    t = _CTRL.sub("", t)
    extras: list[str] = []
    # Gate the extra on an actual ZW character being present in the
    # source text — NOT on comparing the (whitespace-collapsed) spaced
    # variant against the (uncollapsed) primary text. That comparison is
    # true for ANY input containing a newline or a run of >1 spaces, even
    # with zero ZW chars, which used to (a) double the normalized output
    # for every such message and (b) let the collapsed extra's `.{0,N}`
    # rule gaps bridge across what was a newline in the original — e.g.
    # "ignore\nthe previous instructions" doesn't match `.{0,30}` (which
    # excludes newlines) in the primary text, but did in the always-on,
    # whitespace-collapsed "extra".
    if _ZW.search(base):
        zw_spaced = _CTRL.sub("", _ZW.sub(" ", base))
        zw_spaced = re.sub(r"\s+", " ", zw_spaced).strip().lower()
        if zw_spaced != t.lower():
            extras.append(zw_spaced)
    # Depth 1: decode what is visible in the text as written.
    first = _deobfuscate(t)
    extras += first
    # Depth 2: decode what depth 1 uncovered. An attacker who base64-encodes a
    # leetspeak payload (or rot13s one, or base64s a base64) gets exactly one
    # layer peeled by a single pass, and the plaintext underneath — the part the
    # rules can actually read — never appears in any variant. Feeding each
    # first-pass result back through the same extractors once closes that, and
    # stops there: the seed count is capped, so the work stays linear in the
    # input rather than exponential in the nesting an attacker chooses.
    for seed in first[:_MAX_SEEDS]:
        extras += _deobfuscate(seed)
    t = t.lower()
    # Collapse/lowercase each variant on its own and drop empties and exact
    # duplicates (the ZW-spaced extra is compared against the uncollapsed
    # primary, so it can come out identical after collapsing). Order is
    # preserved: element 0 stays the primary form.
    out: list[str] = []
    for v in (t, *extras):
        v = re.sub(r"\s+", " ", v).strip().lower()
        if v and v not in out:
            out.append(v)
        if len(out) >= _MAX_VARIANTS:
            break
    return out


def normalize(text: str) -> str:
    r"""Backwards-compatible single-string form of :func:`normalize_variants`.

    Variants are joined with `" | "`, a separator matched by neither `\s` nor
    `\w`. Prefer `normalize_variants` + `RuleEngine.evaluate_variants`: the
    joined form only exists so existing callers/tests keep working, and
    `RuleEngine.evaluate` splits it back apart on the same separator before
    matching.
    """
    return " | ".join(normalize_variants(text))
