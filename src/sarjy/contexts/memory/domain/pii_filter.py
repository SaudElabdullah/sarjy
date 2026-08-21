from __future__ import annotations

import re
from itertools import pairwise

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Formatted SSN (dash or space between the 3-2-4 groups) is unambiguous enough to
# flag on its own. A bare, unformatted 9-digit run is only flagged when the value
# also names it as a social security number, so plain order/account numbers pass.
_SSN_SEP = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")
_SSN_BARE = re.compile(r"\b\d{9}\b")
_SSN_KEYWORD = re.compile(r"\bssn\b|social security", re.I)
_LONG_DIGITS = re.compile(r"(?:\d[ -]?){16,}")
_CARDISH = re.compile(r"(?:\d[ -]?){13,19}")
_PASSWORD = re.compile(r"\b(password|passcode|passwd|pin code|pin|otp|secret)\b", re.I)
# Key-aware rejection: a *key* like "wifi_password" or "bank_account_number" must be
# rejected even when the value itself (e.g. a 4-digit PIN) doesn't trip any of the
# value-shaped checks above. Matched word-by-word on the normalised (snake_case) key
# plus adjacent two-token joins, so e.g. "token_ring_hobby" and "favorite_pin_collection"
# are also rejected (word-level match, not phrase-level) — a deliberately conservative
# trade-off that a caller with a legitimate "token_ring_hobby" key can work around by
# renaming it.
_SENSITIVE_KEY_TOKENS = {
    "password",
    "passcode",
    "passwd",
    "pin",
    "otp",
    "secret",
    "ssn",
    "social_security",
    "card_number",
    "credit_card",
    "cvv",
    "account_number",
    "api_key",
    "token",
}


def _key_reject_reason(key: str) -> str | None:
    tokens = key.split("_")
    joins = {f"{a}_{b}" for a, b in pairwise(tokens)}
    if (set(tokens) | joins) & _SENSITIVE_KEY_TOKENS:
        return "I don't store passwords or secrets"
    return None


_STREET = re.compile(
    r"\b\d{1,5}[a-z]?\s+[a-z][a-z .'-]{1,40}\s+"
    r"(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|court|ct|place|pl|terrace|square|sq)\b",
    re.I,
)
_POSTCODE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}|\d{5}(-\d{4})?)\b")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def reject_reason(value: str, key: str | None = None) -> str | None:
    """Return a human reason if the value must not be stored, else None."""
    if key:
        key_reason = _key_reject_reason(key)
        if key_reason:
            return key_reason
    v = value.strip()
    for m in _CARDISH.finditer(v):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return "that looks like a payment card number"
    if _LONG_DIGITS.search(v):
        return "that looks like an account or ID number"
    if _SSN_SEP.search(v) or (_SSN_BARE.search(v) and _SSN_KEYWORD.search(v)):
        return "that looks like a government ID number"
    if _PASSWORD.search(v):
        return "I don't store passwords or secret codes"
    if _EMAIL.search(v) and key != "email":
        return "I don't store email addresses unless you ask me to remember your email specifically"
    if _STREET.search(v) and (_POSTCODE.search(v) or "," in v):
        return "I don't store full street addresses"
    return None
