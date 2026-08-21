from __future__ import annotations

import re
import unicodedata

from sarjy.shared.errors import ValidationError

MAX_KEY_LEN = 60

# British → American spellings and common phrasings → canonical keys. Applied
# per-word after snake_casing, then again against the joined key so a multi-word
# phrase (e.g. "dietary_preference") can collapse onto a single canonical word.
_WORD_MAP = {
    "favourite": "favorite",
    "fav": "favorite",
    "colour": "color",
    "hometown": "home_city",
    "home_town": "home_city",
    "city": "city",
    "diet": "dietary",
    "dietary_preference": "dietary",
    "dietary_preferences": "dietary",
    "dob": "birthday",
    "date_of_birth": "birthday",
    "birth_date": "birthday",
}
_PHRASE_MAP = {
    "home_city": "home_city",
    "favorite_color": "favorite_color",
}


def normalize_key(raw: str) -> str:
    # U+2019 (the "curly" apostrophe browsers/keyboards commonly produce) has no
    # NFKD decomposition to ASCII, so it would otherwise survive the ascii-encode
    # below untouched and then get silently dropped instead of triggering the
    # possessive-"'s" collapse below -- folding it onto the ASCII apostrophe first
    # keeps "sister's name" and its U+2019 variant normalising to the same key.
    raw = raw.replace("\u2019", "'")
    s = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    s = s.lower().replace("'s ", " ").replace("'s", "")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = re.sub(r"_{2,}", "_", s)
    if not s:
        raise ValidationError("memory key is empty after normalisation")
    parts = [_WORD_MAP.get(p, p) for p in s.split("_")]
    s = "_".join(parts)
    s = _WORD_MAP.get(s, s)
    s = _PHRASE_MAP.get(s, s)
    return s[:MAX_KEY_LEN]
