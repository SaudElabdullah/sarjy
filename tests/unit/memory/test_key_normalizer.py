import pytest

from sarjy.contexts.memory.domain.key_normalizer import normalize_key
from sarjy.shared.errors import ValidationError


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Favourite Colour", "favorite_color"),
        ("favorite color", "favorite_color"),
        ("fav-colour", "favorite_color"),
        ("Home City!", "home_city"),
        ("hometown", "home_city"),
        ("  sister's name ", "sister_name"),
        ("sister\u2019s name", "sister_name"),  # U+2019 curly apostrophe
        ("Dietary Preference", "dietary"),
        ("diet", "dietary"),
        ("birthday", "birthday"),
        ("DOB", "birthday"),
        ("what_I_like__most", "what_i_like_most"),
    ],
)
def test_normalize_key(raw: str, expected: str) -> None:
    assert normalize_key(raw) == expected


def test_normalize_key_truncates_to_60() -> None:
    assert len(normalize_key("a" * 100)) == 60


def test_normalize_key_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        normalize_key("!!! ???")
