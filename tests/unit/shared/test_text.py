from sarjy.shared.text import SentenceSplitter, to_speech


def _run(chunks: list[str]) -> list[str]:
    s = SentenceSplitter()
    out: list[str] = []
    for c in chunks:
        out += s.feed(c)
    tail = s.flush()
    if tail:
        out.append(tail)
    return out


def test_splits_on_terminal_punctuation_across_chunks() -> None:
    result = _run(["Hello the", "re. How are", " you? Fine!"])
    assert result == ["Hello there.", "How are you?", "Fine!"]


def test_does_not_split_on_decimal_or_abbreviation() -> None:
    result = _run(["It is 22.5 degrees in St. Louis today."])
    assert result == ["It is 22.5 degrees in St. Louis today."]


def test_clause_split_after_60_chars_on_comma() -> None:
    long = (
        "I can help with the weather, remembering things for you, "
        "a quick chat, or a personality test, and more"
    )
    parts = _run([long])
    assert len(parts) >= 2 and all(len(p) <= 110 for p in parts)


def test_flush_returns_remainder() -> None:
    s = SentenceSplitter()
    assert s.feed("no punctuation yet") == []
    assert s.flush() == "no punctuation yet"


def test_to_speech_numbers_and_markdown() -> None:
    result = to_speech("It's **22** degrees, 40% chance of rain.")
    assert result == "It's twenty-two degrees, forty percent chance of rain."
    assert to_speech("Around -3°C") == "Around minus three degrees Celsius"


def test_clause_split_does_not_break_on_number_commas() -> None:
    long = (
        "The prices are 1,000, 2,000, and 3,000 dollars respectively "
        "for each of the three items today"
    )
    parts = _run([long])
    assert len(parts) >= 2


def test_to_speech_ordinal() -> None:
    assert to_speech("the 21st century") == "the twenty-first century"


def test_to_speech_thousands_separator() -> None:
    assert to_speech("It costs 1,000 dollars.") == "It costs one thousand dollars."


def test_to_speech_strips_bullets_and_links() -> None:
    assert to_speech("- item one\n- item two") == "item one item two"
    assert to_speech("Check [this link](https://example.com) out.") == "Check this link out."
