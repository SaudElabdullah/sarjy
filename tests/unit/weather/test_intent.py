import pytest

from sarjy.contexts.weather.application.intent import is_weather_question


@pytest.mark.parametrize(
    "t",
    [
        "what's the weather in Tokyo",
        "Will it rain tomorrow in Berlin?",
        "how hot is it in lisbon",
        "temperature in paris",
        "is it snowing in oslo",
        "forecast for friday",
        "how many degrees is it outside",
        "is it cold in reykjavik today",
        "do I need an umbrella in London",
        "what's the weather like in Cairo",
        "should I take a coat, is it raining?",
        "weather for New York please",
    ],
)
def test_positive(t: str) -> None:
    assert is_weather_question(t)


@pytest.mark.parametrize(
    "t",
    [
        "what's my favorite color",
        "give me a personality test",
        "remember that my sister is Amal",
        "tell me a joke",
        "I feel hot-headed today",
    ],
)
def test_negative(t: str) -> None:
    assert not is_weather_question(t)


@pytest.mark.parametrize(
    "t",
    [
        "Tell me a joke about the rain.",
        "write me a song about the sunny weather",
        "tell me a story about a snowstorm",
        "write a poem about the rain",
    ],
)
def test_weather_words_in_a_creative_request_are_not_a_lookup(t: str) -> None:
    """These used to force a get_weather call, so a request for a joke came back
    with a forecast attached."""
    assert not is_weather_question(t)


@pytest.mark.parametrize(
    "t",
    [
        "I'm feeling a bit under the weather",
        "is it just me or is it raining cats and dogs?",
    ],
)
def test_weather_idioms_are_not_a_lookup(t: str) -> None:
    assert not is_weather_question(t)


@pytest.mark.parametrize(
    "t",
    [
        "the weather was lovely in Rome last summer",
        "I love a rainy day",
        "remember that I hate cold weather",
    ],
)
def test_a_weather_word_without_a_lookup_frame_is_not_a_question(t: str) -> None:
    assert not is_weather_question(t)
