"""Cheap regex-based detector for weather-shaped questions (PRD W-8)."""

from __future__ import annotations

import re

_WEATHER = re.compile(
    r"\b(weather|forecast|temperature|rain(y|ing)?|snow(y|ing)?|sunny|cloudy|windy|"
    r"humid(ity)?|storm(y)?|umbrella|degrees|celsius|fahrenheit|"
    r"how (hot|cold|warm|chilly) (is|will)|is it (hot|cold|warm|chilly|freezing))\b",
    re.I,
)
# A weather word on its own is not a question about the weather — "a joke about
# the rain" and "a song about the sun" are the same word in a different frame.
# Forcing the tool needs an actual lookup: something being asked, or asked about.
_LOOKUP = re.compile(
    r"(\bwhat\b|\bwhat's\b|\bhow\b|\bis it\b|\bwill it\b|\bgoing to\b|\bforecast\b|"
    r"\bdo i need\b|\bshould i\b|\btemperature\b|\bdegrees\b|"
    r"\bweather in\b|\bweather for\b|\bweather like\b|\?)",
    re.I,
)
# Frames that read as a lookup but are not one, plus the idioms that use weather
# words to mean something else entirely.
_NEGATE = re.compile(
    r"\b(joke|story|song|poem|under the weather|raining cats|remember|forget|my favorite)\b",
    re.I,
)


def is_weather_question(text: str) -> bool:
    if not _WEATHER.search(text):
        return False
    if _NEGATE.search(text):
        return False
    return bool(_LOOKUP.search(text))
