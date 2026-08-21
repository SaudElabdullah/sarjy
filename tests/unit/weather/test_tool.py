from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sarjy.contexts.conversation.application.ports import Fact
from sarjy.contexts.weather.application.get_weather import GetWeather
from sarjy.contexts.weather.application.tools import GetWeatherTool
from sarjy.contexts.weather.application.units_resolver import UnitsResolver
from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.contexts.weather.domain.location import Location
from sarjy.contexts.weather.domain.units import Units
from sarjy.contexts.weather.infrastructure.mock_provider import MockProvider
from sarjy.shared.clock import FakeClock
from sarjy.shared.ids import UserId
from tests.unit.weather.test_get_weather import FakeProvider, MemCache

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
UID = UserId(uuid.uuid4())


class FactsWith:
    def __init__(self, facts: list[Fact]) -> None:
        self._f = facts
        self.calls = 0

    async def snapshot(self, user_id: UserId) -> list[Fact]:
        self.calls += 1
        return self._f


class CountingProvider:
    """Wraps a provider and counts geocode/forecast calls, to assert on the
    number of provider round trips a single tool invocation makes."""

    def __init__(self, inner: MockProvider) -> None:
        self._inner = inner
        self.name = inner.name
        self.geocode_calls = 0
        self.forecast_calls = 0

    async def geocode(self, query: str) -> list[Location]:
        self.geocode_calls += 1
        return await self._inner.geocode(query)

    async def forecast(self, loc: Location, when: str) -> Forecast:
        self.forecast_calls += 1
        return await self._inner.forecast(loc, when)


def _tool(facts: list[Fact] | None = None, fail: bool = False) -> GetWeatherTool:
    clock = FakeClock(NOW)
    uc = GetWeather(MockProvider(clock, temp_c=18.0, fail=fail), None, MemCache(), clock)
    f = FactsWith(facts or [])
    return GetWeatherTool(uc, f, UnitsResolver())


def test_declaration_matches_prd() -> None:
    d = _tool().declaration
    assert d["name"] == "get_weather"
    assert d["parameters"]["required"] == ["location"]
    assert set(d["parameters"]["properties"]) == {"location", "when", "units"}


async def test_success_has_grounding_numbers() -> None:
    r = await _tool().invoke(UID, {"location": "Tokyo"})
    assert r.ok and r.data["temp_c"] == 18.0 and r.data["location"]["name"] == "Tokyo"
    assert 18.0 in r.grounding_numbers and 64.4 in r.grounding_numbers


async def test_tomorrow_reports_high_and_low_and_omits_unobserved_fields() -> None:
    r = await _tool().invoke(UID, {"location": "Tokyo", "when": "tomorrow"})
    assert r.ok
    assert "temp_c" not in r.data and "humidity" not in r.data and "wind_kph" not in r.data
    assert r.spoken_summary == (
        "Tomorrow in Tokyo expect a high of twenty-two and a low of twelve "
        "and it will be partly cloudy."
    )
    assert 0.0 not in r.grounding_numbers  # no fabricated wind/humidity to speak


async def test_iso_day_is_spoken_by_weekday_name() -> None:
    r = await _tool().invoke(UID, {"location": "Tokyo", "when": "2026-08-24"})
    assert r.ok and (r.spoken_summary or "").startswith("On Monday in Tokyo expect")


async def test_not_found_spoken_error() -> None:
    r = await _tool().invoke(UID, {"location": "Gondor"})
    assert not r.ok and r.data["error"] == "not_found" and "Gondor" in (r.spoken_error or "")


async def test_here_without_home_city() -> None:
    r = await _tool().invoke(UID, {"location": "here"})
    assert not r.ok and r.data["error"] == "not_found" and r.data["reason"] == "no_home_city"


async def test_here_with_home_city_fact() -> None:
    r = await _tool([Fact("home_city", "Lisbon", "place")]).invoke(UID, {"location": "here"})
    assert r.ok and r.data["location"]["name"] == "Lisbon"


async def test_ambiguous() -> None:
    r = await _tool().invoke(UID, {"location": "Springfield"})
    assert not r.ok and r.data["error"] == "ambiguous" and len(r.data["candidates"]) == 2
    assert "Did you mean" in (r.spoken_error or "")


async def test_ambiguous_three_candidates_uses_oxford_comma() -> None:
    # Comparably sized, or the ambiguity rule would pick the leader outright.
    a = Location("Springfield", "United States", 39.80, -89.64, "Illinois", 116565)
    b = Location("Springfield", "Canada", 45.60, -75.50, "Ontario", 98000)
    c = Location("Springfield", "United Kingdom", 52.73, -0.63, "England", 90000)
    clock = FakeClock(NOW)
    uc = GetWeather(FakeProvider({"springfield": [a, b, c]}), None, MemCache(), clock)
    tool = GetWeatherTool(uc, FactsWith([]), UnitsResolver())

    r = await tool.invoke(UID, {"location": "Springfield"})

    assert not r.ok and r.data["error"] == "ambiguous"
    labels = r.data["candidates"]
    assert len(labels) == 3
    assert r.spoken_error == f"Did you mean {labels[0]}, {labels[1]}, or {labels[2]}?"


async def test_day_beyond_the_window_is_spoken_as_a_seven_day_limit() -> None:
    r = await _tool().invoke(UID, {"location": "Tokyo", "when": "2026-08-29"})
    assert not r.ok and r.data["reason"] == "bad_date"
    assert r.spoken_error == "I can only look up to seven days ahead."


async def test_unavailable() -> None:
    r = await _tool(fail=True).invoke(UID, {"location": "Tokyo"})
    assert not r.ok and r.data["error"] == "unavailable" and "can't reach" in (r.spoken_error or "")


def test_units_resolution() -> None:
    ur = UnitsResolver()
    facts_imperial = [Fact("units", "imperial", "preference")]
    assert ur.resolve(facts_imperial, None, None) == Units.IMPERIAL
    assert ur.resolve(facts_imperial, "metric", None) == Units.METRIC
    assert ur.resolve([], None, "United States") == Units.IMPERIAL
    assert ur.resolve([], None, "Japan") == Units.METRIC


async def test_us_inferred_units_is_a_single_provider_round_trip() -> None:
    clock = FakeClock(NOW)
    provider = CountingProvider(MockProvider(clock, temp_c=18.0))
    cache = MemCache()
    uc = GetWeather(provider, None, cache, clock)
    facts = FactsWith([])
    tool = GetWeatherTool(uc, facts, UnitsResolver())

    r = await tool.invoke(UID, {"location": "New York"})

    assert r.ok and r.data["units"] == "imperial"
    assert provider.geocode_calls == 1
    assert provider.forecast_calls == 1
    # One fetch, two keys: the request token and the resolved local date.
    assert len(cache.d) == 2
    assert facts.calls == 1


async def test_explicit_metric_for_us_city_is_a_single_provider_round_trip() -> None:
    clock = FakeClock(NOW)
    provider = CountingProvider(MockProvider(clock, temp_c=18.0))
    cache = MemCache()
    uc = GetWeather(provider, None, cache, clock)
    facts = FactsWith([])
    tool = GetWeatherTool(uc, facts, UnitsResolver())

    r = await tool.invoke(UID, {"location": "New York", "units": "metric"})

    assert r.ok and r.data["units"] == "metric"
    assert provider.geocode_calls == 1
    assert provider.forecast_calls == 1
    # One fetch, two keys: the request token and the resolved local date.
    assert len(cache.d) == 2
    assert facts.calls == 1


# ---------------------------------------------------------------------------
# I5: the turn's snapshot, handed down rather than re-read.
# ---------------------------------------------------------------------------


async def test_handed_facts_are_used_instead_of_a_second_read() -> None:
    # The turn loads facts in its one RPC and holds them for the whole turn; the
    # tool then went back to the database for the same rows, making a weather
    # turn cost two reads where it owed one.
    tool = _tool([Fact("home_city", "Berlin", "profile")])
    port = tool._facts  # type: ignore[attr-defined]
    res = await tool.invoke(UID, {"location": "here"}, [Fact("home_city", "Lisbon", "profile")])
    assert res.ok
    assert port.calls == 0  # the snapshot port was never touched
    # ...and the handed-down facts are the ones that resolved "here".
    assert res.data["location"]["name"] == "Lisbon"


async def test_units_come_from_the_handed_facts_too() -> None:
    # `UnitsResolver` reads the same snapshot, so handing facts down has to feed
    # both users of them or the tool answers in the wrong scale.
    tool = _tool([])
    res = await tool.invoke(UID, {"location": "Berlin"}, [Fact("units", "imperial", "preference")])
    assert res.ok
    assert res.data["units"] == "imperial"


async def test_no_facts_handed_down_still_reads_the_snapshot() -> None:
    # The port stays for callers that have nothing to hand down — the eval
    # harness drives this tool directly, with no turn around it.
    tool = _tool([Fact("home_city", "Berlin", "profile")])
    port = tool._facts  # type: ignore[attr-defined]
    res = await tool.invoke(UID, {"location": "here"})
    assert res.ok
    assert port.calls == 1
    assert res.data["location"]["name"] == "Berlin"
