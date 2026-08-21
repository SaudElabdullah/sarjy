"""Live smoke test hitting the real Open-Meteo API (network required)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sarjy.contexts.weather.application.get_weather import GetWeather, WeatherSuccess
from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.contexts.weather.domain.units import Units
from sarjy.contexts.weather.infrastructure.in_memory_cache import InMemoryWeatherCache
from sarjy.contexts.weather.infrastructure.open_meteo import OpenMeteoProvider
from sarjy.shared.clock import SystemClock

pytestmark = pytest.mark.integration


async def test_live_tokyo_forecast_is_plausible() -> None:
    """Tokyo is +9h: for eight hours of every UTC day its calendar is a day ahead
    of the server's. "now" must still resolve, and must resolve to Tokyo's own
    today — which is exactly what asking the provider for the token rather than a
    server-computed date buys us (C1)."""
    provider = OpenMeteoProvider()
    try:
        locs = await provider.geocode("Tokyo")
        assert locs, "expected at least one geocode result for Tokyo"
        loc = locs[0]
        assert loc.name == "Tokyo"

        now_utc = datetime.now(UTC)
        fc = await provider.forecast(loc, "now")
        assert isinstance(fc, Forecast)
        assert fc.temp_c is not None and -40.0 <= fc.temp_c <= 60.0
        assert fc.source == "open-meteo"
        # The resolved day is Tokyo's, not the server's — they differ for part of
        # every day, so only assert it is within a day of the server's date.
        assert abs((fc.day - now_utc.date()).days) <= 1

        observed_utc = fc.observed_at.astimezone(UTC)
        assert abs((observed_utc - now_utc).total_seconds()) <= 20 * 60
    finally:
        await provider.aclose()


async def test_live_auckland_now_resolves_regardless_of_server_time() -> None:
    """Auckland is +12/+13h — the furthest-ahead major city, and the one a
    server-UTC "today" is most likely to be a day behind."""
    provider = OpenMeteoProvider()
    try:
        locs = await provider.geocode("Auckland")
        assert locs, "expected at least one geocode result for Auckland"
        now_utc = datetime.now(UTC)
        fc = await provider.forecast(locs[0], "now")
        assert fc.temp_c is not None and -40.0 <= fc.temp_c <= 60.0
        assert abs((fc.day - now_utc.date()).days) <= 1
        assert abs((fc.observed_at.astimezone(UTC) - now_utc).total_seconds()) <= 20 * 60
    finally:
        await provider.aclose()


async def test_live_tomorrow_is_the_day_after_the_locations_today() -> None:
    provider = OpenMeteoProvider()
    try:
        locs = await provider.geocode("Tokyo")
        today = await provider.forecast(locs[0], "now")
        tomorrow = await provider.forecast(locs[0], "tomorrow")
        assert (tomorrow.day - today.day).days == 1
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    ("city", "country"),
    [("London", "United Kingdom"), ("Tokyo", "Japan"), ("Paris", "France")],
)
async def test_live_major_cities_resolve_without_asking_which_one(city: str, country: str) -> None:
    """The old ambiguity rule fired on almost every major city, because almost
    every major city has a small namesake abroad. Against real geocoder output
    these must answer straight through."""
    provider = OpenMeteoProvider()
    clock = SystemClock()
    try:
        uc = GetWeather(provider, None, InMemoryWeatherCache(clock), clock)
        result = await uc.execute(city, "now", Units.METRIC, None)
        assert isinstance(result, WeatherSuccess), f"{city} did not resolve: {result}"
        assert result.location.name == city and result.location.country == country
        assert result.forecast.temp_c is not None
    finally:
        await provider.aclose()
