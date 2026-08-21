from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
import respx

from sarjy.contexts.weather.application.ports import DayOutOfRange, ProviderUnavailable
from sarjy.contexts.weather.domain.location import Location
from sarjy.contexts.weather.infrastructure.open_meteo import OpenMeteoProvider
from sarjy.shared.clock import FakeClock

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
GEO = {
    "results": [
        {
            "id": 1850147,
            "name": "Tokyo",
            "latitude": 35.6895,
            "longitude": 139.69171,
            "country": "Japan",
            "admin1": "Tokyo",
            "country_code": "JP",
        },
        {
            "id": 1,
            "name": "Tokyo",
            "latitude": 36.0,
            "longitude": 140.0,
            "country": "Japan",
            "country_code": "JP",
        },
    ]
}
FORECAST = {
    "latitude": 35.7,
    "longitude": 139.6875,
    "timezone": "Asia/Tokyo",
    # Mirrors a real response: `forecast_days=FORECAST_DAYS` (8) days of daily
    # data, and the offset the provider always sends back under timezone=auto.
    "utc_offset_seconds": 32400,
    "current": {
        "time": "2026-08-21T21:00",
        "temperature_2m": 27.3,
        "apparent_temperature": 30.1,
        "relative_humidity_2m": 74,
        "weather_code": 2,
        "wind_speed_10m": 9.4,
        "precipitation_probability": 20,
    },
    "daily": {
        "time": [
            "2026-08-21",
            "2026-08-22",
            "2026-08-23",
            "2026-08-24",
            "2026-08-25",
            "2026-08-26",
            "2026-08-27",
            "2026-08-28",
        ],
        "temperature_2m_max": [31.2, 30.0, 29.1, 28.0, 27.5, 29.9, 30.4, 26.0],
        "temperature_2m_min": [24.1, 23.8, 23.0, 22.5, 22.0, 23.1, 23.7, 20.0],
        "weather_code": [2, 61, 3, 0, 1, 80, 95, 3],
        "precipitation_probability_max": [20, 70, 35, 5, 10, 60, 80, 15],
    },
}


@pytest.fixture
def provider() -> OpenMeteoProvider:
    return OpenMeteoProvider(httpx.AsyncClient(), FakeClock(NOW))


@respx.mock
async def test_geocode_maps_top_results(provider: OpenMeteoProvider) -> None:
    respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
        return_value=httpx.Response(200, json=GEO)
    )
    locs = await provider.geocode("Tokyo")
    assert locs[0] == Location("Tokyo", "Japan", 35.6895, 139.69171, "Tokyo")
    assert len(locs) == 2


@respx.mock
async def test_geocode_empty(provider: OpenMeteoProvider) -> None:
    respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
        return_value=httpx.Response(200, json={"generationtime_ms": 0.5})
    )
    assert await provider.geocode("Gondor") == []


@respx.mock
async def test_forecast_today_uses_current(provider: OpenMeteoProvider) -> None:
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=FORECAST)
    )
    fc = await provider.forecast(Location("Tokyo", "Japan", 35.6895, 139.69171), "now")
    assert fc.temp_c == 27.3 and fc.condition.code == 2 and fc.high_c == 31.2 and fc.low_c == 24.1
    assert fc.source == "open-meteo" and fc.precip_prob == 20


@respx.mock
async def test_forecast_tomorrow_uses_daily(provider: OpenMeteoProvider) -> None:
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=FORECAST)
    )
    fc = await provider.forecast(Location("Tokyo", "Japan", 35.6895, 139.69171), "tomorrow")
    assert (
        fc.high_c == 30.0 and fc.low_c == 23.8 and fc.condition.code == 61 and fc.precip_prob == 70
    )
    # A future day carries no observation — nothing is synthesised to fill in.
    assert fc.temp_c is None and fc.temp_f is None
    assert fc.feels_like_c is None and fc.wind_kph is None and fc.humidity is None


@respx.mock
async def test_5xx_raises_unavailable(provider: OpenMeteoProvider) -> None:
    respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
        return_value=httpx.Response(503)
    )
    with pytest.raises(ProviderUnavailable):
        await provider.geocode("Tokyo")


@respx.mock
async def test_network_error_raises_unavailable(provider: OpenMeteoProvider) -> None:
    respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
        side_effect=httpx.ConnectError("boom")
    )
    with pytest.raises(ProviderUnavailable):
        await provider.geocode("Tokyo")


@respx.mock
async def test_4xx_on_forecast_raises_unavailable(provider: OpenMeteoProvider) -> None:
    respx.get("https://api.open-meteo.com/v1/forecast").mock(return_value=httpx.Response(400))
    with pytest.raises(ProviderUnavailable):
        await provider.forecast(Location("Tokyo", "Japan", 35.6895, 139.69171), "now")


FORECAST_TZ = {
    "utc_offset_seconds": 32400,  # +9h, e.g. Tokyo
    "current": {
        "time": "2026-08-22T07:30",
        "temperature_2m": 27.3,
        "apparent_temperature": 30.1,
        "relative_humidity_2m": 74,
        "weather_code": 2,
        "wind_speed_10m": 9.4,
        "precipitation_probability": 20,
    },
    "daily": {
        "time": ["2026-08-22", "2026-08-23"],
        "temperature_2m_max": [31.2, 30.0],
        "temperature_2m_min": [24.1, 23.8],
        "weather_code": [2, 61],
        "precipitation_probability_max": [20, 70],
    },
}


@respx.mock
async def test_observed_at_uses_location_offset_not_clock_tz(provider: OpenMeteoProvider) -> None:
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=FORECAST_TZ)
    )
    fc = await provider.forecast(Location("Tokyo", "Japan", 35.6895, 139.69171), "now")
    assert fc.observed_at.utcoffset() == timedelta(hours=9)
    assert fc.observed_at.astimezone(UTC) == datetime(2026, 8, 21, 22, 30, tzinfo=UTC)


FORECAST_ZERO_POP = {
    "current": {
        "time": "2026-08-21T21:00",
        "temperature_2m": 27.3,
        "apparent_temperature": 30.1,
        "relative_humidity_2m": 74,
        "weather_code": 2,
        "wind_speed_10m": 9.4,
        "precipitation_probability": 0,
    },
    "daily": {
        "time": ["2026-08-21", "2026-08-22"],
        "temperature_2m_max": [31.2, 30.0],
        "temperature_2m_min": [24.1, 23.8],
        "weather_code": [2, 61],
        "precipitation_probability_max": [55, 70],
    },
}


@respx.mock
async def test_current_precip_prob_zero_is_not_discarded(provider: OpenMeteoProvider) -> None:
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=FORECAST_ZERO_POP)
    )
    fc = await provider.forecast(Location("Tokyo", "Japan", 35.6895, 139.69171), "now")
    assert fc.precip_prob == 0


@respx.mock
async def test_day_seven_ahead_is_found_in_range(provider: OpenMeteoProvider) -> None:
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=FORECAST)
    )
    fc = await provider.forecast(Location("Tokyo", "Japan", 35.6895, 139.69171), "2026-08-28")
    assert fc.high_c == 26.0 and fc.low_c == 20.0


@respx.mock
async def test_day_out_of_range_raises_day_out_of_range(provider: OpenMeteoProvider) -> None:
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=FORECAST)
    )
    # A distinct exception, not ProviderUnavailable: nothing is wrong with the
    # provider, so there is nothing for a fallback to do about it.
    with pytest.raises(DayOutOfRange):
        await provider.forecast(Location("Tokyo", "Japan", 35.6895, 139.69171), "2026-08-29")


FORECAST_MALFORMED = {
    "current": FORECAST["current"],
    "daily": {
        "time": ["2026-08-21"],
        # "temperature_2m_max" missing -> KeyError while mapping.
        "temperature_2m_min": [24.1],
        "weather_code": [2],
        "precipitation_probability_max": [20],
    },
}


@respx.mock
async def test_malformed_forecast_response_raises_unavailable(
    provider: OpenMeteoProvider,
) -> None:
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=FORECAST_MALFORMED)
    )
    with pytest.raises(ProviderUnavailable):
        await provider.forecast(Location("Tokyo", "Japan", 35.6895, 139.69171), "now")


@respx.mock
async def test_malformed_geocode_response_raises_unavailable(
    provider: OpenMeteoProvider,
) -> None:
    respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(
        return_value=httpx.Response(200, json={"results": [{"name": "Tokyo"}]})
    )
    with pytest.raises(ProviderUnavailable):
        await provider.geocode("Tokyo")  # missing latitude/longitude -> KeyError


@respx.mock
async def test_non_json_body_on_a_200_is_unavailable_not_a_crash(
    provider: OpenMeteoProvider,
) -> None:
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, text="<html>gateway error</html>")
    )
    with pytest.raises(ProviderUnavailable):
        await provider.forecast(Location("Tokyo", "Japan", 35.6895, 139.69171), "now")


async def test_aclose_closes_internally_created_client() -> None:
    provider = OpenMeteoProvider()
    assert provider.client.is_closed is False
    await provider.aclose()
    assert provider.client.is_closed is True


async def test_aclose_leaves_injected_client_open() -> None:
    client = httpx.AsyncClient()
    provider = OpenMeteoProvider(client, FakeClock(NOW))
    await provider.aclose()
    assert client.is_closed is False
    await client.aclose()


# --- C1: the requested day is resolved in the *location's* calendar ----------
# `timezone=auto` makes daily.time the location's own dates, so daily.time[0] is
# its today. Both fixtures below are answers to the same server-side instant
# (2026-08-21 UTC) from locations on either side of the dateline-ish divide.

FORECAST_TOKYO_AHEAD = {
    "utc_offset_seconds": 32400,  # +9h: 16:00 UTC is already the 22nd in Tokyo
    "current": {
        "time": "2026-08-22T01:00",
        "temperature_2m": 27.3,
        "apparent_temperature": 30.1,
        "relative_humidity_2m": 74,
        "weather_code": 2,
        "wind_speed_10m": 9.4,
        "precipitation_probability": 20,
    },
    "daily": {
        "time": ["2026-08-22", "2026-08-23"],
        "temperature_2m_max": [31.2, 30.0],
        "temperature_2m_min": [24.1, 23.8],
        "weather_code": [2, 61],
        "precipitation_probability_max": [20, 70],
    },
}

FORECAST_LA_BEHIND = {
    "utc_offset_seconds": -25200,  # -7h: 06:00 UTC is still the 20th in LA
    "current": {
        "time": "2026-08-20T23:00",
        "temperature_2m": 19.0,
        "apparent_temperature": 18.0,
        "relative_humidity_2m": 60,
        "weather_code": 0,
        "wind_speed_10m": 8.0,
        "precipitation_probability": 5,
    },
    "daily": {
        "time": ["2026-08-20", "2026-08-21"],
        "temperature_2m_max": [24.0, 30.0],
        "temperature_2m_min": [15.0, 18.0],
        "weather_code": [0, 61],
        "precipitation_probability_max": [5, 70],
    },
}


@respx.mock
async def test_now_uses_index_zero_when_the_location_is_a_day_ahead() -> None:
    provider = OpenMeteoProvider(
        httpx.AsyncClient(), FakeClock(datetime(2026, 8, 21, 16, tzinfo=UTC))
    )
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=FORECAST_TOKYO_AHEAD)
    )
    fc = await provider.forecast(Location("Tokyo", "Japan", 35.6895, 139.69171), "now")
    # Server is still on the 21st; Tokyo is not, and Tokyo's calendar wins.
    assert fc.day == date(2026, 8, 22)
    assert fc.temp_c == 27.3 and fc.high_c == 31.2  # current conditions, index 0


@respx.mock
async def test_now_uses_index_zero_when_the_location_is_a_day_behind() -> None:
    provider = OpenMeteoProvider(
        httpx.AsyncClient(), FakeClock(datetime(2026, 8, 21, 6, tzinfo=UTC))
    )
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=FORECAST_LA_BEHIND)
    )
    fc = await provider.forecast(Location("Los Angeles", "United States", 34.05, -118.24), "now")
    # The server's date (the 21st) is index 1 here — taking it would report
    # tomorrow's weather as current.
    assert fc.day == date(2026, 8, 20)
    assert fc.temp_c == 19.0 and fc.high_c == 24.0


@respx.mock
async def test_tomorrow_is_index_one_in_the_locations_calendar() -> None:
    provider = OpenMeteoProvider(
        httpx.AsyncClient(), FakeClock(datetime(2026, 8, 21, 6, tzinfo=UTC))
    )
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=FORECAST_LA_BEHIND)
    )
    fc = await provider.forecast(
        Location("Los Angeles", "United States", 34.05, -118.24), "tomorrow"
    )
    assert fc.day == date(2026, 8, 21) and fc.high_c == 30.0
