from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
import respx

from sarjy.contexts.weather.application.ports import DayOutOfRange, ProviderUnavailable
from sarjy.contexts.weather.domain.location import Location
from sarjy.contexts.weather.infrastructure.owm import OwmProvider
from sarjy.shared.clock import FakeClock

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
GEO = [{"name": "Reykjavik", "lat": 64.1466, "lon": -21.9426, "country": "IS"}]
ONECALL = {
    "timezone_offset": 0,  # Reykjavik: UTC year-round
    "current": {
        # 2026-08-21T12:00:00Z, i.e. `NOW` / the requested day.
        "dt": 1787313600,
        "temp": 11.2,
        "feels_like": 9.0,
        "humidity": 80,
        "wind_speed": 6.1,
        "weather": [{"id": 500, "main": "Rain", "description": "light rain"}],
    },
    "daily": [
        {
            "dt": 1787313600,
            "temp": {"min": 8.0, "max": 13.0},
            "pop": 0.6,
            "weather": [{"id": 500, "description": "light rain"}],
        },
        {
            # 2026-08-22T12:00:00Z
            "dt": 1787400000,
            "temp": {"min": 7.5, "max": 12.0},
            "pop": 0.2,
            "weather": [{"id": 801, "description": "few clouds"}],
        },
    ],
}


@respx.mock
async def test_owm_geocode_and_forecast() -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    respx.get("https://api.openweathermap.org/geo/1.0/direct").mock(
        return_value=httpx.Response(200, json=GEO)
    )
    respx.get("https://api.openweathermap.org/data/3.0/onecall").mock(
        return_value=httpx.Response(200, json=ONECALL)
    )
    locs = await p.geocode("Reykjavik")
    assert locs[0] == Location("Reykjavik", "IS", 64.1466, -21.9426)
    fc = await p.forecast(locs[0], "now")
    assert (
        fc.temp_c == 11.2
        and fc.wind_kph == 22.0
        and fc.precip_prob == 60
        and fc.condition.text == "light rain"
    )
    assert fc.source == "owm"


@respx.mock
async def test_owm_geocode_empty() -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    respx.get("https://api.openweathermap.org/geo/1.0/direct").mock(
        return_value=httpx.Response(200, json=[])
    )
    assert await p.geocode("Gondor") == []


@respx.mock
async def test_owm_5xx_raises_unavailable() -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    respx.get("https://api.openweathermap.org/geo/1.0/direct").mock(
        return_value=httpx.Response(503)
    )
    with pytest.raises(ProviderUnavailable):
        await p.geocode("Reykjavik")


@respx.mock
async def test_owm_network_error_raises_unavailable() -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    respx.get("https://api.openweathermap.org/geo/1.0/direct").mock(
        side_effect=httpx.ConnectError("boom")
    )
    with pytest.raises(ProviderUnavailable):
        await p.geocode("Reykjavik")


@respx.mock
async def test_owm_malformed_geocode_response_raises_unavailable() -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    respx.get("https://api.openweathermap.org/geo/1.0/direct").mock(
        return_value=httpx.Response(200, json=[{"name": "Reykjavik"}])  # missing lat/lon
    )
    with pytest.raises(ProviderUnavailable):
        await p.geocode("Reykjavik")


@respx.mock
async def test_owm_malformed_forecast_response_raises_unavailable() -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    respx.get("https://api.openweathermap.org/data/3.0/onecall").mock(
        return_value=httpx.Response(200, json={"current": {"weather": [{"id": 500}]}, "daily": []})
    )
    with pytest.raises(ProviderUnavailable):
        await p.forecast(Location("Reykjavik", "IS", 64.1466, -21.9426), "now")


@respx.mock
async def test_owm_day_out_of_range_raises_day_out_of_range() -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    respx.get("https://api.openweathermap.org/data/3.0/onecall").mock(
        return_value=httpx.Response(200, json=ONECALL)
    )
    with pytest.raises(DayOutOfRange):
        await p.forecast(Location("Reykjavik", "IS", 64.1466, -21.9426), "2026-08-29")


# Auckland-shaped: every `dt` is still 2026-08-21 in UTC, but +13h puts the
# location on the 22nd — "now" must land on the location's day, not the server's.
ONECALL_AHEAD = {
    "timezone_offset": 46800,
    "current": {
        "dt": 1787313600,  # 2026-08-21T12:00Z == 2026-08-22T01:00 local
        "temp": 11.2,
        "feels_like": 9.0,
        "humidity": 80,
        "wind_speed": 6.1,
        "weather": [{"id": 500, "main": "Rain", "description": "light rain"}],
    },
    "daily": [
        {
            "dt": 1787313600,
            "temp": {"min": 8.0, "max": 13.0},
            "pop": 0.6,
            "weather": [{"id": 500, "description": "light rain"}],
        },
        {
            "dt": 1787400000,
            "temp": {"min": 7.5, "max": 12.0},
            "pop": 0.2,
            "weather": [{"id": 801, "description": "few clouds"}],
        },
    ],
}


@respx.mock
async def test_owm_now_uses_the_locations_calendar_not_utc() -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    respx.get("https://api.openweathermap.org/data/3.0/onecall").mock(
        return_value=httpx.Response(200, json=ONECALL_AHEAD)
    )
    fc = await p.forecast(Location("Auckland", "NZ", -36.85, 174.76), "now")
    assert fc.day == date(2026, 8, 22)
    assert fc.temp_c == 11.2  # current conditions, i.e. index 0
    assert fc.observed_at.utcoffset() == timedelta(hours=13)


@respx.mock
async def test_owm_forecast_day_carries_no_invented_readings() -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    respx.get("https://api.openweathermap.org/data/3.0/onecall").mock(
        return_value=httpx.Response(200, json=ONECALL)
    )
    fc = await p.forecast(Location("Reykjavik", "IS", 64.1466, -21.9426), "tomorrow")
    assert fc.high_c == 12.0 and fc.low_c == 7.5 and fc.precip_prob == 20
    assert fc.temp_c is None and fc.feels_like_c is None
    assert fc.wind_kph is None and fc.humidity is None


@respx.mock
@pytest.mark.parametrize("missing", ["temp", "pop"])
async def test_owm_missing_daily_field_is_malformed_not_zero(missing: str) -> None:
    """A required field that is absent used to read as 0 — a fabricated number
    with nothing behind it. It must fail the request instead."""
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    payload = {
        "timezone_offset": 0,
        "current": ONECALL["current"],
        "daily": [{k: v for k, v in ONECALL["daily"][0].items() if k != missing}],
    }
    respx.get("https://api.openweathermap.org/data/3.0/onecall").mock(
        return_value=httpx.Response(200, json=payload)
    )
    with pytest.raises(ProviderUnavailable):
        await p.forecast(Location("Reykjavik", "IS", 64.1466, -21.9426), "now")


@respx.mock
@pytest.mark.parametrize("missing", ["temp", "feels_like", "humidity", "wind_speed", "weather"])
async def test_owm_missing_current_field_is_malformed_not_zero(missing: str) -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    payload = {
        "timezone_offset": 0,
        "current": {k: v for k, v in ONECALL["current"].items() if k != missing},
        "daily": ONECALL["daily"],
    }
    respx.get("https://api.openweathermap.org/data/3.0/onecall").mock(
        return_value=httpx.Response(200, json=payload)
    )
    with pytest.raises(ProviderUnavailable):
        await p.forecast(Location("Reykjavik", "IS", 64.1466, -21.9426), "now")


@respx.mock
async def test_owm_forecast_day_missing_condition_is_malformed() -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    payload = {
        "timezone_offset": 0,
        "current": ONECALL["current"],
        "daily": [
            ONECALL["daily"][0],
            {k: v for k, v in ONECALL["daily"][1].items() if k != "weather"},
        ],
    }
    respx.get("https://api.openweathermap.org/data/3.0/onecall").mock(
        return_value=httpx.Response(200, json=payload)
    )
    with pytest.raises(ProviderUnavailable):
        await p.forecast(Location("Reykjavik", "IS", 64.1466, -21.9426), "tomorrow")


@respx.mock
async def test_owm_non_json_body_on_a_200_is_unavailable_not_a_crash() -> None:
    p = OwmProvider(httpx.AsyncClient(), "KEY", FakeClock(NOW))
    respx.get("https://api.openweathermap.org/geo/1.0/direct").mock(
        return_value=httpx.Response(200, text="not json")
    )
    with pytest.raises(ProviderUnavailable):
        await p.geocode("Reykjavik")
