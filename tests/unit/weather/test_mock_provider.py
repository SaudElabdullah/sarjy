from __future__ import annotations

import os
import time
from datetime import UTC, date, datetime

import pytest

from sarjy.contexts.weather.application.ports import (
    DayOutOfRange,
    LocationNotFound,
    ProviderUnavailable,
)
from sarjy.contexts.weather.domain.location import Location
from sarjy.contexts.weather.infrastructure.mock_provider import MockProvider
from sarjy.shared.clock import FakeClock

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
TOKYO = Location("Tokyo", "Japan", 35.6895, 139.6917)


async def test_geocode_known_city() -> None:
    p = MockProvider(FakeClock(NOW))
    locs = await p.geocode("Tokyo")
    assert len(locs) == 1 and locs[0].name == "Tokyo" and locs[0].country == "Japan"


async def test_geocode_unknown_city_raises_location_not_found() -> None:
    # No catch-all fixture: an unknown place must not resolve to anywhere.
    p = MockProvider(FakeClock(NOW))
    with pytest.raises(LocationNotFound):
        await p.geocode("Gondor")
    with pytest.raises(LocationNotFound):
        await p.geocode("Testville")


async def test_known_cities_carry_populations() -> None:
    p = MockProvider(FakeClock(NOW))
    assert (await p.geocode("Tokyo"))[0].population == 8336599


async def test_geocode_springfield_ambiguous_across_countries() -> None:
    p = MockProvider(FakeClock(NOW))
    locs = await p.geocode("Springfield")
    assert len(locs) == 2
    assert {loc.country for loc in locs} == {"United States", "Canada"}
    # Comparably sized, which is what makes it a real ambiguity.
    assert all(loc.population is not None for loc in locs)


async def test_forecast_uses_configured_temp() -> None:
    p = MockProvider(FakeClock(NOW), temp_c=22.0)
    locs = await p.geocode("Tokyo")
    fc = await p.forecast(locs[0], "now")
    assert fc.temp_c == 22.0 and fc.source == "mock" and fc.day == date(2026, 8, 21)


async def test_forecast_resolves_relative_and_iso_days() -> None:
    p = MockProvider(FakeClock(NOW))
    assert (await p.forecast(TOKYO, "tomorrow")).day == date(2026, 8, 22)
    assert (await p.forecast(TOKYO, "2026-08-28")).day == date(2026, 8, 28)
    with pytest.raises(DayOutOfRange):
        await p.forecast(TOKYO, "2026-08-29")


async def test_mock_forecast_day_has_no_current_readings() -> None:
    p = MockProvider(FakeClock(NOW))
    fc = await p.forecast(TOKYO, "tomorrow")
    assert fc.temp_c is None and fc.feels_like_c is None
    assert fc.wind_kph is None and fc.humidity is None
    assert fc.high_c == 22.0 and fc.low_c == 12.0


async def test_fail_flag_raises_provider_unavailable() -> None:
    p = MockProvider(FakeClock(NOW), fail=True)
    with pytest.raises(ProviderUnavailable):
        await p.geocode("Tokyo")
    with pytest.raises(ProviderUnavailable):
        await p.forecast(TOKYO, "now")


async def test_mock_delay_ms_env_adds_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_DELAY_MS", "50")
    p = MockProvider(FakeClock(NOW))
    start = time.monotonic()
    await p.geocode("Tokyo")
    elapsed = time.monotonic() - start
    assert elapsed >= 0.045


async def test_no_delay_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOCK_DELAY_MS", raising=False)
    p = MockProvider(FakeClock(NOW))
    start = time.monotonic()
    await p.geocode("Tokyo")
    elapsed = time.monotonic() - start
    assert elapsed < 0.03


async def test_env_mock_delay_ms_not_set_ok() -> None:
    os.environ.pop("MOCK_DELAY_MS", None)
    p = MockProvider(FakeClock(NOW))
    assert await p.geocode("Paris")
