from __future__ import annotations

import asyncio
import time
from datetime import UTC, date, datetime, timedelta

from sarjy.contexts.weather.application.get_weather import (
    GetWeather,
    WeatherAmbiguous,
    WeatherNotFound,
    WeatherSuccess,
    WeatherUnavailable,
    cache_key,
)
from sarjy.contexts.weather.application.ports import (
    DayOutOfRange,
    LocationNotFound,
    ProviderUnavailable,
)
from sarjy.contexts.weather.domain.condition import Condition
from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.contexts.weather.domain.location import Location
from sarjy.contexts.weather.domain.units import Units
from sarjy.contexts.weather.domain.when import MAX_DAYS_AHEAD, index_for_when
from sarjy.shared.clock import FakeClock

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
SERVER_TODAY = date(2026, 8, 21)
TOKYO = Location("Tokyo", "Japan", 35.6895, 139.6917)
SPRINGFIELD_US = Location("Springfield", "United States", 39.80, -89.64, "Illinois", 116565)
SPRINGFIELD_CA = Location("Springfield", "Canada", 45.60, -75.50, "Ontario", 98000)

# Realistic geocoder output for the cities the old rule tripped over.
LONDON_UK = Location("London", "United Kingdom", 51.51, -0.13, "England", 8961989)
LONDON_CA = Location("London", "Canada", 42.98, -81.24, "Ontario", 422324)
TOKYO_JP = Location("Tokyo", "Japan", 35.6895, 139.6917, "Tokyo", 9733276)
TOKYO_UNSIZED = Location("Tokyo", "United States", 35.18, -94.05, "Oklahoma", None)
TIE_A = Location("Twinsville", "Freedonia", 1.0, 1.0, None, 500000)
TIE_B = Location("Twinsville", "Sylvania", 2.0, 2.0, None, 500000)
SPRINGFIELD_MO = Location("Springfield", "United States", 37.21, -93.29, "Missouri", 169176)


def _fc(temp: float = 22.0, day: date = SERVER_TODAY, source: str = "fake") -> Forecast:
    return Forecast.from_metric(
        temp_c=temp,
        feels_like_c=temp,
        condition=Condition.from_wmo(1),
        precip_prob=10,
        wind_kph=5,
        humidity=50,
        high_c=temp + 3,
        low_c=temp - 5,
        day=day,
        observed_at=NOW,
        source=source,
        fetched_at=NOW,
    )


class FakeProvider:
    """Stands in for a real provider, including its own local calendar.

    `local_today` is the location's today as the provider sees it — set it away
    from the server's date to exercise a location whose calendar has already
    rolled over (or has not yet).
    """

    def __init__(
        self,
        geocodes: dict[str, list[Location]],
        fail: bool = False,
        slow: float = 0.0,
        name: str = "fake",
        local_today: date = SERVER_TODAY,
        fail_forecast: bool = False,
    ) -> None:
        self.geocodes, self.fail, self.slow, self.name = geocodes, fail, slow, name
        self.local_today, self.fail_forecast = local_today, fail_forecast
        self.geocode_calls = 0
        # attempts counts every call that arrives; forecast_calls counts the
        # ones that got as far as returning data.
        self.forecast_attempts = 0
        self.forecast_calls = 0

    async def geocode(self, query: str) -> list[Location]:
        if self.fail:
            raise ProviderUnavailable("down")
        await asyncio.sleep(self.slow)
        self.geocode_calls += 1
        return self.geocodes.get(query.lower(), [])

    async def forecast(self, loc: Location, when: str) -> Forecast:
        self.forecast_attempts += 1
        if self.fail or self.fail_forecast:
            raise ProviderUnavailable("down")
        await asyncio.sleep(self.slow)
        self.forecast_calls += 1
        days = [self.local_today + timedelta(days=i) for i in range(MAX_DAYS_AHEAD + 1)]
        idx = index_for_when(when, days)
        if idx is None:
            raise DayOutOfRange(when)
        return _fc(day=days[idx], source=self.name)


class MemCache:
    def __init__(self) -> None:
        self.d: dict[str, Forecast] = {}
        self.ttls: list[int] = []

    async def get(self, key: str) -> Forecast | None:
        return self.d.get(key)

    async def set(self, key: str, forecast: Forecast, ttl_s: int) -> None:
        self.d[key] = forecast
        self.ttls.append(ttl_s)


def _uc(
    provider: FakeProvider,
    fallback: FakeProvider | None = None,
    cache: MemCache | None = None,
    timeout: float = 2.5,
) -> GetWeather:
    return GetWeather(provider, fallback, cache or MemCache(), FakeClock(NOW), timeout_s=timeout)


async def test_success_path_and_cache_hit() -> None:
    p = FakeProvider({"tokyo": [TOKYO]})
    cache = MemCache()
    uc = _uc(p, cache=cache)
    r1 = await uc.execute("Tokyo", "now", Units.METRIC, None)
    assert isinstance(r1, WeatherSuccess) and r1.location == TOKYO and r1.day == date(2026, 8, 21)
    r2 = await uc.execute("Tokyo", "now", Units.METRIC, None)
    assert isinstance(r2, WeatherSuccess) and p.forecast_calls == 1  # second call served from cache
    assert cache_key(TOKYO, date(2026, 8, 21)) == "35.69:139.69:2026-08-21"


async def test_not_found() -> None:
    r = await _uc(FakeProvider({})).execute("Gondor", "now", Units.METRIC, None)
    assert isinstance(r, WeatherNotFound) and r.reason == "not_found" and r.query == "Gondor"


async def test_here_uses_home_city_or_fails() -> None:
    p = FakeProvider({"tokyo": [TOKYO]})
    assert isinstance(await _uc(p).execute("here", "now", Units.METRIC, "Tokyo"), WeatherSuccess)
    r = await _uc(p).execute("here", "now", Units.METRIC, None)
    assert isinstance(r, WeatherNotFound) and r.reason == "no_home_city"


async def test_bad_date() -> None:
    r = await _uc(FakeProvider({"tokyo": [TOKYO]})).execute(
        "Tokyo", "2027-01-01", Units.METRIC, None
    )
    assert isinstance(r, WeatherNotFound) and r.reason == "bad_date"


async def test_ambiguous_when_same_name_different_countries() -> None:
    p = FakeProvider({"springfield": [SPRINGFIELD_US, SPRINGFIELD_CA]})
    r = await _uc(p).execute("Springfield", "now", Units.METRIC, None)
    assert isinstance(r, WeatherAmbiguous) and len(r.candidates) == 2


async def test_a_much_smaller_namesake_is_not_an_ambiguity() -> None:
    # London, Ontario is a real city with a real namesake problem — but nobody
    # asking for "London" unqualified means the one that is 5% of the size.
    p = FakeProvider({"london": [LONDON_UK, LONDON_CA]})
    r = await _uc(p).execute("London", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and r.location == LONDON_UK


async def test_an_unsized_namesake_is_not_an_ambiguity() -> None:
    p = FakeProvider({"tokyo": [TOKYO_JP, TOKYO_UNSIZED]})
    r = await _uc(p).execute("Tokyo", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and r.location == TOKYO_JP


async def test_an_unsized_leader_stays_ambiguous() -> None:
    # Nothing to weigh the runner-up against, so asking is the safe move.
    a = Location("Springfield", "United States", 39.80, -89.64, "Illinois", None)
    p = FakeProvider({"springfield": [a, SPRINGFIELD_CA]})
    r = await _uc(p).execute("Springfield", "now", Units.METRIC, None)
    assert isinstance(r, WeatherAmbiguous)


async def test_comparable_namesakes_are_ambiguous() -> None:
    p = FakeProvider({"twinsville": [TIE_A, TIE_B]})
    r = await _uc(p).execute("Twinsville", "now", Units.METRIC, None)
    assert isinstance(r, WeatherAmbiguous) and len(r.candidates) == 2


async def test_namesakes_within_one_country_are_not_ambiguous() -> None:
    # Every Springfield in the US is still "Springfield" to the caller; picking
    # the top hit is the only sensible answer, and asking would not narrow it.
    p = FakeProvider({"springfield": [SPRINGFIELD_US, SPRINGFIELD_MO]})
    r = await _uc(p).execute("Springfield", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and r.location == SPRINGFIELD_US


async def test_explicit_country_disambiguates() -> None:
    p = FakeProvider(
        {"springfield, canada": [SPRINGFIELD_CA], "springfield": [SPRINGFIELD_US, SPRINGFIELD_CA]}
    )
    r = await _uc(p).execute("Springfield, Canada", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and r.location.country == "Canada"


async def test_a_geocoder_that_sizes_nothing_is_never_ambiguous() -> None:
    # OWM returns no populations at all. Under the previous rule every city on
    # the fallback path came back as "did you mean?".
    owm_shaped = [
        Location("London", "GB", 51.51, -0.13, "England", None),
        Location("London", "CA", 42.98, -81.24, "Ontario", None),
        Location("London", "US", 39.89, -83.45, "Ohio", None),
    ]
    p = FakeProvider({"london": owm_shaped})
    r = await _uc(p).execute("London", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and r.location == owm_shaped[0]


async def test_qualifier_selects_the_candidate_rather_than_just_muting_the_check() -> None:
    # The geocoder ignored the qualifier and put Illinois first; the comma used
    # only to switch the ambiguity check off, so the wrong city was answered.
    p = FakeProvider({"springfield, canada": [SPRINGFIELD_US, SPRINGFIELD_CA]})
    r = await _uc(p).execute("Springfield, Canada", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and r.location == SPRINGFIELD_CA


async def test_qualifier_matches_admin1_too() -> None:
    p = FakeProvider({"springfield, ontario": [SPRINGFIELD_US, SPRINGFIELD_CA]})
    r = await _uc(p).execute("Springfield, Ontario", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and r.location == SPRINGFIELD_CA


async def test_qualifier_that_matches_nothing_falls_back_to_all_candidates() -> None:
    p = FakeProvider({"springfield, atlantis": [SPRINGFIELD_US, SPRINGFIELD_CA]})
    r = await _uc(p).execute("Springfield, Atlantis", "now", Units.METRIC, None)
    assert isinstance(r, WeatherAmbiguous) and len(r.candidates) == 2


async def test_location_not_found_from_provider_is_not_found_not_unavailable() -> None:
    class NoSuchPlace:
        name = "strict"

        async def geocode(self, query: str) -> list[Location]:
            raise LocationNotFound(query)

        async def forecast(self, loc: Location, when: str) -> Forecast:
            raise AssertionError("must not be reached")

    fallback = FakeProvider({"gondor": [TOKYO]}, name="fb")
    uc = GetWeather(NoSuchPlace(), fallback, MemCache(), FakeClock(NOW))
    r = await uc.execute("Gondor", "now", Units.METRIC, None)
    assert isinstance(r, WeatherNotFound) and r.reason == "not_found"
    # A place that does not exist will not start existing on the second provider.
    assert fallback.geocode_calls == 0


async def test_fallback_on_unavailable() -> None:
    primary = FakeProvider({}, fail=True, name="primary")
    fallback = FakeProvider({"tokyo": [TOKYO]}, name="fb")
    r = await _uc(primary, fallback).execute("Tokyo", "now", Units.METRIC, None)
    # `source` names the provider that actually answered — asserting a constant
    # here (as this test used to) could not tell the two apart.
    assert isinstance(r, WeatherSuccess) and r.forecast.source == "fb"
    # The primary failed at geocode, so the sticky resume never asked it again.
    assert primary.forecast_attempts == 0 and fallback.forecast_calls == 1


async def test_primary_geocodes_then_fallback_serves_the_forecast() -> None:
    # The half-failure: the primary answers the geocode and then falls over on
    # the forecast. Each provider should be asked exactly once, for the call it
    # can actually serve.
    primary = FakeProvider({"tokyo": [TOKYO]}, fail_forecast=True, name="primary")
    fallback = FakeProvider({"tokyo": [TOKYO]}, name="fb")
    r = await _uc(primary, fallback).execute("Tokyo", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and r.forecast.source == "fb"
    assert primary.geocode_calls == 1 and fallback.geocode_calls == 0
    assert primary.forecast_attempts == 1 and primary.forecast_calls == 0
    assert fallback.forecast_attempts == 1 and fallback.forecast_calls == 1


async def test_unavailable_when_all_fail() -> None:
    r = await _uc(FakeProvider({}, fail=True), FakeProvider({}, fail=True)).execute(
        "Tokyo", "now", Units.METRIC, None
    )
    assert isinstance(r, WeatherUnavailable)


async def test_timeout_counts_as_unavailable() -> None:
    r = await _uc(FakeProvider({"tokyo": [TOKYO]}, slow=0.3), timeout=0.05).execute(
        "Tokyo", "now", Units.METRIC, None
    )
    assert isinstance(r, WeatherUnavailable) and r.reason == "timeout"


async def test_sticky_fallback_within_request() -> None:
    # Primary times out on every call; fallback is fast. Once geocode has
    # already fallen over to the fallback, the forecast call must not pay
    # another timeout retrying the primary — it should resume from the
    # fallback directly.
    primary = FakeProvider({"tokyo": [TOKYO]}, slow=0.3, name="primary")
    fallback = FakeProvider({"tokyo": [TOKYO]}, name="fallback")
    uc = _uc(primary, fallback, timeout=0.05)
    start = time.monotonic()
    r = await uc.execute("Tokyo", "now", Units.METRIC, None)
    elapsed = time.monotonic() - start
    assert isinstance(r, WeatherSuccess)
    assert elapsed < 0.09  # ~1x timeout, not 2x
    assert fallback.forecast_calls == 1
    assert primary.forecast_calls == 0


class BrokenCache:
    """A cache that is down — `get` and `set` both blow up, as asyncpg would."""

    def __init__(self) -> None:
        self.get_calls = 0
        self.set_calls = 0

    async def get(self, key: str) -> Forecast | None:
        self.get_calls += 1
        raise RuntimeError("cache down")

    async def set(self, key: str, forecast: Forecast, ttl_s: int) -> None:
        self.set_calls += 1
        raise RuntimeError("cache down")


async def test_cache_failure_does_not_take_weather_down() -> None:
    cache = BrokenCache()
    p = FakeProvider({"tokyo": [TOKYO]})
    r = await _uc(p, cache=cache).execute("Tokyo", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and r.forecast.temp_c == 22.0
    assert cache.get_calls == 1 and cache.set_calls  # both were attempted


async def test_cache_write_failure_does_not_discard_an_answer_already_fetched() -> None:
    class WriteOnlyBroken(MemCache):
        async def set(self, key: str, forecast: Forecast, ttl_s: int) -> None:
            raise RuntimeError("disk full")

    p = FakeProvider({"tokyo": [TOKYO]})
    r = await _uc(p, cache=WriteOnlyBroken()).execute("Tokyo", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess)


async def test_cache_set_receives_ttl() -> None:
    cache = MemCache()
    p = FakeProvider({"tokyo": [TOKYO]})
    r = await _uc(p, cache=cache).execute("Tokyo", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess)
    assert cache.ttls == [600, 600]


async def test_relative_request_is_also_stored_under_the_resolved_local_date() -> None:
    # The lookup key can only be the token (the local date is unknown until the
    # provider answers), so the resolved date gets a second entry — that is what
    # lets an explicit ISO date for the same local day reuse the fetch.
    cache = MemCache()
    p = FakeProvider({"tokyo": [TOKYO]}, local_today=date(2026, 8, 22))
    uc = _uc(p, cache=cache)
    await uc.execute("Tokyo", "now", Units.METRIC, None)
    assert set(cache.d) == {"35.69:139.69:today:2026-08-21", "35.69:139.69:2026-08-22"}

    r = await uc.execute("Tokyo", "2026-08-22", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and p.forecast_calls == 1


async def test_relative_cache_slug_rolls_over_with_the_server_date() -> None:
    # Without the server date stamped on it, "tomorrow" would keep pointing at
    # yesterday's tomorrow for as long as the entry lived.
    cache = MemCache()
    p1 = FakeProvider({"tokyo": [TOKYO]}, local_today=SERVER_TODAY)
    await GetWeather(p1, None, cache, FakeClock(NOW)).execute(
        "Tokyo", "tomorrow", Units.METRIC, None
    )
    keys_day_one = set(cache.d)

    cache.d.clear()
    p2 = FakeProvider({"tokyo": [TOKYO]}, local_today=SERVER_TODAY + timedelta(days=1))
    await GetWeather(p2, None, cache, FakeClock(NOW + timedelta(days=1))).execute(
        "Tokyo", "tomorrow", Units.METRIC, None
    )

    assert "35.69:139.69:tomorrow:2026-08-22" in keys_day_one
    assert "35.69:139.69:tomorrow:2026-08-23" in set(cache.d)
    # Nothing carries over, so yesterday's "tomorrow" cannot be served today.
    assert keys_day_one.isdisjoint(set(cache.d))


async def test_now_resolves_in_the_locations_calendar_when_it_is_a_day_ahead() -> None:
    # Server is on 2026-08-21 UTC; Tokyo has already rolled over to the 22nd.
    p = FakeProvider({"tokyo": [TOKYO]}, local_today=date(2026, 8, 22))
    r = await _uc(p).execute("Tokyo", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess)
    assert r.day == date(2026, 8, 22) and r.forecast.day == date(2026, 8, 22)


async def test_now_resolves_in_the_locations_calendar_when_it_is_a_day_behind() -> None:
    # Server is on 2026-08-21 UTC; Los Angeles is still on the 20th.
    p = FakeProvider({"tokyo": [TOKYO]}, local_today=date(2026, 8, 20))
    r = await _uc(p).execute("Tokyo", "now", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and r.day == date(2026, 8, 20)


async def test_iso_date_a_day_either_side_of_the_server_is_accepted() -> None:
    # Yesterday-by-server is today somewhere; +8 days by server is +7 somewhere.
    p = FakeProvider({"tokyo": [TOKYO]}, local_today=date(2026, 8, 20))
    r = await _uc(p).execute("Tokyo", "2026-08-20", Units.METRIC, None)
    assert isinstance(r, WeatherSuccess) and r.day == date(2026, 8, 20)

    p2 = FakeProvider({"tokyo": [TOKYO]}, local_today=date(2026, 8, 22))
    r2 = await _uc(p2).execute("Tokyo", "2026-08-29", Units.METRIC, None)
    assert isinstance(r2, WeatherSuccess) and r2.day == date(2026, 8, 29)


async def test_day_the_location_cannot_cover_is_a_bad_date_not_an_outage() -> None:
    p = FakeProvider({"tokyo": [TOKYO]}, local_today=date(2026, 8, 22))
    r = await _uc(p).execute("Tokyo", "2026-08-21", Units.METRIC, None)
    assert isinstance(r, WeatherNotFound) and r.reason == "bad_date"


async def test_day_beyond_the_forecast_window_is_bad_date_and_spares_the_fallback() -> None:
    # Server today + 8: inside the validation slack, outside anybody's forecast.
    # A second provider would take another round trip to say the same thing.
    primary = FakeProvider({"tokyo": [TOKYO]}, name="primary")
    fallback = FakeProvider({"tokyo": [TOKYO]}, name="fb")
    r = await _uc(primary, fallback).execute("Tokyo", "2026-08-29", Units.METRIC, None)
    assert isinstance(r, WeatherNotFound) and r.reason == "bad_date"
    assert primary.forecast_attempts == 1
    assert fallback.geocode_calls == 0 and fallback.forecast_attempts == 0
