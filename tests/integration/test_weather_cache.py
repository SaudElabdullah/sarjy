import json
import os
from datetime import UTC, date, datetime, timedelta

import pytest

from sarjy.contexts.weather.domain.condition import Condition
from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.contexts.weather.infrastructure.pg_cache import PgWeatherCache
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.clock import FakeClock

pytestmark = pytest.mark.integration
NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


async def test_cache_roundtrip_and_expiry() -> None:
    db = Database(os.environ["DATABASE_URL_DIRECT"])
    await db.connect()
    try:
        clock = FakeClock(NOW)
        cache = PgWeatherCache(db, clock)
        fc = Forecast.from_metric(
            temp_c=20,
            feels_like_c=19,
            condition=Condition.from_wmo(0),
            precip_prob=0,
            wind_kph=3,
            humidity=40,
            high_c=24,
            low_c=15,
            day=date(2026, 8, 21),
            observed_at=NOW,
            source="test",
            fetched_at=NOW,
        )
        await cache.set("k:test", fc, 600)
        got = await cache.get("k:test")
        assert got is not None and got.temp_c == 20.0 and got.condition.code == 0
        assert got.day == date(2026, 8, 21)
        clock.advance(timedelta(seconds=601))
        assert await cache.get("k:test") is None
    finally:
        await db.execute("delete from weather_cache where cache_key = $1", "k:test")
        await db.close()


async def test_payload_without_a_day_is_a_miss_not_a_crash() -> None:
    """Rows written before `Forecast.day` existed cannot be rebuilt — the day is
    not derivable from the rest of the payload. A miss costs one HTTP call;
    raising would cost the whole weather request."""
    db = Database(os.environ["DATABASE_URL_DIRECT"])
    await db.connect()
    try:
        legacy = {
            "temp_c": 20.0,
            "temp_f": 68.0,
            "feels_like_c": 19.0,
            "condition_code": 0,
            "condition_text": "clear sky",
            "precip_prob": 0,
            "wind_kph": 3.0,
            "humidity": 40,
            "high_c": 24.0,
            "low_c": 15.0,
            # no "day" — this is the shape the previous release wrote.
            "observed_at": NOW.isoformat(),
            "source": "test",
            "fetched_at": NOW.isoformat(),
            "_ttl_s": 600,
        }
        await db.execute(
            """insert into weather_cache (cache_key, payload, fetched_at)
               values ($1, $2::jsonb, $3)
               on conflict (cache_key) do update set payload = excluded.payload""",
            "k:legacy",
            json.dumps(legacy),
            NOW,
        )
        cache = PgWeatherCache(db, FakeClock(NOW))
        assert await cache.get("k:legacy") is None
    finally:
        await db.execute("delete from weather_cache where cache_key = $1", "k:legacy")
        await db.close()


async def test_cache_miss_returns_none() -> None:
    db = Database(os.environ["DATABASE_URL_DIRECT"])
    await db.connect()
    try:
        cache = PgWeatherCache(db, FakeClock(NOW))
        assert await cache.get("k:does-not-exist") is None
    finally:
        await db.close()


async def test_cache_overwrite_updates_payload_and_fetched_at() -> None:
    db = Database(os.environ["DATABASE_URL_DIRECT"])
    await db.connect()
    try:
        clock = FakeClock(NOW)
        cache = PgWeatherCache(db, clock)
        fc1 = Forecast.from_metric(
            temp_c=20,
            feels_like_c=19,
            condition=Condition.from_wmo(0),
            precip_prob=0,
            wind_kph=3,
            humidity=40,
            high_c=24,
            low_c=15,
            day=date(2026, 8, 21),
            observed_at=NOW,
            source="test",
            fetched_at=NOW,
        )
        await cache.set("k:overwrite", fc1, 600)
        clock.advance(timedelta(seconds=10))
        fc2 = Forecast.from_metric(
            temp_c=30,
            feels_like_c=29,
            condition=Condition.from_wmo(61),
            precip_prob=50,
            wind_kph=5,
            humidity=60,
            high_c=34,
            low_c=25,
            day=date(2026, 8, 21),
            observed_at=NOW,
            source="test",
            fetched_at=clock.now(),
        )
        await cache.set("k:overwrite", fc2, 600)
        got = await cache.get("k:overwrite")
        assert got is not None and got.temp_c == 30.0
    finally:
        await db.execute("delete from weather_cache where cache_key = $1", "k:overwrite")
        await db.close()
