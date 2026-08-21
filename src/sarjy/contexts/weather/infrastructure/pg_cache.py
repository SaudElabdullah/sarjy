from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from sarjy.contexts.weather.domain.condition import Condition
from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.infrastructure_shared.db import Database
from sarjy.observability.logging import get_logger
from sarjy.shared.clock import Clock

log = get_logger(__name__)


class PgWeatherCache:
    """Postgres-backed WeatherCache. Serialises Forecast to a small JSON payload
    (never the tool-facing dict) and treats an entry as expired once its own
    per-set TTL has elapsed, judged against the injected clock."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self.db, self.clock = db, clock

    async def get(self, key: str) -> Forecast | None:
        row = await self.db.fetchrow(
            "select payload, fetched_at from weather_cache where cache_key=$1", key
        )
        if not row:
            return None
        p: dict[str, Any] = (
            json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        )
        if "day" not in p:
            # Written before Forecast carried a resolved day, which cannot be
            # reconstructed from the rest of the payload. A miss costs one HTTP
            # call; raising would cost the whole request.
            log.debug("weather_cache_legacy_payload", key=key)
            return None
        if (self.clock.now() - row["fetched_at"]).total_seconds() > int(p["_ttl_s"]):
            return None
        return Forecast(
            temp_c=p["temp_c"],
            temp_f=p["temp_f"],
            feels_like_c=p["feels_like_c"],
            condition=Condition(code=p["condition_code"], text=p["condition_text"]),
            precip_prob=p["precip_prob"],
            wind_kph=p["wind_kph"],
            humidity=p["humidity"],
            high_c=p["high_c"],
            low_c=p["low_c"],
            day=date.fromisoformat(p["day"]),
            observed_at=datetime.fromisoformat(p["observed_at"]),
            source=p["source"],
            fetched_at=datetime.fromisoformat(p["fetched_at"]),
        )

    async def set(self, key: str, forecast: Forecast, ttl_s: int) -> None:
        p = {
            "temp_c": forecast.temp_c,
            "temp_f": forecast.temp_f,
            "feels_like_c": forecast.feels_like_c,
            "condition_code": forecast.condition.code,
            "condition_text": forecast.condition.text,
            "precip_prob": forecast.precip_prob,
            "wind_kph": forecast.wind_kph,
            "humidity": forecast.humidity,
            "high_c": forecast.high_c,
            "low_c": forecast.low_c,
            "day": forecast.day.isoformat(),
            "observed_at": forecast.observed_at.isoformat(),
            "source": forecast.source,
            "fetched_at": forecast.fetched_at.isoformat(),
            "_ttl_s": ttl_s,
        }
        await self.db.execute(
            """insert into weather_cache (cache_key, payload, fetched_at) values ($1, $2::jsonb, $3)
               on conflict (cache_key) do update set
                 payload = excluded.payload, fetched_at = excluded.fetched_at""",
            key,
            json.dumps(p),
            self.clock.now(),
        )
