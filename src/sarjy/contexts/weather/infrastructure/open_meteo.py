from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from sarjy.contexts.weather.application.ports import DayOutOfRange, ProviderUnavailable
from sarjy.contexts.weather.domain.condition import Condition
from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.contexts.weather.domain.location import Location
from sarjy.contexts.weather.domain.when import MAX_DAYS_AHEAD, index_for_when
from sarjy.shared.clock import Clock, SystemClock

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FC_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,"
    "wind_speed_10m,precipitation_probability"
)
DAILY = "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max"
# today..today+MAX_DAYS_AHEAD inclusive needs MAX_DAYS_AHEAD + 1 days of daily data.
FORECAST_DAYS = MAX_DAYS_AHEAD + 1


class OpenMeteoProvider:
    """Weather provider backed by the free Open-Meteo geocoding + forecast APIs."""

    name = "open-meteo"

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        timeout_s: float = 2.5,
    ) -> None:
        # client/clock are injectable for tests (respx-mocked client, FakeClock);
        # otherwise a real client and the system clock are created lazily.
        self._owns_client = client is None
        self.client = client if client is not None else httpx.AsyncClient()
        self.clock = clock if clock is not None else SystemClock()
        self.timeout_s = timeout_s

    async def aclose(self) -> None:
        """Close the client, but only if this provider created it itself."""
        if self._owns_client:
            await self.client.aclose()

    async def _get(self, url: str, params: dict[str, str | int | float]) -> dict[str, Any]:
        try:
            r = await self.client.get(url, params=params, timeout=self.timeout_s)
        except httpx.HTTPError as e:
            raise ProviderUnavailable(f"open-meteo network: {e.__class__.__name__}") from e
        if r.status_code >= 500 or r.status_code == 429:
            raise ProviderUnavailable(f"open-meteo http {r.status_code}")
        if r.status_code >= 400:
            # Never fabricate a result for a client error we don't understand.
            raise ProviderUnavailable(f"open-meteo bad request {r.status_code}")
        try:
            # A 200 whose body is not JSON (a proxy error page, a truncated
            # response) is a provider problem like any other, not a crash.
            result: dict[str, Any] = r.json()
        except ValueError as e:
            raise ProviderUnavailable("malformed_response") from e
        return result

    async def geocode(self, query: str) -> list[Location]:
        data = await self._get(
            GEO_URL, {"name": query, "count": 5, "language": "en", "format": "json"}
        )
        try:
            out: list[Location] = []
            for res in data.get("results", []) or []:
                pop = res.get("population")
                out.append(
                    Location(
                        name=str(res.get("name", "")),
                        country=str(res.get("country") or res.get("country_code") or ""),
                        lat=float(res["latitude"]),
                        lon=float(res["longitude"]),
                        admin1=res.get("admin1"),
                        population=int(pop) if pop is not None else None,
                        feature_code=res.get("feature_code"),
                    )
                )
            return out[:3]
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderUnavailable("malformed_response") from e

    async def forecast(self, loc: Location, when: str) -> Forecast:
        data = await self._get(
            FC_URL,
            {
                "latitude": loc.lat,
                "longitude": loc.lon,
                "current": CURRENT,
                "daily": DAILY,
                "timezone": "auto",
                "forecast_days": FORECAST_DAYS,
            },
        )
        try:
            return self._map_forecast(data, when)
        except (DayOutOfRange, ProviderUnavailable):
            raise
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderUnavailable("malformed_response") from e

    def _map_forecast(self, data: dict[str, Any], when: str) -> Forecast:
        daily = data.get("daily") or {}
        # `timezone=auto` makes daily.time the location's own calendar, so
        # daily.time[0] is *its* today — which may be a day either side of the
        # server's UTC date. Resolving the request token against this list is
        # what keeps "now" meaning now wherever the caller is asking about.
        days = [date.fromisoformat(t) for t in (daily.get("time") or [])]
        idx = index_for_when(when, days)
        if idx is None:
            raise DayOutOfRange(when)
        day = days[idx]
        high = float(daily["temperature_2m_max"][idx])
        low = float(daily["temperature_2m_min"][idx])
        d_code = int(daily["weather_code"][idx])
        d_pop = int(daily["precipitation_probability_max"][idx] or 0)
        now = self.clock.now()
        cur = data.get("current") or {}
        if idx == 0 and cur:
            offset_s = data.get("utc_offset_seconds")
            loc_tz = timezone(timedelta(seconds=offset_s)) if offset_s is not None else now.tzinfo
            observed = datetime.fromisoformat(cur["time"]).replace(tzinfo=loc_tz)
            v = cur.get("precipitation_probability")
            precip_prob = int(v) if v is not None else d_pop
            return Forecast.from_metric(
                temp_c=float(cur["temperature_2m"]),
                feels_like_c=float(cur["apparent_temperature"]),
                condition=Condition.from_wmo(int(cur["weather_code"])),
                precip_prob=precip_prob,
                wind_kph=float(cur["wind_speed_10m"]),
                humidity=int(cur["relative_humidity_2m"]),
                high_c=high,
                low_c=low,
                day=day,
                observed_at=observed,
                source=self.name,
                fetched_at=now,
            )
        # A future day has no observation: only high/low/condition and the
        # daily chance of precipitation are real (C2).
        return Forecast.from_metric(
            temp_c=None,
            feels_like_c=None,
            condition=Condition.from_wmo(d_code),
            precip_prob=d_pop,
            wind_kph=None,
            humidity=None,
            high_c=high,
            low_c=low,
            day=day,
            observed_at=datetime.combine(day, datetime.min.time(), tzinfo=now.tzinfo),
            source=self.name,
            fetched_at=now,
        )
