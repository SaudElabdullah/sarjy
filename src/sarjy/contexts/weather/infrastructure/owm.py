from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from sarjy.contexts.weather.application.ports import DayOutOfRange, ProviderUnavailable
from sarjy.contexts.weather.domain.condition import Condition
from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.contexts.weather.domain.location import Location
from sarjy.contexts.weather.domain.when import index_for_when
from sarjy.shared.clock import Clock

GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"
ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

# OWM condition id -> approximate WMO code so Condition text stays consistent
# across providers. Keyed both by exact id and by the id's hundred-group.
_OWM_TO_WMO: dict[int, int] = {
    2: 95,
    3: 53,
    5: 63,
    6: 73,
    7: 45,
    800: 0,
    801: 1,
    802: 2,
    803: 3,
    804: 3,
}


def _wmo(owm_id: int) -> int:
    if owm_id in _OWM_TO_WMO:
        return _OWM_TO_WMO[owm_id]
    return _OWM_TO_WMO.get(owm_id // 100, 3)


class OwmProvider:
    """Fallback weather provider backed by OpenWeatherMap's geocoding + One Call APIs."""

    name = "owm"

    def __init__(
        self, client: httpx.AsyncClient, api_key: str, clock: Clock, timeout_s: float = 2.5
    ) -> None:
        self.client, self.key, self.clock, self.timeout_s = client, api_key, clock, timeout_s

    async def _get(self, url: str, params: dict[str, str | int | float]) -> Any:
        try:
            r = await self.client.get(
                url, params={**params, "appid": self.key}, timeout=self.timeout_s
            )
        except httpx.HTTPError as e:
            raise ProviderUnavailable(f"owm network: {e.__class__.__name__}") from e
        if r.status_code >= 400:
            # Never fabricate a result for any HTTP error, including client errors.
            raise ProviderUnavailable(f"owm http {r.status_code}")
        try:
            return r.json()
        except ValueError as e:
            raise ProviderUnavailable("malformed_response") from e

    async def geocode(self, query: str) -> list[Location]:
        data = await self._get(GEO_URL, {"q": query, "limit": 5})
        try:
            return [
                Location(
                    name=d["name"],
                    country=d.get("country", ""),
                    lat=float(d["lat"]),
                    lon=float(d["lon"]),
                    admin1=d.get("state"),
                )
                for d in data
            ][:3]
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderUnavailable("malformed_response") from e

    async def forecast(self, loc: Location, when: str) -> Forecast:
        data = await self._get(
            ONECALL_URL,
            {
                "lat": loc.lat,
                "lon": loc.lon,
                "units": "metric",
                "exclude": "minutely,hourly,alerts",
            },
        )
        try:
            return self._map_forecast(data, when)
        except (DayOutOfRange, ProviderUnavailable):
            raise
        except (KeyError, TypeError, IndexError, ValueError) as e:
            raise ProviderUnavailable("malformed_response") from e

    def _map_forecast(self, data: Any, when: str) -> Forecast:
        now = self.clock.now()
        daily = data.get("daily") or []
        # One Call stamps every entry in UTC; `timezone_offset` is what turns
        # those into the location's own calendar, so day_dates[0] is its today
        # rather than the server's (see OpenMeteoProvider._map_forecast).
        loc_tz = timezone(timedelta(seconds=int(data["timezone_offset"])))
        day_dates = [datetime.fromtimestamp(entry["dt"], loc_tz).date() for entry in daily]
        idx = index_for_when(when, day_dates)
        if idx is None:
            raise DayOutOfRange(when)
        day = day_dates[idx]
        d = daily[idx]
        # No `.get(..., 0)` fallbacks: a missing required field is a malformed
        # response, not a reading of zero (C2). KeyError/TypeError here is
        # caught by `forecast` and reported as ProviderUnavailable.
        high, low = float(d["temp"]["max"]), float(d["temp"]["min"])
        pop = round(float(d["pop"]) * 100)
        cur = data.get("current") or {}
        if idx == 0 and cur:
            w = cur["weather"][0]
            cond = Condition(code=_wmo(int(w["id"])), text=str(w["description"]))
            return Forecast.from_metric(
                temp_c=float(cur["temp"]),
                feels_like_c=float(cur["feels_like"]),
                condition=cond,
                precip_prob=pop,
                wind_kph=round(float(cur["wind_speed"]) * 3.6, 1),
                humidity=int(cur["humidity"]),
                high_c=high,
                low_c=low,
                day=day,
                observed_at=datetime.fromtimestamp(int(cur["dt"]), loc_tz),
                source=self.name,
                fetched_at=now,
            )
        w = d["weather"][0]
        # A future day has no observation — see OpenMeteoProvider._map_forecast.
        return Forecast.from_metric(
            temp_c=None,
            feels_like_c=None,
            condition=Condition(code=_wmo(int(w["id"])), text=str(w["description"])),
            precip_prob=pop,
            wind_kph=None,
            humidity=None,
            high_c=high,
            low_c=low,
            day=day,
            observed_at=datetime.combine(day, datetime.min.time(), tzinfo=loc_tz),
            source=self.name,
            fetched_at=now,
        )
