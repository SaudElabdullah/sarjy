"""In-process `WeatherCache` for tests and local dev without Postgres.

Production code (not a test double), same as the conversation context's
in-memory repos: useful for `Container.build(..., connect_db=False)` and for
`use_in_memory_repos()`. Mirrors `PgWeatherCache`'s TTL semantics — an entry
is judged expired against the injected clock, per its own per-`set` TTL —
so swapping between the two backends changes nothing about cache behaviour.
"""

from __future__ import annotations

from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.shared.clock import Clock


class InMemoryWeatherCache:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self._entries: dict[str, tuple[Forecast, float]] = {}

    async def get(self, key: str) -> Forecast | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        forecast, expires_at = entry
        if self.clock.now().timestamp() > expires_at:
            return None
        return forecast

    async def set(self, key: str, forecast: Forecast, ttl_s: int) -> None:
        self._entries[key] = (forecast, self.clock.now().timestamp() + ttl_s)
