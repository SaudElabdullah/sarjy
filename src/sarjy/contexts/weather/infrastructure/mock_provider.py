from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta

from sarjy.contexts.weather.application.ports import (
    DayOutOfRange,
    LocationNotFound,
    ProviderUnavailable,
)
from sarjy.contexts.weather.domain.condition import Condition
from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.contexts.weather.domain.location import Location
from sarjy.contexts.weather.domain.when import MAX_DAYS_AHEAD, index_for_when
from sarjy.shared.clock import Clock

# Populations are roughly the real ones: `_is_ambiguous` weighs them, so a
# fixture without them would not exercise the same code path as production.
_KNOWN: dict[str, Location] = {
    "tokyo": Location("Tokyo", "Japan", 35.6895, 139.6917, "Tokyo", 8336599),
    "reykjavik": Location("Reykjavik", "Iceland", 64.1466, -21.9426, None, 118918),
    "lisbon": Location("Lisbon", "Portugal", 38.7223, -9.1393, "Lisbon", 517802),
    "paris": Location("Paris", "France", 48.8566, 2.3522, "Île-de-France", 2138551),
    "berlin": Location("Berlin", "Germany", 52.52, 13.405, "Berlin", 3426354),
    "new york": Location("New York", "United States", 40.7128, -74.006, "New York", 8175133),
}
# The one deliberately ambiguous fixture: two comparably sized namesakes in
# different countries, which is what it takes to be worth a "did you mean?".
_SPRINGFIELD = [
    Location("Springfield", "United States", 39.8, -89.64, "Illinois", 116565),
    Location("Springfield", "Canada", 45.6, -75.5, "Ontario", 98000),
]


def _mock_delay_s() -> float:
    """Simulated latency for testing timeout/fallback behaviour, via MOCK_DELAY_MS."""
    ms = os.environ.get("MOCK_DELAY_MS")
    return float(ms) / 1000.0 if ms else 0.0


class MockProvider:
    """In-process weather provider for tests and local dev without network calls."""

    name = "mock"

    def __init__(self, clock: Clock, temp_c: float = 18.0, fail: bool = False) -> None:
        self.clock, self.temp_c, self.fail = clock, temp_c, fail

    async def geocode(self, query: str) -> list[Location]:
        delay = _mock_delay_s()
        if delay:
            await asyncio.sleep(delay)
        if self.fail:
            raise ProviderUnavailable("mock down")
        q = query.lower().split(",")[0].strip()
        if q == "springfield":
            return list(_SPRINGFIELD)
        known = _KNOWN.get(q)
        if known is None:
            # No catch-all: a mock that invents a plausible city for any string
            # hides exactly the not-found handling the tests exist to check.
            raise LocationNotFound(query)
        return [known]

    async def forecast(self, loc: Location, when: str) -> Forecast:
        delay = _mock_delay_s()
        if delay:
            await asyncio.sleep(delay)
        if self.fail:
            raise ProviderUnavailable("mock down")
        now: datetime = self.clock.now()
        # Stands in for a provider's own day list: the mock has no timezone data,
        # so its calendar is the clock's.
        days = [now.date() + timedelta(days=i) for i in range(MAX_DAYS_AHEAD + 1)]
        idx = index_for_when(when, days)
        if idx is None:
            raise DayOutOfRange(when)
        current = idx == 0
        return Forecast.from_metric(
            temp_c=self.temp_c if current else None,
            feels_like_c=self.temp_c - 1 if current else None,
            condition=Condition.from_wmo(2),
            precip_prob=30,
            wind_kph=14.0 if current else None,
            humidity=60 if current else None,
            high_c=self.temp_c + 4,
            low_c=self.temp_c - 6,
            day=days[idx],
            observed_at=now,
            source=self.name,
            fetched_at=now,
        )
