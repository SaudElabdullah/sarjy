from __future__ import annotations

from typing import Protocol

from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.contexts.weather.domain.location import Location


class LocationNotFound(Exception):  # noqa: N818
    def __init__(self, query: str) -> None:
        super().__init__(query)
        self.query = query


class ProviderUnavailable(Exception):  # noqa: N818
    pass


class DayOutOfRange(Exception):  # noqa: N818
    """The location's own calendar does not cover the requested day.

    Deliberately *not* a `ProviderUnavailable`: nothing is wrong with the
    provider, and the fallback would answer the same thing a round trip later.
    It is a bad request, and the caller is told so.
    """

    def __init__(self, when: str) -> None:
        super().__init__(when)
        self.when = when


class AmbiguousLocation(Exception):  # noqa: N818
    def __init__(self, candidates: list[Location]) -> None:
        super().__init__(f"{len(candidates)} candidates")
        self.candidates = candidates


class WeatherProvider(Protocol):
    name: str

    async def geocode(self, query: str) -> list[Location]: ...

    # `when` is the raw request token ("now"/"today"/"tomorrow"/ISO date), not a
    # resolved date: only the provider knows the location's own calendar, so only
    # it can turn the token into a day (see `Forecast.day`).
    async def forecast(self, loc: Location, when: str) -> Forecast: ...


class WeatherCache(Protocol):
    async def get(self, key: str) -> Forecast | None: ...
    async def set(self, key: str, forecast: Forecast, ttl_s: int) -> None: ...
