from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal, TypeVar

from sarjy.contexts.weather.application.ports import (
    DayOutOfRange,
    LocationNotFound,
    ProviderUnavailable,
    WeatherCache,
    WeatherProvider,
)
from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.contexts.weather.domain.location import Location
from sarjy.contexts.weather.domain.units import Units
from sarjy.contexts.weather.domain.when import parse_when
from sarjy.observability.logging import get_logger
from sarjy.shared.clock import Clock

log = get_logger(__name__)

T = TypeVar("T")

_HOME_ALIASES = ("here", "my city", "home", "")


@dataclass(frozen=True, slots=True)
class WeatherSuccess:
    forecast: Forecast
    location: Location
    units: Units
    day: date


@dataclass(frozen=True, slots=True)
class WeatherNotFound:
    query: str
    reason: Literal["not_found", "no_home_city", "bad_date"]


@dataclass(frozen=True, slots=True)
class WeatherAmbiguous:
    query: str
    candidates: list[Location]


@dataclass(frozen=True, slots=True)
class WeatherUnavailable:
    reason: Literal["timeout", "provider_error"]


WeatherResult = WeatherSuccess | WeatherNotFound | WeatherAmbiguous | WeatherUnavailable


def cache_key(loc: Location, day: date | str) -> str:
    # Units are presentation-only — the stored Forecast carries both C and F —
    # so they are deliberately excluded from the cache key (PRD §7.4 lists
    # units in the key, but that's stricter than the data actually requires).
    return f"{loc.lat:.2f}:{loc.lon:.2f}:{day if isinstance(day, str) else day.isoformat()}"


def _when_slug(when: str, server_day: date) -> str:
    """The lookup half of the cache key for a request that has not run yet.

    Which local date "now"/"tomorrow" mean is only known once the provider has
    answered (`Forecast.day`), so a relative request is looked up under the
    token itself and written back under *both* that token and the resolved
    local date — see `GetWeather._cache_store`.

    The server's date is stamped alongside the token so the key rolls over at
    midnight instead of pointing "tomorrow" at yesterday's tomorrow. An ISO
    request is already absolute and needs no stamp.
    """
    w = (when or "now").strip().lower()
    if w in ("now", "today", ""):
        return f"today:{server_day.isoformat()}"
    if w == "tomorrow":
        return f"tomorrow:{server_day.isoformat()}"
    return w


# How big a namesake has to be, relative to the top hit, before asking which
# one the caller meant is worth the extra turn.
AMBIGUITY_POPULATION_RATIO = 0.3


def _is_ambiguous(cands: list[Location]) -> bool:
    """True only when a namesake is a plausible alternative to the top hit.

    Same name in a different country is necessary but nowhere near sufficient —
    nearly every major city has one, so on its own it made Sarjy ask "did you
    mean?" about London, Paris and Tokyo. The runner-up also has to be within
    reach of the leader's size. A runner-up the geocoder cannot size is treated
    as the small town it almost always is.

    A geocoder that sizes *nothing* (OWM, on the fallback path) tells us nothing
    either way, so it gets the geocoder's own ranking rather than a "did you
    mean?" for every city on Earth. An unsized leader beside a sized rival is a
    different story: there the data exists and the leader is conspicuously
    missing from it, so asking is the safe move.
    """
    if len(cands) < 2:
        return False
    leader = cands[0]
    name = leader.name.casefold()
    rival = next(
        (c for c in cands[1:] if c.name.casefold() == name and c.country != leader.country),
        None,
    )
    if rival is None:
        return False
    if leader.population is None:
        return any(c.population is not None for c in cands)
    return (rival.population or 0) >= AMBIGUITY_POPULATION_RATIO * leader.population


def _narrow_to_qualifier(query: str, cands: list[Location]) -> list[Location]:
    """Apply the "City, X" half of the query, if there is one.

    A comma used to merely switch the ambiguity check off, which let "Springfield,
    Canada" answer with the Illinois one. Now X actually selects: candidates whose
    country or admin1 matches it win, and if none do the qualifier is treated as
    noise rather than as a reason to return nothing.
    """
    _, _, tail = query.partition(",")
    qualifier = tail.strip().casefold()
    if not qualifier:
        return cands
    matched = [
        c
        for c in cands
        if c.country.casefold() == qualifier or (c.admin1 or "").casefold() == qualifier
    ]
    return matched or cands


def _classify(e: ProviderUnavailable) -> Literal["timeout", "provider_error"]:
    return "timeout" if str(e) == "timeout" else "provider_error"


class GetWeather:
    def __init__(
        self,
        provider: WeatherProvider,
        fallback: WeatherProvider | None,
        cache: WeatherCache,
        clock: Clock,
        timeout_s: float = 2.5,
        ttl_s: int = 600,
    ) -> None:
        self.provider = provider
        self.fallback = fallback
        self.cache = cache
        self.clock = clock
        self.timeout_s = timeout_s
        self.ttl_s = ttl_s

    async def execute(
        self, query: str, when: str, units: Units, home_city: str | None
    ) -> WeatherResult:
        q = (query or "").strip()
        if q.lower() in _HOME_ALIASES:
            if not home_city:
                return WeatherNotFound(query=q, reason="no_home_city")
            q = home_city

        # Validation only — the day itself is resolved in the location's own
        # calendar by the provider. A day of slack either side covers a location
        # whose date differs from the server's (C1). The date it returns is used
        # for nothing but the cache slug.
        server_day = parse_when(when, self.clock.now().date(), slack_days=1)
        if server_day is None:
            return WeatherNotFound(query=q, reason="bad_date")

        try:
            cands, provider_idx = await self._with_fallback(lambda p: p.geocode(q))
        except LocationNotFound:
            # A definitive "no such place" — no point asking the fallback.
            return WeatherNotFound(query=q, reason="not_found")
        except ProviderUnavailable as e:
            return WeatherUnavailable(reason=_classify(e))

        if not cands:
            return WeatherNotFound(query=q, reason="not_found")
        cands = _narrow_to_qualifier(q, cands)
        if _is_ambiguous(cands):
            return WeatherAmbiguous(query=q, candidates=cands[:3])
        loc = cands[0]

        key = cache_key(loc, _when_slug(when, server_day))
        cached = await self._cache_load(key)
        if cached is not None:
            return WeatherSuccess(forecast=cached, location=loc, units=units, day=cached.day)

        try:
            # Sticky: resume from the provider that just succeeded (provider_idx)
            # rather than retrying one that already failed/timed out in this request.
            fc, _ = await self._with_fallback(lambda p: p.forecast(loc, when), start=provider_idx)
        except DayOutOfRange:
            # Not a provider failure: the day is simply outside what anyone can
            # forecast, so the fallback is never asked and the caller is told.
            return WeatherNotFound(query=q, reason="bad_date")
        except ProviderUnavailable as e:
            return WeatherUnavailable(reason=_classify(e))

        await self._cache_store(key, loc, fc)
        return WeatherSuccess(forecast=fc, location=loc, units=units, day=fc.day)

    async def _cache_load(self, key: str) -> Forecast | None:
        """A cache outage must not take weather down: it is an optimisation, and
        the provider is right there."""
        try:
            return await self.cache.get(key)
        except Exception as e:
            log.warning("weather_cache_get_failed", key=key, error=str(e))
            return None

    async def _cache_store(self, key: str, loc: Location, fc: Forecast) -> None:
        """Write under the request's own key and under the resolved local date.

        The second key is what makes an explicit ISO date share an entry with the
        relative token that resolved to the same day in the location's calendar.
        """
        for k in dict.fromkeys((key, cache_key(loc, fc.day))):
            try:
                await self.cache.set(k, fc, self.ttl_s)
            except Exception as e:
                # The caller already has their answer; failing now would throw it away.
                log.warning("weather_cache_set_failed", key=k, error=str(e))

    async def _with_fallback(
        self, call: Callable[[WeatherProvider], Awaitable[T]], start: int = 0
    ) -> tuple[T, int]:
        providers = [self.provider, *([self.fallback] if self.fallback else [])]
        last: ProviderUnavailable | None = None
        for idx in range(start, len(providers)):
            p = providers[idx]
            try:
                result = await asyncio.wait_for(call(p), timeout=self.timeout_s)
                return result, idx
            except TimeoutError:
                log.warning("weather_provider_timeout", provider=p.name)
                last = ProviderUnavailable("timeout")
            except ProviderUnavailable as e:
                log.warning("weather_provider_unavailable", provider=p.name, error=str(e))
                last = e
        raise last or ProviderUnavailable("unavailable")
