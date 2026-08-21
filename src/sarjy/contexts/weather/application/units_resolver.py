"""Unit-system inference for `get_weather` (PRD W-9).

Precedence: an explicit tool argument wins; otherwise a stored `units`
preference fact; otherwise the country of the resolved location (United
States -> imperial); otherwise metric.

Pure and synchronous: it takes the already-fetched facts snapshot rather
than a `FactSnapshotPort` + `user_id`, so resolving units never does its own
I/O — the caller fetches facts once and reuses them for both the home-city
lookup and this resolution.
"""

from __future__ import annotations

from sarjy.contexts.conversation.application.ports import Fact
from sarjy.contexts.weather.domain.units import Units

_IMPERIAL_COUNTRIES = {"united states", "us", "usa"}


class UnitsResolver:
    def resolve(self, facts: list[Fact], explicit: str | None, country: str | None) -> Units:
        if explicit in ("metric", "imperial"):
            return Units(explicit)
        for fact in facts:
            if fact.key == "units" and fact.value.lower() in ("metric", "imperial"):
                return Units(fact.value.lower())
        if country is not None and country.strip().lower() in _IMPERIAL_COUNTRIES:
            return Units.IMPERIAL
        return Units.METRIC
