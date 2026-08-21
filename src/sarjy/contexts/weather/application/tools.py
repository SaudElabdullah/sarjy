"""`get_weather` as a conversation `ToolPort` adapter.

Adapts the weather context's `GetWeather` use case to the conversation
context's `ToolPort` protocol so the tool router can drive it without
depending on weather internals. `name` and `declaration` are verbatim from
PRD §9.3.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sarjy.contexts.conversation.application.ports import (
    Fact,
    FactSnapshotPort,
    ToolResult,
)
from sarjy.contexts.weather.application.get_weather import (
    GetWeather,
    WeatherAmbiguous,
    WeatherNotFound,
    WeatherSuccess,
    WeatherUnavailable,
)
from sarjy.contexts.weather.application.units_resolver import UnitsResolver
from sarjy.contexts.weather.domain.units import Units
from sarjy.contexts.weather.domain.when import day_label
from sarjy.observability.logging import get_logger
from sarjy.shared.ids import UserId

log = get_logger(__name__)

_CANT_REACH = "I can't reach the weather service right now."

_NOT_FOUND_SPOKEN: dict[str, str] = {
    "no_home_city": "I don't know where you are yet — which city should I check?",
    "bad_date": "I can only look up to seven days ahead.",
}


class GetWeatherTool:
    name = "get_weather"
    mutating: ClassVar[bool] = False  # a forecast read changes nothing
    declaration: ClassVar[dict[str, Any]] = {
        "name": "get_weather",
        "description": (
            "Get current conditions or a forecast for a real place on Earth. "
            "ALWAYS call this before stating any weather information. Never guess weather."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "City or 'city, country'. Use 'here' for the user's home city."
                    ),
                },
                "when": {
                    "type": "string",
                    "enum": ["now", "today", "tomorrow"],
                    "description": "Or an ISO date within 7 days.",
                },
                "units": {"type": "string", "enum": ["metric", "imperial"]},
            },
            "required": ["location"],
        },
    }

    def __init__(self, use_case: GetWeather, facts: FactSnapshotPort, units: UnitsResolver) -> None:
        self._uc = use_case
        self._facts = facts
        self._units = units

    async def invoke(
        self, user_id: UserId, args: dict[str, Any], facts: list[Fact] | None = None
    ) -> ToolResult:
        try:
            return await self._invoke(user_id, args, facts)
        except Exception:
            log.exception("get_weather_tool_error")
            return ToolResult(ok=False, data={"error": "unavailable"}, spoken_error=_CANT_REACH)

    async def _invoke(
        self, user_id: UserId, args: dict[str, Any], facts: list[Fact] | None
    ) -> ToolResult:
        query = str(args.get("location") or "").strip()
        when = str(args.get("when") or "now")
        raw_units = args.get("units")
        explicit_units = raw_units if isinstance(raw_units, str) else None

        # The turn already loaded these — one RPC, before the model was asked for
        # a token — and was holding them in memory while this tool went back to
        # the database for the same two rows (I5). A weather turn therefore cost
        # two reads where it owed one, on the single hottest path in the app. The
        # snapshot port stays for callers with nothing to hand down (the eval
        # harness drives the tool directly), but the turn hands them down.
        if facts is None:
            facts = await self._facts.snapshot(user_id)
        home_city = next((f.value for f in facts if f.key == "home_city"), None)

        # A single provider round trip per tool call. Units are presentation-only
        # (the stored Forecast carries both C and F, and the cache key is
        # unit-agnostic — see get_weather.cache_key), so there is nothing to gain
        # from executing twice: we fetch once, then resolve the final display
        # units from the already-fetched facts plus whatever country the
        # location resolved to (UnitsResolver.resolve is synchronous — no I/O).
        result = await self._uc.execute(query, when, Units.METRIC, home_city)
        if isinstance(result, WeatherSuccess):
            units = self._units.resolve(facts, explicit_units, result.location.country)
            return self._success_result(result, units, day_label(when))
        return self._to_tool_result(result)

    def _success_result(self, result: WeatherSuccess, units: Units, label: str) -> ToolResult:
        return ToolResult(
            ok=True,
            data=result.forecast.to_tool_payload(result.location, units, label),
            grounding_numbers=result.forecast.grounding_numbers(),
            spoken_summary=result.forecast.spoken_summary(result.location, units, label),
        )

    def _to_tool_result(
        self, result: WeatherNotFound | WeatherAmbiguous | WeatherUnavailable
    ) -> ToolResult:
        if isinstance(result, WeatherNotFound):
            if result.reason == "not_found":
                spoken = (
                    f"I couldn't find a place called {result.query}. "
                    "Could you add the country or check the spelling?"
                )
            else:
                spoken = _NOT_FOUND_SPOKEN[result.reason]
            return ToolResult(
                ok=False,
                data={"error": "not_found", "reason": result.reason, "query": result.query},
                spoken_error=spoken,
            )
        if isinstance(result, WeatherAmbiguous):
            labels = [c.label for c in result.candidates]
            return ToolResult(
                ok=False,
                data={"error": "ambiguous", "query": result.query, "candidates": labels},
                spoken_error=f"Did you mean {_or_join(labels)}?",
            )
        assert isinstance(result, WeatherUnavailable)
        return ToolResult(ok=False, data={"error": "unavailable"}, spoken_error=_CANT_REACH)


def _or_join(items: list[str]) -> str:
    """English list join with an Oxford comma: 'A' / 'A or B' / 'A, B, or C'."""
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return f"{', '.join(items[:-1])}, or {items[-1]}"
