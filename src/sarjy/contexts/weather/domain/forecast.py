from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sarjy.contexts.weather.domain.condition import Condition
from sarjy.contexts.weather.domain.location import Location
from sarjy.contexts.weather.domain.units import Units
from sarjy.shared.text import to_speech


def c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def kph_to_mph(k: float) -> float:
    return round(k / 1.609344, 1)


def _opt_f(c: float | None) -> float | None:
    return None if c is None else c_to_f(c)


# Condition texts that name the stuff falling out of the sky read as nouns
# ("...with heavy rain"); everything else describes the sky itself and needs a
# verb ("...and it will be partly cloudy").
_NOUN_CONDITIONS = ("rain", "snow", "showers", "drizzle", "thunderstorm", "fog")


@dataclass(frozen=True, slots=True)
class Forecast:
    """One day's weather for one place.

    The optional fields are the ones only a *current* observation carries. A
    forecast for a future day has a high, a low, a condition and a chance of
    precipitation and nothing else — so those fields are None rather than
    filled with a plausible-looking midpoint or a zero. Nothing downstream may
    invent them: they are omitted from the tool payload and from the grounding
    set, which is what stops the model speaking them.
    """

    temp_c: float | None
    temp_f: float | None
    feels_like_c: float | None
    condition: Condition
    precip_prob: int
    wind_kph: float | None
    humidity: int | None
    high_c: float
    low_c: float
    day: date
    observed_at: datetime
    source: str
    fetched_at: datetime

    @classmethod
    def from_metric(
        cls,
        *,
        temp_c: float | None,
        feels_like_c: float | None,
        condition: Condition,
        precip_prob: int,
        wind_kph: float | None,
        humidity: int | None,
        high_c: float,
        low_c: float,
        day: date,
        observed_at: datetime,
        source: str,
        fetched_at: datetime,
    ) -> Forecast:
        return cls(
            temp_c=None if temp_c is None else round(temp_c, 1),
            temp_f=_opt_f(temp_c),
            feels_like_c=None if feels_like_c is None else round(feels_like_c, 1),
            condition=condition,
            precip_prob=int(precip_prob),
            wind_kph=None if wind_kph is None else round(wind_kph, 1),
            humidity=None if humidity is None else int(humidity),
            high_c=round(high_c, 1),
            low_c=round(low_c, 1),
            day=day,
            observed_at=observed_at,
            source=source,
            fetched_at=fetched_at,
        )

    def grounding_numbers(self) -> tuple[float, ...]:
        """Every number Sarjy may legitimately speak, in both unit systems, raw and rounded.

        Unobserved fields contribute nothing — a number that was never measured
        must not become sayable just because it has a slot on this class.
        """
        base: list[float | None] = [
            self.temp_c,
            self.temp_f,
            self.feels_like_c,
            _opt_f(self.feels_like_c),
            float(self.precip_prob),
            self.wind_kph,
            None if self.wind_kph is None else kph_to_mph(self.wind_kph),
            None if self.humidity is None else float(self.humidity),
            self.high_c,
            c_to_f(self.high_c),
            self.low_c,
            c_to_f(self.low_c),
        ]
        out: set[float] = set()
        for n in base:
            if n is None:
                continue
            out.add(float(n))
            out.add(float(round(n)))
        return tuple(sorted(out))

    def to_tool_payload(
        self, location: Location, units: Units, day_label: str = "right now"
    ) -> dict[str, Any]:
        """What the model is allowed to see about this forecast.

        Only fields that are either sayable or needed to say something: every
        number here is in `grounding_numbers`, so there is nothing in the payload
        the output guard would have to reject. Coordinates, the numeric condition
        code and the timestamps are all ungrounded numbers that the model could
        read back as weather — they are deliberately absent.
        """
        payload: dict[str, Any] = {
            "location": {"name": location.name, "country": location.country},
            "units": units.value,
            "day": self.day.isoformat(),
            "day_label": day_label,
            "condition_text": self.condition.text,
            "precip_prob": self.precip_prob,
            "high_c": self.high_c,
            "high_f": c_to_f(self.high_c),
            "low_c": self.low_c,
            "low_f": c_to_f(self.low_c),
            "source": self.source,
        }
        # Absent keys, not null ones: a key with a null beside a dozen real
        # readings is an invitation to fill it in.
        optional: dict[str, Any] = {
            "temp_c": self.temp_c,
            "temp_f": self.temp_f,
            "feels_like_c": self.feels_like_c,
            "feels_like_f": _opt_f(self.feels_like_c),
            "wind_kph": self.wind_kph,
            "wind_mph": None if self.wind_kph is None else kph_to_mph(self.wind_kph),
            "humidity": self.humidity,
        }
        payload.update({k: v for k, v in optional.items() if v is not None})
        return payload

    def spoken_summary(self, location: Location, units: Units, day_label: str = "right now") -> str:
        """A single grounded sentence for the output guard's fallback path.

        `day_label` is how the requested day is said out loud ("right now",
        "tomorrow", "on Monday"). With no current temperature there is nothing
        to report *as* the temperature, so the sentence is built from the high
        and the low instead.
        """
        imperial = units == Units.IMPERIAL
        if self.temp_c is not None and self.temp_f is not None:
            temp = self.temp_f if imperial else self.temp_c
            sentence = (
                f"It's {round(temp)} degrees and {self.condition.text} "
                f"in {location.name} {day_label}."
            )
        else:
            high = c_to_f(self.high_c) if imperial else self.high_c
            low = c_to_f(self.low_c) if imperial else self.low_c
            lead = day_label[:1].upper() + day_label[1:]
            text = self.condition.text
            tail = f"with {text}" if text.endswith(_NOUN_CONDITIONS) else f"and it will be {text}"
            sentence = (
                f"{lead} in {location.name} expect a high of {round(high)} "
                f"and a low of {round(low)} {tail}."
            )
        return to_speech(sentence)
