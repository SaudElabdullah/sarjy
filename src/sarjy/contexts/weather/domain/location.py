from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Location:
    name: str
    country: str
    lat: float
    lon: float
    admin1: str | None = None
    # Both are geocoder metadata, kept because they are what separates a real
    # ambiguity from a namesake: "London, Canada" (≈420k) is not a plausible
    # answer to "London" (≈9M), but two 500k cities are. Providers without the
    # data (OWM) leave them None, which is read as "cannot judge".
    population: int | None = None
    feature_code: str | None = None

    @property
    def label(self) -> str:
        return f"{self.name}, {self.country}"
