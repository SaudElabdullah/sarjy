from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Item:
    no: int
    trait: str
    reverse: bool
    text: str


@dataclass(frozen=True, slots=True)
class Instrument:
    id: str
    version: int
    scale_labels: list[str]
    traits: dict[str, str]
    bands: dict[str, tuple[float, float]]
    items: list[Item]

    @classmethod
    def from_definition(cls, d: dict[str, Any]) -> Instrument:
        items = [
            Item(int(i["no"]), str(i["trait"]), bool(i["reverse"]), str(i["text"]))
            for i in d["items"]
        ]
        items.sort(key=lambda i: i.no)
        bands = {k: (float(v[0]), float(v[1])) for k, v in d["bands"].items()}
        return cls(
            id=str(d["id"]),
            version=int(d["version"]),
            scale_labels=list(d["scale"]["labels"]),
            traits=dict(d["traits"]),
            bands=bands,
            items=items,
        )

    @property
    def total_items(self) -> int:
        return len(self.items)

    def item(self, no: int) -> Item:
        for i in self.items:
            if i.no == no:
                return i
        raise KeyError(no)

    def items_for_trait(self, code: str) -> list[Item]:
        return [i for i in self.items if i.trait == code]
