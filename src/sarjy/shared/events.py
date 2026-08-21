from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainEvent:
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))


Handler = Callable[[DomainEvent], Awaitable[None]]


class EventBus(Protocol):
    def subscribe(self, event_type: type[DomainEvent], handler: Handler) -> None: ...
    async def publish(self, event: DomainEvent) -> None: ...


class InMemoryEventBus:
    def __init__(self) -> None:
        self._handlers: dict[type[DomainEvent], list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: type[DomainEvent], handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: DomainEvent) -> None:
        for h in self._handlers[type(event)]:
            await h(event)
