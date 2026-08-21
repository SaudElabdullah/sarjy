from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):  # noqa: UP046
    value: T
    is_ok: bool = True

    def unwrap(self) -> T:
        return self.value


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):  # noqa: UP046
    error: E
    is_ok: bool = False

    def unwrap(self) -> None:
        raise RuntimeError(f"unwrap on Err: {self.error!r}")


Result = Ok[T] | Err[E]
