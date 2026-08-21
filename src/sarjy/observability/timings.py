from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

# Every per-tool stage is recorded as `t_tool_<name>`, and `as_dict` rolls them
# up into a single `t_tool`. The prefix is the contract between the two.
_TOOL_PREFIX = "t_tool_"


class Timings:
    """Stage stopwatch. All values are integer milliseconds since construction or stage start."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._data: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._data[f"t_{name}"] = int((time.perf_counter() - start) * 1000)

    def mark(self, name: str) -> None:
        self._data[f"t_{name}"] = int((time.perf_counter() - self._t0) * 1000)

    def add(self, name: str, ms: int) -> None:
        """Record a duration this stopwatch did not measure itself.

        The HTTP gate is the case (I5): the JWT verify and the rate limiter's
        round trip both happen before `RunTurn` exists, so their cost cannot be
        a `stage` here — but leaving it out meant `t_total` and the client's own
        `t_request_ms` disagreed by a gap nothing on the server accounted for.
        """
        self._data[f"t_{name}"] = ms

    def as_dict(self) -> dict[str, int]:
        """The recorded stages, plus `t_tool`: everything the turn spent in tools.

        The per-tool keys stay — "which tool was slow" is the question a
        breakdown exists to answer — but a turn with two parallel calls reported
        two keys and no total, so "how much of this turn was tool time?" had to
        be reassembled by whoever read it, differently each time. Derived on
        read rather than stored, so it cannot drift from the stages it sums, and
        omitted entirely when no tool ran: a `t_tool: 0` on every chat turn is a
        column of noise in a table read for outliers.
        """
        d = dict(self._data)
        tool_ms = [v for k, v in d.items() if k.startswith(_TOOL_PREFIX)]
        if tool_ms:
            d["t_tool"] = sum(tool_ms)
        return d
