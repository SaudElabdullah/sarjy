"""Fire-and-forget work that must not be lost at shutdown.

`asyncio.create_task` on its own is not enough for this: the loop keeps only a
weak reference to a running task, so a task nothing else holds can be garbage
collected mid-flight and vanish without a trace. `spawn` keeps a strong
reference until the task finishes, logs anything it raised, and `drain` is what
`Container.shutdown` calls so the last turn's writes land before the database
pool goes away.

Nothing here is a queue or a retry: a failed background write is logged and
dropped. That is the deliberate trade for taking it off the hot path — the
work put here (the assistant transcript row, the session touch) is worth a
round trip of latency to the caller, but not worth failing a turn the user has
already heard the answer to.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from sarjy.observability.logging import get_logger

log = get_logger(__name__)


class BackgroundTasks:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        t = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._done)
        return t

    async def run(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Register `coro` here AND wait for it — a write whose ordering matters
        but whose survival must not depend on the caller.

        `RunTurn._finish` is the case this exists for: the assistant row is
        awaited (so the next turn can read it back), but the thing awaiting it is
        a response generator that a client disconnect can close and cancel at
        exactly that moment. A plain `await` would take the write down with it.
        Spawned and shielded, the write is a task of its own: the cancellation
        stops the waiting, not the writing, and `drain` still covers it at
        shutdown.
        """
        await asyncio.shield(self.spawn(coro))

    def _done(self, t: asyncio.Task[Any]) -> None:
        self._tasks.discard(t)
        # A cancelled task has no exception to read (asking for it re-raises).
        if not t.cancelled() and t.exception():
            log.error("background_task_failed", error=repr(t.exception()))

    @property
    def pending(self) -> int:
        return len(self._tasks)

    # ASYNC109 wants the caller to wrap this in `asyncio.timeout` instead. Here the
    # budget belongs to the callee: `asyncio.wait(timeout=...)` RETURNS on expiry,
    # leaving the stragglers running, whereas a cancel scope around the call would
    # cancel them — which is the one thing a drain must not do to a half-written
    # insert. Same reason `OutputGuard.drain` owns its own waiting.
    async def drain(self, timeout: float = 5.0) -> None:  # noqa: ASYNC109
        """Wait for everything currently in flight, up to `timeout` seconds.

        Bounded on purpose: shutdown must finish. A task still running past the
        timeout is left alone rather than cancelled — the loop is about to go
        away either way, and cancelling a half-written insert buys nothing.
        """
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=timeout)
