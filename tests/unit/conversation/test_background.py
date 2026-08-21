from __future__ import annotations

import asyncio

import pytest

from sarjy.infrastructure_shared.background import BackgroundTasks


async def test_spawn_and_drain_swallow_errors() -> None:
    bg = BackgroundTasks()
    done = []

    async def ok() -> None:
        done.append(1)

    async def boom() -> None:
        raise RuntimeError("x")

    bg.spawn(ok())
    bg.spawn(boom())
    await bg.drain(timeout=1)
    # One task raising must not take the other with it, and neither may reach
    # the caller: `drain` is called from shutdown, where an exception would
    # abort the rest of it (the database close, in `Container.shutdown`).
    assert done == [1] and bg.pending == 0


async def test_pending_counts_work_in_flight_and_clears_as_it_finishes() -> None:
    bg = BackgroundTasks()
    gate = asyncio.Event()

    async def waits() -> None:
        await gate.wait()

    bg.spawn(waits())
    # A task nothing else references can be garbage collected mid-flight, which
    # is exactly how a deferred write vanishes without a log line; the set is
    # the strong reference that stops it.
    assert bg.pending == 1
    gate.set()
    await bg.drain(timeout=1)
    assert bg.pending == 0


async def test_drain_returns_at_the_timeout_rather_than_hanging_shutdown() -> None:
    bg = BackgroundTasks()

    async def never() -> None:
        await asyncio.Event().wait()

    bg.spawn(never())
    await bg.drain(timeout=0.01)
    # Still running — deliberately not cancelled (a half-written insert is
    # worse than a slow one) — but shutdown got its control back.
    assert bg.pending == 1


async def test_drain_with_nothing_pending_is_a_no_op() -> None:
    bg = BackgroundTasks()
    await bg.drain(timeout=0.01)
    assert bg.pending == 0


def test_spawn_without_a_running_loop_raises_rather_than_dropping_the_work() -> None:
    coro = asyncio.sleep(0)
    with pytest.raises(RuntimeError):
        BackgroundTasks().spawn(coro)
    coro.close()
