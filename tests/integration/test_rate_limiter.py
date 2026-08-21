import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from sarjy.contexts.guardrails.infrastructure.pg_rate_limiter import PgRateLimiter
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.clock import FakeClock
from sarjy.shared.ids import UserId

pytestmark = pytest.mark.integration


@pytest.fixture
async def db():  # type: ignore[no-untyped-def]
    d = Database(os.environ["DATABASE_URL_DIRECT"])
    await d.connect()
    yield d
    await d.close()


@pytest.fixture
async def user(db: Database) -> UserId:
    u = uuid.uuid4()
    await db.execute("insert into auth.users (id,email) values ($1,$2)", u, f"{u}@x.test")
    return UserId(u)


async def test_rate_limiter_blocks_after_limit(db: Database, user: UserId) -> None:
    rl = PgRateLimiter(db, per_10min=3, per_day=100)
    results = [await rl.hit(user, is_anonymous=False) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert results[-1].retry_after_s > 0


async def test_rate_limiter_halves_the_limit_for_anonymous_users(
    db: Database, user: UserId
) -> None:
    rl = PgRateLimiter(db, per_10min=4, per_day=100)
    results = [await rl.hit(user, is_anonymous=True) for _ in range(3)]
    assert [r.allowed for r in results] == [True, True, False]
    assert results[-1].retry_after_s > 0


# ---------------------------------------------------------------------------
# I2/I3: window namespacing and the boundary burst.
# ---------------------------------------------------------------------------


async def _rows(db: Database, user: UserId) -> list[dict]:  # type: ignore[type-arg]
    rows = await db.fetch(
        'select "window", window_start, count from rate_limits where user_id = $1'
        ' order by "window"',
        user,
    )
    return [dict(r) for r in rows]


async def test_midnight_does_not_collide_the_day_and_short_buckets(
    db: Database, user: UserId
) -> None:
    # I2: at 00:00 the day bucket and the short bucket share a start timestamp.
    # Keyed on (user_id, window_start) alone they were one row, so a single hit
    # incremented it twice and the daily allowance drained ten times too fast.
    midnight = datetime(2026, 8, 21, 0, 0, 0, tzinfo=UTC)
    rl = PgRateLimiter(db, per_10min=100, per_day=100, clock=FakeClock(midnight))
    assert (await rl.hit(user, is_anonymous=False)).allowed

    rows = await _rows(db, user)
    assert [r["window"] for r in rows] == ["chat:1d", "chat:5m"]
    assert all(r["window_start"] == midnight for r in rows)
    assert [r["count"] for r in rows] == [1, 1]


async def test_burst_across_the_window_boundary_is_refused(db: Database, user: UserId) -> None:
    # I3: a fixed 10-minute window resets on its boundary, so ten requests at
    # 09:59 plus ten at 10:01 were all allowed — twice the intended rate inside
    # two minutes. Two 5-minute buckets summed together close that.
    clock = FakeClock(datetime(2026, 8, 21, 9, 59, 0, tzinfo=UTC))
    rl = PgRateLimiter(db, per_10min=10, per_day=1000, clock=clock)

    first = [await rl.hit(user, is_anonymous=False) for _ in range(10)]
    assert all(r.allowed for r in first)

    clock.advance(timedelta(minutes=2))  # 10:01 — a new 5-minute bucket
    second = [await rl.hit(user, is_anonymous=False) for _ in range(10)]
    assert not any(r.allowed for r in second)
    assert second[0].retry_after_s > 0


async def test_the_short_window_recovers_once_both_buckets_roll_off(
    db: Database, user: UserId
) -> None:
    clock = FakeClock(datetime(2026, 8, 21, 9, 59, 0, tzinfo=UTC))
    rl = PgRateLimiter(db, per_10min=3, per_day=1000, clock=clock)
    assert [(await rl.hit(user, is_anonymous=False)).allowed for _ in range(4)] == [
        True,
        True,
        True,
        False,
    ]
    clock.advance(timedelta(minutes=11))
    assert (await rl.hit(user, is_anonymous=False)).allowed


async def test_anonymous_limit_of_one_is_floored_not_zeroed(db: Database, user: UserId) -> None:
    # //2 of a limit of 1 is 0, which would refuse an anonymous user's very
    # first request rather than rate-limiting them.
    rl = PgRateLimiter(
        db, per_10min=1, per_day=1, clock=FakeClock(datetime(2026, 8, 21, 12, 0, tzinfo=UTC))
    )
    assert (await rl.hit(user, is_anonymous=True)).allowed


async def test_advertised_retry_lands_in_an_allowed_window(db: Database, user: UserId) -> None:
    # R5: with 25 hits in one 5-minute bucket against a limit of 10, waiting for
    # the *previous* bucket to age out achieves nothing — the offending bucket
    # merely becomes the previous one and still refuses. Both have to roll off.
    clock = FakeClock(datetime(2026, 8, 21, 12, 0, 0, tzinfo=UTC))
    rl = PgRateLimiter(db, per_10min=10, per_day=1000, clock=clock)
    results = [await rl.hit(user, is_anonymous=False) for _ in range(25)]
    assert sum(r.allowed for r in results) == 10
    retry = results[-1].retry_after_s

    # The old answer — the end of the current bucket — is still refused.
    clock.advance(timedelta(seconds=300))
    assert not (await rl.hit(user, is_anonymous=False)).allowed

    clock = FakeClock(datetime(2026, 8, 21, 12, 0, 0, tzinfo=UTC) + timedelta(seconds=retry))
    rl = PgRateLimiter(db, per_10min=10, per_day=1000, clock=clock)
    assert (await rl.hit(user, is_anonymous=False)).allowed, (
        f"advertised Retry-After of {retry}s still refuses"
    )


async def test_retry_after_waits_only_for_the_previous_bucket_when_that_is_enough(
    db: Database, user: UserId
) -> None:
    # 6 in the previous bucket, 5 in the current one, limit 10: the current
    # bucket is under the limit on its own, so one bucket of waiting is correct.
    clock = FakeClock(datetime(2026, 8, 21, 12, 0, 0, tzinfo=UTC))
    rl = PgRateLimiter(db, per_10min=10, per_day=1000, clock=clock)
    for _ in range(6):
        await rl.hit(user, is_anonymous=False)
    clock.advance(timedelta(minutes=5))
    results = [await rl.hit(user, is_anonymous=False) for _ in range(6)]
    assert not results[-1].allowed
    assert results[-1].retry_after_s <= 300


# ---------------------------------------------------------------------------
# Namespaced buckets: one endpoint's traffic must not spend another's allowance.
# ---------------------------------------------------------------------------


async def _counts(db: Database, user: UserId) -> dict[str, int]:
    rows = await db.fetch(
        'select "window", count from rate_limits where user_id = $1 order by 1', user
    )
    return {r["window"]: r["count"] for r in rows}


async def test_each_namespace_counts_into_its_own_bucket(db: Database, user: UserId) -> None:
    # One /chat hit and one /telemetry hit. On a shared bucket that is two
    # against one allowance, which is how adding the limiter to /telemetry
    # silently halved conversational throughput.
    rl = PgRateLimiter(db, per_10min=60, per_day=500)
    await rl.hit(user, is_anonymous=False)  # window_ns defaults to "chat"
    await rl.hit(user, is_anonymous=False, window_ns="tele")

    counts = await _counts(db, user)
    assert counts == {"chat:5m": 1, "chat:1d": 1, "tele:5m": 1, "tele:1d": 1}


async def test_exhausting_one_namespace_leaves_the_others_open(db: Database, user: UserId) -> None:
    # The point of the separation: a client that floods /chat/confirm cannot
    # stop the same user from holding a conversation.
    rl = PgRateLimiter(db, per_10min=2, per_day=100)
    assert [(await rl.hit(user, is_anonymous=False, window_ns="confirm")).allowed] == [True]
    for _ in range(3):
        await rl.hit(user, is_anonymous=False, window_ns="confirm")
    assert not (await rl.hit(user, is_anonymous=False, window_ns="confirm")).allowed
    assert (await rl.hit(user, is_anonymous=False)).allowed
    assert (await rl.hit(user, is_anonymous=False, window_ns="tele")).allowed
