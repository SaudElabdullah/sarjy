"""Per-user request rate limiting, backed by the `rate_limits` table.

Two limits are enforced: a short-term one (nominally "per 10 minutes") and
a daily one. Both are counted in fixed buckets keyed by
`(user_id, window, window_start)` — the `window` discriminator is what
keeps the two kinds of bucket in separate keyspaces. Without it they
collided at midnight, where the day bucket and the short bucket share a
start timestamp, and every hit incremented one row twice (I2).

That discriminator carries an endpoint namespace too: `chat:5m`, `tele:1d`
and so on. Each endpoint gets the same limits on its own counter, so
metering `/telemetry` and `/chat/confirm` does not come out of the budget
`/chat` was sized against — a voice turn posts its marks and may confirm,
which on one shared counter made a single turn cost three hits. See `hit`.

The short-term limit is an *approximate sliding* window rather than a
fixed one: hits are counted in 5-minute buckets and the check sums the
current bucket and the one before it. A fixed 10-minute window resets to
zero on its boundary, so `limit` requests at 09:59 plus `limit` more at
10:01 were all allowed — double the intended rate inside two minutes
(I3). Summing two 5-minute buckets means the allowance covers a trailing
5-10 minutes at any instant, so a boundary burst is refused. It costs one
extra read per hit and never under-counts; the approximation is that a
request can be held against a window up to 10 minutes long rather than
exactly 10, which errs towards the limit rather than past it.

Both buckets are bumped on every `hit()` — including hits that end up
refused — on a single acquired connection, so one call never leaves the
two counters out of step with each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sarjy.infrastructure_shared.db import Database
from sarjy.shared.clock import Clock, SystemClock
from sarjy.shared.ids import UserId

# "window" is a reserved word in Postgres — quoted here and in the migration.
_UPSERT = """
insert into rate_limits (user_id, "window", window_start, count) values ($1,$2,$3,1)
on conflict (user_id, "window", window_start) do update set count = rate_limits.count + 1
returning count
"""
_PEEK = """
select count from rate_limits where user_id = $1 and "window" = $2 and window_start = $3
"""

_SHORT = "5m"
_DAY = "1d"
_BUCKET = timedelta(minutes=5)

# The default namespace: `/chat`, the endpoint the configured limits were sized
# against. See `hit`.
CHAT_NS = "chat"


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    retry_after_s: int = 0


class PgRateLimiter:
    """Sliding-ish short-term limit plus a fixed daily one."""

    def __init__(
        self,
        db: Database,
        per_10min: int = 60,
        per_day: int = 500,
        clock: Clock | None = None,
    ) -> None:
        self.db, self.per_10min, self.per_day = db, per_10min, per_day
        # Injectable so the boundary-burst behaviour can be tested at a chosen
        # instant instead of by waiting five real minutes.
        self.clock: Clock = clock or SystemClock()

    async def hit(
        self, user_id: UserId, is_anonymous: bool, *, window_ns: str = CHAT_NS
    ) -> RateLimitResult:
        """Count one request against `user_id`'s allowance and say whether to serve it.

        `window_ns` namespaces the buckets, so each endpoint gets its own
        allowance rather than eating into a shared one. Without it, adding the
        limiter to `/telemetry` silently halved conversational throughput: a
        voice turn posts its marks, so every turn cost two hits of the same
        60-per-10-minutes budget that was sized for one. The limits are
        deliberately the SAME per namespace — the point is separation, not
        generosity — and the shape of the bucket key is `<ns>:5m` / `<ns>:1d`,
        which the `"window"` column is plain `text` and takes as-is.

        Namespaces in use: `chat` (the default, `/chat`), `tele` (`/telemetry`)
        and `confirm` (`/chat/confirm`).
        """
        short, day = f"{window_ns}:{_SHORT}", f"{window_ns}:{_DAY}"
        now = self.clock.now()
        w5 = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        prev5 = w5 - _BUCKET
        wday = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # Anonymous users get half the allowance, floored at 1: integer division
        # of a limit of 1 is 0, which would refuse every anonymous request
        # outright rather than rate-limiting it.
        divisor = 2 if is_anonymous else 1
        lim10 = max(1, self.per_10min // divisor)
        limday = max(1, self.per_day // divisor)

        async with self.db.acquire() as conn:
            c5 = await conn.fetchval(_UPSERT, user_id, short, w5)
            cday = await conn.fetchval(_UPSERT, user_id, day, wday)
            prev = await conn.fetchval(_PEEK, user_id, short, prev5) or 0

        if c5 + prev > lim10:
            # R5: advertise a retry that actually lands in an allowed window.
            # The refusal clears when enough of the counted history has aged
            # out, and which bucket that is depends on where the count sits.
            # If the CURRENT bucket alone is already at the limit, waiting for
            # the previous one to roll off changes nothing — the current one
            # merely becomes the previous one and still refuses. Both have to
            # age out, which is one full bucket beyond the end of this one.
            # Otherwise the previous bucket is what tips it over, and that
            # stops being counted at the end of the current bucket.
            until_bucket_end = (w5 + _BUCKET - now).total_seconds()
            wait = until_bucket_end + (_BUCKET.total_seconds() if c5 >= lim10 else 0)
            # Floored at a second: a sub-second `Retry-After: 0` reads as
            # "go ahead immediately".
            return RateLimitResult(False, max(1, int(wait)))
        if cday > limday:
            return RateLimitResult(
                False, max(1, int((wday + timedelta(days=1) - now).total_seconds()))
            )
        return RateLimitResult(True)
