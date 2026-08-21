"""`AuditQueuePort`, backed by `public.audit_queue` (Layer 7, PRD §Layer 7).

`claim` hands out unprocessed rows that have not already failed three times
(`attempts < 3`, incremented by `mark_failed` — see `_MARK_FAILED` below), and
locks the ones it hands out (`for update ... skip locked`), which is enough to
keep two `claim` calls landing at the same instant from handing out the same
row — see the held-only-for-the-`SELECT` caveat below for what it does NOT
cover. The lock is scoped `of aq` (the joined `messages` rows are not locked)
because Postgres refuses `FOR UPDATE` on the nullable side of an outer join,
and the preceding-user-message lookup below is exactly that: a
`LEFT JOIN LATERAL`, since an
assistant turn with no prior user message in its session (should not happen,
but is not this repo's job to assume) must still be returned rather than
silently dropped from the batch.

The lock is held only for the duration of the `SELECT` itself — asyncpg runs
it as its own implicit transaction, same as every other single-statement call
through `Database` — not across the gap between `claim` and the later
`mark_processed`. That gap is a real, accepted race: a second `claim` landing
in that window would see the same rows (still `processed_at is null`) and
hand them out again. Given the cron cadence (every 10 minutes) and that
`/internal/audit/run` is not a user-facing endpoint, a duplicate audit run on
the same handful of messages costs an extra `guardrail_events` row, not a
correctness bug — the alternative (holding a transaction open across the
whole batch, spanning a network round trip to the classifier) would turn a
slow classifier call into a long-held row lock, which is worse.
"""

from __future__ import annotations

from sarjy.contexts.guardrails.application.ports import AuditItem
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import MessageId, UserId

_CLAIM = """
select aq.id, aq.message_id, aq.user_id,
       coalesce(um.content, '') as user_text, am.content as assistant_text
from public.audit_queue aq
join public.messages am on am.id = aq.message_id
left join lateral (
  select m.content
  from public.messages m
  where m.session_id = am.session_id and m.role = 'user' and m.created_at < am.created_at
  order by m.created_at desc, m.id desc
  limit 1
) um on true
where aq.processed_at is null and aq.attempts < 3
order by aq.id
limit $1
for update of aq skip locked
"""

_MARK_PROCESSED = "update public.audit_queue set processed_at = now() where id = any($1::bigint[])"

# Leaves `processed_at` null — the item is still unprocessed and still worth
# retrying — but counts the attempt, so `_CLAIM`'s `attempts < 3` eventually
# stops handing it out. Without this an item the classifier fails on every
# time (a message it chokes on permanently) is claimed, failed and re-claimed
# on every 10-minute cron run indefinitely, and it sits at the head of the
# `order by aq.id` batch crowding out newer items.
_MARK_FAILED = "update public.audit_queue set attempts = attempts + 1 where id = any($1::bigint[])"


class PgAuditRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def claim(self, limit: int) -> list[AuditItem]:
        rows = await self.db.fetch(_CLAIM, limit)
        return [
            AuditItem(
                id=r["id"],
                message_id=MessageId(r["message_id"]),
                user_id=UserId(r["user_id"]),
                user_text=r["user_text"],
                assistant_text=r["assistant_text"],
            )
            for r in rows
        ]

    async def mark_processed(self, ids: list[int]) -> None:
        await self.db.execute(_MARK_PROCESSED, ids)

    async def mark_failed(self, ids: list[int]) -> None:
        await self.db.execute(_MARK_FAILED, ids)
