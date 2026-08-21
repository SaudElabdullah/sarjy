"""`20260821000700_retention_cron.sql` / `20260821000800_audit_sample.sql` —
retention/alert/audit `pg_cron` jobs and the `check_alerts` no-op path
(PRD §11 retention, §13 alerts, Layer 7 audit sampling).
"""

import os
import uuid

import asyncpg
import pytest

pytestmark = pytest.mark.integration

DB = os.environ["DATABASE_URL_DIRECT"]

_EXPECTED_JOBS = {
    "retention_messages",
    "retention_tool_calls",
    "retention_guard_events",
    "retention_telemetry",
    "retention_weather_cache",
    "retention_rate_limits",
    "retention_sessions",
    "check_alerts",
    "audit",
    "retention_memories_history",
}


async def _conn() -> asyncpg.Connection:
    return await asyncpg.connect(DB)


async def test_retention_and_alert_and_audit_jobs_are_scheduled() -> None:
    c = await _conn()
    try:
        rows = await c.fetch("select jobname from cron.job")
    finally:
        await c.close()
    assert {r["jobname"] for r in rows} >= _EXPECTED_JOBS


async def test_check_alerts_runs_without_error_on_an_empty_db() -> None:
    # `app.alert_webhook` is unset on this database (nothing in the migrations
    # sets it — see the runbook note in 20260821000700_retention_cron.sql), so
    # `fire_alert` no-ops on every threshold check: no `net.http_post`, no
    # write to `alert_state`. This is exactly the local/CI shape, and the
    # point of the test is that `check_alerts()` runs clean either way —
    # whether or not any threshold is currently tripped by leftover rows from
    # other integration tests in the same run.
    c = await _conn()
    try:
        await c.execute("select public.check_alerts()")
    finally:
        await c.close()


async def test_audit_queue_is_locked_down_for_authenticated() -> None:
    c = await _conn()
    try:
        async with c.transaction():
            await c.execute("set role authenticated")
            await c.execute(
                "select set_config('request.jwt.claims', $1, true)",
                f'{{"sub":"{uuid.uuid4()}","role":"authenticated"}}',
            )
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await c.execute("delete from public.audit_queue")
        await c.execute("reset role")
    finally:
        await c.close()


async def test_alert_state_is_locked_down_for_authenticated() -> None:
    c = await _conn()
    try:
        async with c.transaction():
            await c.execute("set role authenticated")
            await c.execute(
                "select set_config('request.jwt.claims', $1, true)",
                f'{{"sub":"{uuid.uuid4()}","role":"authenticated"}}',
            )
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await c.execute("delete from public.alert_state")
        await c.execute("reset role")
    finally:
        await c.close()


_LOCKED_DOWN_FUNCTIONS = {
    "fire_alert(text,jsonb)": "fire_alert",
    "check_alerts()": "check_alerts",
    "enqueue_audit_sample()": "enqueue_audit_sample",
    "run_audit_cron()": "run_audit_cron",
    # M6: the one `security definer` function the round-1 lockdown missed —
    # revoked in 20260821000900. Revoking EXECUTE does not affect the
    # `on_auth_user_created` trigger, which runs as the table owner.
    "handle_new_user()": "handle_new_user",
}


@pytest.mark.parametrize("signature,proname", _LOCKED_DOWN_FUNCTIONS.items())
async def test_security_definer_functions_are_not_executable_by_anon_or_authenticated(
    signature: str, proname: str
) -> None:
    # Fix round 1, Critical 1: a `security definer` function with an unpinned
    # search_path is a privilege-escalation hole, and one still EXECUTE-able by
    # anon/authenticated defeats the ops-table lockdown around it.
    c = await _conn()
    try:
        for role in ("anon", "authenticated"):
            can_execute = await c.fetchval(
                f"select has_function_privilege($1, 'public.{signature}', 'EXECUTE')", role
            )
            assert can_execute is False, f"{proname} is EXECUTE-able by {role}"
        config = await c.fetchval(
            "select proconfig from pg_proc where proname = $1 "
            "and pronamespace = 'public'::regnamespace",
            proname,
        )
        assert config is not None and "search_path=public" in config
    finally:
        await c.close()


async def test_audit_sample_trigger_failure_does_not_roll_back_the_message_insert() -> None:
    # Fix round 1, Critical 2: audit_queue.user_id FKs to auth.users, tighter
    # than messages.user_id itself (no FK — see core_tables.sql), so an
    # assistant message for a user_id that isn't in auth.users used to make
    # the sampling trigger's own insert raise a foreign-key violation that
    # aborted the whole `messages` insert with it (intermittently, since
    # sampling is `random() < 0.2`). The trigger now swallows any error and
    # always returns `new`, so this always succeeds and never samples.
    c = await _conn()
    try:
        owner = uuid.uuid4()
        await c.execute(
            "insert into auth.users (id,email) values ($1,$2)", owner, f"{owner}@x.test"
        )
        session_id = await c.fetchval(
            "insert into public.sessions (user_id) values ($1) returning id", owner
        )
        ghost = uuid.uuid4()  # never inserted into auth.users
        for _ in range(30):  # well past the 20% sample rate, to force at least one hit
            await c.execute(
                "insert into public.messages (session_id,user_id,role,content,guard_decision) "
                "values ($1,$2,'assistant','hi',null)",
                session_id,
                ghost,
            )
        count = await c.fetchval("select count(*) from public.messages where user_id = $1", ghost)
        assert count == 30
        sampled = await c.fetchval(
            "select count(*) from public.audit_queue where user_id = $1", ghost
        )
        assert sampled == 0
    finally:
        await c.close()


async def test_run_audit_cron_no_ops_when_url_and_token_are_unset() -> None:
    # Fix round 1, Important 2: `app.audit_run_url`/`app.internal_token` are
    # unset on every database this migration runs against until an operator
    # sets both — `run_audit_cron` must not call `net.http_post` (queueing a
    # request to nowhere, or with a null token) until then.
    c = await _conn()
    try:
        before = await c.fetchval("select count(*) from net.http_request_queue")
        await c.execute("select public.run_audit_cron()")
        after = await c.fetchval("select count(*) from net.http_request_queue")
        assert after == before
    finally:
        await c.close()


async def test_retention_memories_history_deletes_only_orphaned_expired_rows() -> None:
    # I2: the job collects history rows whose `memories` row is gone for real
    # (the user was deleted, or `rescreen_memories.py --delete` removed it) and
    # that are past the 30-day window. History for a memory that still exists —
    # live OR soft-deleted — is what "when did this fact change" is answered
    # from and must survive; so must a recent orphan.
    c = await _conn()
    try:
        owner = uuid.uuid4()
        await c.execute(
            "insert into auth.users (id,email) values ($1,$2)", owner, f"{owner}@x.test"
        )
        kept_id = await c.fetchval(
            "insert into public.memories (user_id,key,value,kind) "
            "values ($1,'k','v','fact') returning id",
            owner,
        )
        rows = {
            # (memory_id, age) -> should it survive?
            "live_old": (kept_id, "40 days", True),
            "orphan_old": (uuid.uuid4(), "40 days", False),
            "orphan_new": (uuid.uuid4(), "1 day", True),
        }
        ids = {}
        for name, (memory_id, age, _) in rows.items():
            ids[name] = await c.fetchval(
                "insert into public.memories_history (memory_id,user_id,old_value,new_value,"
                "action,at) values ($1,$2,null,'v','create',now() - ($3::text)::interval) "
                "returning id",
                memory_id,
                owner,
                age,
            )

        command = await c.fetchval(
            "select command from cron.job where jobname = 'retention_memories_history'"
        )
        assert command is not None
        await c.execute(command)

        for name, (_, _, survives) in rows.items():
            still_there = await c.fetchval(
                "select exists (select 1 from public.memories_history where id = $1)", ids[name]
            )
            assert still_there is survives, name
    finally:
        await c.close()


async def test_weather_cache_is_locked_down_for_authenticated() -> None:
    # M8: RLS-with-no-policies was the only thing standing between an
    # authenticated user and this table; 20260821000900 revokes the grants too,
    # matching every other server-only table (20260821000450).
    c = await _conn()
    try:
        async with c.transaction():
            await c.execute("set role authenticated")
            await c.execute(
                "select set_config('request.jwt.claims', $1, true)",
                f'{{"sub":"{uuid.uuid4()}","role":"authenticated"}}',
            )
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await c.execute("select * from public.weather_cache")
        await c.execute("reset role")
    finally:
        await c.close()


async def test_audit_queue_has_the_attempts_column_and_unprocessed_index() -> None:
    # I5/M7: `attempts` bounds the retry of an item the classifier permanently
    # fails on (`PgAuditRepo._CLAIM` filters `attempts < 3`), and the partial
    # index covers the claim query's one predicate without growing with the
    # processed archive.
    c = await _conn()
    try:
        col = await c.fetchrow(
            "select data_type, column_default, is_nullable from information_schema.columns "
            "where table_schema='public' and table_name='audit_queue' and column_name='attempts'"
        )
        assert col is not None
        assert col["data_type"] == "integer"
        assert col["is_nullable"] == "NO"
        assert col["column_default"] == "0"

        indexdef = await c.fetchval(
            "select indexdef from pg_indexes where schemaname='public' "
            "and indexname='audit_queue_unprocessed'"
        )
        assert indexdef is not None
        assert "processed_at" in indexdef
        assert "WHERE (processed_at IS NULL)" in indexdef
    finally:
        await c.close()


async def test_claim_skips_items_that_have_already_failed_three_times() -> None:
    # The DB half of the I5 fix: `PgAuditRepo._CLAIM`'s `attempts < 3` against
    # a real `audit_queue`, and `mark_failed` incrementing it.
    from sarjy.contexts.guardrails.infrastructure.pg_audit_repo import PgAuditRepo
    from sarjy.infrastructure_shared.db import Database

    db = Database(DB)
    await db.connect()
    try:
        owner = uuid.uuid4()
        await db.execute(
            "insert into auth.users (id,email) values ($1,$2)", owner, f"{owner}@x.test"
        )
        session_id = await db.fetchval(
            "insert into public.sessions (user_id) values ($1) returning id", owner
        )
        message_id = await db.fetchval(
            "insert into public.messages (session_id,user_id,role,content,guard_decision) "
            "values ($1,$2,'assistant','hi','block') returning id",  # 'block' → not sampled
            session_id,
            owner,
        )
        item_id = await db.fetchval(
            "insert into public.audit_queue (message_id,user_id) values ($1,$2) returning id",
            message_id,
            owner,
        )
        repo = PgAuditRepo(db)

        def mine(items: list[object]) -> list[object]:
            return [i for i in items if getattr(i, "id", None) == item_id]

        for expected in (1, 2, 3):
            assert mine(list(await repo.claim(100))), f"not claimable at attempt {expected}"
            await repo.mark_failed([item_id])
            attempts = await db.fetchval(
                "select attempts from public.audit_queue where id=$1", item_id
            )
            assert attempts == expected

        assert not mine(list(await repo.claim(100)))
    finally:
        await db.close()
