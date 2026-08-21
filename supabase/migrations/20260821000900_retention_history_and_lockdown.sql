-- 20260821000900_retention_history_and_lockdown.sql
--   (Phase 8 whole-phase review: I2/I5/M6/M7/M8)
--
-- Four unrelated-but-small gaps left by the earlier ops migrations, batched
-- into one file because each is a couple of statements:
--
--   1. `memories_history` had no retention job at all (I2). Every other
--      user-text-bearing table got one in `20260821000700_retention_cron.sql`;
--      this one kept `old_value`/`new_value` — the literal remembered facts —
--      forever.
--   2. `handle_new_user()` was the one `security definer` function the
--      round-1 lockdown (`20260821000700`, `20260821000800`) missed (M6).
--   3. `weather_cache` had RLS enabled with no policies but never had its
--      table grants revoked the way `20260821000450_ops_tables_lockdown.sql`
--      revoked them for every other server-only table (M8).
--   4. `audit_queue` had no failure accounting and no index for the one
--      predicate `PgAuditRepo._CLAIM` filters on (I5/M7).

-- 1. Retention for `memories_history` (I2) ---------------------------------
--
-- Scoped to history rows whose `memory_id` no longer names a row in
-- `memories`: while the memory itself still exists (live OR soft-deleted —
-- `memories` rows are soft-deleted, `deleted_at` set, not removed), its audit
-- trail is what "when did this fact change" is answered from, and PRD §11's
-- 30-day clock is about text we no longer have a reason to hold. Once the
-- `memories` row is gone for real — the user was deleted (FK cascade) or
-- `scripts/rescreen_memories.py --delete` removed it — the history rows are
-- orphaned user text and nothing reads them again.
--
-- `at` (not `created_at`) is this table's timestamp column; see
-- `20260821000200_core_tables.sql`.
--
-- `not exists` rather than a left join for the same reason the other
-- retention jobs are plain deletes: this runs as a cron statement with no
-- plan tuning available, and the anti-join is what Postgres picks anyway.
--
-- `unschedule`-before-`schedule` for the idempotency reason spelled out at
-- the top of `20260821000700_retention_cron.sql`.
select cron.unschedule(jobid) from cron.job where jobname = 'retention_memories_history';
select cron.schedule('retention_memories_history', '40 3 * * *',
  $$delete from public.memories_history h
     where h.at < now() - interval '30 days'
       and not exists (select 1 from public.memories m where m.id = h.memory_id)$$);

-- 2. `handle_new_user()` lockdown (M6) -------------------------------------
--
-- The profile-creation trigger function from `20260821000200_core_tables.sql`:
-- `security definer`, `set search_path = public` (so the escalation half was
-- already covered), but it kept Postgres's default EXECUTE grant to `public`.
-- It is only ever meant to run from the `on_auth_user_created` trigger —
-- revoking EXECUTE does not affect trigger invocation, which runs as the
-- table owner, not as the caller. Same treatment `fire_alert`/`check_alerts`/
-- `enqueue_audit_sample`/`run_audit_cron` got in the round-1 lockdown.
revoke execute on function public.handle_new_user() from public, anon, authenticated;

-- 3. `weather_cache` grants (M8) -------------------------------------------
--
-- `20260821000300_rls.sql` enabled RLS with no policies ("service role only"),
-- which is deny-all for anon/authenticated *as long as RLS stays on*. The
-- other server-only tables got belt-and-braces `revoke all` in
-- `20260821000450_ops_tables_lockdown.sql`; this one was missed. Matching it
-- means a future `alter table ... disable row level security` (or a table
-- rebuild that loses it) does not silently open the cache to every signed-in
-- user.
revoke all on public.weather_cache from anon, authenticated;

-- 4. `audit_queue` failure accounting + claim index (I5/M7) -----------------
--
-- `attempts`: `AuditWorker` leaves an item unmarked when the classifier fails,
-- so `claim` hands it out again on the next run — correct for a transient
-- failure, an infinite retry loop for an item that fails every time (a
-- message the classifier chokes on permanently). The worker now increments
-- this via `mark_failed`, and `PgAuditRepo._CLAIM` filters `attempts < 3`, so
-- a permanently-failing item drops out of the queue after three tries instead
-- of being re-fetched (and re-charged to the classifier) every ten minutes
-- forever. Rows that predate this column start at 0, which is the same as a
-- fresh row — nothing needs backfilling.
--
-- `if not exists` so a re-run of this migration on a database that already
-- took it (a retried `db push`) is a no-op rather than an error, matching the
-- `create table if not exists` / `unschedule`-first idempotency the other ops
-- migrations are written for.
alter table public.audit_queue add column if not exists attempts int not null default 0;

-- The claim query's one predicate is `processed_at is null`, over a table
-- whose processed rows accumulate (nothing deletes them) — so an unpartial
-- index would grow with the archive while only the unprocessed tail is ever
-- read. Partial keeps it proportional to the backlog instead. `attempts` is
-- deliberately not in the index: with the queue drained to a handful of rows,
-- the extra filter is cheaper applied on the heap than carried in the index.
create index if not exists audit_queue_unprocessed
  on public.audit_queue (processed_at) where processed_at is null;
