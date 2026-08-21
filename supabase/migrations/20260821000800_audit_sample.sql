-- 20260821000800_audit_sample.sql  (PRD Layer 7: 20% sample of allowed turns for async audit)
--
-- Server-only ops table, same lockdown pattern as `20260821000450_ops_tables_
-- lockdown.sql` and `alert_state` above: RLS enabled, no policies, grants
-- revoked from anon/authenticated. `enqueue_audit_sample` is `security
-- definer` so it can insert here regardless of who/what triggered the
-- `messages` insert it fires on (RLS on `audit_queue` would not otherwise
-- block the service role, which is what the API writes `messages` as, but
-- making the trigger function security definer is the explicit, not-
-- incidental, reason inserts always succeed). The FKs mirror
-- `tool_calls`/`rate_limits` in `20260821000450`: an audit row is worthless
-- once the message or the user it refers to is gone, so both cascade.
create table if not exists public.audit_queue (
  id bigserial primary key,
  message_id uuid not null references public.messages(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz default now(),
  processed_at timestamptz
);
alter table public.audit_queue enable row level security;
revoke all on public.audit_queue from anon, authenticated;

-- `enqueue_audit_sample` runs AFTER INSERT on `messages`, on the same row and
-- in the same transaction as the chat write it's sampling — so any error it
-- raises aborts that write too. `audit_queue.user_id` FKs to `auth.users`,
-- which is a tighter constraint than `messages.user_id` itself carries (that
-- column is bare `uuid not null`, no FK — see `20260821000200_core_tables.
-- sql`), so a `messages` row whose `user_id` is not (yet, or ever) a real
-- `auth.users` row inserts fine on its own but raises a foreign-key violation
-- the moment this trigger tries to mirror it into `audit_queue`. Combined
-- with `random() < 0.2` sampling, that failure is intermittent — most such
-- rows insert fine, one in five (of the ones the sample would have taken)
-- takes the whole chat turn down with it. The `begin ... exception when
-- others then return new; end` below is a PL/pgSQL sub-transaction: it
-- catches ANY error the sampling logic raises (this one and any other) and
-- returns `new` regardless, so a broken or unlucky audit sample can only ever
-- cost an audit row, never the write it's riding on. The FKs stay — losing an
-- audit sample silently is fine; a `messages` row it's not fine to lose.
create or replace function public.enqueue_audit_sample() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  if new.role = 'assistant' and (new.guard_decision is null or new.guard_decision = 'allow') and random() < 0.2 then
    insert into public.audit_queue (message_id, user_id) values (new.id, new.user_id);
  end if;
  return new;
exception when others then
  return new;
end $$;

revoke all on function public.enqueue_audit_sample() from public, anon, authenticated;

drop trigger if exists messages_audit_sample on public.messages;
create trigger messages_audit_sample after insert on public.messages
  for each row execute function public.enqueue_audit_sample();

-- Scheduled sampling run: POSTs to the internal audit endpoint (`POST
-- /internal/audit/run`, `src/sarjy/interfaces/http/internal.py`),
-- authenticated with a shared secret compared via `hmac.compare_digest`
-- (`Settings.internal_token`).
--
-- Both the URL and the token come from GUCs, read inside `run_audit_cron`
-- rather than embedded in the cron job's command text: the job itself
-- (`cron.job.command`) is the same on every database this migration runs
-- against, so a hardcoded prod URL in it would mean every environment —
-- local, staging, a developer's laptop — schedules a job that, the moment
-- someone DOES set `app.internal_token` for unrelated reasons, starts
-- POSTing to production every 10 minutes. Routing through a function keyed
-- on two GUCs means the job is inert (no `net.http_post` call at all, not
-- even to a wrong/empty target) until BOTH are explicitly set for that
-- database:
--   alter database postgres set app.audit_run_url = 'https://sarjy-prod.fly.dev/internal/audit/run';
--   alter database postgres set app.internal_token = '...';
-- `security definer set search_path = public`, and EXECUTE revoked from
-- public/anon/authenticated, for the same reasons as `fire_alert`/
-- `check_alerts` (see `20260821000700_retention_cron.sql`).
create or replace function public.run_audit_cron() returns void
language plpgsql security definer set search_path = public as $$
declare
  url   text := current_setting('app.audit_run_url', true);
  token text := current_setting('app.internal_token', true);
begin
  if url is null or url = '' or token is null or token = '' then
    return;
  end if;
  perform net.http_post(url, headers := jsonb_build_object('X-Internal-Token', token));
end $$;

revoke all on function public.run_audit_cron() from public, anon, authenticated;

select cron.unschedule(jobid) from cron.job where jobname = 'audit';
select cron.schedule('audit', '*/10 * * * *', $$select public.run_audit_cron()$$);
