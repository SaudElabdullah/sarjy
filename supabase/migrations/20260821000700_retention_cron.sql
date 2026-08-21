-- 20260821000700_retention_cron.sql  (PRD §11 retention; §13 alerts)
--
-- `select cron.unschedule(jobid) from cron.job where jobname = '…'` ahead of
-- every `cron.schedule` below makes this file idempotent: `supabase db reset`
-- re-applies every migration from scratch, but a staging/prod `db push` can
-- re-run a migration that already landed (a partial apply retried, a manual
-- re-run), and `cron.schedule` errors on a jobname that already exists rather
-- than upserting it. A `select ... where jobname = '…'` with no matching row
-- is a normal zero-row result, not an error, so this is safe on a database
-- that has never seen these jobs at all too.
select cron.unschedule(jobid) from cron.job where jobname = 'retention_messages';
select cron.schedule('retention_messages', '15 3 * * *',
  $$delete from public.messages where created_at < now() - interval '30 days'$$);

select cron.unschedule(jobid) from cron.job where jobname = 'retention_tool_calls';
select cron.schedule('retention_tool_calls', '20 3 * * *',
  $$delete from public.tool_calls where created_at < now() - interval '30 days'$$);

select cron.unschedule(jobid) from cron.job where jobname = 'retention_guard_events';
select cron.schedule('retention_guard_events', '25 3 * * *',
  $$delete from public.guardrail_events where created_at < now() - interval '180 days'$$);

select cron.unschedule(jobid) from cron.job where jobname = 'retention_telemetry';
select cron.schedule('retention_telemetry', '30 3 * * *',
  $$delete from public.telemetry_turns where created_at < now() - interval '90 days'$$);

select cron.unschedule(jobid) from cron.job where jobname = 'retention_weather_cache';
select cron.schedule('retention_weather_cache', '*/30 * * * *',
  $$delete from public.weather_cache where fetched_at < now() - interval '1 hour'$$);

select cron.unschedule(jobid) from cron.job where jobname = 'retention_rate_limits';
select cron.schedule('retention_rate_limits', '0 4 * * *',
  $$delete from public.rate_limits where window_start < now() - interval '2 days'$$);

select cron.unschedule(jobid) from cron.job where jobname = 'retention_sessions';
select cron.schedule('retention_sessions', '35 3 * * *',
  $$delete from public.sessions s where last_active_at < now() - interval '30 days' and not exists (select 1 from public.messages m where m.session_id = s.id)$$);

-- Alerts: evaluated every 15 min; POSTs to a webhook stored in vault.
--
-- Server-only ops table, same lockdown pattern as `20260821000450_ops_tables_
-- lockdown.sql`: RLS enabled, no policies, and grants explicitly revoked from
-- anon/authenticated on top of that — `fire_alert`/`check_alerts` are
-- `security definer` and run as the function owner, which bypasses RLS
-- entirely, so nothing but those functions (and the service role) ever
-- touches this table.
--
-- Both functions also get `set search_path = public` (the `load_turn_context`
-- pattern from `20260821000400_rpc_load_turn_context.sql`/`...000610`) and an
-- explicit `revoke ... from public, anon, authenticated` on top of the
-- default EXECUTE grant Supabase gives every new function: a `security
-- definer` function with an unpinned search_path is a classic privilege-
-- escalation hole (a caller with `CREATE` on some schema earlier in their own
-- `search_path` can shadow an unqualified identifier the function resolves at
-- call time), and one callable by anon/authenticated defeats the point of the
-- lockdown above — either role could invoke `fire_alert`/`check_alerts`
-- directly today, running as their owner, RLS or no RLS.
create table if not exists public.alert_state (key text primary key, last_fired timestamptz);
alter table public.alert_state enable row level security;
revoke all on public.alert_state from anon, authenticated;

create or replace function public.fire_alert(p_key text, p_payload jsonb) returns void
language plpgsql security definer set search_path = public as $$
declare url text := current_setting('app.alert_webhook', true);
begin
  if url is null then return; end if;
  if exists (select 1 from public.alert_state where key = p_key and last_fired > now() - interval '1 hour') then return; end if;
  perform net.http_post(url := url, body := p_payload, headers := '{"Content-Type":"application/json"}'::jsonb);
  insert into public.alert_state (key, last_fired) values (p_key, now()) on conflict (key) do update set last_fired = now();
end $$;

create or replace function public.check_alerts() returns void
language plpgsql security definer set search_path = public as $$
declare p95 numeric; err_rate numeric; sev3 int;
begin
  select percentile_cont(0.95) within group (order by ttfa_ms) into p95
    from public.telemetry_turns where created_at > now() - interval '15 minutes';
  if p95 > 2000 then perform public.fire_alert('ttfa_p95', jsonb_build_object('text', format('Sarjy TTFA p95 = %s ms over last 15 min', p95))); end if;

  select count(*) into sev3 from public.guardrail_events where severity >= 3 and created_at > now() - interval '15 minutes';
  if sev3 > 0 then perform public.fire_alert('guard_sev3', jsonb_build_object('text', format('%s severity-3 guardrail events in last 15 min', sev3))); end if;

  select coalesce(avg(case when guard_decision like 'error:%' then 1 else 0 end), 0) into err_rate
    from public.messages where role = 'assistant' and created_at > now() - interval '15 minutes';
  if err_rate > 0.02 then perform public.fire_alert('error_rate', jsonb_build_object('text', format('Sarjy error rate %s%%', round(err_rate*100,1)))); end if;
end $$;

revoke all on function public.fire_alert(text, jsonb), public.check_alerts()
  from public, anon, authenticated;

select cron.unschedule(jobid) from cron.job where jobname = 'check_alerts';
select cron.schedule('check_alerts', '*/15 * * * *', $$select public.check_alerts()$$);

-- Set the webhook once per project (not committed — see the runbook):
--   alter database postgres set app.alert_webhook = 'https://hooks.slack.com/services/...';
-- Left unset here: `fire_alert` treats a null `app.alert_webhook` as a no-op,
-- which is exactly what every environment without that `alter database` — every
-- local/dev/test database — should do.
