-- Server-only ops tables: the API writes these with the service role, which
-- bypasses RLS entirely. anon/authenticated must have no access at all.
revoke all on public.rate_limits, public.guardrail_events, public.telemetry_turns,
              public.memories_history, public.tool_calls from anon, authenticated;
drop policy if exists rate_limits_owner on public.rate_limits;
drop policy if exists guardrail_events_owner on public.guardrail_events;
drop policy if exists telemetry_turns_owner on public.telemetry_turns;
drop policy if exists memories_history_owner on public.memories_history;
drop policy if exists tool_calls_owner on public.tool_calls;
-- RLS stays enabled with no policies (deny-all); the API writes with the service role.

-- Foreign keys tying ops rows back to auth.users, so orphaned rows are cleaned
-- up (or nulled) when a user is deleted.
alter table public.tool_calls add constraint tool_calls_user_fk foreign key (user_id) references auth.users(id) on delete cascade;
alter table public.rate_limits add constraint rate_limits_user_fk foreign key (user_id) references auth.users(id) on delete cascade;
alter table public.guardrail_events add constraint guardrail_events_user_fk foreign key (user_id) references auth.users(id) on delete set null;
alter table public.telemetry_turns add constraint telemetry_turns_user_fk foreign key (user_id) references auth.users(id) on delete set null;
