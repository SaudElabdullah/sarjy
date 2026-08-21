do $$
declare t text;
begin
  foreach t in array array['profiles','memories','memories_history','sessions','messages','tool_calls',
                           'workflow_runs','guardrail_events','rate_limits','telemetry_turns']
  loop
    execute format('alter table public.%I enable row level security', t);
    execute format('create policy %I_owner on public.%I for all using (auth.uid() = user_id) with check (auth.uid() = user_id)', t, t);
  end loop;
end $$;

-- workflow_answers has no user_id column; scope through run
alter table public.workflow_answers enable row level security;
create policy workflow_answers_owner on public.workflow_answers for all
  using (exists (select 1 from public.workflow_runs r where r.id = run_id and r.user_id = auth.uid()))
  with check (exists (select 1 from public.workflow_runs r where r.id = run_id and r.user_id = auth.uid()));

-- Read-only shared tables
alter table public.workflow_definitions enable row level security;
create policy workflow_definitions_read on public.workflow_definitions for select using (true);
alter table public.weather_cache enable row level security;   -- service role only; no policies
