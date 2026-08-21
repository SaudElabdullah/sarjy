create or replace view public.v_latency_daily as
select date_trunc('day', created_at) as day,
       client_info->>'mode' as mode,
       count(*) as turns,
       percentile_cont(0.5) within group (order by ttfa_ms) as ttfa_p50,
       percentile_cont(0.95) within group (order by ttfa_ms) as ttfa_p95,
       percentile_cont(0.5) within group (order by t_first_byte_ms) as first_byte_p50,
       percentile_cont(0.5) within group (order by (server_timings->>'t_gemini_first_token')::int) as gemini_first_token_p50,
       percentile_cont(0.5) within group (order by t_last_audio_ms) as turn_p50
from public.telemetry_turns where ttfa_ms is not null group by 1,2 order by 1 desc;

create or replace view public.v_latency_by_browser as
select case when client_info->>'ua' ilike '%chrome%' and client_info->>'ua' not ilike '%edg%' then 'chrome'
            when client_info->>'ua' ilike '%edg%' then 'edge'
            when client_info->>'ua' ilike '%safari%' then 'safari' else 'other' end as browser,
       count(*) turns,
       percentile_cont(0.5) within group (order by ttfa_ms) as ttfa_p50,
       percentile_cont(0.95) within group (order by ttfa_ms) as ttfa_p95
from public.telemetry_turns where created_at > now() - interval '7 days' group by 1;

create or replace view public.v_guard_daily as
select date_trunc('day', created_at) as day, layer, kind, action, count(*) n
from public.guardrail_events group by 1,2,3,4 order by 1 desc, 5 desc;

create or replace view public.v_ocean_funnel as
select date_trunc('day', started_at) as day,
       count(*) filter (where status in ('proposed','abandoned','active','paused','scoring','complete')) proposed,
       count(*) filter (where status in ('active','paused','scoring','complete')) started,
       count(*) filter (where status = 'complete') completed
from public.workflow_runs group by 1 order by 1 desc;

revoke all on public.v_latency_daily, public.v_latency_by_browser, public.v_guard_daily, public.v_ocean_funnel from anon, authenticated;
