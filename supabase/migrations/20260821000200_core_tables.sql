-- Users
create table public.profiles (
  user_id         uuid primary key references auth.users(id) on delete cascade,
  display_name    text,
  preferred_voice text,
  units           text check (units in ('metric','imperial')),
  locale          text default 'en-US',
  continuous_mode boolean default false,
  created_at      timestamptz default now(),
  updated_at      timestamptz default now()
);

-- Memory
create type public.memory_kind as enum ('fact','preference','person','place','note');

create table public.memories (
  id                uuid primary key default gen_random_uuid(),
  user_id           uuid not null references auth.users(id) on delete cascade,
  key               text,
  value             text not null check (char_length(value) <= 200),
  kind              public.memory_kind not null default 'fact',
  embedding         extensions.vector(768),
  source_message_id uuid,
  created_at        timestamptz default now(),
  updated_at        timestamptz default now(),
  deleted_at        timestamptz
);
create unique index memories_user_key_live on public.memories (user_id, key) where deleted_at is null and key is not null;
create index memories_user_live on public.memories (user_id) where deleted_at is null;

create table public.memories_history (
  id         bigserial primary key,
  memory_id  uuid not null,
  user_id    uuid not null,
  old_value  text,
  new_value  text,
  action     text check (action in ('create','update','delete')),
  at         timestamptz default now()
);

-- Conversation
create table public.sessions (
  id             uuid primary key default gen_random_uuid(),
  user_id        uuid not null references auth.users(id) on delete cascade,
  started_at     timestamptz default now(),
  last_active_at timestamptz default now(),
  summary        text,
  client_info    jsonb
);
create index sessions_user on public.sessions (user_id, last_active_at desc);

create table public.messages (
  id             uuid primary key default gen_random_uuid(),
  session_id     uuid not null references public.sessions(id) on delete cascade,
  user_id        uuid not null,
  role           text not null check (role in ('user','assistant','tool','system_event')),
  content        text not null,
  speech_content text,
  client_turn_id text,
  guard_decision text,
  timings        jsonb,
  prompt_hash    text,
  created_at     timestamptz default now()
);
create unique index messages_user_turn on public.messages (user_id, client_turn_id, role) where client_turn_id is not null;
create index messages_session on public.messages (session_id, created_at);

-- Tools
create table public.tool_calls (
  id         uuid primary key default gen_random_uuid(),
  message_id uuid references public.messages(id) on delete cascade,
  user_id    uuid not null,
  tool_name  text not null,
  args       jsonb not null,
  result     jsonb,
  status     text check (status in ('ok','error','timeout')),
  latency_ms int,
  created_at timestamptz default now()
);

create table public.weather_cache (
  cache_key  text primary key,
  payload    jsonb not null,
  fetched_at timestamptz default now()
);

-- Workflows
create table public.workflow_definitions (
  id         text primary key,
  version    int not null,
  definition jsonb not null,
  active     boolean default true
);

create type public.workflow_status as enum ('proposed','active','paused','scoring','complete','abandoned');

create table public.workflow_runs (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid not null references auth.users(id) on delete cascade,
  definition_id      text not null references public.workflow_definitions(id),
  definition_version int not null,
  status             public.workflow_status not null default 'proposed',
  current_item       int not null default 1,
  skips_used         int not null default 0,
  results            jsonb,
  narrative          text,
  started_at         timestamptz default now(),
  updated_at         timestamptz default now(),
  completed_at       timestamptz
);
create index workflow_runs_user_status on public.workflow_runs (user_id, status);

create table public.workflow_answers (
  run_id      uuid not null references public.workflow_runs(id) on delete cascade,
  item_no     int not null,
  raw_text    text,
  value       int check (value between 1 and 5),
  confidence  real,
  answered_at timestamptz default now(),
  primary key (run_id, item_no)
);

-- Guardrails & ops
create table public.guardrail_events (
  id         bigserial primary key,
  user_id    uuid,
  message_id uuid,
  layer      smallint not null,
  kind       text not null,
  action     text not null,
  severity   smallint default 1,
  detail     jsonb,
  created_at timestamptz default now()
);
create index guardrail_events_created on public.guardrail_events (created_at desc);

create table public.rate_limits (
  user_id      uuid not null,
  window_start timestamptz not null,
  count        int not null default 0,
  primary key (user_id, window_start)
);

create table public.telemetry_turns (
  id                  bigserial primary key,
  user_id             uuid,
  message_id          uuid,
  ttfa_ms             int,
  t_request_ms        int,
  t_first_byte_ms     int,
  t_first_sentence_ms int,
  t_last_audio_ms     int,
  server_timings      jsonb,
  client_info         jsonb,
  created_at          timestamptz default now()
);
create index telemetry_created on public.telemetry_turns (created_at desc);

-- Auto-create profile on signup
create or replace function public.handle_new_user() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (user_id) values (new.id) on conflict do nothing;
  return new;
end; $$;
create trigger on_auth_user_created after insert on auth.users
  for each row execute function public.handle_new_user();
