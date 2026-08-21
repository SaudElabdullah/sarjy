-- A run that was mid-question when the process restarted has to come back to
-- exactly where it was: `pending_confirmation` holds the low-confidence answer
-- awaiting a yes/no ("I'll put that as a 4 — right?"), and `resume_hint` marks
-- a run whose user wandered off-topic so the next prompt block nudges them back.
alter table public.workflow_runs
  add column if not exists pending_confirmation jsonb,
  add column if not exists resume_hint boolean not null default false;
