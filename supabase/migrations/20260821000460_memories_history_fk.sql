-- Phase 1 residual: memories_history was left out of the ops-tables FK
-- lockdown (20260821000450), so orphaned rows survived user deletion.
alter table public.memories_history add constraint memories_history_user_fk foreign key (user_id) references auth.users(id) on delete cascade;
