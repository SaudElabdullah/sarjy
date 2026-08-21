-- load_turn_context v2 (PRD L-7): one round trip is now the ONLY context read a
-- turn makes, so what it returns has to be everything `RunTurn` used to fetch for
-- itself — and nothing it would have to fetch again.
--
-- Four changes over v1:
--
-- 1. `history` excludes rows the input guard blocked (`guard_decision like
--    'block:%'`). `RunTurn` already dropped them when building the prompt (I6/R4:
--    a refused injection must not be quietly re-delivered every subsequent turn),
--    but it dropped them AFTER paying to ship them over the wire and AFTER they
--    had eaten slots in the `limit`. Filtering in SQL means the limit buys twelve
--    usable turns rather than twelve rows of which some are unusable. The
--    application-side filter stays as a defensive second pass.
--
-- 2. `history` orders by `(created_at, id)`, not `created_at` alone. Two rows
--    written inside the same millisecond (a user row and its refusal) otherwise
--    come back in an arbitrary order, and a `model` turn ahead of the `user` turn
--    it answers is a prompt Gemini rejects outright. `id` is the same tiebreaker
--    `PgMessageRepo.history` uses, so both readers agree.
--
-- 3. `history` carries `guard_decision` and `client_turn_id`. Without the former
--    the defensive filter in (1) has nothing to filter on; the latter is what a
--    resumed client uses to match a stored turn to one it already spoke.
--
-- 4. New key `session`: the row for `p_session` itself. `RunTurn` opened every
--    turn by reading it (ownership and expiry are decided in code, not in SQL —
--    a foreign or expired id must start a NEW session rather than error), which
--    was the one read left outside this call. `null` when the id is unknown,
--    which the caller reads as "start a fresh one".
--
-- 5. New key `last_results`: the user's most recent COMPLETE run, and only when
--    no run is open. That is the grounding for follow-up Q&A about a finished
--    Big Five test ("how did I score on openness?") — see PRD P-9/P-11. Without
--    it the model is answering from a transcript that has already scrolled out of
--    the history window, which is exactly how a confident, invented "4.9" gets
--    spoken. `null` whenever a run is open, because then the live `workflow`
--    block is the truth and last time's numbers are noise.
--
-- The open-status list also gains `scoring` here, matching
-- `pg_run_repo.OPEN_STATUSES` and `MemRunRepo.OPEN`. v1 omitted it, which was
-- harmless while `RunTurn` read the workflow through `ActiveRunPort.active_run`
-- (which uses `get_open`, and does include it) — but this RPC is now that read,
-- and a run stranded mid-scoring must still produce its prompt block (I1/I2)
-- rather than silently looking like no run at all.
create or replace function public.load_turn_context(p_user uuid, p_session uuid, p_history_limit int default 12)
returns jsonb language sql stable security definer set search_path = public as $$
  select jsonb_build_object(
    'memories', (select coalesce(jsonb_agg(jsonb_build_object('k',key,'v',value,'kind',kind) order by updated_at desc),'[]'::jsonb)
                 from (select key, value, kind, updated_at from memories
                       where user_id = p_user and deleted_at is null and kind <> 'note'
                       order by updated_at desc limit 60) m),
    'history',  (select coalesce(jsonb_agg(jsonb_build_object(
                          'role',role,'content',content,
                          'guard_decision',guard_decision,'client_turn_id',client_turn_id)
                        order by created_at, id),'[]'::jsonb)
                 from (select role, content, guard_decision, client_turn_id, created_at, id
                       from messages
                       where session_id = p_session and user_id = p_user
                         and role in ('user','assistant')
                         and (guard_decision is null or guard_decision not like 'block:%')
                       order by created_at desc, id desc limit p_history_limit) h),
    'workflow', (select to_jsonb(w) from workflow_runs w where user_id = p_user
                 and status in ('proposed','active','paused','scoring')
                 order by updated_at desc limit 1),
    'profile',  (select to_jsonb(p) from profiles p where user_id = p_user),
    'session',  (select jsonb_build_object('id',id,'user_id',user_id,'started_at',started_at,
                          'last_active_at',last_active_at,'summary',summary)
                 from sessions where id = p_session),
    'last_results', (
      select to_jsonb(r)
      from (select results, narrative, completed_at from workflow_runs
            where user_id = p_user and status = 'complete' and results is not null
            order by completed_at desc nulls last limit 1) r
      where not exists (select 1 from workflow_runs o
                        where o.user_id = p_user
                          and o.status in ('proposed','active','paused','scoring')))
  );
$$;
revoke all on function public.load_turn_context(uuid, uuid, int) from public, anon, authenticated;
