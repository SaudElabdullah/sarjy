create or replace function public.load_turn_context(p_user uuid, p_session uuid, p_history_limit int default 12)
returns jsonb language sql stable security definer set search_path = public as $$
  select jsonb_build_object(
    'memories', (select coalesce(jsonb_agg(jsonb_build_object('k',key,'v',value,'kind',kind) order by updated_at desc),'[]'::jsonb)
                 from (select key, value, kind, updated_at from memories
                       where user_id = p_user and deleted_at is null and kind <> 'note'
                       order by updated_at desc limit 60) m),
    'history',  (select coalesce(jsonb_agg(jsonb_build_object('role',role,'content',content) order by created_at),'[]'::jsonb)
                 from (select role, content, created_at from messages
                       where session_id = p_session and user_id = p_user and role in ('user','assistant')
                       order by created_at desc limit p_history_limit) h),
    'workflow', (select to_jsonb(w) from workflow_runs w where user_id = p_user
                 and status in ('proposed','active','paused') order by updated_at desc limit 1),
    'profile',  (select to_jsonb(p) from profiles p where user_id = p_user)
  );
$$;
revoke all on function public.load_turn_context(uuid, uuid, int) from public, anon, authenticated;
