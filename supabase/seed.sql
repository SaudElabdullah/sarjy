-- Seeds the Mini-IPIP Big Five instrument (supabase/mini_ipip.json, verbatim).
-- Re-run safe: `supabase db reset` replays this after every migration.
insert into public.workflow_definitions (id, version, definition, active)
values ('ocean_mini_ipip', 1, $$
{
  "id": "ocean_mini_ipip",
  "version": 1,
  "scale": {"min":1,"max":5,"labels":["Very inaccurate","Moderately inaccurate","Neither","Moderately accurate","Very accurate"]},
  "traits": {"O":"Openness","C":"Conscientiousness","E":"Extraversion","A":"Agreeableness","N":"Neuroticism"},
  "bands": {"low":[1,2.4],"moderate":[2.5,3.5],"high":[3.6,5]},
  "items": [
    {"no":1,"trait":"E","reverse":false,"text":"I am the life of the party."},
    {"no":2,"trait":"A","reverse":false,"text":"I sympathize with others' feelings."},
    {"no":3,"trait":"C","reverse":false,"text":"I get chores done right away."},
    {"no":4,"trait":"N","reverse":false,"text":"I have frequent mood swings."},
    {"no":5,"trait":"O","reverse":false,"text":"I have a vivid imagination."},
    {"no":6,"trait":"E","reverse":true, "text":"I don't talk a lot."},
    {"no":7,"trait":"A","reverse":true, "text":"I am not interested in other people's problems."},
    {"no":8,"trait":"C","reverse":true, "text":"I often forget to put things back in their proper place."},
    {"no":9,"trait":"N","reverse":true, "text":"I am relaxed most of the time."},
    {"no":10,"trait":"O","reverse":true,"text":"I am not interested in abstract ideas."},
    {"no":11,"trait":"E","reverse":false,"text":"I talk to a lot of different people at parties."},
    {"no":12,"trait":"A","reverse":false,"text":"I feel others' emotions."},
    {"no":13,"trait":"C","reverse":false,"text":"I like order."},
    {"no":14,"trait":"N","reverse":false,"text":"I get upset easily."},
    {"no":15,"trait":"O","reverse":true, "text":"I have difficulty understanding abstract ideas."},
    {"no":16,"trait":"E","reverse":true, "text":"I keep in the background."},
    {"no":17,"trait":"A","reverse":true, "text":"I am not really interested in others."},
    {"no":18,"trait":"C","reverse":true, "text":"I make a mess of things."},
    {"no":19,"trait":"N","reverse":true, "text":"I seldom feel blue."},
    {"no":20,"trait":"O","reverse":true, "text":"I do not have a good imagination."}
  ],
  "scoring": "trait = mean(value if !reverse else 6 - value) over answered items; require >= 3 answered per trait"
}
$$::jsonb, true)
on conflict (id) do update set
  definition = excluded.definition,
  version = excluded.version,
  active = excluded.active;
