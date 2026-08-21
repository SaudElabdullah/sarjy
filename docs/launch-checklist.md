# Sarjy launch checklist

PRD §2.3 gates + §11 + acceptance tests. See `docs/runbook.md` for the
procedures each item below refers to.

## Functional acceptance (run on prod URL, Chrome desktop + Chrome Android)
- [ ] V: mic permission → weather question → interrupt mid-sentence → transcript correct; Firefox shows text fallback
- [ ] M: favorite color teal → new session next day → recalled; forget → "I don't have that stored"
- [ ] W: Reykjavik matches tool_calls payload; Gondor → not found; provider down → "can't reach the weather service"
- [ ] P: 7 items with mixed phrasings, pause, new tab, resume, finish; scores match hand calculation from workflow_answers
- [ ] G: four PRD §7.6 scripts refused; guardrail_events rows present with expected layer/kind
- [ ] L: tests/latency/measure.py against prod, n=50 → p50 ≤ 800 ms, p95 ≤ 1500 ms (paste numbers)

## Quality gates (paste numbers from the latest evals artifact)
- [ ] red-team ≥ 99 %   - [ ] benign FP ≤ 2 %   - [ ] memory ≥ 95 %   - [ ] weather 100 %   - [ ] OCEAN 100 %

## Security & privacy
- [ ] Secret scan run locally: `git grep -nE 'AIz[a]|eyJhbG[c]i' -- ':!tests' ':!*.md' ':!src/sarjy/interfaces/web/static/supabase.js'` is empty
- [ ] All 7 required Fly secrets set on staging AND prod (`fly secrets list -c fly.{staging,prod}.toml`): `GEMINI_API_KEY SUPABASE_URL SUPABASE_ANON_KEY SUPABASE_SERVICE_ROLE_KEY SUPABASE_JWT_SECRET DATABASE_URL DATABASE_URL_DIRECT` — the app refuses to start without any one of them
- [ ] Deployment-shaped Fly settings also set: `CORS_ORIGINS` (includes the Storage origin), `PUBLIC_API_BASE` (this app's absolute URL), `ADMIN_USER_IDS`, `INTERNAL_TOKEN`, `TURNSTILE_SITE_KEY` — each has a default that is wrong or inert in production
- [ ] RLS verified on prod: `tests/integration/test_db_rls.py` run against prod DB (read-only user)
- [ ] CSP/HSTS headers present on prod (curl -I)
- [ ] Anonymous sign-in captcha enabled; SMTP configured; Site URL correct
- [ ] `TURNSTILE_SITE_KEY` set on Fly (staging + prod) and as a shell env var for `make release-*`; `TURNSTILE_TOKEN` set as a shell env var for `make release-*`/`make evals-staging` (Cloudflare's always-pass dummy token, for smoke/evals) — see runbook "Turnstile captcha"
- [ ] DELETE /account works on prod with a throwaway user
- [ ] Retention crons listed in `select * from cron.job;`
- [ ] Gemini project on paid tier (no-training) — screenshot in ticket
- [ ] `INTERNAL_TOKEN` set on Fly and `app.internal_token`/`app.audit_run_url`/`app.alert_webhook` set on the DB

## Ops
- [ ] Alerts webhook configured (`app.alert_webhook`) and test alert fired
- [ ] Fly: min_machines_running=2, health checks green, rolling strategy
- [ ] Runbook reviewed by a second engineer
- [ ] Budget alert set in Google Cloud billing for the Gemini key
- [ ] `make evals-staging` numbers pasted below

Signed off by: ____________   Date: ________
