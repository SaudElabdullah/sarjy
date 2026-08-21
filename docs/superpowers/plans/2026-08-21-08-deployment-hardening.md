# Phase 8 — Deployment, CI/CD & Launch Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the staging system to a production-grade deployment: separate prod Supabase + Fly apps, static client served from Supabase Storage CDN, staging→prod pipeline with eval gates and manual approval, data retention and audit crons, alerting, security headers, a user data-deletion path, and a signed launch checklist.

**Architecture:** Two environments (`staging`, `prod`), each = one Supabase project + one Fly app. The FastAPI container serves the API; the static client (`index.html`, JS, CSS) is uploaded to a public Supabase Storage bucket and served from the Supabase CDN, pointing at the Fly API via `apiBase`. GitHub Actions: `ci` (every PR) → `deploy-staging` (merge to main) → `evals` → `deploy-prod` (manual approval environment).

**Tech Stack:** GitHub Actions environments, flyctl, Supabase CLI, Supabase Storage, pg_cron + pg_net, Fly checks/metrics.

**Spec:** `PRD.md` §11, §12, §13, §14, §15, §16 M5.

## Global Constraints

- Secrets only in Fly secrets / GitHub environment secrets; never in repo or Docker image.
- Transcript retention 30 days; guardrail events 180 days; telemetry raw 90 days (PRD §11).
- Prod deploy requires: CI green, staging evals pass gates, manual approval.
- CSP allows only `self`, the Supabase project origin, and `cdn.jsdelivr.net` (supabase-js) — or vendor supabase-js into `/static` to drop the CDN (do this; see Task 3).

---

## File structure

```
.github/workflows/ci.yml            (extended)
.github/workflows/deploy.yml        (new)
fly.staging.toml, fly.prod.toml     (rename from fly.toml)
scripts/upload_static.py            # uploads interfaces/web/static + rendered index.html to Storage
scripts/smoke.py                    # post-deploy smoke (health, anon signup, one /chat turn)
supabase/migrations/20260821000700_retention_cron.sql
supabase/migrations/20260821000800_audit_sample.sql
src/sarjy/interfaces/http/security.py   # headers middleware
src/sarjy/interfaces/http/account.py    # DELETE /account (data deletion), GET /account/export (P2 stub returns 501)
docs/runbook.md
docs/launch-checklist.md
```

---

### Task 1: Production environment provisioning

- [ ] **Supabase prod project**: create `sarjy-prod` in the same region as staging's Fly region. Enable anonymous sign-ins, `vector`, `pg_cron`, `pg_net`. Auth → Bot protection → Cloudflare Turnstile with `TURNSTILE_SECRET_KEY`. Auth → SMTP → configure Resend (or other) so magic links are not rate-limited. Auth → URL configuration → Site URL = prod web URL, redirect URLs include it.
- [ ] `supabase link --project-ref <prod-ref> && supabase db push` (all migrations), then `psql "$DATABASE_URL_DIRECT" -f supabase/seed.sql`.
- [ ] **Fly prod app**: `cp fly.toml fly.prod.toml`, set `app = "sarjy-prod"`, `APP_ENV = "prod"`, `[[vm]] memory = "1gb"`, and `min_machines_running = 2` in two regions? Keep one region (DB locality) with `min_machines_running = 2` for zero-downtime deploys (`[deploy] strategy = "rolling"`). `fly apps create sarjy-prod`, `fly secrets set …` (prod keys, `GUARD_MODE=enforce`, `SPECULATIVE_ENABLED=true`, `CORS_ORIGINS=https://<prod-web-origin>`).
- [ ] Rename the staging file to `fly.staging.toml`; update `Makefile` with `deploy-staging: fly deploy -c fly.staging.toml` and `deploy-prod: fly deploy -c fly.prod.toml`.
- [ ] Commit `chore(deploy): prod environment configs`.

---

### Task 2: Security headers and account deletion

**Files:**
- Create: `src/sarjy/interfaces/http/security.py`, `src/sarjy/interfaces/http/account.py`
- Test: `tests/unit/interfaces/test_security_headers.py`, `tests/integration/test_account_delete.py`

- [ ] **Step 1: Failing tests**

```python
# tests/unit/interfaces/test_security_headers.py
from fastapi.testclient import TestClient
from sarjy.config import Settings
from sarjy.main import create_app


def test_headers_present() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.headers["strict-transport-security"].startswith("max-age=")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "microphone=(self)" in r.headers["permissions-policy"]
```

```python
# tests/integration/test_account_delete.py
import os, time, uuid
import jwt, pytest
from httpx import ASGITransport, AsyncClient
from asgi_lifespan import LifespanManager
from sarjy.config import Settings
from sarjy.main import create_app

pytestmark = pytest.mark.integration
SECRET = os.environ["SUPABASE_JWT_SECRET"]


async def test_delete_account_cascades() -> None:
    app = create_app(Settings())
    u = uuid.uuid4()
    async with LifespanManager(app):
        db = app.state.container.db
        await db.execute("insert into auth.users (id,email) values ($1,$2)", u, f"{u}@x.test")
        await db.execute("insert into memories (user_id,key,value) values ($1,'k','v')", u)
        tok = jwt.encode({"sub": str(u), "aud": "authenticated", "exp": int(time.time()) + 60}, SECRET, algorithm="HS256")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.delete("/account", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 204
        assert await db.fetchval("select count(*) from memories where user_id=$1", u) == 0
        assert await db.fetchval("select count(*) from auth.users where id=$1", u) == 0
```

- [ ] **Step 2–3: Implement**

```python
# src/sarjy/interfaces/http/security.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, supabase_origin: str) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.csp = ("default-src 'self'; "
                    f"connect-src 'self' {supabase_origin} wss://{supabase_origin.split('//')[1]}; "
                    "script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                    "font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        resp = await call_next(request)
        resp.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        resp.headers["Content-Security-Policy"] = self.csp
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        resp.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
        resp.headers["X-Frame-Options"] = "DENY"
        return resp
```
Register in `create_app`: `app.add_middleware(SecurityHeadersMiddleware, supabase_origin=settings.supabase_url)`. Because CSP forbids inline scripts, move the `window.SARJY = {...}` inline block from `index.html` into a `GET /config.js` route that returns `application/javascript` built from settings (no secrets beyond the anon key, which is public by design).

```python
# src/sarjy/interfaces/http/account.py
import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from sarjy.interfaces.http.auth import CurrentUserDep

router = APIRouter()


@router.delete("/account", status_code=204)
async def delete_account(user: CurrentUserDep, request: Request) -> Response:
    s = request.app.state.settings
    # Deleting the auth user cascades to every table via FK (PRD §8). Use the Admin API so auth internals stay consistent.
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.delete(f"{s.supabase_url}/auth/v1/admin/users/{user.user_id}",
                           headers={"apikey": s.supabase_service_role_key, "Authorization": f"Bearer {s.supabase_service_role_key}"})
    if r.status_code not in (200, 204):
        raise HTTPException(502, "deletion failed")
    return Response(status_code=204)


@router.get("/account/export")
async def export_account(user: CurrentUserDep) -> Response:
    return Response(status_code=501, content="export is planned for a later release")
```
(For the integration test against local Supabase the Admin API is available at `http://localhost:54321`; the test's `SUPABASE_SERVICE_ROLE_KEY` must be the local one printed by `supabase start`.)

Web: add a "Delete my data" button in the settings panel (memory.js) that confirms twice, calls `DELETE /account`, then `sb.auth.signOut()` and reloads; Sarjy speaks "All your data is gone. Starting fresh." (PRD §11 user rights).

- [ ] **Step 4: Run; Step 5: Commit** `feat(security): headers middleware, CSP-safe config, account deletion`.

---

### Task 3: Static client on Supabase Storage CDN

**Files:**
- Create: `scripts/upload_static.py`
- Modify: `voice.js` (vendor supabase-js: `curl -o src/sarjy/interfaces/web/static/supabase.js https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/dist/umd/supabase.js`; import locally), `index.html` (absolute `apiBase`)

- [ ] Create a public bucket `web` in both projects (Storage → New bucket → public). Set `Cache-Control` on upload: HTML `no-cache`, JS/CSS `public, max-age=31536000, immutable` with content-hashed filenames.
- [ ] `scripts/upload_static.py`: renders `index.html` via the same Jinja2 template with `api_base=https://sarjy-<env>.fly.dev`, rewrites asset names to `name.<sha8>.ext`, uploads all files with the Storage REST API (`POST /storage/v1/object/web/<path>` with service-role key, `x-upsert: true`). Prints the public URL `https://<ref>.supabase.co/storage/v1/object/public/web/index.html`.
- [ ] CORS: set `CORS_ORIGINS` on each Fly app to the Storage public origin (`https://<ref>.supabase.co`). Supabase Auth → Site URL = the same.
- [ ] Optional custom domain: point a CNAME at the Supabase custom-domain feature or front the bucket with Cloudflare; out of scope for v1 unless already owned.
- [ ] Keep `GET /` on the Fly app as well (useful for local/dev) — both origins serve the same build.
- [ ] Commit `chore(deploy): static client published to supabase storage`.

---

### Task 4: Retention, audit sampling and alert crons (pg_cron + pg_net)

**Files:**
- Create: `supabase/migrations/20260821000700_retention_cron.sql`, `20260821000800_audit_sample.sql`

```sql
-- 20260821000700_retention_cron.sql  (PRD §11 retention; §13 alerts)
select cron.schedule('retention_messages', '15 3 * * *',
  $$delete from public.messages where created_at < now() - interval '30 days'$$);
select cron.schedule('retention_tool_calls', '20 3 * * *',
  $$delete from public.tool_calls where created_at < now() - interval '30 days'$$);
select cron.schedule('retention_guard_events', '25 3 * * *',
  $$delete from public.guardrail_events where created_at < now() - interval '180 days'$$);
select cron.schedule('retention_telemetry', '30 3 * * *',
  $$delete from public.telemetry_turns where created_at < now() - interval '90 days'$$);
select cron.schedule('retention_weather_cache', '*/30 * * * *',
  $$delete from public.weather_cache where fetched_at < now() - interval '1 hour'$$);
select cron.schedule('retention_rate_limits', '0 4 * * *',
  $$delete from public.rate_limits where window_start < now() - interval '2 days'$$);
select cron.schedule('retention_sessions', '35 3 * * *',
  $$delete from public.sessions s where last_active_at < now() - interval '30 days' and not exists (select 1 from public.messages m where m.session_id = s.id)$$);

-- Alerts: evaluated every 15 min; POSTs to a webhook stored in vault
create table if not exists public.alert_state (key text primary key, last_fired timestamptz);

create or replace function public.fire_alert(p_key text, p_payload jsonb) returns void
language plpgsql security definer as $$
declare url text := current_setting('app.alert_webhook', true);
begin
  if url is null then return; end if;
  if exists (select 1 from public.alert_state where key = p_key and last_fired > now() - interval '1 hour') then return; end if;
  perform net.http_post(url := url, body := p_payload, headers := '{"Content-Type":"application/json"}'::jsonb);
  insert into public.alert_state (key, last_fired) values (p_key, now()) on conflict (key) do update set last_fired = now();
end $$;

create or replace function public.check_alerts() returns void language plpgsql security definer as $$
declare p95 numeric; err_rate numeric; sev3 int;
begin
  select percentile_cont(0.95) within group (order by ttfa_ms) into p95
    from public.telemetry_turns where created_at > now() - interval '15 minutes';
  if p95 > 2000 then perform public.fire_alert('ttfa_p95', jsonb_build_object('text', format('Sarjy TTFA p95 = %s ms over last 15 min', p95))); end if;

  select count(*) into sev3 from public.guardrail_events where severity >= 3 and created_at > now() - interval '15 minutes';
  if sev3 > 0 then perform public.fire_alert('guard_sev3', jsonb_build_object('text', format('%s severity-3 guardrail events in last 15 min', sev3))); end if;

  select coalesce(avg(case when guard_decision like 'error:%' then 1 else 0 end), 0) into err_rate
    from public.messages where role = 'assistant' and created_at > now() - interval '15 minutes';
  if err_rate > 0.02 then perform public.fire_alert('error_rate', jsonb_build_object('text', format('Sarjy error rate %s%%', round(err_rate*100,1)))); end if;
end $$;

select cron.schedule('check_alerts', '*/15 * * * *', $$select public.check_alerts()$$);
```
Set the webhook once per project: `alter database postgres set app.alert_webhook = 'https://hooks.slack.com/services/...';` (documented in runbook; not committed). `RunTurn` must record `guard_decision = "error:<code>"` on the assistant message when it yields an `ErrorEvent` (small change; add a unit assertion).

```sql
-- 20260821000800_audit_sample.sql  (PRD Layer 7: 20% sample of allowed turns for async audit)
create table if not exists public.audit_queue (
  id bigserial primary key, message_id uuid not null, user_id uuid not null, created_at timestamptz default now(), processed_at timestamptz);
create or replace function public.enqueue_audit_sample() returns trigger language plpgsql as $$
begin
  if new.role = 'assistant' and (new.guard_decision is null or new.guard_decision = 'allow') and random() < 0.2 then
    insert into public.audit_queue (message_id, user_id) values (new.id, new.user_id);
  end if;
  return new;
end $$;
create trigger messages_audit_sample after insert on public.messages for each row execute function public.enqueue_audit_sample();
```
Python side: add `src/sarjy/contexts/guardrails/application/audit.py` with `AuditWorker(db, classifier).run_once(limit=50)` that loads queued assistant messages + their preceding user message, calls the classifier on the pair, writes a `guardrail_events` row (`layer=7`, `kind="audit:<category>"`, `action="allow_flagged"` or `"audit_clean"`), marks `processed_at`. Expose `POST /internal/audit/run` protected by a shared `INTERNAL_TOKEN` header and schedule it with `cron.schedule('audit', '*/10 * * * *', $$select net.http_post('https://sarjy-prod.fly.dev/internal/audit/run', headers := jsonb_build_object('X-Internal-Token', current_setting('app.internal_token')))$$)`. Unit test the worker with fakes.

- [ ] Apply migrations to staging + prod; commit `feat(ops): retention crons, alerting, audit sampling worker`.

---

### Task 5: CI/CD pipeline with eval gates and manual prod approval

**Files:**
- Modify: `.github/workflows/ci.yml`; Create: `.github/workflows/deploy.yml`, `scripts/smoke.py`

- [ ] `scripts/smoke.py <base_url> <supabase_url> <anon_key>`: health check; anonymous sign-up via `POST /auth/v1/signup`; one `/chat` turn "say hi"; asserts a `sentence` event arrives within 5 s and `done` within 20 s; exits non-zero otherwise.

- [ ] `deploy.yml`:

```yaml
name: deploy
on:
  push: { branches: [main] }
  workflow_dispatch:
concurrency: deploy-${{ github.ref }}
jobs:
  staging:
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - uses: supabase/setup-cli@v1
      - run: supabase link --project-ref ${{ secrets.SUPABASE_PROJECT_REF }} && supabase db push
        env: { SUPABASE_ACCESS_TOKEN: "${{ secrets.SUPABASE_ACCESS_TOKEN }}", SUPABASE_DB_PASSWORD: "${{ secrets.SUPABASE_DB_PASSWORD }}" }
      - uses: superfly/flyctl-actions/setup-flyctl@master
      - run: flyctl deploy -c fly.staging.toml --remote-only
        env: { FLY_API_TOKEN: "${{ secrets.FLY_API_TOKEN }}" }
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-extras
      - run: uv run python scripts/upload_static.py --env staging
        env: { SUPABASE_URL: "${{ secrets.SUPABASE_URL }}", SUPABASE_SERVICE_ROLE_KEY: "${{ secrets.SUPABASE_SERVICE_ROLE_KEY }}", API_BASE: https://sarjy-staging.fly.dev }
      - run: uv run python scripts/smoke.py https://sarjy-staging.fly.dev ${{ secrets.SUPABASE_URL }} ${{ secrets.SUPABASE_ANON_KEY }}
  evals:
    needs: staging
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-extras
      - run: uv run python tests/evals/run_evals.py --target https://sarjy-staging.fly.dev --suites redteam,benign,memory,weather,ocean
        env: { SUPABASE_URL: "${{ secrets.SUPABASE_URL }}", SUPABASE_ANON_KEY: "${{ secrets.SUPABASE_ANON_KEY }}", GEMINI_API_KEY: "${{ secrets.GEMINI_API_KEY }}" }
      - uses: actions/upload-artifact@v4
        with: { name: eval-results, path: evals/out/ }
  prod:
    needs: evals
    runs-on: ubuntu-latest
    environment: production          # configure "Required reviewers" on this environment in GitHub → manual approval gate
    steps:
      - uses: actions/checkout@v4
      - uses: supabase/setup-cli@v1
      - run: supabase link --project-ref ${{ secrets.SUPABASE_PROJECT_REF }} && supabase db push
        env: { SUPABASE_ACCESS_TOKEN: "${{ secrets.SUPABASE_ACCESS_TOKEN }}", SUPABASE_DB_PASSWORD: "${{ secrets.SUPABASE_DB_PASSWORD }}" }
      - uses: superfly/flyctl-actions/setup-flyctl@master
      - run: flyctl deploy -c fly.prod.toml --remote-only --strategy rolling
        env: { FLY_API_TOKEN: "${{ secrets.FLY_API_TOKEN }}" }
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-extras
      - run: uv run python scripts/upload_static.py --env prod
        env: { SUPABASE_URL: "${{ secrets.SUPABASE_URL }}", SUPABASE_SERVICE_ROLE_KEY: "${{ secrets.SUPABASE_SERVICE_ROLE_KEY }}", API_BASE: https://sarjy-prod.fly.dev }
      - run: uv run python scripts/smoke.py https://sarjy-prod.fly.dev ${{ secrets.SUPABASE_URL }} ${{ secrets.SUPABASE_ANON_KEY }}
```
- [ ] `run_evals.py` gains `--target <url>` (remote mode: uses anon sign-up for tokens instead of signing HS256 locally) and `--suites`. Evals run against staging with `WEATHER_PROVIDER=open-meteo` live; weather suite asserts numbers ⊆ tool result (not fixed values) so it works with live data.
- [ ] Create GitHub environments `staging` and `production` with their secrets; add required reviewer on `production`.
- [ ] Rollback runbook: `flyctl releases -c fly.prod.toml` → `flyctl deploy -c fly.prod.toml --image <previous image>`; static: re-run `upload_static.py` from the previous commit. Migrations are forward-only; write a compensating migration if needed.
- [ ] Commit `ci: staging→evals→prod pipeline with manual approval`.

---

### Task 6: Runbook and launch checklist

**Files:**
- Create: `docs/runbook.md`, `docs/launch-checklist.md`

`docs/runbook.md` sections: environments & URLs; secrets inventory (names only, where each lives — from master plan §2); deploy & rollback; how to read the latency views (`select * from v_latency_daily limit 7;`); how to hot-fix a guardrail rule (edit `rules.py` → PR → CI evals gate → merge → auto-deploy); how to rotate `GEMINI_API_KEY` (set new secret on Fly → `fly deploy` restarts machines); how to purge a user on request (`DELETE /account` or Admin API); common alerts and first actions (TTFA p95 spike → check Fly region/CPU, Gemini status page; sev-3 guard events → query `guardrail_events` by `message_id` and review); quota monitoring (Gemini console spend, Open-Meteo limits).

`docs/launch-checklist.md` (PRD §2.3 gates + §11 + acceptance tests):

```
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
- [ ] No secrets in repo (`git grep -i "AIza\|service_role\|eyJhbGci"` is empty); Docker image has no .env
- [ ] RLS verified on prod: `tests/integration/test_db_rls.py` run against prod DB (read-only user)
- [ ] CSP/HSTS headers present on prod (curl -I)
- [ ] Anonymous sign-in captcha enabled; SMTP configured; Site URL correct
- [ ] DELETE /account works on prod with a throwaway user
- [ ] Retention crons listed in `select * from cron.job;`
- [ ] Gemini project on paid tier (no-training) — screenshot in ticket

## Ops
- [ ] Alerts webhook configured (`app.alert_webhook`) and test alert fired
- [ ] Fly: min_machines_running=2, health checks green, rolling strategy
- [ ] Runbook reviewed by a second engineer
- [ ] Budget alert set in Google Cloud billing for the Gemini key

Signed off by: ____________   Date: ________
```

- [ ] Commit `docs: runbook and launch checklist`.

---

### Task 7: Production launch

- [ ] Merge to `main` → pipeline runs → approve `production`.
- [ ] Walk the launch checklist on prod; record numbers; fix blockers; re-run.
- [ ] Tag: `git tag -a v1.0.0 -m "Sarjy v1.0 launch" && git push --tags`.

---

## Phase 8 self-review

- Spec coverage: §12 environments/CI/rollback ✔ T1, T5; static hosting on Supabase ✔ T3; §11 secrets/CSP/retention/user rights/abuse ✔ T2, T4 (+ Phase 5 rate limits, Turnstile in T1); §13 alerts ✔ T4, dashboards (Phase 7 views + `/admin/latency`); §14 eval gates in pipeline ✔ T5; §15 availability via 2 machines + rolling deploys ✔ T1; §16 M5 ✔ T6–T7. Layer-7 20 % audit sample ✔ T4 (closes the gap noted in Phase 5's self-review).
- Consistency: `run_evals.py --target/--suites` flags extend Phase 5's runner; `guard_decision = "error:<code>"` is a small `RunTurn` addition noted in T4.
- No placeholders.
