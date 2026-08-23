# Sarjy operations runbook

Grounded in the actual deploy config as of this writing: `Makefile`,
`fly.staging.toml`, `fly.prod.toml`, `scripts/{smoke,upload_static}.py`,
`.env.example`, and the
`supabase/migrations/20260821000600_latency_views.sql` /
`...000700_retention_cron.sql` / `...000800_audit_sample.sql` migrations.

## Environments & URLs

| Environment | Fly app | Base URL | `fly*.toml` | Deploy trigger |
|---|---|---|---|---|
| staging | `sarjy-staging` | `https://sarjy-staging.fly.dev` | `fly.staging.toml` (`min_machines_running = 1`) | manual: `make release-staging` |
| production | `sarjy-prod` | `https://sarjy-prod.fly.dev` | `fly.prod.toml` (`min_machines_running = 2`, rolling strategy) | manual: `make release-prod`, run only after `make evals-staging` passes and a human signs off |

Both apps run `primary_region = "ams"` (matches the Supabase project region),
`internal_port = 8080`, `force_https = true`, and a `GET /healthz` check
(`interval 15s`, `timeout 3s`, `grace_period 10s`).

Static client: published separately to Supabase Storage's public `web`
bucket by `scripts/upload_static.py` (see "Static hosting" under
Operational notes below). `GET /` on the Fly app itself also serves the
same template locally-asset'd — that's the primary entry point end users
should be pointed at.

## Secrets inventory

Names only — see each variable's own comment in `.env.example` /
`src/sarjy/config.py` for validation rules and defaults. "Shell env var for
`make release-*`/`make evals-staging`" means it needs to be exported in the
shell that runs those targets (see "Deploy & rollback" below); the unit and
integration test suites run against an ephemeral local Supabase stack and use
fixed, non-secret placeholder values for the DB/JWT-shaped vars (never
real credentials).

| Variable | Lives |
|---|---|
| `GEMINI_API_KEY` | Fly secret (staging + prod) · shell env var for `make release-*`/`make evals-staging` (`evals-staging` and optionally `smoke-*`) · local `.env` |
| `GEMINI_CHAT_MODEL` | Has a code default (`gemini-3.6-flash`); not set in either `fly*.toml [env]`. Set via `fly secrets set` / local `.env` only to override |
| `GEMINI_GUARD_MODEL` | Same as above; default `gemini-3.5-flash-lite` |
| `GEMINI_EMBED_MODEL` | Same as above; default `gemini-embedding-001` |
| `SUPABASE_URL` | Fly secret (staging + prod) · shell env var for `make release-*`/`make evals-staging` (`upload_static.py`/`smoke.py`/evals steps) · local `.env` |
| `SUPABASE_ANON_KEY` | Fly secret · shell env var for `make release-*`/`make evals-staging` (used by `smoke.py`/evals steps) · local `.env` |
| `SUPABASE_SERVICE_ROLE_KEY` | Fly secret · shell env var for `make release-*` (used by `upload_static.py` steps) · local `.env`. The unit/integration test suites use a fixed placeholder (`s`) for the local Supabase stack, never a real key; `tests/conftest.py` also `setdefault`s a placeholder for the unit suite (see Testing quirk below) |
| `SUPABASE_JWT_SECRET` | Only used for HS256 tokens (legacy projects, local stack). Projects created since late 2025 sign access tokens with ES256; the API verifies those against `<SUPABASE_URL>/auth/v1/.well-known/jwks.json` automatically (`interfaces/http/auth.py`, cached, refetch on unknown `kid` at most once a minute). Still required at boot. Fly secret · test env uses a fixed placeholder for the local stack · local `.env` |
| `DATABASE_URL` | Fly secret (pooled connection string) · test/evals env uses a fixed local Postgres URL · local `.env` |
| `DATABASE_URL_DIRECT` | Same as `DATABASE_URL`, direct (non-pooled) connection |
| `WEATHER_PROVIDER` | Fly secret/env var, default `open-meteo`. Set to `mock` for offline/dry-run evals. Local `.env` |
| `OWM_API_KEY` | Fly secret, optional — only read when `WEATHER_PROVIDER=owm`. Local `.env` |
| `GUARD_MODE` | Fly secret/env var, default `enforce`; set to `shadow` temporarily via `fly secrets set` after a guardrail rule change (see "Hot-fix a guardrail rule" below). Local `.env` |
| `HISTORY_LIMIT` | Fly secret/env var, default `12`. Local `.env` |
| `SPECULATIVE_ENABLED` | Fly secret/env var, default `false`; the lever for the per-process speculation caveat (see Operational notes). Local `.env` |
| `AUDIO_MODE` | Fly secret/env var, default `webspeech`; `gemini-tts` is Phase 9/optional. Local `.env` |
| `APP_ENV` | Set directly (not secret) in `fly.staging.toml`/`fly.prod.toml` `[env]` (`staging`/`prod`). Local `.env` default `dev` |
| `APP_BASE_URL` | Defined in `Settings` (default `http://localhost:8000`) but not read anywhere else in `src/` today — currently inert/reserved. Local `.env` only |
| `PUBLIC_API_BASE` | Fly secret/env var. Empty by default (relative fetches, correct for same-origin `GET /`). Must be set on Fly to the app's own absolute URL (`https://sarjy-{staging,prod}.fly.dev`) so `GET /config.js`'s `apiBase` matches what `scripts/upload_static.py --api-base` baked into the Storage-hosted page's `<script src>` — see `src/sarjy/interfaces/http/web.py` |
| `CORS_ORIGINS` | Fly secret/env var, default `http://localhost:8000`. Must include the Storage bucket's public origin once the client is Storage-hosted (cross-origin `fetch`s from the static page to the Fly API). Rejects a literal `*` (Bearer tokens, not cookies) |
| `LOG_LEVEL` | Set directly (not secret) in `fly.staging.toml`/`fly.prod.toml` `[env]` = `INFO`. Local `.env` |
| `SENTRY_DSN` | Optional `Settings` field, read but **not currently wired to any Sentry SDK init** anywhere in `src/` (no `sentry-sdk` dependency, no `sentry_sdk.init` call) — parked, see Known gaps. If/when wired: Fly secret |
| `TURNSTILE_SITE_KEY` | Fly secret/env var (staging + prod) · shell env var for `make release-*` (read by the `upload_static.py` steps so the Storage-hosted page matches the Fly-served one). Surfaced to the client via `GET /config.js`'s `turnstileSiteKey` (`web.py`), and the single switch for the whole captcha path: page markup, CSP, and the client's sign-in call. Local `.env` (leave empty for local dev — see "Turnstile captcha" below) |
| `TURNSTILE_TOKEN` | Shell env var for `make release-*`/`make evals-staging` only — never a Fly secret and never read by the app. Read by `scripts/_supabase_auth.py` so `smoke.py` and the remote eval runs can sign up against a captcha-protected project. Its value is Cloudflare's always-pass dummy token (see "Turnstile captcha" below) |
| `INTERNAL_TOKEN` | Fly secret (staging + prod). Must match the DB GUC `app.internal_token` (`alter database postgres set app.internal_token = '...'`) — see `supabase/migrations/20260821000800_audit_sample.sql` and Operational notes. Unset → `POST /internal/audit/run` returns 503 |

Not in the fixed 19-variable list above but load-bearing for ops: `ADMIN_USER_IDS`
(Fly secret, comma-separated UUIDs — gates `GET /admin/latency`, see
`src/sarjy/interfaces/http/admin.py`), and the DB-only GUCs `app.alert_webhook`
and `app.audit_run_url` (set via `alter database postgres set ...`, never
committed — see Operational notes).

## Turnstile captcha (anonymous sign-in)

PRD §11 requires anonymous sign-in to be captcha-gated. The whole path is
gated on one setting, `TURNSTILE_SITE_KEY`:

- **Unset (local/dev default).** `GET /` renders no Cloudflare script and no
  widget host, the CSP advertises no third-party origin, `voice.js` calls
  `signInAnonymously()` with no arguments, and `scripts/_supabase_auth.py`
  posts a bare `{}`. Byte-for-byte what the app did before captcha existed.
- **Set.** `index.html` loads
  `https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit`
  (external, never inline — the CSP still forbids inline scripts) plus a
  `<div id="turnstile">` host; `SecurityHeadersMiddleware` adds
  `https://challenges.cloudflare.com` to `script-src`, `frame-src` and
  `connect-src`, and only then; `voice.js` renders the widget explicitly on
  the first sign-in, awaits the token, and passes it as
  `signInAnonymously({ options: { captchaToken } })`. A widget that fails to
  load or a challenge that fails shows "Verification failed — reload the page
  to try again." rather than a dead mic button.

Turning it on for a project takes **both** halves — they are independent and
either alone is a broken state:

1. Supabase dashboard → Authentication → Attack Protection → enable captcha,
   provider Turnstile, paste the **secret** key. (This is what actually
   enforces it; GoTrue does the verification, not this app.)
2. `fly secrets set TURNSTILE_SITE_KEY=0x... -c fly.prod.toml`, and export the
   same value as the `TURNSTILE_SITE_KEY` shell env var before running `make
   release-*` so `upload_static.py` bakes it into the Storage-hosted page too.

**Smoke / evals.** Cloudflare publishes always-pass test credentials, so
a captcha-protected staging project can still be signed into by a headless
script. Configure the *staging* project with the test **secret**
`1x0000000000000000000000000000000AA` (its matching sitekey is
`1x00000000000000000000AA`, which always passes in a browser), and export
`TURNSTILE_TOKEN=XXXX.DUMMY.TOKEN.XXXX` as a shell env var before running
`make release-*`/`make evals-staging`. `scripts/_supabase_auth.py` forwards
it to GoTrue as `{"gotrue_meta_security": {"captcha_token": ...}}`. Unset,
that body stays a bare `{}` — so a project without captcha is unaffected
either way.

**Local Supabase does NOT enforce captcha.** `supabase/config.toml` has no
`[auth.captcha]` block on purpose: the pinned CLI (v1.190.0) silently ignores
that section — verified by adding it, restarting the stack, and finding no
`GOTRUE_SECURITY_CAPTCHA_*` variables on the `supabase_auth` container and a
token-less `POST /auth/v1/signup` still returning 200. Adding a block the CLI
drops on the floor would be worse than leaving it out: it reads like local
parity that does not exist. Local/dev and the integration suite therefore run
uncaptcha'd (the integration tests create `auth.users` rows over direct SQL
and never touch GoTrue's signup endpoint at all), and captcha is verified
against a real staging project instead. Revisit when the CLI is upgraded.

## Deploy & rollback

**Deploy is entirely manual — there is no CI/CD.** From a shell with the
Fly/Supabase vars exported (`FLY_API_TOKEN`, plus the relevant `SUPABASE_*`
and `TURNSTILE_*` vars — see the Secrets inventory above):

1. `make release-staging` — `supabase db push` (migrations; assumes
   `supabase link` has already been run once against the staging project) →
   `flyctl deploy` (`deploy-staging`) → `upload_static.py`
   (`static-staging`, publish the static client) → `smoke.py`
   (`smoke-staging`: health + anon sign-up + one `/chat` turn).
2. `make evals-staging` — the live suite (`redteam,benign,memory,weather,
   ocean`) against the staging URL. Review the numbers.
3. A human decides whether to proceed, based on the eval numbers and any
   manual checks.
4. `make release-prod` — same four steps as `release-staging`, against the
   prod project/app.

Each of `deploy-{staging,prod}`, `smoke-{staging,prod}`,
`static-{staging,prod}`, and `evals-staging` can also be run individually —
see `Makefile` for the exact commands each target runs.

**Rollback** (exactly as Task 5's brief specifies):

```bash
flyctl releases -c fly.prod.toml                       # find the previous image
flyctl deploy -c fly.prod.toml --image <previous image>
```

Static: re-run `upload_static.py` from the previous commit (`git checkout
<previous sha> -- src/sarjy/interfaces/web` then `uv run python
scripts/upload_static.py --env prod`, or just check out that commit
entirely and run it).

**Migrations are forward-only.** Do not attempt to "roll back" a migration —
write a compensating migration instead (e.g. a new file that reverses the
effect) and push it forward.

## How to read the latency views

All four views live in `supabase/migrations/20260821000600_latency_views.sql`,
revoked from `anon`/`authenticated` (service-role / SQL editor / `GET
/admin/latency` only):

```sql
select * from v_latency_daily limit 7;       -- day, mode, turns, ttfa_p50/p95, first_byte_p50, gemini_first_token_p50, turn_p50
select * from v_latency_by_browser;          -- browser, turns, ttfa_p50/p95 (trailing 7 days)
select * from v_guard_daily limit 20;        -- day, layer, kind, action, n
select * from v_ocean_funnel limit 7;        -- day, proposed, started, completed
```

Same data is exposed at `GET /admin/latency` (requires the caller's
`user_id` to be in `ADMIN_USER_IDS`; 403 otherwise) — see "Purge a user" /
Operational notes for the auth shape.

## How to hot-fix a guardrail rule

1. Edit `src/sarjy/contexts/guardrails/domain/rules.py` (the deterministic
   Layer-2 regex engine — self-harm rules must stay first per the module
   docstring).
2. Open a PR. Locally, run `make check && make evals-offline` (lint/type/
   test, then the JSONL shape + coverage matrix and the Layer-2-only,
   no-API-key run) before asking for review.
3. Get it reviewed and merged, then run `make evals-staging` (the full live
   gate: red-team ≥ 99%, benign FP ≤ 2%) against staging.
4. Once those numbers look right, run `make release-prod` to deploy.
5. For a risky rule change, deploy with `GUARD_MODE=shadow` first (`fly
   secrets set GUARD_MODE=shadow -c fly.<env>.toml`, then `flyctl deploy`):
   Layer 2/3 (`InputGuard`) and Layer 6 (`OutputGuard`) both log what they
   *would* have done (`shadow_block`, `shadow_cut`, `shadow_replace`)
   without acting on it. Run this against real traffic for ~24h, check
   `select * from v_guard_daily where action like 'shadow_%' order by day
   desc;`, then flip back to `GUARD_MODE=enforce` and redeploy.

## How to rotate GEMINI_API_KEY

```bash
fly secrets set GEMINI_API_KEY=<new key> -c fly.staging.toml
fly secrets set GEMINI_API_KEY=<new key> -c fly.prod.toml
```

`fly secrets set` triggers a new release and restarts the app's machines
with the new value (no separate `fly deploy` needed, though running one is
harmless). Also update the `GEMINI_API_KEY` shell env var wherever
`make evals-staging` is run from, and any local `.env` files. Revoke the old
key in the Gemini console only after confirming the new one is live
(`smoke.py` passing, or a manual `/chat` turn).

## How to purge a user on request

- Self-service: `DELETE /account` (Bearer token, the user's own JWT) —
  deletes the GoTrue user via the Supabase Admin API, which cascades to
  every FK'd table (PRD §8). Idempotent: a second call (or one that races
  an out-of-band deletion) gets a 404 from the Admin API and still returns
  204 to the caller (`src/sarjy/interfaces/http/account.py`).
- Support/ops path: the Supabase Auth Admin API directly (`DELETE
  /auth/v1/admin/users/{user_id}` with the service-role key), or the
  Supabase dashboard's Auth panel — same cascade.
- **What deliberately survives a purge.** The FK cascade in
  `20260821000450_ops_tables_lockdown.sql` is `on delete set null` — not
  `cascade` — for `guardrail_events` and `telemetry_turns`, so those rows stay
  with `user_id = NULL` after the user is gone. That is by design: they are the
  safety audit trail (which layers fired, at what severity) and the latency
  series, and both are aggregate signals that would silently rewrite history if
  a deletion could remove them. Neither table stores user text — `detail` is a
  small JSON object of rule ids, categories, confidences and timings, never the
  utterance — so a de-linked row is not personal data. Everything that *does*
  hold user text (`messages`, `tool_calls`, `memories`, `memories_history`,
  `sessions`, `workflow_*`) cascades and is gone.
- **Forgetting one fact** (as opposed to purging the user) also erases that
  memory's `memories_history` rows — the history table stores `old_value`/
  `new_value` verbatim, so a soft delete alone would leave a full copy of the
  forgotten fact behind. See `ForgetFact._forget`. Orphaned history rows (from
  a hard-deleted memory) are collected after 30 days by the
  `retention_memories_history` cron.
- `GET /account/export` is **not implemented** (returns 501 "export is
  planned for a later release"). For now, run a manual SQL export instead:

  ```sql
  select * from public.memories where user_id = '<uuid>';
  select * from public.messages where user_id = '<uuid>';
  select * from public.sessions where user_id = '<uuid>';
  select * from public.workflow_runs where user_id = '<uuid>';
  -- add any other user-scoped table as needed
  ```

## Common alerts and first actions

Alerts fire from `public.check_alerts()` (pg_cron, every 15 minutes,
`supabase/migrations/20260821000700_retention_cron.sql`), POSTing to the
webhook in the `app.alert_webhook` GUC via `fire_alert()` (rate-limited to
once per key per hour).

- **TTFA p95 spike** (`ttfa_p95 > 2000` over the trailing 15 min): check the
  Fly region/CPU (`fly status -c fly.prod.toml`, `fly logs -c
  fly.prod.toml`) for saturation or a bad machine; check the [Gemini status
  page](https://status.cloud.google.com/) for a provider-side incident;
  cross-check `select * from v_latency_daily limit 3;` and `v_latency_by_
  browser` to see if it's one browser/region or global.
- **Sev-3 guard events** (`guardrail_events.severity >= 3`, any in the
  trailing 15 min — self-harm and prompt-leak categories are severity 3):
  query `select * from guardrail_events where message_id = '<id>' order by
  created_at;` for the flagged turn, review the message/transcript, and
  escalate per the self-harm response process if it's a genuine disclosure
  rather than a rule false-positive.
- **Error rate** (`> 2%` of assistant messages with `guard_decision like
  'error:%'` over 15 min): check Fly logs for stack traces, then Gemini/
  Supabase status.

## Quota monitoring

- **Gemini**: check spend/usage in the Google Cloud console (the project
  billing the `GEMINI_API_KEY` belongs to); confirm the project is on the
  paid, no-training tier (see launch checklist) and that a budget alert is
  configured.
- **Open-Meteo**: free tier has a request-rate limit; `weather_cache`
  (retention-cron'd every 30 minutes, `retention_weather_cache` job) keeps
  repeat lookups cheap. If the OWM fallback (`WEATHER_PROVIDER=owm`) is
  ever the primary provider, watch `OWM_API_KEY`'s quota in the OpenWeatherMap
  dashboard instead.

## Operational notes

1. **Speculative turns are per-process.** The early-confirm registry and
   `SpeculativeTurnCache`
   (`src/sarjy/contexts/conversation/application/speculation.py`) live in
   one worker's memory by design — a shared store would put a network round
   trip in the path of the write speculation exists to avoid. With prod's
   `min_machines_running = 2` and no `fly-replay`/sticky routing configured,
   a `/chat` and its confirming `/chat/confirm` can land on different
   machines, in which case the confirm finds nothing parked under that id
   and `RunTurn.confirm` returns `"pending"` — HTTP **202**, not 409 (see
   `chat.py` and `RunTurn.confirm`'s docstring, `run_turn.py`). The client
   does nothing on a 202: `voice.js`'s `confirm()` only retries on a network
   failure (the fetch itself throwing) or on a 409, never on 202
   (`if (await post() !== 409) return;`), and the mismatch-triggered resend
   in `onFinal` runs *before* `confirm()` is ever called, on a text mismatch
   — it is not a response to this. So the early-confirm transcript this
   case records simply expires unread, and the speculative latency win is
   silently lost for that turn: no error, no retry, no duplicate. Accepted
   as-is. If `/chat/confirm`'s 202 rate shows up meaningfully in telemetry,
   either set `SPECULATIVE_ENABLED=false` or pin the app to one machine.

2. **Rate limiting.** `rate_limits` rows are namespaced by window/endpoint
   (`chat`, `tele`, `confirm`, ...) in 5-minute buckets, and a hit checks the
   current bucket plus the one before it (an approximate sliding window —
   see `pg_rate_limiter.py`). Anonymous users get half the configured
   allowance (floored at 1). The limiter is entirely off when the container
   is built with `connect_db=False` (no database, so nothing to count
   against).

3. **Guardrails.** The Layer-3 classifier has a 0.4s first-token budget
   (`InputGuard.timeout`, default `classifier_timeout_s=0.4`) and fails
   CLOSED (blocks, recorded as `classifier_timeout`/`classifier_error`) on
   timeout or any other classifier failure, for every rule family that
   routes to it — G-12. `GUARD_MODE=shadow` logs `shadow_block` (Layer 2/3)
   and `shadow_cut`/`shadow_replace` (Layer 6) without acting; use it for
   ~24h after a rule change (see "Hot-fix a guardrail rule" above).
   `uncertain`-verdict rules route to the classifier. Bare `"kms"` is
   deliberately NOT classifier-reachable on its own — the rule
   (`selfharm.slang`) requires a preceding first-person intent marker
   ("i/imma/wanna/gonna/... kms") specifically because a bare `\bkms\b`
   collides with "kilometres" and a digit-based fix doesn't survive the
   de-leeted normalisation variant.

   **Memory writes are screened by the Layer-2 rule engine** (Phase 8
   Task 6b, + fix round 1). Two entry points reach a stored fact, and both go
   through the same screening:

   - `RememberFact` (`memory/application/remember.py`) — the `remember` tool,
     i.e. the model deciding mid-turn to store something. There is no REST
     endpoint that creates a memory.
   - `EditFact` (`memory/application/edit.py`) — `PATCH /memory/{id}`. This
     used to write straight through the repo after only the PII filter, so a
     value the tool path would refuse went in unscreened through the REST
     door; `EditFact` is now the one place both paths share.

   Both call `screen_reason` (`memory/application/screening.py`), which screens
   **the key and the value as two separate calls** — never concatenated. That
   is not a style choice: the rule engine has a memory-set-frame carve-out
   (R2 in `guardrails/domain/rules.py`) that lets a quoted fact value skip the
   low-precision `unc.*` family on a normal chat turn, and concatenating a key
   that normalises to "note"/"remember"/"save"/"store" with a value shaped like
   `"that my X is '<payload>'"` reconstructs that frame at screening time and
   smuggles an uncertain-severity payload straight past the screen. Screening
   each field alone closes it.

   The adapter behind the port is `RuleEngineValueScreen`
   (`guardrails/infrastructure/value_screen.py`), wired by
   `Container.rebuild_memory`. It runs the same Layer-2 `RuleEngine` a normal
   turn runs, over `normalize_variants(...)`, with
   **`honor_memory_set_frame=False`** — the carve-out can never fire here,
   because the string being screened IS the value being stored, not the
   framing around it. It refuses on **either** a `block` **or an `uncertain`**
   verdict: there is no Layer-3 classifier in this synchronous write path to
   resolve an `uncertain`, and a value ambiguous enough to reach the
   classifier on a normal turn is not safe to store and re-inject into every
   later prompt. Same fail-closed posture as `InputGuard` on a classifier
   timeout.

   A refusal writes nothing to the repo and is recorded as a `guardrail_events`
   row with `layer=2`, `action=refuse` and `kind=memory_write:<rule_id>` —
   that is the string to grep when someone reports "it won't remember X".
   The event write is best-effort (fire-and-forget through `BackgroundTasks`);
   the refusal itself does not depend on it. The `remember` tool speaks the
   refusal in the same "I won't store that — ..." style as a PII refusal
   (`memory/application/tools.py`).

   The memory context still does not import `sarjy.contexts.guardrails`
   directly — `RuleEngineValueScreen` is a guardrails-infrastructure adapter
   plugged in behind `ValueScreenPort`
   (`src/sarjy/contexts/memory/application/ports.py`).

   Facts stored **before** this landed were never screened — see Known gaps
   and `scripts/rescreen_memories.py`.

4. **Static hosting.** Supabase Storage serves uploaded `.html` objects as
   `Content-Type: text/plain`, not `text/html` — so a browser navigated
   straight at the Storage `index.html` URL downloads/renders it as plain
   text rather than a page. `GET /` on the Fly app is therefore the
   primary, correct entry point for end users; Storage's role is to host
   the *hashed, immutably-cached* JS/CSS assets the page then loads
   (`Cache-Control: public, max-age=31536000, immutable` — see
   `scripts/upload_static.py`), plus a copy of `index.html` for anyone who
   does end up there directly. `upload_static.py` runs as part of every
   `make release-staging`/`make release-prod` (via the `static-staging`/
   `static-prod` targets), publishing that asset bundle.

   The `static-staging`/`static-prod` targets don't pass `--api-base`
   explicitly, so the script falls back to its own per-`--env` default
   (`DEFAULT_API_BASE` in `upload_static.py` — the Fly app's own URL for
   staging/prod); pass `--api-base` by hand if that ever needs overriding.

5. **Testing quirk.** `tests/conftest.py` sets a placeholder
   `SUPABASE_SERVICE_ROLE_KEY` (`os.environ.setdefault(...)`) so the unit
   suite can boot `Settings` without a real key. Running the app locally
   against a real Supabase project for manual/integration testing (e.g. to
   exercise `GET /admin/latency` or `DELETE /account` for real) needs the
   *real* key in the environment, or the Admin API calls come back 403.
   Export your real `.env` first: `set -a && source .env && set +a`.

6. **Account deletion / export.** `DELETE /account` is idempotent (a
   already-deleted or racing-delete auth user gets a 404 from GoTrue, which
   the endpoint turns into a 204 to the caller either way). `GET
   /account/export` is a stub — 501 — do the export manually via the SQL in
   "Purge a user" above.

7. **`DELETE /account`** and **`GET /admin/latency`** usage are documented
   above. Audit worker: `POST /internal/audit/run` (header
   `X-Internal-Token: <INTERNAL_TOKEN>`) is called by pg_cron's `audit` job
   every 10 minutes via `pg_net`, and is also safe to call manually for a
   backfill/ops check. The `audit` cron job (`run_audit_cron()`,
   `20260821000800_audit_sample.sql`) only actually POSTs anything once
   **both** `app.audit_run_url` and `app.internal_token` are set on the
   database (`alter database postgres set app.audit_run_url = '...'` /
   `... set app.internal_token = '...'`); with either unset it's a silent
   no-op, on every environment including local/dev, by design.

8. **Fly rollback** is exactly the procedure in "Deploy & rollback" above
   (Task 5 brief). Migrations are forward-only — never attempt a schema
   rollback; write a compensating migration instead.

## Known gaps / parked

- **No live Gemini evals run has been executed and checked in yet** for
  this deployment-hardening phase — both `make evals` (local, full suite)
  and `make evals-staging` (against a deployed target) depend on a real
  `GEMINI_API_KEY` being exported in the shell, which happens at rollout
  time, not at doc-writing time. Numbers to paste into the launch checklist
  come from that first real run.
- **Facts stored before Phase 8 Task 6b were never guardrail-screened.**
  Screening was added to the write path in 6b (`RememberFact`, `EditFact` —
  see Operational notes); anything already in `memories` at that point went in
  unscreened and is still re-injected into the prompt on every recall. The
  backfill is `scripts/rescreen_memories.py`: dry run by default, `--delete` to
  remove the refused rows and their history. Needs `DATABASE_URL_DIRECT` (a
  cross-user sweep, so service-role). It never prints a refused value in full —
  40 characters, ellipsised — so its output is safe to paste into a ticket.
  Run it once per environment before launch and note the counts here.
- **No Playwright TTFA run has been executed** (`tests/latency/measure.py`,
  PRD acceptance test L) — needs a deployed instance with a real Gemini key
  and network egress for live Chromium speech recognition, neither
  available in the environment this runbook was written in. See
  `tests/latency/README.md` ("Status: run parked") for exactly what's
  ready to go the moment a staging URL is available.
- **Storage serves `.html` as `text/plain`.** Documented above under
  "Static hosting" — not something this task fixes (it's a Supabase
  Storage content-type behaviour, not a bug in this repo), just something
  the runbook makes sure nobody rediscovers the hard way by linking
  directly to the Storage `index.html` URL.
- **Speculative turns are per-process**, so a multi-machine prod
  (`min_machines_running = 2`) can lose the speculative latency win on a
  `/chat`/`/chat/confirm` pair that lands on two different machines: the
  confirm returns 202 (not 409), the client makes no retry on 202, and the
  parked transcript silently expires unread. Documented above under
  Operational notes item 1; accepted as-is per the comment in
  `fly.prod.toml`.
