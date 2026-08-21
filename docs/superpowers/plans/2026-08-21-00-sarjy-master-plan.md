# Sarjy — Master Implementation Plan (Python / DDD)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy Sarjy — the voice assistant specified in `PRD.md` — as a fully Python backend organised with Domain-Driven Design, a thin browser client, Supabase for data/auth/hosting, and Gemini as the LLM.

**Architecture:** Hexagonal (ports & adapters) inside DDD bounded contexts. Five contexts — Conversation, Memory, Weather, Assessment, Guardrails — each with `domain / application / infrastructure` layers and a tiny shared kernel. A FastAPI "interfaces" layer exposes SSE and JSON endpoints and serves the web client. Supabase Postgres (+pgvector) is the only database; Supabase Auth issues the JWTs the API verifies; the static client is hosted from Supabase Storage; the Python API runs as a container on Fly.io in the same region as the Supabase project.

**Tech Stack:** Python 3.12 · FastAPI · uvicorn · httpx · pydantic v2 · asyncpg · pgvector · google-genai SDK · PyJWT · Jinja2 · pytest + pytest-asyncio + respx · ruff · mypy · uv · Supabase CLI · Docker · Fly.io · GitHub Actions.

**Spec:** `PRD.md` (v1.0, 2026-08-21) — read it first; every task below cites PRD requirement IDs (V-1, M-4, G-7, …).

---

## Global Constraints

Copied from the PRD; every phase plan inherits these.

- Python ≥ 3.12, type-checked with `mypy --strict` on `src/`; `ruff` clean.
- All LLM calls go through Gemini API only (PRD G4). Model IDs come from config: `GEMINI_CHAT_MODEL=gemini-2.5-flash`, `GEMINI_GUARD_MODEL=gemini-2.5-flash-lite` (PRD C-2).
- Chat path: `temperature 0.6`, `max_output_tokens 300`, `thinking_budget 0` (PRD C-3).
- Secrets never reach the browser (PRD AD1). Client holds only a Supabase JWT.
- Every user-owned table has RLS `auth.uid() = user_id` (PRD §8).
- Latency targets: TTFA p50 ≤ 800 ms, p95 ≤ 1500 ms (PRD G5).
- Guardrail gates: ≥ 99 % on red-team suite, ≤ 2 % false refusals on benign suite (PRD G-15).
- Voice replies ≤ 2 sentences by default; numbers spoken as words (PRD C-11).
- Max 600 chars per utterance; 60 turns / 10 min / user; 500 turns / day (PRD G-11, Layer 0/1).
- Memory values ≤ 200 chars, sanitised, PII-filtered (PRD G-4, M-10).
- OCEAN instrument = 20-item Mini-IPIP; scoring is deterministic Python, never LLM (PRD P-8).
- Commit after every green test cycle; conventional commit messages.

---

## 1. Deviation from the PRD you must know about

| PRD said | This plan does | Why |
|---|---|---|
| Backend = Supabase Edge Functions (Deno) | Backend = **Python FastAPI container on Fly.io** (region `ams`/`iad` matched to the Supabase project) | You asked for fully Python; Supabase cannot execute Python. Supabase still provides Postgres, pgvector, Auth, Storage/hosting, pg_cron. |
| Client = React/Vite | Client = **Jinja2 templates served by FastAPI + one vanilla-JS module `voice.js`** for mic/STT/TTS/SSE | Web Speech API is browser-JS-only; we isolate it to one file with no business logic. Everything else is Python. |
| `EdgeRuntime.waitUntil` for deferred writes | `asyncio.create_task` + a bounded background task registry | Same effect in Python. |
| Warm-ping via pg_cron → Edge Function | Fly.io `min_machines_running = 1` | Simpler; removes cold starts entirely. |

Everything else in the PRD (data model, API contract, guardrail layers, workflow engine, latency techniques, eval suites) is implemented as specified.

---

## 2. External services, APIs and keys

| Service | Used for | Credential(s) | Where it lives | How to obtain |
|---|---|---|---|---|
| **Google Gemini API** | Chat generation, guard classifier, OCEAN answer interpreter, narrative, embeddings | `GEMINI_API_KEY` | Fly secrets; `.env` locally | aistudio.google.com → Get API key. Use a **paid** project so prompts are excluded from training (PRD §11). Enable billing. |
| **Supabase** (project) | Postgres + pgvector, Auth (anonymous + magic link), Storage (static client), pg_cron | `SUPABASE_URL`, `SUPABASE_ANON_KEY` (client-safe), `SUPABASE_SERVICE_ROLE_KEY` (server only), `SUPABASE_JWT_SECRET` (server only, for verifying user JWTs), `DATABASE_URL` (pooled, port 6543, `?pgbouncer=true`), `DATABASE_URL_DIRECT` (port 5432, migrations) | Fly secrets; `.env` locally; anon key is templated into the client HTML | supabase.com → New project → Settings → API and Settings → Database. Enable Anonymous sign-ins under Auth → Providers. Enable `vector`, `pg_cron`, `pg_net` under Database → Extensions. |
| **Supabase CLI** | Migrations, local stack | `SUPABASE_ACCESS_TOKEN` (CI only) | GitHub secrets | `supabase login` locally; generate token at supabase.com/dashboard/account/tokens for CI. |
| **Open-Meteo** | Geocoding + forecast (primary weather provider) | none | — | No key. Rate limit ~10k req/day free; we cache 10 min. |
| **OpenWeatherMap** (optional fallback) | Weather fallback | `OWM_API_KEY` (optional) | Fly secrets | openweathermap.org → API keys. Leave unset to disable. |
| **Cloudflare Turnstile** | Captcha on anonymous sign-in (abuse control, PRD §11) | `TURNSTILE_SITE_KEY` (client), `TURNSTILE_SECRET_KEY` (configured in Supabase Auth → Bot protection) | Supabase Auth settings + client HTML | dash.cloudflare.com → Turnstile → Add site. |
| **SMTP for magic links** | Supabase Auth emails in prod (built-in sender is rate-limited to 3/hour) | SMTP host/user/pass (e.g. Resend: `RESEND_API_KEY`) | Supabase Auth → SMTP settings | resend.com or any SMTP. Not needed for local dev (Inbucket). |
| **Fly.io** | Hosting the Python API | `FLY_API_TOKEN` (CI) | GitHub secrets | `fly auth token`. Install `flyctl`. |
| **GitHub Actions** | CI/CD | the above CI secrets | repo settings | — |
| **Sentry** (optional) | Error tracking | `SENTRY_DSN` | Fly secrets | sentry.io. Skip if not wanted; code path is guarded. |

Environment variable reference (complete):

```
# LLM
GEMINI_API_KEY=
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_GUARD_MODEL=gemini-2.5-flash-lite
GEMINI_EMBED_MODEL=gemini-embedding-001

# Supabase
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_JWT_SECRET=
DATABASE_URL=postgresql://postgres.xxxx:PASS@aws-0-eu-central-1.pooler.supabase.com:6543/postgres
DATABASE_URL_DIRECT=postgresql://postgres:PASS@db.xxxx.supabase.co:5432/postgres

# Weather
WEATHER_PROVIDER=open-meteo        # open-meteo | owm | mock
OWM_API_KEY=

# Guardrails / behaviour
GUARD_MODE=enforce                 # enforce | shadow
HISTORY_LIMIT=12
SPECULATIVE_ENABLED=false
AUDIO_MODE=webspeech               # webspeech | gemini-tts (phase 9, optional)

# App
APP_ENV=dev                        # dev | staging | prod
APP_BASE_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:8000
LOG_LEVEL=INFO
SENTRY_DSN=
TURNSTILE_SITE_KEY=
```

---

## 3. DDD design

### 3.1 Bounded contexts and ubiquitous language

| Context | Core aggregate(s) | Key domain terms | Depends on (via ports only) |
|---|---|---|---|
| **Conversation** | `Session`, `Turn` (aggregate root for one exchange), `Message` | turn, utterance, reply, sentence, tool invocation, filler | Guardrails (InputGuard/OutputGuard ports), Memory (FactSnapshot port), Weather (tool), Assessment (ActiveRunSnapshot port), LLM port |
| **Memory** | `Memory` (fact) | fact, key, value, kind, recall, forget, snapshot | Embedding port (for notes, P1) |
| **Weather** | none (stateless) — value objects `Location`, `Forecast` | geocode, forecast, condition, units, grounding facts | WeatherProvider port, Cache port |
| **Assessment** | `WorkflowRun` (aggregate), `Instrument` (entity, versioned), `Answer` | item, scale, control (repeat/skip/back/explain/pause/quit), trait, band, narrative | AnswerInterpreter port (LLM), NarrativeWriter port (LLM) |
| **Guardrails** | `GuardDecision` (value object), `GuardEvent` | allow/block/uncertain, category, layer, grounding, leak, persona break | Classifier port (LLM) |
| **Shared kernel** | `UserId`, `SessionId`, `TurnId`, `Clock`, `Result`, `DomainEvent`, `Sentence` | — | — |

Rules:
- Domain layer imports **nothing** outside its own context and the shared kernel. No FastAPI, no asyncpg, no Gemini in `domain/`.
- Application layer defines **ports** (Protocols) and **use cases**; it depends on domain + shared kernel only.
- Infrastructure implements ports (Postgres repositories, Gemini client, Open-Meteo client).
- Contexts talk to each other only through application-level ports, wired in `container.py`. Conversation is the orchestrator and the only context that knows the others exist.

### 3.2 Repository layout (locked — later phases reference these paths)

```
sarjy/
├── pyproject.toml                 # uv-managed; ruff/mypy/pytest config
├── uv.lock
├── .env.example
├── Dockerfile
├── fly.toml
├── Makefile
├── PRD.md
├── docs/superpowers/plans/        # these plans
├── supabase/
│   ├── config.toml
│   ├── migrations/                # SQL, applied by supabase CLI
│   └── seed.sql                   # workflow_definitions seed
├── src/sarjy/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app factory
│   ├── config.py                  # pydantic-settings Settings
│   ├── container.py               # composition root (DI wiring)
│   ├── shared/                    # shared kernel
│   │   ├── ids.py                 # UserId, SessionId, TurnId, MessageId (NewType over UUID)
│   │   ├── clock.py               # Clock protocol + SystemClock + FakeClock
│   │   ├── result.py              # Ok/Err result type
│   │   ├── events.py              # DomainEvent base, EventBus protocol
│   │   ├── text.py                # Sentence VO, sentence splitter, number-to-words
│   │   └── errors.py              # DomainError hierarchy
│   ├── contexts/
│   │   ├── conversation/
│   │   │   ├── domain/            # session.py, turn.py, message.py, events.py
│   │   │   ├── application/       # ports.py, run_turn.py (use case), prompt_builder.py, tool_router.py
│   │   │   └── infrastructure/    # pg_session_repo.py, pg_message_repo.py, gemini_llm.py, prompts/*.md
│   │   ├── memory/
│   │   │   ├── domain/            # memory.py (aggregate), key_normalizer.py, pii_filter.py
│   │   │   ├── application/       # ports.py, remember.py, forget.py, recall.py, snapshot.py
│   │   │   └── infrastructure/    # pg_memory_repo.py, gemini_embedder.py
│   │   ├── weather/
│   │   │   ├── domain/            # location.py, forecast.py, units.py
│   │   │   ├── application/       # ports.py, get_weather.py
│   │   │   └── infrastructure/    # open_meteo.py, owm.py, mock_provider.py, pg_cache.py
│   │   ├── assessment/
│   │   │   ├── domain/            # instrument.py, workflow_run.py, answer.py, scoring.py, state_machine.py
│   │   │   ├── application/       # ports.py, start_run.py, answer_item.py, control_run.py, complete_run.py
│   │   │   └── infrastructure/    # pg_run_repo.py, pg_instrument_repo.py, gemini_interpreter.py, gemini_narrator.py
│   │   └── guardrails/
│   │       ├── domain/            # decision.py, categories.py, rules.py, leak_detector.py, grounding.py, persona.py, templates.py
│   │       ├── application/       # ports.py, input_guard.py, output_guard.py
│   │       └── infrastructure/    # gemini_classifier.py, pg_event_repo.py, pg_rate_limiter.py
│   ├── interfaces/
│   │   ├── http/                  # routers: chat.py (SSE), memory.py, telemetry.py, health.py; auth.py (JWT dep); sse.py
│   │   └── web/                   # templates/index.html, static/voice.js, static/app.css
│   └── observability/             # logging.py (structlog JSON), timings.py, sentry.py
├── tests/
│   ├── unit/<context>/...
│   ├── integration/               # needs local supabase
│   ├── evals/                     # redteam.jsonl, benign.jsonl, memory.jsonl, weather.jsonl, ocean.jsonl, run_evals.py
│   └── conftest.py
└── .github/workflows/ci.yml
```

### 3.3 Cross-context ports (the contracts every phase must honour)

```python
# src/sarjy/contexts/conversation/application/ports.py  (defined in Phase 2, implemented across phases)

class LLMPort(Protocol):
    def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]: ...
    async def generate_json(self, req: LLMRequest, schema: type[BaseModel]) -> BaseModel: ...

class FactSnapshotPort(Protocol):            # implemented by Memory context
    async def snapshot(self, user_id: UserId) -> list[Fact]: ...

class ToolPort(Protocol):                    # one per tool; registered in ToolRouter
    name: str
    declaration: dict                        # Gemini function declaration
    async def invoke(self, user_id: UserId, args: dict) -> ToolResult: ...

class InputGuardPort(Protocol):
    async def check(self, user_id: UserId, text: str, recent_user_turns: list[str]) -> GuardDecision: ...

class OutputGuardPort(Protocol):
    def check_sentence(self, sentence: str, ctx: GuardContext) -> SentenceVerdict: ...

class ActiveRunPort(Protocol):               # implemented by Assessment context
    async def active_run(self, user_id: UserId) -> ActiveRunSnapshot | None: ...
    async def handle_turn(self, user_id: UserId, text: str) -> AssessmentReply | None: ...
```

---

## 4. Phase plans (execute in order; each is independently testable and deployable)

| # | Plan file | Delivers | PRD coverage |
|---|---|---|---|
| 1 | `2026-08-21-01-foundation.md` | Repo, tooling, shared kernel, config, Supabase project + migrations + RLS, JWT auth dependency, FastAPI skeleton, Docker, first Fly deploy, CI skeleton | §8, §11 (auth/secrets), §12 |
| 2 | `2026-08-21-02-conversation-core.md` | Gemini streaming adapter, sentence splitter, SSE endpoint, `RunTurn` use case, session/message persistence, web client (`voice.js`) with STT/TTS/barge-in | §7.1, §7.2, §9.1, §10 (static blocks) |
| 3 | `2026-08-21-03-memory.md` | Memory aggregate, remember/forget/recall tools, snapshot injection, PII filter, memory REST + settings panel | §7.3, §9.2 |
| 4 | `2026-08-21-04-weather.md` | Open-Meteo adapter, cache, `get_weather` tool, disambiguation, numeric grounding facts | §7.4 |
| 5 | `2026-08-21-05-guardrails.md` | Rules engine, Gemini classifier, templates, stream-aware output guard (leak/persona/grounding), rate limiter, guard events, red-team & benign eval suites | §7.6, §14.3 |
| 6 | `2026-08-21-06-assessment-ocean.md` | Workflow engine state machine, Mini-IPIP instrument, answer interpreter, scoring, narrative, resume, results UI | §7.5 |
| 7 | `2026-08-21-07-latency-telemetry.md` | Client marks, server timings, single-RPC context load, fillers, speculative requests, prompt trimming, dashboard SQL | §7.7, §13 |
| 8 | `2026-08-21-08-deployment-hardening.md` | Full CI/CD (staging→prod), eval gates, retention cron, static hosting on Supabase Storage, alerts, launch checklist | §12, §13, §14, §15 |
| 9 | *(optional)* `gemini-tts` audio mode | Server-side Gemini TTS streamed as PCM | §7.7 deep-dive |

Each phase plan has its own header, file list, interfaces, and bite-sized TDD tasks. Phases 3–6 are independent of each other after Phase 2 and can be executed in parallel by separate workers if desired (they touch disjoint context directories and only register into `container.py` and `tool_router.py`).

---

## 5. Definition of Done (whole project)

- [ ] All phase plans' tasks checked.
- [ ] `make check` (ruff + mypy --strict + pytest unit) green.
- [ ] `make test-integration` green against local Supabase.
- [ ] `make evals` meets gates: red-team ≥ 99 %, benign ≤ 2 % FP, memory ≥ 95 %, weather 100 %, OCEAN 100 %.
- [ ] Latency dashboard shows p50 TTFA ≤ 800 ms / p95 ≤ 1500 ms over ≥ 50 real-browser turns.
- [ ] Prod URL reachable; PRD acceptance tests V, M, W, P, G, L pass manually.
- [ ] Launch checklist in Phase 8 signed.
