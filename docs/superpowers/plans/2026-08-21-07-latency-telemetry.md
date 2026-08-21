# Phase 7 — Latency & Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure time-to-first-audio end to end, then drive it to p50 ≤ 800 ms / p95 ≤ 1500 ms with the techniques in PRD §7.7: single-RPC context load, deferred writes, spoken fillers on tool turns, speculative requests on stable interim transcripts, prompt trimming, and a latency dashboard.

**Architecture:** Telemetry is a thin interface-layer concern (`/telemetry` endpoint + `telemetry_turns` table); latency work is mostly inside `RunTurn` (context load via `load_turn_context`, background persistence) and `voice.js` (marks, fillers, speculative send). No new bounded context — speculative-turn confirmation is a small addition to the Conversation application layer.

**Tech Stack:** `performance.mark`/`sendBeacon` (client), asyncpg RPC, `asyncio.create_task` registry, SQL views for percentiles.

**Spec:** `PRD.md` §7.7 (L-1…L-9, budget table), §13 (dashboards/alerts), §9.4.

## Global Constraints

- TTFA = `first_audio − speech_end` measured on the client (PRD §7.7 definition).
- Exactly one DB read round-trip on the hot path (L-7).
- Filler only if no first sentence within 700 ms of tool start (L-4).
- Speculative turns never persist until confirmed (L-3).
- Static prompt prefix ≤ 1,200 tokens (budget table).

---

## File structure

```
src/sarjy/contexts/conversation/application/context_loader.py      # single-RPC TurnContext
src/sarjy/contexts/conversation/application/speculation.py         # SpeculativeTurnCache
src/sarjy/contexts/conversation/infrastructure/pg_context_loader.py
src/sarjy/infrastructure_shared/background.py                      # BackgroundTasks registry
src/sarjy/interfaces/http/telemetry.py
src/sarjy/interfaces/web/static/fillers.js
supabase/migrations/20260821000600_latency_views.sql
tests/unit/conversation/test_context_loader.py, test_speculation.py
tests/unit/interfaces/test_telemetry.py
tests/integration/test_pg_context_loader.py
tests/latency/measure.py                                           # Playwright real-browser harness
```

Modified: `run_turn.py`, `ports.py` (`ContextLoaderPort`), `container.py`, `chat.py`, `voice.js`, `system_static.md`.

---

### Task 1: Client marks and `/telemetry` endpoint (L-1, L-2, §9.4)

**Files:**
- Create: `src/sarjy/interfaces/http/telemetry.py`
- Modify: `voice.js` (marks already partially collected in Phase 2 — complete them), `run_turn.py` (`DoneEvent.timings` already populated), `main.py`
- Test: `tests/unit/interfaces/test_telemetry.py`

**Interfaces:**
- `POST /telemetry` body `TelemetryIn{message_id: UUID|None, marks: {speech_end, request_sent, first_byte, first_sentence, first_audio, last_audio}: float ms (performance.now values), server_timings: dict|None, client_info: {ua, stt, tts, mode}}` → `204`. Derives `ttfa_ms = first_audio - speech_end`, `t_request_ms = request_sent - speech_end`, `t_first_byte_ms = first_byte - speech_end`, `t_first_sentence_ms = first_sentence - speech_end`, `t_last_audio_ms = last_audio - speech_end`. Inserts into `telemetry_turns` with `user_id`.

- [ ] **Step 1: Failing test**

```python
# tests/unit/interfaces/test_telemetry.py
import time, uuid
import jwt
from fastapi.testclient import TestClient
from sarjy.config import Settings
from sarjy.main import create_app

SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"


class MemTelemetry:
    def __init__(self) -> None:
        self.rows: list[dict] = []  # type: ignore[type-arg]

    async def save(self, **kw) -> None:  # type: ignore[no-untyped-def]
        self.rows.append(kw)


def test_telemetry_derives_ttfa() -> None:
    app = create_app(Settings(), connect_db=False)
    app.state.container.telemetry = MemTelemetry()
    tok = jwt.encode({"sub": str(uuid.uuid4()), "aud": "authenticated", "exp": int(time.time()) + 60}, SECRET, algorithm="HS256")
    with TestClient(app) as c:
        r = c.post("/telemetry", headers={"Authorization": f"Bearer {tok}"}, json={
            "message_id": None,
            "marks": {"speech_end": 1000.0, "request_sent": 1010.0, "first_byte": 1400.0, "first_sentence": 1500.0, "first_audio": 1650.5, "last_audio": 3000.0},
            "server_timings": {"t_gemini_first_token": 320}, "client_info": {"ua": "x", "stt": True, "tts": True, "mode": "voice"}})
    assert r.status_code == 204
    row = app.state.container.telemetry.rows[0]
    assert row["ttfa_ms"] == 650 and row["t_first_sentence_ms"] == 500
```

- [ ] **Step 2: Run → fails.** **Step 3: Implement**

```python
# src/sarjy/interfaces/http/telemetry.py
from __future__ import annotations

import json, uuid
from typing import Any, Protocol

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from sarjy.infrastructure_shared.db import Database
from sarjy.interfaces.http.auth import CurrentUserDep
from sarjy.shared.ids import UserId

router = APIRouter()


class Marks(BaseModel):
    speech_end: float
    request_sent: float | None = None
    first_byte: float | None = None
    first_sentence: float | None = None
    first_audio: float | None = None
    last_audio: float | None = None


class TelemetryIn(BaseModel):
    message_id: uuid.UUID | None = None
    marks: Marks
    server_timings: dict[str, int] | None = None
    client_info: dict[str, Any] = {}


class TelemetryRepo(Protocol):
    async def save(self, **kw: Any) -> None: ...


class PgTelemetryRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def save(self, **kw: Any) -> None:
        await self.db.execute(
            """insert into telemetry_turns (user_id,message_id,ttfa_ms,t_request_ms,t_first_byte_ms,t_first_sentence_ms,t_last_audio_ms,server_timings,client_info)
               values ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb)""",
            kw["user_id"], kw["message_id"], kw["ttfa_ms"], kw["t_request_ms"], kw["t_first_byte_ms"], kw["t_first_sentence_ms"],
            kw["t_last_audio_ms"], json.dumps(kw["server_timings"]), json.dumps(kw["client_info"]))


def _d(a: float | None, b: float) -> int | None:
    return None if a is None else int(round(a - b))


@router.post("/telemetry", status_code=204)
async def telemetry(body: TelemetryIn, user: CurrentUserDep, request: Request) -> Response:
    m = body.marks
    await request.app.state.container.telemetry.save(
        user_id=UserId(user.user_id), message_id=body.message_id, ttfa_ms=_d(m.first_audio, m.speech_end),
        t_request_ms=_d(m.request_sent, m.speech_end), t_first_byte_ms=_d(m.first_byte, m.speech_end),
        t_first_sentence_ms=_d(m.first_sentence, m.speech_end), t_last_audio_ms=_d(m.last_audio, m.speech_end),
        server_timings=body.server_timings or {}, client_info=body.client_info)
    return Response(status_code=204)
```

Container: `self.telemetry = PgTelemetryRepo(self.db)`; `main.py` includes router.

`voice.js` changes (inside `controller`): record `first_byte` on first chunk read, `first_sentence` on first `sentence` event, `last_audio` in `onSpeechQueueDrained`, keep `server_timings` from `done`; then:
```javascript
flushTelemetry() {
  const m = this.marks; if (!m.speech_end) return;
  const body = JSON.stringify({ message_id: this.lastMessageId, marks: m, server_timings: this.serverTimings,
    client_info: { ua: navigator.userAgent, stt: hasSTT, tts: hasTTS, mode: this.lastMode } });
  ensureSession().then(({ access_token }) => fetch(`${apiBase}/telemetry`, { method: "POST", keepalive: true,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${access_token}` }, body }));
  this.marks = {};
}
```
called from `onSpeechQueueDrained`. (`sendBeacon` cannot set the Authorization header, so use `fetch` with `keepalive`.)

- [ ] **Step 4: Run → pass.** **Step 5: Commit** `feat(telemetry): client marks and /telemetry ingestion`.

---

### Task 2: Percentile views and dashboard SQL (§13)

**Files:**
- Create: `supabase/migrations/20260821000600_latency_views.sql`

```sql
create or replace view public.v_latency_daily as
select date_trunc('day', created_at) as day,
       client_info->>'mode' as mode,
       count(*) as turns,
       percentile_cont(0.5) within group (order by ttfa_ms) as ttfa_p50,
       percentile_cont(0.95) within group (order by ttfa_ms) as ttfa_p95,
       percentile_cont(0.5) within group (order by t_first_byte_ms) as first_byte_p50,
       percentile_cont(0.5) within group (order by (server_timings->>'t_gemini_first_token')::int) as gemini_first_token_p50,
       percentile_cont(0.5) within group (order by t_last_audio_ms) as turn_p50
from public.telemetry_turns where ttfa_ms is not null group by 1,2 order by 1 desc;

create or replace view public.v_latency_by_browser as
select case when client_info->>'ua' ilike '%chrome%' and client_info->>'ua' not ilike '%edg%' then 'chrome'
            when client_info->>'ua' ilike '%edg%' then 'edge'
            when client_info->>'ua' ilike '%safari%' then 'safari' else 'other' end as browser,
       count(*) turns,
       percentile_cont(0.5) within group (order by ttfa_ms) as ttfa_p50,
       percentile_cont(0.95) within group (order by ttfa_ms) as ttfa_p95
from public.telemetry_turns where created_at > now() - interval '7 days' group by 1;

create or replace view public.v_guard_daily as
select date_trunc('day', created_at) day, layer, kind, action, count(*) n
from public.guardrail_events group by 1,2,3,4 order by 1 desc, 5 desc;

create or replace view public.v_ocean_funnel as
select date_trunc('day', started_at) day,
       count(*) filter (where status in ('proposed','abandoned','active','paused','scoring','complete')) proposed,
       count(*) filter (where status in ('active','paused','scoring','complete')) started,
       count(*) filter (where status = 'complete') completed
from public.workflow_runs group by 1 order by 1 desc;

revoke all on public.v_latency_daily, public.v_latency_by_browser, public.v_guard_daily, public.v_ocean_funnel from anon, authenticated;
```

- [ ] Apply (`supabase db reset` locally, `supabase db push` staging); add a `GET /admin/latency` JSON endpoint in `telemetry.py` guarded by a `ADMIN_USER_IDS` setting (comma-separated UUIDs) returning the four views — that is the "internal page" (PRD §13) for v1. Commit `feat(observability): latency, guard and funnel views`.

---

### Task 3: Single-RPC context loader and deferred persistence (L-7)

**Files:**
- Create: `application/context_loader.py`, `infrastructure/pg_context_loader.py`, `src/sarjy/infrastructure_shared/background.py`
- Modify: `ports.py` (add `ContextLoaderPort`), `run_turn.py`, `container.py`
- Test: `tests/unit/conversation/test_context_loader.py`, `tests/integration/test_pg_context_loader.py`

**Interfaces:**
- `TurnContext(facts: list[Fact], history: list[Message], workflow: ActiveRunSnapshot|None, profile: dict)`.
- `ContextLoaderPort.load(user_id, session_id, history_limit) -> TurnContext`.
- `PgContextLoader(db, active_run_port)` calls `select load_turn_context($1,$2,$3)` and maps JSON → `TurnContext`; the `workflow` JSON is converted to `ActiveRunSnapshot` via `active_run_port.snapshot_from_row(row)` (add this method to `ActiveRunPort`; Phase 6's adapter implements it, `NoActiveRun` returns None).
- `BackgroundTasks.spawn(coro)` keeps strong refs, logs exceptions, `await drain(timeout)` on shutdown.
- `RunTurn` changes: replace the three awaits in `_run` with one `await self.context.load(...)`; move `messages.save(assistant)` + `sessions.save(touch)` into `self.bg.spawn(...)` after `DoneEvent` is yielded (user message is still saved synchronously before tool calls because `tool_calls` FK needs it — keep it, it's one insert).

- [ ] **Step 1: Failing tests**

```python
# tests/unit/conversation/test_context_loader.py
import uuid
from sarjy.contexts.conversation.application.context_loader import TurnContext, context_from_rpc
from sarjy.contexts.conversation.infrastructure.noop_guards import NoActiveRun
from sarjy.shared.ids import SessionId, UserId


def test_maps_rpc_json() -> None:
    raw = {"memories": [{"k": "favorite_color", "v": "teal", "kind": "fact"}],
           "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
           "workflow": None, "profile": {"units": "metric"}}
    ctx = context_from_rpc(raw, UserId(uuid.uuid4()), SessionId(uuid.uuid4()), NoActiveRun())
    assert ctx.facts[0].key == "favorite_color" and [m.role for m in ctx.history] == ["user", "assistant"]
    assert ctx.workflow is None and ctx.profile["units"] == "metric"
```

```python
# tests/unit/conversation/test_background.py
import asyncio
from sarjy.infrastructure_shared.background import BackgroundTasks


async def test_spawn_and_drain_swallow_errors() -> None:
    bg = BackgroundTasks()
    done = []
    async def ok() -> None: done.append(1)
    async def boom() -> None: raise RuntimeError("x")
    bg.spawn(ok()); bg.spawn(boom())
    await bg.drain(timeout=1)
    assert done == [1] and bg.pending == 0
```

- [ ] **Step 2: Run → fails.** **Step 3: Implement**

```python
# src/sarjy/infrastructure_shared/background.py
from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from sarjy.observability.logging import get_logger

log = get_logger(__name__)


class BackgroundTasks:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        t = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._done)

    def _done(self, t: asyncio.Task[Any]) -> None:
        self._tasks.discard(t)
        if not t.cancelled() and t.exception():
            log.error("background_task_failed", error=repr(t.exception()))

    @property
    def pending(self) -> int:
        return len(self._tasks)

    async def drain(self, timeout: float = 5.0) -> None:
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=timeout)
```

```python
# src/sarjy/contexts/conversation/application/context_loader.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from sarjy.contexts.conversation.application.ports import ActiveRunPort, ActiveRunSnapshot, Fact
from sarjy.contexts.conversation.domain.message import Message
from sarjy.shared.ids import MessageId, SessionId, UserId, new_id


@dataclass(slots=True)
class TurnContext:
    facts: list[Fact]
    history: list[Message]
    workflow: ActiveRunSnapshot | None
    profile: dict[str, Any] = field(default_factory=dict)


class ContextLoaderPort(Protocol):
    async def load(self, user_id: UserId, session_id: SessionId, history_limit: int) -> TurnContext: ...


def context_from_rpc(raw: dict[str, Any], user_id: UserId, session_id: SessionId, runs: ActiveRunPort) -> TurnContext:
    facts = [Fact(m["k"], m["v"], m["kind"]) for m in raw.get("memories") or []]
    now = datetime.now(UTC)
    history = [Message(id=new_id(MessageId), session_id=session_id, user_id=user_id, role=h["role"], content=h["content"], created_at=now)
               for h in raw.get("history") or []]
    wf = runs.snapshot_from_row(raw["workflow"]) if raw.get("workflow") else None
    return TurnContext(facts=facts, history=history, workflow=wf, profile=raw.get("profile") or {})
```

```python
# src/sarjy/contexts/conversation/infrastructure/pg_context_loader.py
import json
from sarjy.contexts.conversation.application.context_loader import TurnContext, context_from_rpc
from sarjy.contexts.conversation.application.ports import ActiveRunPort
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import SessionId, UserId


class PgContextLoader:
    def __init__(self, db: Database, runs: ActiveRunPort) -> None:
        self.db, self.runs = db, runs

    async def load(self, user_id: UserId, session_id: SessionId, history_limit: int) -> TurnContext:
        raw = await self.db.fetchval("select public.load_turn_context($1,$2,$3)", user_id, session_id, history_limit)
        return context_from_rpc(json.loads(raw) if isinstance(raw, str) else raw, user_id, session_id, self.runs)
```

`RunTurn` diff (in `_run`):
```python
        with t.stage("context"):
            ctx = await self.context.load(inp.user_id, session.id, self.s.history_limit)
        history, facts, run = ctx.history, ctx.facts, ctx.workflow
        ...
        # at the end, instead of awaiting assistant persistence:
        self.bg.spawn(self._persist_assistant(...))
        yield DoneEvent(mid, t.as_dict())
```
`RunTurn.__init__` gains `context: ContextLoaderPort` and `bg: BackgroundTasks`. An in-memory `InMemoryContextLoader(facts_port, messages, active_run)` is added to `infrastructure/memory_repos.py` for unit tests (it composes the three calls). `Container.shutdown` awaits `self.bg.drain()`.

- [ ] **Step 4: Run unit + integration (`test_pg_context_loader.py`: insert a memory + 2 messages, assert loader returns them).** **Step 5: Commit** `perf(conversation): single-RPC context load and deferred persistence`.

---

### Task 4: Spoken fillers on tool turns (L-4)

**Files:**
- Create: `src/sarjy/interfaces/web/static/fillers.js`
- Modify: `voice.js`

```javascript
// fillers.js — local, never from the server; spoken only if no sentence arrives within 700 ms of tool start
export const FILLERS = { get_weather: ["Let me check.", "One sec, checking the weather.", "Checking now."],
                         default: ["One moment.", "Let me see.", "Just a sec."] };
export function pickFiller(tool) { const l = FILLERS[tool] || FILLERS.default; return l[Math.floor(Math.random() * l.length)]; }
```
`voice.js` `handle()`:
```javascript
case "tool_status":
  if (data.state === "start") { chip(`⚙ ${data.tool}`);
    this.fillerTimer = setTimeout(() => { if (!this.firstSentenceSeen) { tts.enqueue(pickFiller(data.tool), onFirstAudio); this.set("speaking"); } }, 700); }
  break;
case "sentence":
  clearTimeout(this.fillerTimer); this.firstSentenceSeen = true; ...
```
Reset `firstSentenceSeen=false` at `send()`. Manual test: weather question with provider slowed (`WEATHER_PROVIDER=mock` has a `MOCK_DELAY_MS` env, add it in the mock provider) → filler is spoken, then the answer. Commit `feat(web): spoken fillers while tools run`.

---

### Task 5: Speculative requests on stable interim transcript (L-3)

**Files:**
- Create: `application/speculation.py`
- Modify: `run_turn.py`, `chat.py` (`POST /chat/confirm`), `voice.js`
- Test: `tests/unit/conversation/test_speculation.py`

**Interfaces:**
- `SpeculativeTurnCache(ttl_s=10)`: `put(client_turn_id, normalized_text, pending: PendingPersist)`, `take(client_turn_id, final_text) -> PendingPersist | None` (returns only if `normalize(final) == normalize(stored)`; normalisation = lowercase, strip punctuation/whitespace).
- `PendingPersist(user_msg: Message, assistant_msg: Message, tool_calls: list[...])`.
- Server flow: `speculative=true` → `RunTurn` runs normally but `_persist` stores `PendingPersist` into the cache instead of writing. Client, on final STT result: if it matches the speculative text → `POST /chat/confirm {client_turn_id, text}` → server `take()` and persists (204) — audio already played. If it doesn't match → client aborts the speculative stream (if still open), sends a normal turn with a **new** `client_turn_id`; the orphaned speculative entry expires via TTL (never persisted — PRD L-3 guarantee).
- Client: in `listenOnce.onresult`, keep `stableSince`; if interim unchanged for 400 ms and ≥ 3 words and `SPECULATIVE_ENABLED` (templated from settings into `window.SARJY.speculative`), call `controller.send(interim, "voice", {speculative: true})` once; on final, compare.

- [ ] **Step 1: Failing test**

```python
# tests/unit/conversation/test_speculation.py
from sarjy.contexts.conversation.application.speculation import SpeculativeTurnCache


def test_take_matches_normalised_text() -> None:
    c = SpeculativeTurnCache()
    c.put("t1", "What's the weather in Paris", pending="P")  # type: ignore[arg-type]
    assert c.take("t1", "whats the weather in paris?") == "P"
    assert c.take("t1", "again") is None  # consumed


def test_take_rejects_mismatch() -> None:
    c = SpeculativeTurnCache()
    c.put("t2", "what's the weather in Paris", pending="P")  # type: ignore[arg-type]
    assert c.take("t2", "what's the weather in Rome") is None
```

- [ ] **Step 2–3: Implement**

```python
# src/sarjy/contexts/conversation/application/speculation.py
from __future__ import annotations

import re, time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")
_P = re.compile(r"[^a-z0-9 ]+")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", _P.sub("", s.lower())).strip()


@dataclass(slots=True)
class _Entry(Generic[T]):
    text: str
    pending: T
    at: float


class SpeculativeTurnCache(Generic[T]):
    def __init__(self, ttl_s: float = 10.0) -> None:
        self._ttl, self._d = ttl_s, {}  # type: ignore[var-annotated]

    def put(self, client_turn_id: str, text: str, pending: T) -> None:
        self._gc()
        self._d[client_turn_id] = _Entry(_norm(text), pending, time.monotonic())

    def take(self, client_turn_id: str, final_text: str) -> T | None:
        self._gc()
        e = self._d.pop(client_turn_id, None)
        return e.pending if e and e.text == _norm(final_text) else None

    def _gc(self) -> None:
        now = time.monotonic()
        for k in [k for k, v in self._d.items() if now - v.at > self._ttl]:
            del self._d[k]
```

`RunTurn`: new kwarg `speculation: SpeculativeTurnCache[PendingPersist]`; in `_persist`, when `inp.speculative`, build `PendingPersist` and `put`; add method `async confirm(client_turn_id, final_text) -> bool` that `take`s and writes. Note: for speculative turns, the user message must also not be written before tool calls; buffer tool-call rows in `PendingPersist.tool_calls` instead (branch on `inp.speculative` at the two write sites).

`chat.py`:
```python
class ConfirmIn(BaseModel):
    client_turn_id: str
    text: str

@router.post("/chat/confirm", status_code=204)
async def confirm(body: ConfirmIn, user: CurrentUserDep, request: Request) -> Response:
    ok = await request.app.state.container.run_turn.confirm(body.client_turn_id, body.text)
    return Response(status_code=204 if ok else 409)
```

- [ ] **Step 4: Run; manual test with `SPECULATIVE_ENABLED=true`:** speak a sentence and pause — the reply starts before the recogniser finalises; the transcript/DB shows exactly one user + one assistant row. **Step 5: Commit** `perf: speculative turns on stable interim transcripts`.

---

### Task 6: Prompt trimming and Gemini context caching (L-6, budget)

- [ ] Count tokens of `system_static.md` with `client.models.count_tokens`; if > 1,200, tighten wording (keep every rule, cut examples). Record the count in the commit.
- [ ] If the static prefix ≥ 1,024 tokens (current Gemini minimum for explicit caching on Flash), enable explicit caching: in `GeminiLLM`, add `cached_system: str | None`; on first use create `client.aio.caches.create(model, config={"system_instruction": static, "tools": tools, "ttl": "3600s"})`, store the cache name, and pass `cached_content=<name>` in `GenerateContentConfig` while appending only the dynamic blocks (`<facts>…`, `<workflow>…`) as a leading user message prefixed `Context for this turn:`. Refresh the cache when `PromptBuilder.static_text` hash changes or the cache returns 404. If the prefix is under the minimum, skip caching (document the decision in `container.py`).
- [ ] Commit `perf(llm): static prompt ≤1.2k tokens; explicit context caching when eligible`.

---

### Task 7: Real-browser latency harness (PRD acceptance test L)

**Files:**
- Create: `tests/latency/measure.py` (Playwright, Chromium with `--use-fake-device-for-media-stream --use-fake-ui-for-media-stream --use-file-for-fake-audio-capture=tests/latency/fixtures/weather_paris.wav`), 5 WAV prompts, runs N=50 turns against a target URL, reads `telemetry_turns` afterwards (or intercepts `/telemetry` requests), prints p50/p95 per stage.

- [ ] `uv add --dev playwright && uv run playwright install chromium`.
- [ ] Write `measure.py` (~120 lines): opens the page, clicks mic, waits for `done`, records `marks` via `page.evaluate(() => controller.lastMarks)` (expose `window.__sarjy = controller` in `voice.js` when `?debug=1`), loops.
- [ ] Run against staging: `uv run python tests/latency/measure.py https://sarjy-staging.fly.dev --n 50`. Expected after Tasks 3–6: p50 ≤ 800 ms, p95 ≤ 1500 ms. If not, use the per-stage table to find the dominant stage (the budget in PRD §7.7) and fix it before moving on: typical culprits are Gemini first-token (> 500 ms → check region, thinking budget, prompt size), or TTS start (> 250 ms → voice choice, priming).
- [ ] Commit `test(latency): playwright TTFA harness + results (p50 X ms, p95 Y ms)`.

---

## Phase 7 self-review

- Spec coverage: L-1 ✔ T1, L-2 ✔ (Phase 2 `DoneEvent.timings` + T1 storage), L-3 ✔ T5, L-4 ✔ T4, L-5 ✔ (Phase 5 templates), L-6 ✔ T6, L-7 ✔ T3, L-8 ✔ (Fly `min_machines_running=1`, Phase 1), L-9 ✔ (`preconnect` Phase 2; proactive token refresh — add `sb.auth.onAuthStateChange` no-op subscription which keeps the session refreshed; supabase-js auto-refreshes), §13 views ✔ T2, alerts → Phase 8.
- Type consistency: `ActiveRunPort.snapshot_from_row(row: dict) -> ActiveRunSnapshot | None` is a new port method — Phase 6 adapter and `NoActiveRun` must implement it (noted for executors). `RunTurn` kwargs now include `context`, `bg`, `speculation`.
