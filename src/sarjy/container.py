from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import httpx

from sarjy.config import Settings
from sarjy.contexts.assessment.application.active_run_adapter import ActiveRunAdapter
from sarjy.contexts.assessment.application.control_run import ControlRun
from sarjy.contexts.assessment.application.handle_turn import HandleAssessmentTurn
from sarjy.contexts.assessment.application.ports import (
    AnswerInterpreterPort,
    InstrumentRepo,
    NarratorPort,
    RunRepo,
)
from sarjy.contexts.assessment.application.start_run import StartRun
from sarjy.contexts.assessment.application.tools import StartWorkflowTool, WorkflowControlTool
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.infrastructure.gemini_interpreter import GeminiAnswerInterpreter
from sarjy.contexts.assessment.infrastructure.gemini_narrator import GeminiNarrator
from sarjy.contexts.assessment.infrastructure.memory_repos import MemInstrumentRepo, MemRunRepo
from sarjy.contexts.assessment.infrastructure.offline_interpreter import OfflineInterpreter
from sarjy.contexts.assessment.infrastructure.offline_narrator import OfflineNarrator
from sarjy.contexts.assessment.infrastructure.pg_instrument_repo import PgInstrumentRepo
from sarjy.contexts.assessment.infrastructure.pg_run_repo import PgRunRepo
from sarjy.contexts.conversation.application.context_loader import ContextLoaderPort
from sarjy.contexts.conversation.application.ports import (
    ActiveRunPort,
    FactSnapshotPort,
    InputGuardPort,
    LLMPort,
    MessageRepo,
    OutputGuardPort,
    RefusalPort,
    SessionRepo,
)
from sarjy.contexts.conversation.application.prompt_builder import PromptBuilder
from sarjy.contexts.conversation.application.run_turn import RunTurn
from sarjy.contexts.conversation.application.tool_router import ToolRouter
from sarjy.contexts.conversation.infrastructure.gemini_llm import GeminiLLM
from sarjy.contexts.conversation.infrastructure.memory_repos import (
    InMemoryContextLoader,
    MemMessages,
    MemSessions,
)
from sarjy.contexts.conversation.infrastructure.noop_guards import NoActiveRun, NoFacts
from sarjy.contexts.conversation.infrastructure.pg_context_loader import PgContextLoader
from sarjy.contexts.conversation.infrastructure.pg_message_repo import PgMessageRepo
from sarjy.contexts.conversation.infrastructure.pg_session_repo import PgSessionRepo
from sarjy.contexts.guardrails.application.audit import AuditWorker
from sarjy.contexts.guardrails.application.input_guard import InputGuard
from sarjy.contexts.guardrails.application.output_guard import OutputGuard
from sarjy.contexts.guardrails.application.ports import (
    AuditQueuePort,
    ClassifierPort,
    GuardEventRepo,
)
from sarjy.contexts.guardrails.application.refusals import TemplateRefusals
from sarjy.contexts.guardrails.domain.rules import DEFAULT_RULES, RuleEngine
from sarjy.contexts.guardrails.infrastructure.gemini_classifier import GeminiClassifier
from sarjy.contexts.guardrails.infrastructure.memory_event_repo import MemGuardEvents
from sarjy.contexts.guardrails.infrastructure.offline_classifier import OfflineClassifier
from sarjy.contexts.guardrails.infrastructure.pg_audit_repo import PgAuditRepo
from sarjy.contexts.guardrails.infrastructure.pg_event_repo import PgGuardEventRepo
from sarjy.contexts.guardrails.infrastructure.pg_rate_limiter import PgRateLimiter
from sarjy.contexts.guardrails.infrastructure.value_screen import RuleEngineValueScreen
from sarjy.contexts.memory.application.edit import EditFact
from sarjy.contexts.memory.application.forget import ForgetFact
from sarjy.contexts.memory.application.ports import MemoryRepo
from sarjy.contexts.memory.application.recall import RecallFacts
from sarjy.contexts.memory.application.remember import RememberFact
from sarjy.contexts.memory.application.snapshot import FactSnapshot
from sarjy.contexts.memory.application.tools import ForgetTool, RecallTool, RememberTool
from sarjy.contexts.memory.infrastructure.in_memory_repo import InMemoryMemoryRepo
from sarjy.contexts.memory.infrastructure.pg_memory_repo import PgMemoryRepo
from sarjy.contexts.weather.application.get_weather import GetWeather
from sarjy.contexts.weather.application.intent import is_weather_question
from sarjy.contexts.weather.application.ports import WeatherCache, WeatherProvider
from sarjy.contexts.weather.application.tools import GetWeatherTool
from sarjy.contexts.weather.application.units_resolver import UnitsResolver
from sarjy.contexts.weather.infrastructure.in_memory_cache import InMemoryWeatherCache
from sarjy.contexts.weather.infrastructure.mock_provider import MockProvider
from sarjy.contexts.weather.infrastructure.open_meteo import OpenMeteoProvider
from sarjy.contexts.weather.infrastructure.owm import OwmProvider
from sarjy.contexts.weather.infrastructure.pg_cache import PgWeatherCache
from sarjy.infrastructure_shared.background import BackgroundTasks
from sarjy.infrastructure_shared.db import Database
from sarjy.observability.admin_repo import AdminLatencyRepo, PgAdminLatencyRepo
from sarjy.observability.telemetry_repo import MemTelemetry, PgTelemetryRepo, TelemetryRepo
from sarjy.shared.clock import Clock, SystemClock
from sarjy.shared.events import EventBus, InMemoryEventBus

# container.py -> sarjy -> src -> repo root, same three hops every assessment
# unit test uses (`Path(__file__).parents[3]` from a test three levels under
# tests/).
_MINI_IPIP_PATH = Path(__file__).resolve().parent.parent.parent / "supabase" / "mini_ipip.json"


@lru_cache(maxsize=1)
def _mini_ipip_seed() -> dict[str, Instrument]:
    """The Mini-IPIP instrument, loaded once and cached process-wide.

    `MemInstrumentRepo` — the in-memory stand-in for `PgInstrumentRepo` used
    whenever `connect_db` is false — has nothing else to serve `Instrument
    Repo.get` from; this is the same `supabase/mini_ipip.json` the Postgres
    seed migration loads. Read-only after construction, so sharing the one
    cached `Instrument` across every container built in a process (tests
    build many) is safe.
    """
    definition = json.loads(_MINI_IPIP_PATH.read_text(encoding="utf-8"))
    ins = Instrument.from_definition(definition)
    return {ins.id: ins}


@dataclass
class Container:
    """Composition root. Phases add adapters/use cases here; nothing else does wiring."""

    settings: Settings
    db: Database
    clock: Clock = field(default_factory=SystemClock)
    event_bus: EventBus = field(default_factory=InMemoryEventBus)
    connect_db: bool = True

    llm: LLMPort | None = None
    tools: ToolRouter = field(default_factory=ToolRouter)
    prompt_builder: PromptBuilder = field(default_factory=PromptBuilder)
    input_guard: InputGuardPort | None = None
    output_guard: OutputGuardPort | None = None
    rule_engine: RuleEngine | None = None
    classifier: ClassifierPort | None = None
    guard_events: GuardEventRepo | None = None
    refusals: RefusalPort | None = None
    facts: FactSnapshotPort = field(default_factory=NoFacts)
    active_run: ActiveRunPort = field(default_factory=NoActiveRun)
    context: ContextLoaderPort | None = None
    # Every fire-and-forget write a turn defers (the assistant transcript row,
    # the session touch) lands here, so `shutdown` has one thing to drain.
    bg: BackgroundTasks = field(default_factory=BackgroundTasks)
    sessions: SessionRepo | None = None
    messages: MessageRepo | None = None
    memory_repo: MemoryRepo | None = None
    http_client: httpx.AsyncClient = field(default_factory=lambda: httpx.AsyncClient(timeout=3.0))
    weather_provider: WeatherProvider | None = None
    weather_fallback: WeatherProvider | None = None
    weather_cache: WeatherCache | None = None
    rate_limiter: PgRateLimiter | None = None
    force_tool_when: Callable[[str], str | None] | None = None
    # Repos/adapters for the assessment (OCEAN) workflow — see `rebuild_assessment`.
    # A caller (a test, the eval runner) can pre-set any of these before
    # `__post_init__`/`rebuild_assessment` runs and have that win.
    run_repo: RunRepo | None = None
    instrument_repo: InstrumentRepo | None = None
    interpreter: AnswerInterpreterPort | None = None
    narrator: NarratorPort | None = None
    telemetry: TelemetryRepo | None = None
    admin_repo: AdminLatencyRepo | None = None
    audit_queue: AuditQueuePort | None = None
    audit_worker: AuditWorker | None = None
    remember_fact: RememberFact = field(init=False)
    edit_fact: EditFact = field(init=False)
    forget_fact: ForgetFact = field(init=False)
    recall_facts: RecallFacts = field(init=False)
    get_weather: GetWeather = field(init=False)
    run_turn: RunTurn = field(init=False)

    @classmethod
    def build(cls, settings: Settings, connect_db: bool = True) -> Container:
        return cls(
            settings=settings,
            db=Database(settings.database_url.get_secret_value()),
            connect_db=connect_db,
        )

    def __post_init__(self) -> None:
        # Explicit context caching (Phase 7 Task 6, L-6): `GeminiLLM.__init__` takes a
        # `cached_content` cache-name hook, but nothing here creates or passes one yet.
        # `PromptBuilder.static_text` sits at ~1,056 estimated tokens — above the
        # ~1,024-token minimum Flash requires for explicit caching — so this prefix
        # would in principle be eligible. It stays unset because there is no Gemini
        # API key in this environment to create a cache against, verify its `ttl`/404
        # refresh behaviour, or confirm the estimate against `client.models.
        # count_tokens` (the heuristic in `sarjy.shared.tokens` is not a substitute for
        # that check — see its docstring). Wire `client.aio.caches.create(...)` here,
        # keyed on `PromptBuilder.build(...).hash`'s static portion, once a key is
        # available to test it against.
        self.llm = self.llm or GeminiLLM(
            self.settings.gemini_api_key.get_secret_value(),
            self.settings.gemini_chat_model,
            self.settings.gemini_first_token_timeout_s,
            self.settings.gemini_total_timeout_s,
        )
        self.sessions = self.sessions or PgSessionRepo(self.db)
        self.messages = self.messages or PgMessageRepo(self.db)
        self.memory_repo = self.memory_repo or PgMemoryRepo(self.db)
        self.telemetry = self.telemetry or PgTelemetryRepo(self.db)
        self.admin_repo = self.admin_repo or PgAdminLatencyRepo(self.db)
        self.rebuild_guards()
        self.rebuild_audit()
        if self.connect_db and self.rate_limiter is None:
            self.rate_limiter = PgRateLimiter(
                self.db,
                per_10min=self.settings.rate_limit_per_10min,
                per_day=self.settings.rate_limit_per_day,
                clock=self.clock,
            )
        self.rebuild_memory()
        self.rebuild_weather()
        self.rebuild_assessment()
        # Every rebuild_* above leaves run_turn stale on purpose: it depends on
        # all of them, so it is built once, here, rather than three times.
        self.rebuild_run_turn()

    def rebuild_guards(self, classifier: ClassifierPort | None = None) -> None:
        """Build the Layer 2/3 input guard, the Layer 6 output guard, and refusals.

        Like `rebuild_weather`, this only fills in pieces that are `None`, so a
        caller can pre-set `guard_events`/`input_guard`/`output_guard` and have
        those win — set the field back to `None` first to force a fresh one.
        Passing `classifier` replaces the Layer-3 classifier and forces the
        input guard to be rebuilt around it.

        The guards stay real even with no database behind them: without Postgres
        the event repo becomes `MemGuardEvents` rather than the guards becoming
        no-ops, so a local run (or a test through `/chat`) exercises the same
        rules, leak checks and grounding checks production does.

        Does NOT rebuild `run_turn` — see `rebuild_run_turn`.
        """
        if classifier is not None:
            self.classifier = classifier
            self.input_guard = None  # it holds the old classifier by reference
        if self.guard_events is None:
            self.guard_events = PgGuardEventRepo(self.db) if self.connect_db else MemGuardEvents()
        if self.classifier is None:
            # A separate, cheaper model on a far tighter first-token budget than
            # the chat LLM: the classifier sits in front of every ambiguous turn,
            # so a slow one is felt as latency on the whole conversation. Past
            # that budget `InputGuard` fails closed (G-12).
            self.classifier = GeminiClassifier(
                GeminiLLM(
                    self.settings.gemini_api_key.get_secret_value(),
                    self.settings.gemini_guard_model,
                    first_token_timeout_s=0.4,
                )
            )
        if self.rule_engine is None:
            self.rule_engine = RuleEngine(DEFAULT_RULES)
        if self.input_guard is None:
            self.input_guard = InputGuard(
                self.rule_engine,
                self.classifier,
                self.guard_events,
                mode=self.settings.guard_mode,
            )
        if self.output_guard is None:
            self.output_guard = OutputGuard(self.guard_events, mode=self.settings.guard_mode)
        self.refusals = self.refusals or TemplateRefusals()

    def rebuild_audit(self) -> None:
        """Build the Layer-7 audit sampling worker (PRD Layer 7), backing
        `POST /internal/audit/run`.

        Only when `connect_db`: the worker reads/writes `public.audit_queue`,
        which does not exist without Postgres, and — unlike guards/weather/
        assessment — there is no in-memory stand-in worth building here:
        nothing exercises the worker without a database behind it, so
        `audit_worker` simply stays `None` and the endpoint answers 503.

        Must run AFTER `rebuild_guards`: it reuses `self.classifier` (the same
        Layer-3 model `InputGuard` escalates to) and `self.guard_events`, both
        of which `rebuild_guards` is what fills in.

        The classifier is `self.classifier` in every environment except
        `app_env == "test"`, where it is `OfflineClassifier` instead:
        `/internal/audit/run` is reachable over HTTP, and an automated test
        run must never be one stray request away from a live Gemini call —
        the same reasoning `use_in_memory_repos` applies to the classifier
        `rebuild_guards` would otherwise pick.
        """
        if not self.connect_db:
            return
        if self.audit_queue is None:
            self.audit_queue = PgAuditRepo(self.db)
        if self.audit_worker is None:
            assert self.guard_events is not None and self.classifier is not None
            clf = OfflineClassifier() if self.settings.app_env == "test" else self.classifier
            self.audit_worker = AuditWorker(self.audit_queue, clf, self.guard_events)

    def rebuild_memory(self) -> None:
        """Build the memory use cases/tools off `self.memory_repo`.

        Must run AFTER `rebuild_guards`: `RememberFact` and `EditFact` (the
        latter behind `PATCH /memory/{id}` — fix round 1, Critical 1: that
        route used to write straight through the repo, unscreened) are both
        wired with the same `RuleEngineValueScreen` over `self.rule_engine`
        (Phase 8 Task 6b), so a value smuggled into a "remember that ..."
        fact OR a PATCH edit is screened by the same Layer-2 rules a normal
        turn would run, before it is stored and re-injected into every later
        prompt. `self.guard_events`/`self.bg` are passed through too, so a
        refusal writes a `guardrail_events` row the same way `InputGuard`
        does (fire-and-forget, drained by `self.bg.drain()` at shutdown). The
        memory context itself never imports the guardrails context —
        `RuleEngineValueScreen` is a guardrails-infrastructure adapter behind
        `ValueScreenPort`.

        Does NOT rebuild `run_turn` — see `rebuild_run_turn`.
        """
        assert self.memory_repo is not None
        assert self.rule_engine is not None
        assert self.guard_events is not None
        screen = RuleEngineValueScreen(self.rule_engine, self.guard_events, self.bg)
        self.remember_fact = RememberFact(self.memory_repo, self.clock, screen=screen)
        self.edit_fact = EditFact(self.memory_repo, self.clock, screen=screen)
        self.forget_fact = ForgetFact(self.memory_repo, self.clock)
        self.recall_facts = RecallFacts(self.memory_repo)
        self.facts = FactSnapshot(self.memory_repo)
        for tool in (
            RememberTool(self.remember_fact),
            ForgetTool(self.forget_fact),
            RecallTool(self.recall_facts),
        ):
            self.tools.register(tool)

    def rebuild_weather(self) -> None:
        """Build the weather provider(s)/cache/use case and register `GetWeatherTool`.

        Only fills in pieces that are `None`, so a caller (a test, the eval
        runner) can pre-set `weather_provider`/`weather_fallback`/`weather_cache`
        before calling this and have those win — set the field back to `None`
        first to force a fresh one to be built instead.

        Does NOT rebuild `run_turn` — see `rebuild_run_turn`.
        """
        s = self.settings
        if self.weather_provider is None:
            if s.weather_provider == "mock":
                self.weather_provider = MockProvider(self.clock)
            elif s.weather_provider == "owm":
                if not s.owm_api_key:
                    raise ValueError("owm_api_key must be set when weather_provider='owm'")
                self.weather_provider = OwmProvider(
                    self.http_client, s.owm_api_key.get_secret_value(), self.clock
                )
            else:
                self.weather_provider = OpenMeteoProvider(self.http_client, self.clock)
        if self.weather_fallback is None and s.owm_api_key and s.weather_provider == "open-meteo":
            self.weather_fallback = OwmProvider(
                self.http_client, s.owm_api_key.get_secret_value(), self.clock
            )
        if self.weather_cache is None:
            self.weather_cache = (
                PgWeatherCache(self.db, self.clock)
                if self.connect_db
                else InMemoryWeatherCache(self.clock)
            )
        self.get_weather = GetWeather(
            self.weather_provider, self.weather_fallback, self.weather_cache, self.clock
        )
        self.tools.register(GetWeatherTool(self.get_weather, self.facts, UnitsResolver()))
        self.force_tool_when = lambda t: "get_weather" if is_weather_question(t) else None

    def rebuild_assessment(self) -> None:
        """Build the assessment (OCEAN workflow) repos/use cases and register the
        `start_workflow` / `workflow_control` tools.

        `run_repo`/`instrument_repo`/`interpreter`/`narrator` only get filled in
        when `None` — the same rule `rebuild_weather` follows — so a caller (a
        test, the eval runner) can pre-set any of them before calling this and
        have those win. `interpreter`/`narrator` stay Gemini-backed even with no
        database behind them (`self.connect_db` only decides the *repos*): like
        the Layer-3 classifier, a local run without Postgres should still
        exercise the real interpreter/narrator, not a no-op standing in for
        them. `use_in_memory_repos()` is the one caller that wants the no-op —
        it sets `interpreter`/`narrator` to the offline adapters itself before
        calling this, the same way it forces `OfflineClassifier` on
        `rebuild_guards`.

        Does NOT rebuild `run_turn` — see `rebuild_run_turn`.
        """
        self.run_repo = self.run_repo or (PgRunRepo(self.db) if self.connect_db else MemRunRepo())
        self.instrument_repo = self.instrument_repo or (
            PgInstrumentRepo(self.db) if self.connect_db else MemInstrumentRepo(_mini_ipip_seed())
        )
        if self.interpreter is None:
            self.interpreter = GeminiAnswerInterpreter(
                GeminiLLM(
                    self.settings.gemini_api_key.get_secret_value(),
                    self.settings.gemini_guard_model,
                    first_token_timeout_s=1.5,
                )
            )
        if self.narrator is None:
            self.narrator = GeminiNarrator(
                GeminiLLM(
                    self.settings.gemini_api_key.get_secret_value(),
                    self.settings.gemini_chat_model,
                )
            )
        handle = HandleAssessmentTurn(
            self.run_repo, self.instrument_repo, self.interpreter, self.narrator, self.clock
        )
        start = StartRun(self.run_repo, self.instrument_repo, self.clock)
        control = ControlRun(self.run_repo, self.instrument_repo, self.clock, handle)
        self.active_run = ActiveRunAdapter(self.run_repo, self.instrument_repo, handle)
        self.tools.register(StartWorkflowTool(start))
        self.tools.register(WorkflowControlTool(control))

    def rebuild_context_loader(self) -> None:
        """Build the single-RPC turn-context loader (L-7).

        Fills `context` when it is `None`, the same rule the other `rebuild_*`
        builders follow, and additionally refreshes an in-memory loader the
        container built itself once the repos behind it have been replaced (see
        below) — so a caller can swap `messages`/`facts` and just call
        `rebuild_run_turn()`. Setting `context = None` forces a fresh one
        regardless. With a database behind us that
        is `PgContextLoader`: one `load_turn_context` round trip instead of the
        three sequential reads `RunTurn` used to open a turn with. Without one,
        `InMemoryContextLoader` composes those same three ports, so the seam is
        identical either way.

        `self.active_run` is passed twice on purpose: once as the thing that
        turns the RPC's `workflow` JSON into a snapshot, once as the source of a
        finished run's results for the in-memory path (the RPC returns those
        itself).
        """
        # A loader the container built itself holds `messages`/`facts`/
        # `active_run` by reference, exactly as `run_turn` does — so a caller
        # that swapped one of them (the eval runners do, directly) gets a fresh
        # loader rather than one still reading the objects it replaced. A loader
        # the caller supplied is left alone: they chose it, stale or not.
        if isinstance(self.context, InMemoryContextLoader) and (
            self.context.messages is not self.messages
            or self.context.facts is not self.facts
            or self.context.active_run is not self.active_run
        ):
            self.context = None
        if self.context is None:
            assert self.messages is not None
            self.context = (
                PgContextLoader(self.db, self.active_run)
                if self.connect_db
                else InMemoryContextLoader(
                    self.facts, self.messages, self.active_run, self.active_run
                )
            )

    def rebuild_run_turn(self) -> None:
        """Rebuild the turn orchestrator from the current adapters.

        `RunTurn` captures every dependency by reference at construction, so
        this has to run after anything it depends on is replaced. The
        `rebuild_guards`/`rebuild_memory`/`rebuild_weather` builders each leave
        it stale deliberately: a caller usually runs several of them in a row,
        and rebuilding after each one would build (and throw away) the same
        object three times. Call this once when done.
        """
        assert self.llm and self.sessions and self.messages
        assert self.input_guard and self.output_guard
        self.rebuild_context_loader()
        assert self.context is not None
        self.run_turn = RunTurn(
            llm=self.llm,
            prompt_builder=self.prompt_builder,
            tools=self.tools,
            input_guard=self.input_guard,
            output_guard=self.output_guard,
            context=self.context,
            active_run=self.active_run,
            sessions=self.sessions,
            messages=self.messages,
            clock=self.clock,
            settings=self.settings,
            refusals=self.refusals,
            force_tool_when=self.force_tool_when,
            bg=self.bg,
        )

    def use_in_memory_repos(self) -> None:
        """Test helper: swap Postgres-backed repos for in-memory ones."""
        self.sessions, self.messages, self.memory_repo = (
            MemSessions(),
            MemMessages(),
            InMemoryMemoryRepo(),
        )
        self.telemetry = MemTelemetry()
        # The guards themselves stay real — only where they *record* changes, so
        # a test through /chat still runs the production rules and can then read
        # back what was logged. Nulling the two guards forces `rebuild_guards` to
        # rebuild them against the new event repo (they hold it by reference).
        self.guard_events = MemGuardEvents()
        self.input_guard = None
        self.output_guard = None
        # `OfflineClassifier` rather than the Gemini one: an in-memory container
        # is by definition not talking to anything, and a fixture that happens to
        # trip an `uncertain` rule must not fire a real HTTPS request mid-test.
        # It raises rather than returning a benign verdict, so InputGuard fails
        # closed exactly as it would against an unreachable classifier.
        self.rebuild_guards(classifier=OfflineClassifier())
        # run_turn (and the memory use cases/tools/facts snapshot) hold the repos by
        # reference, so they have to be rebuilt or they keep writing to Postgres
        # after the swap. rebuild_weather() also needs to run again afterwards:
        # GetWeatherTool was built against the pre-swap `self.facts`, so it would
        # otherwise keep reading from the old (Postgres-backed) FactSnapshot.
        self.rebuild_memory()
        self.rebuild_weather()
        # Same reasoning as the classifier swap above, applied to the assessment
        # interpreter/narrator: an in-memory container must never place a live
        # Gemini call mid-test, and a bad guess recorded into someone's results
        # would be worse than the guard's fail-closed, so these degrade to
        # deterministic stand-ins instead — see `OfflineInterpreter`/
        # `OfflineNarrator`. Forcing fresh Mem repos (rather than leaving
        # whatever was already set) matches `rebuild_assessment`'s "only fills
        # in `None`" contract: without this, a container built with
        # `connect_db=True` and then switched to in-memory would keep its
        # Postgres-backed run/instrument repos.
        self.run_repo, self.instrument_repo = MemRunRepo(), MemInstrumentRepo(_mini_ipip_seed())
        self.interpreter, self.narrator = OfflineInterpreter(), OfflineNarrator()
        self.rebuild_assessment()
        # Built explicitly rather than via `rebuild_context_loader`: that one
        # picks its implementation off `connect_db`, which is still true for a
        # container that had a database and has just been switched away from it
        # — and a `PgContextLoader` here would read straight past every repo
        # this method just swapped.
        self.context = InMemoryContextLoader(
            self.facts, self.messages, self.active_run, self.active_run
        )
        self.rebuild_run_turn()

    async def startup(self) -> None:
        if self.connect_db:
            await self.db.connect()

    async def shutdown(self) -> None:
        # BOTH guards record as fire-and-forget background tasks — the output
        # guard because its caller is sync, the input guard because a user is
        # waiting on the decision (I1) — so drain them before the database goes
        # away, or the last turn's blocks and cuts are never written.
        for guard in (self.input_guard, self.output_guard):
            drain = getattr(guard, "drain", None)
            if drain is not None:
                await drain()
        # And the same for the turn's own deferred writes (L-7): the assistant
        # row and the session touch are spawned just before DoneEvent, so the
        # last turn of a process is precisely the one whose transcript is lost
        # if the pool closes first. Drained BEFORE `db.close()`, never after.
        await self.bg.drain()
        if self.connect_db:
            await self.db.close()
        await self.http_client.aclose()
