"""`RunTurn` — the conversation-turn orchestrator use case."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Literal

from sarjy.config import Settings
from sarjy.contexts.conversation.application.context_loader import (
    ContextLoaderPort,
    LastResults,
    TurnContext,
    asks_about_results,
    mentions_trait,
)
from sarjy.contexts.conversation.application.ports import (
    ActiveRunPort,
    ActiveRunSnapshot,
    Fact,
    FunctionCall,
    FunctionResponse,
    GuardContext,
    InputGuardPort,
    LLMFinished,
    LLMFunctionCall,
    LLMMessage,
    LLMPort,
    LLMRequest,
    LLMText,
    LLMTimeout,
    LLMUnavailable,
    MessageRepo,
    OutputGuardPort,
    RefusalPort,
    SessionRepo,
)
from sarjy.contexts.conversation.application.prompt_builder import PromptBuilder, sanitise_value
from sarjy.contexts.conversation.application.speculation import (
    PendingPersist,
    SpeculativeTurnCache,
    normalise,
)
from sarjy.contexts.conversation.application.tool_router import ToolRouter
from sarjy.contexts.conversation.domain.events import (
    DoneEvent,
    ErrorEvent,
    GuardEvent,
    SentenceEvent,
    SessionEvent,
    ToolStatusEvent,
    TurnEvent,
)
from sarjy.contexts.conversation.domain.message import Message
from sarjy.contexts.conversation.domain.session import Session
from sarjy.contexts.conversation.domain.turn import TurnInput
from sarjy.infrastructure_shared.background import BackgroundTasks
from sarjy.observability.logging import get_logger
from sarjy.observability.timings import Timings
from sarjy.shared.clock import Clock
from sarjy.shared.ids import MessageId, SessionId, UserId, new_id
from sarjy.shared.text import Sentence, SentenceSplitter, to_speech

log = get_logger(__name__)
MAX_TOOL_HOPS = 3

GENERIC_BLOCK = (
    "That's outside what I can help with. I'm good for weather, remembering things "
    "for you, a chat, or a quick personality test."
)
NO_OUTPUT_FALLBACK = "Sorry, I didn't come up with anything — could you try again?"
MEMORY_UNAVAILABLE = "My memory is temporarily unavailable, but I can still help."
TIMEOUT_CLOSER = "…let me stop there."
ERROR_CLOSER = "Sorry, I lost my train of thought."
# The subset of ErrorEvent codes a mid-turn failure can degrade into.
DegradeCode = Literal["gemini_unavailable", "timeout", "internal"]
ERROR_SPOKEN: dict[DegradeCode, str] = {
    "gemini_unavailable": "Sorry, I'm having trouble thinking right now. Try again in a moment?",
    "timeout": "Sorry, that took too long. Could you ask again?",
    "internal": "Something went wrong on my end.",
}


# What `RunTurn.confirm` can answer, and what `/chat/confirm` maps to a status:
# True -> 204 (written), "pending" -> 202 (recorded, the turn will settle it),
# False -> 409 (a parked guess the final transcript contradicts). See `confirm`.
ConfirmOutcome = bool | Literal["pending"]


def _spec_key(user_id: UserId, client_turn_id: str) -> str:
    """The key a speculative turn is parked under: a turn id is only ever its
    owner's, so a caller guessing at ids looks in a namespace of their own (C1)."""
    return f"{user_id}:{client_turn_id}"


@dataclass
class TurnState:
    """Everything `__call__` needs to salvage a turn that failed part-way through.

    `_run` mutates this in place as it goes, so when it raises `__call__` still knows
    what was already spoken, whether the paired user row was written, and which prompt
    produced the partial answer.
    """

    sentences: list[str] = field(default_factory=list)
    idx: int = 0
    prompt_hash: str | None = None
    user_saved: bool = False
    # State for the grounded-summary substitution in `_speak`.
    # `last_tool_summary` is the `spoken_summary` of the tool result the next
    # cut sentence can be attributed to — `None` whenever that attribution is
    # not safe to make (no summary yet, or two summaries in one round).
    # `sentences_since_tool` counts what has been spoken since that result
    # arrived: only the FIRST sentence after a tool call is plausibly the
    # answer that tool was called for, so a later one is silently cut instead
    # of answered with a summary about a different city or day.
    # `summary_used` caps substitution at one per turn.
    last_tool_summary: str | None = None
    sentences_since_tool: int = 0
    summary_used: bool = False
    # L-3. Whether this turn is ACTUALLY running as a speculation, which is not
    # the same question as whether the client asked for one (`inp.speculative`).
    # A turn that would mutate anything outside itself is disqualified — before
    # it starts if a workflow run is open, mid-flight if the model reaches for a
    # mutating tool (see `_promote`) — and every write site branches on this,
    # never on the request. It lives on the state rather than on the (frozen)
    # input because `__call__`'s degrade path has to see a promotion that
    # happened deep inside `_run`.
    speculative: bool = False
    # Every row this turn is holding back, collected instead of written and
    # parked in the speculation cache at DoneEvent, to be written only if the
    # client confirms the guess. Minted at the user row (the first write there
    # is) and cleared by a promotion, which flushes it.
    pending: PendingPersist | None = None
    # The session row a speculative turn is holding back — see `_resolve_session`.
    # Moved into `pending` when that is created, so a guess nobody confirms
    # leaves no orphan session behind either.
    session_touch: Session | None = None


class RunTurn:
    def __init__(
        self,
        *,
        llm: LLMPort,
        prompt_builder: PromptBuilder,
        tools: ToolRouter,
        input_guard: InputGuardPort,
        output_guard: OutputGuardPort,
        context: ContextLoaderPort,
        active_run: ActiveRunPort,
        sessions: SessionRepo,
        messages: MessageRepo,
        clock: Clock,
        settings: Settings,
        refusals: RefusalPort | None = None,
        force_tool_when: Callable[[str], str | None] | None = None,
        bg: BackgroundTasks | None = None,
        speculation: SpeculativeTurnCache[PendingPersist] | None = None,
    ) -> None:
        self.llm, self.pb, self.tools = llm, prompt_builder, tools
        self.input_guard, self.output_guard = input_guard, output_guard
        # One port for history + facts + workflow + last results: over Postgres
        # that is a single RPC rather than three sequential round trips (L-7).
        self.context, self.active_run = context, active_run
        self.sessions, self.messages, self.clock, self.s = sessions, messages, clock, settings
        self.refusals = refusals
        self.force_tool_when = force_tool_when
        # Owned by the container in production, so `Container.shutdown` can drain
        # it; a private one by default keeps a hand-built `RunTurn` usable.
        self.bg = bg or BackgroundTasks()
        # L-3: where a speculative turn's writes wait for the confirmation that
        # turns them into rows. Injectable, but a private one is the norm — the
        # confirmation is served by this same object (`confirm`), so nothing
        # else needs a handle on it.
        self.speculation = speculation or SpeculativeTurnCache[PendingPersist]()

    async def __call__(self, inp: TurnInput) -> AsyncIterator[TurnEvent]:
        t = Timings()
        # The gate's own cost, measured by the interface layer and carried in on
        # the input (I5). Recorded unconditionally, zero included: a breakdown
        # with a stage that is sometimes absent is a breakdown that cannot be
        # aggregated, and zero is the true answer for a caller with no gate.
        t.add("auth", inp.t_auth_ms)
        st = TurnState()
        session: Session | None = None
        try:
            # Inside the try so a repo failure here becomes a clean ErrorEvent
            # instead of an exception escaping mid-stream.
            #
            # The context load comes FIRST now: it carries the session row, so
            # resolving the session costs no read of its own (L-7). A turn that
            # is not resuming anything mints its id here rather than after the
            # load, so the id asked for is the id used and the loader is never
            # handed a null.
            sid = inp.session_id or new_id(SessionId)
            ctx, memory_down = await self._load_context(inp, sid, t)
            # L-3, before anything is buffered: an OPEN workflow run disqualifies
            # speculation outright. The assessment engine takes the whole turn
            # and answering an item advances the run — a write with no
            # confirmation step behind it, so a misheard "three" would move the
            # test on and there would be no way to move it back. A run in
            # progress is also the case where the user is saying single words,
            # which is exactly where an interim transcript is least trustworthy.
            # The client's later confirmation finds nothing and gets a 409,
            # which costs it nothing: the turn was persisted normally.
            st.speculative = inp.speculative and ctx.workflow is None
            session = await self._resolve_session(inp, sid, ctx, st, memory_down)
            if session.id != sid:
                # The id was refused (someone else's, or expired) and a new
                # session started. The history that came back belongs to the
                # session that was refused, so it goes with it — C1: guessing a
                # session id must not be a way to read its transcript.
                ctx.history = []
            yield SessionEvent(session.id)

            text = inp.text.strip()
            if not text or len(text) > self.s.max_utterance_chars:
                yield ErrorEvent("invalid_input", "I didn't catch that.")
                return

            # aclosing, not a bare `async for` (C2): a client that hangs up is a
            # `GeneratorExit` thrown into THIS generator, and a nested generator
            # is not closed by its consumer being closed — it is left to be
            # collected whenever. `_finish`'s finally is where the assistant row
            # is written, so "whenever" meant the turn's own answer went missing
            # from the transcript. Closing explicitly propagates the close down
            # the chain, which is what makes those finallys run now.
            async with aclosing(self._run(inp, session, text, t, st, ctx, memory_down)) as run:
                async for ev in run:
                    yield ev
        except LLMUnavailable as e:
            # Not an exception-with-traceback log: this is an expected degrade
            # path (429 quota, 5xx, or first-token deadline) — but ops still
            # needs to see WHICH of those it was.
            log.warning("llm_unavailable", reason=str(e)[:300])
            async for ev in self._degrade(inp, session, st, t, "gemini_unavailable"):
                yield ev
        except LLMTimeout as e:
            log.warning("llm_timeout", stage=str(e))
            async for ev in self._degrade(inp, session, st, t, "timeout"):
                yield ev
        except Exception:
            log.exception("turn_failed")
            async for ev in self._degrade(inp, session, st, t, "internal"):
                yield ev

    async def _degrade(
        self,
        inp: TurnInput,
        session: Session | None,
        st: TurnState,
        t: Timings,
        code: DegradeCode,
    ) -> AsyncIterator[TurnEvent]:
        """End a failed turn as gracefully as what was already spoken allows.

        Once sentences have gone out the caller has heard a partial answer, so
        replacing it with an error is both jarring and lossy: close it off instead
        and persist what was actually spoken. With nothing spoken the caller still
        gets the ErrorEvent, plus an empty assistant row so the transcript stays
        paired and Phase 8's `guard_decision like 'error:%'` alert can find it.

        A speculative turn degrades the same way, but its rows are buffered like
        any others (L-3): a failed guess is still a guess, and writing an error
        row for a question the user may never have asked would put an utterance
        they did not make into their transcript.
        """
        guard = f"error:{code}"
        if session is None or not st.user_saved:
            # No session (or no user row yet) means there is nothing to pair with.
            yield ErrorEvent(code, ERROR_SPOKEN[code])
            return
        if st.sentences:
            closer = TIMEOUT_CLOSER if code == "timeout" else ERROR_CLOSER
            yield SentenceEvent(Sentence(st.idx, closer, to_speech(closer)))
            st.sentences.append(closer)
            st.idx += 1
            t.mark("total")
            mid = await self._persist_or_log(
                session, inp, st, st.sentences, t, guard=guard, prompt_hash=st.prompt_hash
            )
            yield DoneEvent(mid, t.as_dict())
            return
        await self._persist_or_log(session, inp, st, [], t, guard=guard, prompt_hash=st.prompt_hash)
        yield ErrorEvent(code, ERROR_SPOKEN[code])

    async def _persist_or_log(
        self,
        session: Session,
        inp: TurnInput,
        st: TurnState,
        sentences: list[str],
        t: Timings,
        guard: str | None = None,
        prompt_hash: str | None = None,
    ) -> MessageId:
        """`_persist` that swallows its own failure.

        This is only for the degrade path: it is already handling one failure, and
        letting a second (a repo that is down for the same reason) escape `__call__`
        would end the stream with a bare exception — the client would hear the closer
        and then hang, never seeing DoneEvent/ErrorEvent. Losing the row is the lesser
        harm, so log it and let the caller finish the stream.
        """
        try:
            return await self._persist(
                session, inp, st, sentences, t, guard=guard, prompt_hash=prompt_hash
            )
        except Exception:
            log.exception("degrade_persist_failed")
            return new_id(MessageId)

    # ------------------------------------------------------------------
    async def _load_context(
        self, inp: TurnInput, sid: SessionId, t: Timings
    ) -> tuple[TurnContext, bool]:
        """The turn's one read, and what to do when it fails.

        I8: a turn that cannot load its context is degraded, not dead. History,
        facts, the session row and any active workflow are all *enrichment* —
        without them Sarjy can still answer "what's the weather in Lisbon", and
        answering it beats an error page. The empty context returned here says
        `session_loaded=False`, so the caller falls back to reading the session
        on its own rather than silently starting a new one for a caller who was
        mid-conversation.
        """
        try:
            with t.stage("context"):
                return await self.context.load(inp.user_id, sid, self.s.history_limit), False
        except Exception:
            log.exception("context_load_failed")
            return TurnContext(facts=[], history=[], workflow=None), True

    async def _resolve_session(
        self,
        inp: TurnInput,
        sid: SessionId,
        ctx: TurnContext,
        st: TurnState,
        memory_down: bool,
    ) -> Session:
        """The session this turn belongs to, from the context load where possible.

        `ctx.session` is the row as stored — the RPC does not decide whether it
        may be used, because "refuse and start a new one" is not something SQL
        can express as a row. The two rules that decide it are here: ownership
        (a caller may only resume their own session — guessing or replaying
        someone else's id must start a fresh one, never leak its history) and
        expiry. A loader that does not carry sessions (the in-memory one) says
        so, and only then is the repo read.

        `memory_down` is the third case, and it starts a fresh session rather
        than reading the repo. The context load has just failed; the session row
        lives in the same database, so the fallback read is a second call to
        something already known to be down — it buys a slower failure, not a
        resumed conversation. This turn has no history either way (I8), so it is
        already a fresh start in every sense but the id — and the id is minted
        too, because an unvalidated one may be someone else's.
        """
        now = self.clock.now()
        if inp.session_id:
            s = (
                None
                if memory_down
                else (
                    ctx.session if ctx.session_loaded else await self.sessions.get(inp.session_id)
                )
            )
            if s and s.user_id == inp.user_id and not s.is_expired(now):
                s.touch(now)
                # Deferred (L-7): this write moves `last_active_at` forward and
                # nothing in the turn reads it back, so making the caller wait a
                # round trip for it before the first token is pure latency. The
                # row already exists, so there is no FK for a later insert to
                # race. A brand-new session below is a different matter — the
                # user message's FK needs it, so that one is awaited.
                #
                # A speculative turn defers it further still: to the confirmation
                # (L-3). Moving `last_active_at` for a question the user may not
                # have asked is harmless but pointless, and keeping every write
                # of the turn in one place is what makes "nothing was written"
                # checkable.
                if st.speculative:
                    st.session_touch = s
                else:
                    self.bg.spawn(self._touch_session(s))
                return s
            # A refused id is never reused as the new session's id: it may be a
            # live session belonging to someone else — and `sessions.save` is an
            # upsert on that id, so reusing it would reach into that row. The
            # same applies to an id that could not be checked at all, because the
            # read that would have validated it is the one that just failed.
            sid = new_id(SessionId)
        s = Session.start(sid, inp.user_id, now)
        if st.speculative:
            # Held back with the rest (L-3). Nothing in the turn reads the row —
            # only the messages' foreign key needs it, and those are buffered
            # too, so `confirm` writes the session first and the ordering the
            # database cares about is preserved.
            st.session_touch = s
        else:
            await self.sessions.save(s)
        return s

    async def _touch_session(self, s: Session) -> None:
        """The deferred `last_active_at` write. Swallows its own failure — see
        `_save_assistant`; a session that keeps a stale timestamp is not worth
        ending a turn over."""
        try:
            await self.sessions.save(s)
        except Exception:
            log.exception("session_touch_failed", session_id=str(s.id))

    async def _run(
        self,
        inp: TurnInput,
        session: Session,
        text: str,
        t: Timings,
        st: TurnState,
        ctx: TurnContext,
        memory_down: bool,
    ) -> AsyncGenerator[TurnEvent, None]:
        history: list[Message] = ctx.history
        facts: list[Fact] = ctx.facts
        run: ActiveRunSnapshot | None = ctx.workflow
        # A finished run only enters the turn when the conversation is about it.
        # Otherwise the block is prompt budget spent on nothing and the score
        # check is armed over a conversation it has no business judging — see
        # `asks_about_results`.
        last: LastResults | None = (
            ctx.last_results if self._results_in_play(text, history) else None
        )
        recent_user = [*[m.content for m in history if m.role == "user"][-3:], text]

        # Minted before the guard runs, not after, so the guardrail_events rows
        # the guard writes carry the id of the very message they are about (I4).
        # A guard event that cannot be joined back to its message is a count,
        # not an investigation.
        user_msg_id = new_id(MessageId)
        with t.stage("guard"):
            decision = await self.input_guard.check(
                inp.user_id,
                text,
                recent_user,
                message_id=user_msg_id,
                speculative=st.speculative,
            )
        if decision.action == "block":
            user_guard = f"block:{decision.category}"
            user_msg = self._user_message(session, inp, text, user_guard, user_msg_id)
            yield GuardEvent("block", decision.category)
            block_text = (
                self.refusals.refusal(decision.category) if self.refusals else GENERIC_BLOCK
            )
            yield SentenceEvent(Sentence(st.idx, block_text, to_speech(block_text)))
            st.sentences.append(block_text)
            st.idx += 1
            await self._save_user(user_msg, inp, st)
            t.mark("total")
            # Synchronous, unlike the answer paths below: this row IS the
            # record that a refusal happened, and Phase 8's
            # `guard_decision like 'block:%'` alert reads it. A write ordered
            # after the stream ends is a write a disconnect can lose.
            mid = await self._persist(session, inp, st, st.sentences, t, guard=user_guard)
            yield DoneEvent(mid, t.as_dict())
            return
        user_msg = self._user_message(session, inp, text, decision.action, user_msg_id)
        yield GuardEvent("allow")
        await self._save_user(user_msg, inp, st)

        if memory_down:
            # Ahead of everything else, so the caller hears the caveat before the
            # answer it qualifies rather than after it.
            yield SentenceEvent(self._take(MEMORY_UNAVAILABLE, st))

        # Only an OPEN run can take the turn, and the context load already said
        # whether there is one — so an ordinary chat turn no longer pays a read
        # to be told "no" (L-7). With the context down there is nothing to hand
        # the turn to either: the read would fail the same way it just did.
        reply = await self.active_run.handle_turn(inp.user_id, text) if run else None
        if reply is not None:
            # Workflow replies are spoken output too, so they go through the output
            # guard like model text. There is no workflow block or summary to build
            # against yet — the static prompt plus this turn's facts is the context.
            wf_ctx = GuardContext(
                system_prompt=self.pb.build(facts, None, None).system,
                # The scores the workflow computed, so the grounding check is
                # armed for a workflow reply the same way it is for a tool
                # result (I9). An empty tuple leaves it off, which is the right
                # answer for a reply that states no figures.
                tool_numbers=list(reply.grounding_numbers),
                facts=facts,
                user_id=inp.user_id,
                leak_corpus=self.pb.confidential_text,
                message_id=user_msg_id,
                speculative=st.speculative,
            )
            for s_ in reply.sentences:
                for sent in self._speak(s_, wf_ctx, st):
                    yield SentenceEvent(sent)
            if not st.sentences:
                # The guard cut everything the workflow wanted to say. Persisting an
                # empty assistant turn would leave the caller with silence and a
                # DoneEvent, so fall back to the same "nothing to say" line the model
                # path uses.
                for sent in self._speak(NO_OUTPUT_FALLBACK, wf_ctx, st):
                    yield SentenceEvent(sent)
            # aclosing at every `_finish` site, for the reason in `__call__`.
            async with aclosing(self._finish(session, inp, st, t, workflow=reply.workflow)) as fin:
                async for done in fin:
                    yield done
            return

        prompt = self.pb.build(
            facts=facts,
            workflow_block=run.prompt_block if run else None,
            summary=session.summary,
            # P-9/P-11: a finished test's scores, stated for the model, so
            # "how did I score on openness?" is answered from the row rather
            # than from a transcript that has scrolled out of the window.
            results_block=last.prompt_block if last else None,
        )
        st.prompt_hash = prompt.hash
        msgs: list[LLMMessage] = []
        for m in history:
            if m.role == "user":
                # I6: a turn the guard already refused does not get replayed into
                # the next prompt. It was stored (the transcript has to stay
                # paired, and Phase 8 alerts read those rows), but feeding a
                # refused injection back to the model every subsequent turn hands
                # the attacker exactly the persistence they were after — one
                # blocked message, quietly re-delivered for the rest of the
                # session. The paired assistant row is the refusal itself, which
                # is harmless and stays.
                if (m.guard_decision or "").startswith("block:"):
                    continue
                # Replayed history is as untrusted as the live turn: wrap and sanitise
                # it identically so an injection stored last turn cannot escape now.
                msgs.append(LLMMessage(role="user", text=self._wrap_user(m.content)))
            elif m.role == "assistant":
                # R4: the refusal PAIRED with a dropped user row goes with it.
                # Both rows carry the same `block:<category>` decision, so this
                # is the same predicate. Keeping it would leave a refusal
                # template answering a question the model can no longer see —
                # and if the blocked turn was the first of the session, a
                # leading `model` turn with no `user` turn before it, which
                # Gemini rejects outright.
                if (m.guard_decision or "").startswith("block:"):
                    continue
                msgs.append(LLMMessage(role="model", text=m.content))
        msgs.append(LLMMessage(role="user", text=self._wrap_user(text)))
        gctx = GuardContext(
            system_prompt=prompt.system,
            tool_numbers=[],
            # The finished run's figures (each score, the integer it rounds to,
            # and the top of the scale), so the answer the results block invites
            # is speakable and a "you scored 4.9" it did not is cut. Their own
            # list, checked only against sentences about the results and to the
            # decimal — see `GuardContext.results_numbers`.
            results_numbers=list(last.grounding_numbers) if last else [],
            facts=facts,
            user_id=inp.user_id,
            leak_corpus=self.pb.confidential_text,
            message_id=user_msg_id,
            speculative=st.speculative,
        )
        splitter = SentenceSplitter()
        first_token_marked = False

        # W-8 forces the tool on hop 0 only, which covers the opening question but
        # not a follow-up ("and tomorrow?") whose weather shape is only visible in
        # context. Catching those needs a post-hoc assertion — did the answer state
        # weather without a tool call this turn? — which is the output guard's job,
        # not the orchestrator's. Deferred to Phase 5 T7 with the guard (review I4).
        forced = self.force_tool_when(text) if self.force_tool_when else None
        if forced and not self.tools.has(forced):
            forced = None

        for hop in range(MAX_TOOL_HOPS + 1):
            req = LLMRequest(
                system=prompt.system,
                messages=msgs,
                tools=self.tools.declarations(),
                temperature=self.s.chat_temperature,
                max_output_tokens=self.s.chat_max_output_tokens,
                force_tool=forced if hop == 0 else None,
            )
            # A model round may emit several function calls at once (parallel calling).
            # Collect them all, then invoke them in order once the stream has ended.
            pending_calls: list[FunctionCall] = []
            async for ev in self.llm.stream(req):
                if isinstance(ev, LLMText):
                    if not first_token_marked:
                        t.mark("gemini_first_token")
                        first_token_marked = True
                    for s_ in splitter.feed(ev.text):
                        for sent in self._speak(s_, gctx, st):
                            yield SentenceEvent(sent)
                elif isinstance(ev, LLMFunctionCall):
                    pending_calls.append(ev.call)
                elif isinstance(ev, LLMFinished):
                    pass
            if not pending_calls:
                break

            # Flush any preamble text ("Let me check.") buffered ahead of the
            # tool call so it is spoken before the tool status/cap sentence,
            # rather than getting glued onto whatever text streams next.
            for sent in self._flush(splitter, gctx, st):
                yield SentenceEvent(sent)

            # The budget counts model rounds, not calls: a round carrying three
            # parallel calls costs one hop, the same as a round carrying one.
            if hop == MAX_TOOL_HOPS:
                s_ = "I couldn't finish that — could you rephrase?"
                yield SentenceEvent(Sentence(st.idx, s_, to_speech(s_)))
                st.sentences.append(s_)
                st.idx += 1
                break

            # Gemini expects a parallel-call round as one model turn holding every
            # call, followed by one user turn holding every response -- not as
            # interleaved call/response pairs. Collect the responses, then append.
            responses: list[FunctionResponse] = []
            hop_summaries: list[str] = []
            for call in pending_calls:
                yield ToolStatusEvent(call.name, "start")
                if st.speculative and self.tools.mutates(call.name):
                    await self._promote(st, gctx)
                with t.stage(f"tool_{call.name}"):
                    # The turn's own snapshot, handed down rather than re-read
                    # (I5). `ctx.facts` is what the prompt was built from, so a
                    # tool resolving "here" against it resolves it against
                    # exactly what the model was told.
                    res = await self.tools.invoke(inp.user_id, call, facts=facts)
                await self._save_tool_call(
                    inp,
                    st,
                    (
                        user_msg.id,
                        inp.user_id,
                        call.name,
                        call.args,
                        res.data,
                        "ok" if res.ok else "error",
                        res.latency_ms,
                    ),
                )
                gctx.tool_numbers.extend(res.grounding_numbers)
                if res.spoken_summary:
                    hop_summaries.append(res.spoken_summary)
                    # A summary can only stand in for a cut sentence if it is
                    # unambiguously the answer that sentence got wrong. One
                    # summary in the round: attributable. Two (parallel calls
                    # for Tokyo and Lisbon) : there is no way to tell which one
                    # the model was talking about, and speaking the wrong city's
                    # weather confidently is worse than saying nothing.
                    st.last_tool_summary = hop_summaries[0] if len(hop_summaries) == 1 else None
                    st.sentences_since_tool = 0
                yield ToolStatusEvent(call.name, "end", ok=res.ok)

                if res.direct_sentences is not None:
                    for s_ in res.direct_sentences:
                        for sent in self._speak(s_, gctx, st):
                            yield SentenceEvent(sent)
                    async with aclosing(
                        self._finish(
                            session,
                            inp,
                            st,
                            t,
                            prompt_hash=prompt.hash,
                            workflow=res.data.get("workflow"),
                        )
                    ) as fin:
                        async for done in fin:
                            yield done
                    return

                responses.append(FunctionResponse(call.name, res.data, call.id))

            msgs += [LLMMessage(role="model", function_call=c) for c in pending_calls]
            msgs += [LLMMessage(role="tool", function_response=r) for r in responses]

        for sent in self._flush(splitter, gctx, st):
            yield SentenceEvent(sent)
        if not st.sentences:
            yield SentenceEvent(Sentence(st.idx, NO_OUTPUT_FALLBACK, to_speech(NO_OUTPUT_FALLBACK)))
            st.sentences.append(NO_OUTPUT_FALLBACK)
            st.idx += 1
        async with aclosing(self._finish(session, inp, st, t, prompt_hash=prompt.hash)) as fin:
            async for done in fin:
                yield done

    def _wrap_user(self, content: str) -> str:
        return f"<user>{sanitise_value(content, limit=self.s.max_utterance_chars)}</user>"

    def _flush(
        self, splitter: SentenceSplitter, ctx: GuardContext, st: TurnState
    ) -> list[Sentence]:
        """Flush the splitter's buffered tail and speak it (see `_speak`).

        Returns an empty list (leaving `st` untouched) when there is nothing
        buffered; otherwise the tail goes through the guard like any other
        sentence.
        """
        tail = splitter.flush()
        if not tail:
            return []
        return self._speak(tail, ctx, st)

    def _speak(self, sentence: str, ctx: GuardContext, st: TurnState) -> list[Sentence]:
        """Guard `sentence` and return what should actually be spoken for it.

        Zero or one sentence: what the guard passed/replaced, nothing when it
        cut, or — for the one repairable cut — the tool's own grounded summary
        in its place (see `_grounded_substitute`).
        """
        out, kind = self._guard(sentence, ctx)
        if out:
            return [self._take(out, st)]
        if kind == "ungrounded_number":
            sub = self._grounded_substitute(ctx, st)
            if sub is not None:
                return [sub]
        return []

    def _grounded_substitute(self, ctx: GuardContext, st: TurnState) -> Sentence | None:
        """The tool's own sentence, standing in for one the guard just cut.

        An `ungrounded_number` cut means the model stated a figure no tool
        returned. The guard cannot repair that — it has no idea which number,
        unit or wording was meant — but the tool result can: `spoken_summary`
        is a sentence built *from* the payload. Substituting is only honest
        under three conditions, all of which have to hold:

        * a summary is attributable at all (`last_tool_summary`) — a round that
          returned two of them is ambiguous and clears it;
        * nothing has been spoken since that tool result
          (`sentences_since_tool`), so the cut sentence really is the answer
          the tool was called for, not a later aside about another day or city;
        * no substitution has happened yet this turn (`summary_used`), so a
          model fabricating three numbers in a row doesn't get the same
          sentence back three times.

        The substitute still goes through the guard. It is grounded by
        construction, so that is a formality — but a sentence that skips the
        guard is a hole in it. `summary_used` is only set once it survives,
        so a substitute that is itself cut doesn't burn the turn's one chance.
        """
        if st.summary_used or st.last_tool_summary is None or st.sentences_since_tool:
            return None
        out, _ = self._guard(st.last_tool_summary, ctx)
        if not out:
            return None
        st.summary_used = True
        return self._take(out, st)

    def _take(self, text: str, st: TurnState) -> Sentence:
        """Record `text` as spoken and mint the `Sentence` carrying it."""
        st.sentences.append(text)
        st.sentences_since_tool += 1
        s = Sentence(st.idx, text, to_speech(text))
        st.idx += 1
        return s

    def _guard(self, sentence: str, ctx: GuardContext) -> tuple[str | None, str | None]:
        """Run the output guard, returning `(text_to_speak_or_None, verdict_kind)`.

        The `kind` is passed back rather than swallowed so `_speak` can react to
        *why* a sentence was cut, which is what makes the grounded-summary
        fallback possible.
        """
        v = self.output_guard.check_sentence(sentence, ctx)
        if v.action == "pass":
            return sentence, v.kind
        if v.action == "replace":
            return v.replacement, v.kind
        return None, v.kind

    def _user_message(
        self, session: Session, inp: TurnInput, text: str, guard: str, mid: MessageId
    ) -> Message:
        return Message(
            id=mid,
            session_id=session.id,
            user_id=inp.user_id,
            role="user",
            content=text,
            created_at=self.clock.now(),
            client_turn_id=inp.client_turn_id,
            guard_decision=guard,
        )

    def _assistant_message(
        self,
        session: Session,
        inp: TurnInput,
        sentences: list[str],
        t: Timings,
        guard: str | None = None,
        prompt_hash: str | None = None,
    ) -> Message:
        content = " ".join(sentences)
        return Message(
            id=new_id(MessageId),
            session_id=session.id,
            user_id=inp.user_id,
            role="assistant",
            content=content,
            speech_content=to_speech(content),
            created_at=self.clock.now(),
            client_turn_id=inp.client_turn_id,
            guard_decision=guard,
            timings=t.as_dict(),
            prompt_hash=prompt_hash,
        )

    async def _persist(
        self,
        session: Session,
        inp: TurnInput,
        st: TurnState,
        sentences: list[str],
        t: Timings,
        guard: str | None = None,
        prompt_hash: str | None = None,
    ) -> MessageId:
        """Write the assistant row and wait for it.

        The two paths that end a turn without an answer use this — a blocked
        turn and a degraded one. Both are recording *why* something did not
        happen, and Phase 8's `guard_decision like 'block:%'`/`'error:%'` alerts
        read exactly these rows: deferring them to a background task the caller
        may never drain is how an incident becomes invisible. The answer path
        ends in `_finish` instead, which orders its write after DoneEvent.

        A speculative turn buffers instead of writing, here as everywhere (L-3),
        and this is where its guess is parked for confirmation.
        """
        a = self._assistant_message(session, inp, sentences, t, guard, prompt_hash)
        if st.speculative:
            await self._park(inp, st, a)
            return a.id
        await self.messages.save(a)
        return a.id

    # --- L-3: the four write sites, and the one exit they share -----------

    async def _save_user(self, user_msg: Message, inp: TurnInput, st: TurnState) -> None:
        """Write the user row, or buffer it for a speculative turn.

        `user_saved` is set either way: it means "this turn has a user row to
        pair an assistant row with", which is as true of a buffered row as a
        written one — the degrade path's question is whether the transcript
        would be left half a conversation, and for a speculative turn the answer
        is decided together at `confirm`.
        """
        if st.speculative:
            st.pending = PendingPersist(user_msg=user_msg, session_touch=st.session_touch)
        else:
            await self.messages.save(user_msg)
        st.user_saved = True

    async def _save_tool_call(self, inp: TurnInput, st: TurnState, row: tuple[Any, ...]) -> None:
        """Write a tool-call row, or buffer it. Buffered rows carry the same
        `user_msg.id` they would have been written with, so the foreign key is
        satisfied by the buffered user row `confirm` writes just before them."""
        if st.speculative:
            if st.pending is None:
                # Unreachable by construction: `st.speculative` is only true
                # between the user row (which mints the buffer) and a promotion
                # (which clears it and sets this false together). If it ever
                # happens, writing the row would put half a speculative turn in
                # the database — the one outcome this whole path exists to
                # prevent — so the turn fails instead and degrades.
                log.error("speculative_tool_call_without_buffer", tool=row[2])
                raise RuntimeError("speculative turn has no pending buffer")
            st.pending.tool_calls.append(row)
            return
        await self.messages.save_tool_call(*row)

    async def _park(self, inp: TurnInput, st: TurnState, assistant: Message) -> None:
        """Park this speculative turn's buffered rows for confirmation (L-3).

        Called at the end of the turn, immediately BEFORE the terminal event
        goes out — not after. The client posts `/chat/confirm` the instant its
        recogniser finalises, which can be the moment after DoneEvent reaches
        it; parking afterwards would leave a window in which that confirmation
        finds nothing, answers 409, and the client politely asks the same
        question a second time out loud.

        Moving the park earlier narrows that window but cannot close it: a
        recogniser can finalise while the model is still streaming, so the
        confirmation can beat the park by the length of the whole answer. That
        is the case `take_early` handles (C1). A confirmation already recorded
        for this id settles the turn HERE, in the same order `confirm` uses —
        matching text writes the rows now, a mismatched one drops the buffer and
        the turn is discarded. Nothing is parked in either case: the turn has
        had its one answer, and a second confirmation for it must find nothing.

        A turn with no `pending` never got as far as its user row (the guard
        blocked before it, or the context load failed), so there is nothing to
        park and nothing that could ever be written — which is the guarantee.
        """
        p = st.pending
        if p is None:
            return
        p.assistant_msg = assistant
        # Keyed by the id the client will confirm under, matched on the text the
        # turn was actually run on — the user row's content, which is `inp.text`
        # stripped exactly as the turn read it.
        key = _spec_key(inp.user_id, inp.client_turn_id)
        early = self.speculation.take_early(key)
        if early is None:
            self.speculation.put(key, p.user_msg.content, p)
            return
        if early != normalise(p.user_msg.content):
            st.pending = None
            log.info(
                "speculation_turn_discarded",
                reason="early_confirm_mismatch",
                client_turn_id=inp.client_turn_id,
                user_id=str(inp.user_id),
            )
            return
        # The buffer is held until the write RETURNS, the same rule `_promote`
        # follows and for the same reason (I1). If the flush raises, `st` is left
        # holding it and `user_saved` goes back to False, so the degrade path
        # that catches the exception takes its no-user-row branch: a bare
        # ErrorEvent, no assistant row to orphan, and — the part a listener
        # notices — no "sorry, I lost my train of thought" appended to an answer
        # they have just heard in full. Nothing about the turn was wrong; only
        # the write of it failed.
        try:
            await self._write_pending(p)
        except Exception:
            st.user_saved = False
            log.exception(
                "speculation_early_confirm_write_failed",
                client_turn_id=inp.client_turn_id,
                user_id=str(inp.user_id),
            )
            raise
        st.pending = None
        log.info(
            "speculation_early_confirm_written",
            client_turn_id=inp.client_turn_id,
            user_id=str(inp.user_id),
        )

    async def _write_pending(self, p: PendingPersist) -> None:
        """Turn a buffered turn into rows, in the order the database needs them.

        The session row first (the messages' foreign key), then the user row,
        then its tool-call rows (whose foreign key is that user row), then the
        assistant row. Shared by `confirm` and by `_park`'s early-confirm branch
        precisely so there is one such order rather than two that drift.
        """
        if p.session_touch is not None:
            await self.sessions.save(p.session_touch)
        await self.messages.save(p.user_msg)
        for row in p.tool_calls:
            await self.messages.save_tool_call(*row)
        if p.assistant_msg is not None:
            await self.messages.save(p.assistant_msg)

    async def _promote(self, st: TurnState, gctx: GuardContext | None = None) -> None:
        """Stop speculating, mid-turn, and write what was buffered (L-3).

        The trigger is a tool that mutates something outside the turn: a fact
        stored or deleted, a run opened or ended. Those cannot be buffered and
        replayed on confirmation — the tool has to run now to answer this turn,
        and there is no undo for a wrong guess. So the turn stops being a guess:
        the buffered rows are written on the spot, and everything after this
        persists normally. Nothing is parked, so the client's confirmation gets
        a 409 and (by design) does not re-run anything — the turn is already
        recorded under the id it was sent with.

        Ordered session → user → tool calls, the same order `confirm` uses and
        for the same reason: the messages' foreign key.

        The FLUSH COMES FIRST and the flags follow it, which is the whole of I1.
        Clearing them up front meant a repo that failed mid-flush left a turn
        that believed it had persisted its user row: the exception unwound into
        `__call__`, the degrade path saw `speculative=False` and `user_saved=
        True`, and wrote an assistant row pointing at a user row that had never
        landed — an orphan half of an exchange, in the one code path whose
        entire job is not to leave those. Failing before the flags move leaves
        the turn exactly as it was, still speculative and still holding its
        buffer, so the degrade path buffers its error row like any other
        speculative write and the database gains nothing it cannot explain.
        """
        p = st.pending
        if p is None:
            st.speculative = False
            if gctx is not None:
                gctx.speculative = False
            return
        if p.session_touch is not None:
            await self.sessions.save(p.session_touch)
        await self.messages.save(p.user_msg)
        for row in p.tool_calls:
            await self.messages.save_tool_call(*row)
        # Only now, and only together.
        st.user_saved = True
        st.pending = None
        st.speculative = False
        if gctx is not None:
            gctx.speculative = False

    async def confirm(
        self, client_turn_id: str, final_text: str, user_id: UserId
    ) -> ConfirmOutcome:
        """Write a speculative turn's parked rows, if the guess was right.

        Three outcomes, because "nothing parked under that id" and "parked, and
        the guess was wrong" are not the same situation and were being answered
        the same way (C1):

        * `True` — `final_text` is (normalised) the text the turn was run on,
          and its rows are now written.
        * `False` — a guess IS parked under that id and the final transcript
          says something else. The guess is consumed and dropped; the client
          sends an ordinary turn under a new id.
        * `"pending"` — nothing is parked. Most often that is a confirmation
          that overtook its own turn, which is still streaming and has not
          reached the park yet, so the transcript is recorded (`early_confirm`)
          and `_park` settles the turn with it a moment later. It also covers
          the cases that genuinely have nothing left to do — a turn promoted
          mid-flight (already persisted normally), an expired guess, a
          confirmation that landed on another worker's process — and for those
          the recorded transcript simply expires unread. The client does
          nothing on any of them, which is the correct response to all four.

        `user_id` scopes the lookup rather than being checked after it, so a
        caller guessing at turn ids cannot consume — and thereby destroy —
        someone else's pending turn on a wrong guess (C1); the early confirm is
        recorded under the same scoped key, so a guessed id cannot pre-load a
        transcript onto a turn it does not own either. Required, not defaulted:
        a caller that has no user to scope by has no business confirming
        anything.
        """
        key = _spec_key(user_id, client_turn_id)
        parked = self.speculation.has(key)
        p = self.speculation.take(key, final_text)
        if p is None:
            if parked:
                # Logged at info, not warning: a transcript that changed between
                # the guess and the final is the feature working, not a fault. A
                # RATE of these is the signal worth watching, which is what a
                # counted log line gives.
                log.info(
                    "speculation_confirm_mismatch",
                    client_turn_id=client_turn_id,
                    user_id=str(user_id),
                )
                return False
            self.speculation.early_confirm(key, final_text, owner=str(user_id))
            log.info(
                "speculation_confirm_early",
                client_turn_id=client_turn_id,
                user_id=str(user_id),
            )
            return "pending"
        await self._write_pending(p)
        return True

    def _results_in_play(self, text: str, history: list[Message]) -> bool:
        """Is this turn about the user's assessment results?

        The user's own words settle it most of the time (`asks_about_results`).
        What that misses is the follow-up that carries no vocabulary of its own:
        "and the other four?", "what does that mean?", "is that high?" — which
        are about the results only because the sentence before them was. So the
        last thing Sarjy said counts too: if it named a trait, the results are
        still the subject, and the block and the decimal-tight grounding check
        stay armed for one more turn rather than leaving the follow-up to be
        answered from a history window the scores have scrolled out of.

        Deliberately the LAST assistant turn only. Anything wider would keep the
        results in play for the rest of the session on the strength of one
        sentence, which is the prompt budget and the tight number check spent on
        a conversation that has moved on.
        """
        if asks_about_results(text):
            return True
        last_assistant = next((m for m in reversed(history) if m.role == "assistant"), None)
        return last_assistant is not None and mentions_trait(last_assistant.content)

    async def _finish(
        self,
        session: Session,
        inp: TurnInput,
        st: TurnState,
        t: Timings,
        prompt_hash: str | None = None,
        workflow: dict[str, Any] | None = None,
    ) -> AsyncGenerator[TurnEvent, None]:
        """End the turn: DoneEvent, then the assistant row, awaited (L-7, I4).

        The write is ordered AFTER the last event but still awaited before the
        turn ends. The caller cannot perceive the wait — every sentence and the
        DoneEvent are already on the wire — and the turn not ending until the row
        lands is what lets the very next turn read its own predecessor back out
        of history. A plain background task would have made that a race, and
        "the model forgot what it just said" is not a bug anyone enjoys
        reproducing.

        It is a `finally` so that a caller who disconnects at DoneEvent leaves a
        transcript behind rather than half a conversation — but a `finally` only
        runs if something closes this generator, and until C2 nothing did.
        Starlette closes the *response* generator; that close reaches here only
        because every consumer in the chain (`__call__` around `_run`, `_run`
        around this) now closes what it is iterating via `aclosing`. Without that
        the write waited on garbage collection, which is to say it waited until
        after the turn's own reply had already gone missing from history.

        The disconnect is also why the write is handed to `bg` rather than
        plainly awaited: the same disconnect can cancel whatever is awaiting
        here, and a write that only existed as this coroutine would go with it.
        `bg.run` spawns it as a task FIRST and then awaits it under a shield, so
        the cancellation stops the waiting rather than the writing, and
        `Container.shutdown` drains it either way. `_save_assistant` swallows its
        own failure: nothing downstream is waiting on it, and there is no stream
        left to fail.

        A speculative turn buffers the row instead and parks the turn for
        confirmation (L-3). DoneEvent still carries the assistant id it will be
        written under, so the client's telemetry points at a real row once the
        confirmation lands.
        """
        # Marked before the row is built, so the assistant row's stored timings
        # and the DoneEvent's are the same dict rather than two that differ by
        # however long the write took to arrange.
        t.mark("total")
        a = self._assistant_message(session, inp, st.sentences, t, None, prompt_hash)
        if st.speculative:
            await self._park(inp, st, a)
            yield DoneEvent(a.id, t.as_dict(), workflow=workflow)
            return
        try:
            yield DoneEvent(a.id, t.as_dict(), workflow=workflow)
        finally:
            await self.bg.run(self._save_assistant(a))

    async def _save_assistant(self, a: Message) -> None:
        """The deferred write itself — it swallows its own failure.

        Nothing is waiting on it and there is no stream left to fail, so a repo
        that is down costs this one transcript row and a log line. `Background
        Tasks` would log an escaping exception anyway; catching it here is what
        puts the message id in that line.
        """
        try:
            await self.messages.save(a)
        except Exception:
            log.exception("assistant_persist_failed", message_id=str(a.id))
