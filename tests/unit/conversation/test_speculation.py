"""L-3: a turn run on a guess writes nothing until the guess is confirmed."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, ClassVar

from sarjy.contexts.conversation.application.ports import (
    AssessmentReply,
    Fact,
    FunctionCall,
    GuardContext,
    LLMFinished,
    LLMFunctionCall,
    LLMText,
    LLMUnavailable,
    SentenceVerdict,
    ToolResult,
)
from sarjy.contexts.conversation.application.speculation import (
    PendingPersist,
    SpeculativeTurnCache,
    normalise,
)
from sarjy.contexts.conversation.application.tool_router import ToolRouter
from sarjy.contexts.conversation.domain.events import (
    DoneEvent,
    ErrorEvent,
    SentenceEvent,
    SessionEvent,
)
from sarjy.contexts.conversation.domain.message import Message
from sarjy.contexts.conversation.domain.session import Session
from sarjy.contexts.conversation.domain.turn import TurnInput
from sarjy.contexts.conversation.infrastructure.memory_repos import MemMessages, MemSessions
from sarjy.shared.ids import SessionId, UserId
from tests.unit.conversation.test_run_turn import (
    BlockGuard,
    FakeLLM,
    RaisingLLM,
    Weather,
    WorkflowRun,
    _make,
    _turn,
)

# ---------------------------------------------------------------------------
# The cache itself.
# ---------------------------------------------------------------------------


def test_take_matches_normalised_text() -> None:
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache()
    c.put("t1", "What's the weather in Paris", pending="P")
    assert c.take("t1", "whats the weather in paris?") == "P"
    assert c.take("t1", "again") is None  # consumed


def test_take_rejects_mismatch() -> None:
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache()
    c.put("t2", "what's the weather in Paris", pending="P")
    assert c.take("t2", "what's the weather in Rome") is None


def test_a_mismatch_consumes_the_entry_too() -> None:
    # The client is about to send a fresh turn under a NEW id; leaving the wrong
    # guess parked would only give it a second chance to be written by mistake.
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache()
    c.put("t3", "hello there", pending="P")
    assert c.take("t3", "goodbye") is None
    assert c.take("t3", "hello there") is None
    assert c.size == 0


def test_expired_entries_are_never_returned() -> None:
    # A NEGATIVE ttl rather than 0.0: expiry is `now - at > ttl`, so a zero TTL
    # only expires once the monotonic clock has actually moved between the put
    # and the take — true in practice, not guaranteed, and a test that depends
    # on it either sleeps or flakes. Below zero the entry is expired the instant
    # it is written, which is what "already expired" should mean in a test.
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache(ttl_s=-1.0)
    c.put("t4", "hello there", pending="P")
    assert c.take("t4", "hello there") is None


def test_normalise_matches_the_clients_rule() -> None:
    # Kept in step with `normalise()` in voice.js — lowercase, punctuation gone,
    # whitespace collapsed, and nothing else.
    assert normalise("  What's  the WEATHER, in Paris? ") == "whats the weather in paris"
    # Both sides strip before they collapse, so a newline disappears rather than
    # becoming a space. Odd in isolation, identical in the two places it matters.
    assert normalise("weather,\nin Paris") == "weatherin paris"


# ---------------------------------------------------------------------------
# RunTurn: buffered writes.
# ---------------------------------------------------------------------------


def _spec(user: UserId, text: str = "what is the weather", tid: str = "spec-1") -> TurnInput:
    return TurnInput(user, None, tid, text, speculative=True)


async def test_a_speculative_turn_writes_nothing() -> None:
    user = UserId(uuid.uuid4())
    sessions, msgs = MemSessions(), MemMessages()
    llm = FakeLLM([[LLMText("It's mild in Paris."), LLMFinished("stop")]])
    rt, _ = _make(llm, sessions=sessions, messages=msgs)

    events = await _turn(rt, _spec(user))

    # The answer was still streamed and spoken — that is the whole point.
    assert [e.sentence.text for e in events if isinstance(e, SentenceEvent)] == [
        "It's mild in Paris."
    ]
    assert isinstance(events[-1], DoneEvent)
    # ...and not one row of it exists yet.
    assert msgs.items == []
    assert msgs.tool_calls == []
    assert sessions.items == {}


async def test_confirming_writes_the_turn_under_its_original_client_turn_id() -> None:
    user = UserId(uuid.uuid4())
    sessions, msgs = MemSessions(), MemMessages()
    llm = FakeLLM([[LLMText("It's mild in Paris."), LLMFinished("stop")]])
    rt, _ = _make(llm, sessions=sessions, messages=msgs)

    events = await _turn(rt, _spec(user))
    done = next(e for e in events if isinstance(e, DoneEvent))

    assert await rt.confirm("spec-1", "What is the weather?", user) is True

    assert [m.role for m in msgs.items] == ["user", "assistant"]
    assert [m.client_turn_id for m in msgs.items] == ["spec-1", "spec-1"]
    assert msgs.items[0].content == "what is the weather"
    assert msgs.items[1].content == "It's mild in Paris."
    # The id DoneEvent carried is the id the row was written under, so the
    # client's telemetry post points at a real message.
    assert msgs.items[1].id == done.message_id
    # The session the messages point at was written too, and only now.
    assert list(sessions.items) == [msgs.items[0].session_id]


async def test_confirming_writes_tool_call_rows_between_the_two_messages() -> None:
    user = UserId(uuid.uuid4())
    msgs = MemMessages()
    llm = FakeLLM(
        [
            [
                LLMFunctionCall(FunctionCall("get_weather", {"location": "Paris"})),
                LLMFinished("stop"),
            ],
            [LLMText("It's 22 degrees in Paris."), LLMFinished("stop")],
        ]
    )
    tools = ToolRouter()
    tools.register(Weather())
    rt, _ = _make(llm, tools, messages=msgs)

    await _turn(rt, _spec(user))
    assert msgs.items == [] and msgs.tool_calls == []

    assert await rt.confirm("spec-1", "what is the weather", user) is True
    assert [m.role for m in msgs.items] == ["user", "assistant"]
    assert len(msgs.tool_calls) == 1
    # The buffered row points at the user row that was written just before it.
    assert msgs.tool_calls[0][0] == msgs.items[0].id


async def test_a_mismatched_transcript_writes_nothing() -> None:
    user = UserId(uuid.uuid4())
    sessions, msgs = MemSessions(), MemMessages()
    llm = FakeLLM([[LLMText("It's mild in Paris."), LLMFinished("stop")]])
    rt, _ = _make(llm, sessions=sessions, messages=msgs)

    await _turn(rt, _spec(user))
    assert await rt.confirm("spec-1", "what is the weather in Rome", user) is False
    assert msgs.items == [] and sessions.items == {}


async def test_an_unconfirmed_turn_expires_unwritten() -> None:
    # The PRD L-3 guarantee: a guess nobody confirms is never persisted. There is
    # no path out of the cache other than `take`, and the TTL closes that one.
    user = UserId(uuid.uuid4())
    msgs = MemMessages()
    rt, _ = _make(FakeLLM([[LLMText("Sure."), LLMFinished("stop")]]), messages=msgs)
    rt.speculation = SpeculativeTurnCache[PendingPersist](ttl_s=-1.0)

    await _turn(rt, _spec(user))

    # "pending", not False: from `confirm`'s side an expired guess is
    # indistinguishable from one that has not parked yet, and both are answered
    # by recording the transcript and doing nothing else. The guarantee under
    # test is the row count, and it is zero.
    assert await rt.confirm("spec-1", "what is the weather", user) == "pending"
    assert msgs.items == []


async def test_another_user_cannot_confirm_or_destroy_a_pending_turn() -> None:
    # C1: a turn id belongs to its owner. A caller guessing at ids must not be
    # able to write someone else's turn — nor to consume the pending entry and
    # leave the real owner's confirmation to 409 into a duplicate answer.
    user, other = UserId(uuid.uuid4()), UserId(uuid.uuid4())
    msgs = MemMessages()
    rt, _ = _make(FakeLLM([[LLMText("Sure."), LLMFinished("stop")]]), messages=msgs)

    await _turn(rt, _spec(user))
    # Scoped by caller, so the thief's confirmation finds nothing of its own
    # (202/"pending") and — crucially — leaves the owner's parked turn intact.
    assert await rt.confirm("spec-1", "what is the weather", other) == "pending"
    assert msgs.items == []
    assert await rt.confirm("spec-1", "what is the weather", user) is True
    assert [m.role for m in msgs.items] == ["user", "assistant"]


class CountingSessions(MemSessions):
    """A `MemSessions` that remembers how often it was written to — the in-memory
    repo stores the very object `touch()` mutates, so "was it saved?" cannot be
    read back off the row."""

    def __init__(self) -> None:
        super().__init__()
        self.saves = 0

    async def save(self, s: Session) -> None:
        self.saves += 1
        await super().save(s)


async def test_a_resumed_session_is_not_touched_until_the_turn_is_confirmed() -> None:
    user = UserId(uuid.uuid4())
    sessions, msgs = CountingSessions(), MemMessages()
    session = Session.start(
        SessionId(uuid.uuid4()), user, datetime(2026, 8, 20, 23, 50, tzinfo=UTC)
    )
    await sessions.save(session)
    rt, _ = _make(
        FakeLLM([[LLMText("Sure."), LLMFinished("stop")]]), sessions=sessions, messages=msgs
    )

    events = await _turn(
        rt, TurnInput(user, session.id, "spec-2", "what is the weather", speculative=True)
    )
    assert next(e for e in events if isinstance(e, SessionEvent)).session_id == session.id
    assert rt.bg.pending == 0  # no deferred touch was spawned
    assert sessions.saves == 1  # only the one this test made

    assert await rt.confirm("spec-2", "what is the weather", user) is True
    assert sessions.saves == 2
    assert sessions.items[session.id].last_active_at == datetime(2026, 8, 21, tzinfo=UTC)


async def test_a_blocked_speculative_turn_is_buffered_like_any_other() -> None:
    # A refusal is a row too: writing it before the transcript is confirmed would
    # put an utterance the user may never have made into their audit trail.
    user = UserId(uuid.uuid4())
    msgs = MemMessages()
    rt, _ = _make(FakeLLM([]), input_guard=BlockGuard(), messages=msgs)

    await _turn(rt, _spec(user, "how much ibuprofen can I take"))
    assert msgs.items == []

    assert await rt.confirm("spec-1", "how much ibuprofen can I take", user) is True
    assert [m.guard_decision for m in msgs.items] == ["block:medical", "block:medical"]


async def test_a_degraded_speculative_turn_is_buffered_too() -> None:
    user = UserId(uuid.uuid4())
    msgs = MemMessages()
    rt, _ = _make(RaisingLLM(LLMUnavailable), messages=msgs)

    await _turn(rt, _spec(user))
    assert msgs.items == []

    assert await rt.confirm("spec-1", "what is the weather", user) is True
    assert [m.role for m in msgs.items] == ["user", "assistant"]
    assert msgs.items[1].guard_decision == "error:gemini_unavailable"


async def test_a_turn_that_never_reached_its_user_row_parks_nothing() -> None:
    # An empty utterance ends before the guard runs, so there is no pending turn
    # to confirm — and nothing that could ever be written for it.
    user = UserId(uuid.uuid4())
    msgs = MemMessages()
    rt, _ = _make(FakeLLM([]), messages=msgs)

    await _turn(rt, _spec(user, "   "))
    assert rt.speculation.size == 0
    assert await rt.confirm("spec-1", "   ", user) == "pending"
    assert msgs.items == []


async def test_an_ordinary_turn_is_unaffected() -> None:
    # Regression guard for the branching: with `speculative=False` nothing is
    # parked and every row is written on the spot, as before.
    user = UserId(uuid.uuid4())
    msgs = MemMessages()
    rt, _ = _make(FakeLLM([[LLMText("Sure."), LLMFinished("stop")]]), messages=msgs)

    await _turn(rt, TurnInput(user, None, "t-plain", "what is the weather"))
    assert [m.role for m in msgs.items] == ["user", "assistant"]
    assert rt.speculation.size == 0


# ---------------------------------------------------------------------------
# Nothing outside the turn may change on a guess.
# ---------------------------------------------------------------------------


class CountingWorkflowRun(WorkflowRun):
    """An open run that counts the turns handed to it."""

    def __init__(self) -> None:
        self.handled = 0

    async def handle_turn(self, user_id: UserId, text: str) -> AssessmentReply | None:
        self.handled += 1
        return AssessmentReply(sentences=["Question two of twenty."], workflow={"item": 2})


async def test_an_open_run_disqualifies_speculation_entirely() -> None:
    # Answering an item advances the run, and there is no confirmation step that
    # could move it back — so a turn the assessment engine will take is never a
    # guess. It runs and persists like any ordinary turn.
    user = UserId(uuid.uuid4())
    sessions, msgs = MemSessions(), MemMessages()
    runs = CountingWorkflowRun()
    rt, _ = _make(FakeLLM([]), active_run=runs, sessions=sessions, messages=msgs)

    events = await _turn(rt, _spec(user, "three", tid="spec-run"))

    assert runs.handled == 1
    assert [e.sentence.text for e in events if isinstance(e, SentenceEvent)] == [
        "Question two of twenty."
    ]
    assert [m.role for m in msgs.items] == ["user", "assistant"]
    assert [m.client_turn_id for m in msgs.items] == ["spec-run", "spec-run"]
    assert list(sessions.items) == [msgs.items[0].session_id]
    # Nothing was parked, so the client's confirmation is a harmless 202.
    assert rt.speculation.size == 0
    assert await rt.confirm("spec-run", "three", user) == "pending"


class RememberStub:
    """A mutating tool: what it does cannot be un-done by not confirming."""

    name: ClassVar[str] = "remember"
    mutating: ClassVar[bool] = True
    declaration: ClassVar[dict[str, Any]] = {
        "name": "remember",
        "description": "r",
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
    }

    def __init__(self) -> None:
        self.stored: list[str] = []

    async def invoke(
        self, user_id: UserId, args: dict[str, Any], facts: list[Fact] | None = None
    ) -> ToolResult:
        self.stored.append(str(args.get("value")))
        return ToolResult(ok=True, data={"status": "stored"})


async def test_a_mutating_tool_promotes_the_turn_and_persists_it() -> None:
    # The tool has to run now to answer the turn, and a stored fact has no undo.
    # So the turn stops being a guess at the moment it reaches for one: the
    # buffered rows are written on the spot, under the id they were minted with.
    user = UserId(uuid.uuid4())
    sessions, msgs = MemSessions(), MemMessages()
    tool = RememberStub()
    llm = FakeLLM(
        [
            [LLMFunctionCall(FunctionCall("remember", {"value": "teal"})), LLMFinished("stop")],
            [LLMText("Got it, teal it is."), LLMFinished("stop")],
        ]
    )
    tools = ToolRouter()
    tools.register(tool)
    rt, _ = _make(llm, tools, sessions=sessions, messages=msgs)

    events = await _turn(rt, _spec(user, "remember teal please", tid="spec-mut"))

    assert tool.stored == ["teal"]
    assert [m.role for m in msgs.items] == ["user", "assistant"]
    assert [m.client_turn_id for m in msgs.items] == ["spec-mut", "spec-mut"]
    assert len(msgs.tool_calls) == 1 and msgs.tool_calls[0][0] == msgs.items[0].id
    assert list(sessions.items) == [msgs.items[0].session_id]
    assert isinstance(events[-1], DoneEvent)
    # Nothing parked: the confirmation 202s and the client does not re-run it.
    assert rt.speculation.size == 0
    assert await rt.confirm("spec-mut", "remember teal please", user) == "pending"


async def test_a_read_only_tool_leaves_the_turn_speculative() -> None:
    # The other half of the rule: a forecast read changes nothing outside the
    # turn, so a weather turn is still a guess and still writes nothing.
    user = UserId(uuid.uuid4())
    sessions, msgs = MemSessions(), MemMessages()
    llm = FakeLLM(
        [
            [
                LLMFunctionCall(FunctionCall("get_weather", {"location": "Paris"})),
                LLMFinished("stop"),
            ],
            [LLMText("It's 22 degrees in Paris."), LLMFinished("stop")],
        ]
    )
    tools = ToolRouter()
    tools.register(Weather())
    rt, _ = _make(llm, tools, sessions=sessions, messages=msgs)

    await _turn(rt, _spec(user, tid="spec-ro"))

    assert msgs.items == [] and msgs.tool_calls == [] and sessions.items == {}
    assert await rt.confirm("spec-ro", "what is the weather", user) is True
    assert [m.role for m in msgs.items] == ["user", "assistant"]


class RecordingOutputGuard:
    def __init__(self) -> None:
        self.seen: list[GuardContext] = []

    def check_sentence(self, sentence: str, ctx: GuardContext) -> SentenceVerdict:
        # The context is mutated in place as the turn goes, so what matters is
        # its value AT the call — record that, not the object.
        self.seen.append(replace(ctx))
        return SentenceVerdict(action="pass")


async def test_a_promoted_turn_stamps_no_speculative_flag_on_later_guard_events() -> None:
    # The stamp says "this event was raised on an unconfirmed guess". After a
    # promotion that is no longer true, so the context is corrected too.
    user = UserId(uuid.uuid4())
    guard = RecordingOutputGuard()
    tools = ToolRouter()
    tools.register(RememberStub())
    llm = FakeLLM(
        [
            [LLMFunctionCall(FunctionCall("remember", {"value": "teal"})), LLMFinished("stop")],
            [LLMText("Got it."), LLMFinished("stop")],
        ]
    )
    rt, _ = _make(llm, tools, output_guard=guard)
    await _turn(rt, _spec(user, "remember teal please", tid="spec-flag"))
    assert guard.seen and all(not c.speculative for c in guard.seen)


async def test_a_guess_stamps_its_guard_events() -> None:
    user = UserId(uuid.uuid4())
    guard = RecordingOutputGuard()
    rt, _ = _make(FakeLLM([[LLMText("Sure."), LLMFinished("stop")]]), output_guard=guard)
    await _turn(rt, _spec(user))
    assert guard.seen and all(c.speculative for c in guard.seen)


async def test_a_blocked_guess_tells_the_input_guard_it_is_one() -> None:
    user = UserId(uuid.uuid4())
    g = BlockGuard()
    rt, _ = _make(FakeLLM([]), input_guard=g)
    await _turn(rt, _spec(user, "how much ibuprofen can I take", tid="spec-blk"))
    assert g.seen_speculative is True


# ---------------------------------------------------------------------------
# C1: the confirmation that overtakes its own turn.
# ---------------------------------------------------------------------------


def test_an_early_confirm_is_kept_until_the_turn_parks() -> None:
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache()
    c.early_confirm("t5", "What's the weather in Paris?", owner="u1")
    assert c.early_size == 1
    # Stored normalised, because that is the only form it is ever compared in.
    assert c.take_early("t5") == "whats the weather in paris"
    # Consumed: a turn parks exactly once, so a second reader would be a bug.
    assert c.take_early("t5") is None
    assert c.early_size == 0


def test_an_early_confirm_expires_like_a_parked_one() -> None:
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache(ttl_s=-1.0)
    c.early_confirm("t6", "hello there", owner="u1")
    assert c.take_early("t6") is None


def test_has_tells_a_parked_guess_from_an_absent_one() -> None:
    # What `take` cannot say: it returns None for both, and `confirm` has to
    # answer 409 for the first and 202 for the second.
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache()
    c.put("t7", "hello there", pending="P")
    assert c.has("t7") is True
    assert c.has("t8") is False
    c.take("t7", "goodbye")
    assert c.has("t7") is False


class _GateLLM:
    """An LLM whose stream stops mid-turn until the test lets it finish.

    This is the C1 race made deterministic: the confirmation lands while the
    model is still streaming, which is exactly when a fast recogniser finalises.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.gate = asyncio.Event()
        self.streaming = asyncio.Event()
        self.requests: list[Any] = []

    async def stream(self, req: Any) -> Any:
        self.requests.append(req)
        yield LLMText(self.text)
        self.streaming.set()
        await self.gate.wait()
        yield LLMFinished("stop")

    async def generate_json(self, req: Any, schema: Any) -> Any:
        raise NotImplementedError


async def _run_gated(rt: Any, inp: TurnInput, llm: _GateLLM, confirm_text: str) -> Any:
    """Drive a turn, confirm it mid-stream, then let it finish."""
    events: list[Any] = []
    outcome: list[Any] = []

    async def drive() -> None:
        async for ev in rt(inp):
            events.append(ev)

    task = asyncio.create_task(drive())
    await llm.streaming.wait()
    outcome.append(await rt.confirm(inp.client_turn_id, confirm_text, inp.user_id))
    llm.gate.set()
    await task
    await rt.bg.drain()
    return events, outcome[0]


async def test_a_confirm_that_beats_the_park_still_writes_the_turn() -> None:
    # The bug this closes: the confirmation arrived while the turn was still
    # streaming, found nothing parked, and answered 409. The park that followed a
    # moment later then waited out its ten seconds for a client that had already
    # confirmed — and the turn was lost, silently, with the answer spoken aloud.
    user = UserId(uuid.uuid4())
    sessions, msgs = MemSessions(), MemMessages()
    llm = _GateLLM("It's mild in Paris.")
    rt, _ = _make(llm, sessions=sessions, messages=msgs)

    events, outcome = await _run_gated(rt, _spec(user), llm, "What is the weather?")

    assert outcome == "pending"  # 202: accepted, nothing written yet
    assert isinstance(events[-1], DoneEvent)
    # Written at the park, in the same order `confirm` uses.
    assert [m.role for m in msgs.items] == ["user", "assistant"]
    assert [m.client_turn_id for m in msgs.items] == ["spec-1", "spec-1"]
    assert list(sessions.items) == [msgs.items[0].session_id]
    # Nothing is left parked, so a client that retries cannot write it twice.
    assert rt.speculation.size == 0
    assert rt.speculation.early_size == 0
    assert await rt.confirm("spec-1", "what is the weather", user) == "pending"
    assert len(msgs.items) == 2


async def test_a_confirm_that_beats_the_park_and_disagrees_writes_nothing() -> None:
    # The other half: the final transcript says something else, so the turn is
    # discarded at the park rather than parked for a confirmation that has
    # already been and gone.
    user = UserId(uuid.uuid4())
    sessions, msgs = MemSessions(), MemMessages()
    llm = _GateLLM("It's mild in Paris.")
    rt, _ = _make(llm, sessions=sessions, messages=msgs)

    events, outcome = await _run_gated(rt, _spec(user), llm, "what is the time")

    assert outcome == "pending"
    assert isinstance(events[-1], DoneEvent)
    assert msgs.items == [] and msgs.tool_calls == [] and sessions.items == {}
    # Nothing parked and nothing held: a later confirmation finds the turn gone.
    assert rt.speculation.size == 0
    assert rt.speculation.early_size == 0
    assert rt.speculation.take(f"{user}:spec-1", "what is the weather") is None
    assert await rt.confirm("spec-1", "what is the weather", user) == "pending"
    assert msgs.items == []


async def test_an_early_confirm_cannot_be_planted_on_someone_elses_turn() -> None:
    # C1 again, from the other side: the early confirm is recorded under the
    # caller-scoped key, so a stranger cannot pre-load a transcript that the
    # owner's park would then match against.
    user, other = UserId(uuid.uuid4()), UserId(uuid.uuid4())
    msgs = MemMessages()
    llm = _GateLLM("Sure.")
    rt, _ = _make(llm, messages=msgs)

    inp = _spec(user)
    events: list[Any] = []

    async def drive() -> None:
        async for ev in rt(inp):
            events.append(ev)

    task = asyncio.create_task(drive())
    await llm.streaming.wait()
    assert await rt.confirm("spec-1", "what is the weather", other) == "pending"
    llm.gate.set()
    await task
    await rt.bg.drain()

    # The owner's turn parked normally, unaffected by the stranger's entry.
    assert msgs.items == []
    assert rt.speculation.size == 1
    assert await rt.confirm("spec-1", "what is the weather", user) is True
    assert [m.role for m in msgs.items] == ["user", "assistant"]


async def test_the_after_done_confirm_path_is_unchanged() -> None:
    # Regression guard for the ordinary case: a confirmation that arrives after
    # the turn parked still writes on the spot, and a mismatched one still 409s.
    user = UserId(uuid.uuid4())
    msgs = MemMessages()
    rt, _ = _make(FakeLLM([[LLMText("Sure."), LLMFinished("stop")]]), messages=msgs)
    await _turn(rt, _spec(user))
    assert rt.speculation.size == 1
    assert await rt.confirm("spec-1", "what is the weather", user) is True
    assert [m.role for m in msgs.items] == ["user", "assistant"]


# ---------------------------------------------------------------------------
# I1: a promotion that fails leaves nothing orphaned.
# ---------------------------------------------------------------------------


class FlakyMessages(MemMessages):
    """A `MemMessages` whose flush fails at a chosen point.

    `fail_on_role` fails the message write for that role; `fail_tool_calls`
    fails `save_tool_call`. Between them they cover both ends of `_promote`'s
    flush: one where nothing lands, one where the user row lands and the rest
    does not.
    """

    def __init__(self, *, fail_on_role: str | None = None, fail_tool_calls: bool = False) -> None:
        super().__init__()
        self.fail_on_role = fail_on_role
        self.fail_tool_calls = fail_tool_calls

    async def save(self, m: Message) -> None:
        if m.role == self.fail_on_role:
            raise RuntimeError("messages repo is down")
        await super().save(m)

    async def save_tool_call(self, *row: Any) -> None:
        if self.fail_tool_calls:
            raise RuntimeError("tool_calls repo is down")
        await super().save_tool_call(*row)


def _promoting_turn(
    msgs: MemMessages, sessions: MemSessions, *, buffer_a_tool_call: bool = False
) -> tuple[Any, RememberStub]:
    """A turn that reaches for `remember` and so promotes itself mid-flight.

    With `buffer_a_tool_call` it reads the weather first, so by the time the
    promotion happens the buffer holds a tool-call row as well as the user row —
    which is the only way to make the flush fail PART of the way through.
    """
    tool = RememberStub()
    tools = ToolRouter()
    tools.register(tool)
    tools.register(Weather())
    scripts = [
        [LLMFunctionCall(FunctionCall("remember", {"value": "teal"})), LLMFinished("stop")],
        [LLMText("Got it, teal it is."), LLMFinished("stop")],
    ]
    if buffer_a_tool_call:
        scripts.insert(
            0,
            [
                LLMFunctionCall(FunctionCall("get_weather", {"location": "Paris"})),
                LLMFinished("stop"),
            ],
        )
    rt, _ = _make(FakeLLM(scripts), tools, sessions=sessions, messages=msgs)
    return rt, tool


async def test_a_promotion_that_fails_writes_no_orphan_assistant_row() -> None:
    # I1: the flags used to move before the flush, so a repo that failed
    # mid-promotion left the turn believing its user row was written. The degrade
    # path then wrote an assistant row paired with nothing — the one thing it
    # exists to avoid. With the flush first, the failure leaves the turn as it
    # was: still speculative, still holding its buffer, still writing nothing.
    msgs = FlakyMessages(fail_on_role="user")
    sessions = MemSessions()
    rt, tool = _promoting_turn(msgs, sessions)

    events = await _turn(rt, _spec(UserId(uuid.uuid4()), "remember teal", tid="spec-f1"))

    assert isinstance(events[-1], ErrorEvent)
    assert msgs.items == []  # no user row, and above all no assistant row
    assert tool.stored == []  # the promotion failed before the tool could run
    # Still a guess: its error row went into the buffer and was parked, not
    # written, which is what "the turn is exactly as it was" means.
    assert rt.speculation.size == 1


async def test_a_half_written_promotion_leaves_the_assistant_row_unwritten() -> None:
    # The other end of the flush: the session and user rows land and the buffered
    # tool-call row does not. That is half an exchange either way, but the half
    # that must NOT appear is an assistant row with no user row — and it does not.
    msgs = FlakyMessages(fail_tool_calls=True)
    sessions = MemSessions()
    rt, tool = _promoting_turn(msgs, sessions, buffer_a_tool_call=True)

    events = await _turn(rt, _spec(UserId(uuid.uuid4()), "remember teal", tid="spec-f2"))

    assert isinstance(events[-1], ErrorEvent)
    assert [m.role for m in msgs.items] == ["user"]
    assert msgs.tool_calls == []
    assert tool.stored == []  # the mutating tool never ran either


async def test_a_promotion_that_succeeds_still_clears_every_flag() -> None:
    # The ordering change must not cost the happy path anything: on success the
    # three flags move together, right after the last write.
    msgs, sessions = MemMessages(), MemSessions()
    rt, tool = _promoting_turn(msgs, sessions)

    await _turn(rt, _spec(user := UserId(uuid.uuid4()), "remember teal", tid="spec-ok"))

    assert tool.stored == ["teal"]
    assert [m.role for m in msgs.items] == ["user", "assistant"]
    assert len(msgs.tool_calls) == 1
    assert rt.speculation.size == 0
    assert await rt.confirm("spec-ok", "remember teal", user) == "pending"
    assert len(msgs.items) == 2


# ---------------------------------------------------------------------------
# The early-confirm write, when the repo is down.
# ---------------------------------------------------------------------------


async def _gated_spec_turn(rt: Any, inp: TurnInput, llm: _GateLLM, confirm_text: str) -> list[Any]:
    """Confirm mid-stream, then let the turn reach its park."""
    events: list[Any] = []

    async def drive() -> None:
        async for ev in rt(inp):
            events.append(ev)

    task = asyncio.create_task(drive())
    await llm.streaming.wait()
    assert await rt.confirm(inp.client_turn_id, confirm_text, inp.user_id) == "pending"
    llm.gate.set()
    await task
    await rt.bg.drain()
    return events


async def test_a_failed_early_confirm_write_does_not_apologise_for_the_answer() -> None:
    # The turn was right, the answer was spoken in full, and only the WRITE of it
    # failed. Clearing the buffer before the write meant the degrade path saw a
    # turn that still believed its user row existed: it appended "Sorry, I lost my
    # train of thought." to an answer the listener had just heard end to end, and
    # wrote an assistant row with nothing to pair it to.
    user = UserId(uuid.uuid4())
    msgs = FlakyMessages(fail_on_role="assistant")
    sessions = MemSessions()
    llm = _GateLLM("It's mild in Paris.")
    rt, _ = _make(llm, sessions=sessions, messages=msgs)

    events = await _gated_spec_turn(rt, _spec(user), llm, "what is the weather")

    spoken = [e.sentence.text for e in events if isinstance(e, SentenceEvent)]
    assert spoken == ["It's mild in Paris."]  # no closer bolted onto the answer
    assert isinstance(events[-1], ErrorEvent)
    # No assistant row — the write that failed is not retried into an orphan.
    assert not any(m.role == "assistant" for m in msgs.items)
    # The buffer is still held, so nothing about the turn was thrown away either.
    assert rt.speculation.size == 0 and rt.speculation.early_size == 0


async def test_a_failed_early_confirm_write_at_the_user_row_writes_nothing() -> None:
    # The first message write is the user row, so a repo that is down for the
    # whole flush leaves the database exactly as it found it.
    user = UserId(uuid.uuid4())
    msgs = FlakyMessages(fail_on_role="user")
    sessions = MemSessions()
    llm = _GateLLM("It's mild in Paris.")
    rt, _ = _make(llm, sessions=sessions, messages=msgs)

    events = await _gated_spec_turn(rt, _spec(user), llm, "what is the weather")

    assert [e.sentence.text for e in events if isinstance(e, SentenceEvent)] == [
        "It's mild in Paris."
    ]
    assert isinstance(events[-1], ErrorEvent)
    assert msgs.items == [] and msgs.tool_calls == []


async def test_a_successful_early_confirm_write_still_clears_the_buffer() -> None:
    # The other side of holding the buffer until the write returns: on success it
    # is released, so a second confirmation cannot write the same turn twice.
    user = UserId(uuid.uuid4())
    msgs = MemMessages()
    llm = _GateLLM("It's mild in Paris.")
    rt, _ = _make(llm, messages=msgs)

    events = await _gated_spec_turn(rt, _spec(user), llm, "what is the weather")

    assert isinstance(events[-1], DoneEvent)
    assert [m.role for m in msgs.items] == ["user", "assistant"]
    assert await rt.confirm("spec-1", "what is the weather", user) == "pending"
    assert len(msgs.items) == 2


# ---------------------------------------------------------------------------
# The early-confirm store is bounded.
# ---------------------------------------------------------------------------


def test_the_early_store_is_capped_per_owner_and_evicts_oldest_first() -> None:
    # `/chat/confirm` is the one way into this process's memory with no turn
    # behind it. 300 junk confirmations from one caller must not leave 300
    # entries — nor evict the newest, which is the one most likely to be waiting
    # for a turn that is still streaming.
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache()
    for i in range(300):
        c.early_confirm(f"u:{i}", "hello there", owner="u")
    assert c.early_size <= 256
    assert c.take_early("u:0") is None  # the oldest went
    assert c.take_early("u:299") == "hello there"  # the newest stayed


def test_one_owner_cannot_evict_anothers_early_confirms() -> None:
    # The per-owner cap has to bite before the global one, or a single caller
    # filling the store pushes out every other caller's pending confirmation.
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache()
    c.early_confirm("victim:1", "hello there", owner="victim")
    for i in range(300):
        c.early_confirm(f"flood:{i}", "junk", owner="flood")
    assert c.take_early("victim:1") == "hello there"


def test_the_early_store_is_capped_globally_across_owners() -> None:
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache()
    # 60 owners x 200 each = 12,000 attempted, each owner under the per-user cap.
    for owner in range(60):
        for i in range(200):
            c.early_confirm(f"o{owner}:{i}", "hello there", owner=f"o{owner}")
    assert c.early_size <= 10_000


def test_taking_an_early_confirm_frees_the_owners_allowance() -> None:
    # The per-owner count is bookkeeping, and bookkeeping that only ever counts
    # up is a cap that eventually refuses a caller who is behaving.
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache()
    for i in range(256):
        c.early_confirm(f"u:{i}", "hello there", owner="u")
    for i in range(256):
        assert c.take_early(f"u:{i}") == "hello there"
    c.early_confirm("u:new", "hello there", owner="u")
    assert c.take_early("u:new") == "hello there"


def test_expiry_does_not_depend_on_the_sweep() -> None:
    # `_gc` is throttled now, so a read that trusted it would hand back an entry
    # whose ten-second window has gone — which for `take` means writing a turn
    # the client stopped waiting for.
    c: SpeculativeTurnCache[str] = SpeculativeTurnCache(ttl_s=-1.0)
    c.put("t-exp", "hello there", pending="P")
    c.early_confirm("t-exp", "hello there", owner="u")
    assert c.has("t-exp") is False
    assert c.take("t-exp", "hello there") is None
    assert c.take_early("t-exp") is None
    assert c.size == 0 and c.early_size == 0
