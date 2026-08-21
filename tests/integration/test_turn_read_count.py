"""How many times a turn reads the database (PRD L-7).

The target is one. A turn used to open with a session read, a history read, a
facts read and an active-run read — four round trips, in sequence, before the
model could be asked for a token. `load_turn_context` answers all four, and the
only way to keep it that way is to count.

The probe wraps `Database`'s read methods and records every statement, so a
failure here names the query that crept back in rather than just a number — and
each case below asserts the exact list, not just its length. A budget expressed
as a number is a budget that silently accepts one read being swapped for
another; expressed as a list, a regression has to say which query it added.

Not every turn owes exactly one read, and the cases here are deliberately honest
about which ones do not:

* an ordinary turn, new session or resumed — one, the RPC.
* an open OCEAN run — the RPC, plus what the assessment engine reads to advance
  the run. That work cannot come out of `load_turn_context`: the RPC returns the
  run's *snapshot* for the prompt, while advancing it needs the aggregate and
  its instrument, and reading them speculatively on every chat turn would cost
  the other 99% of turns a round trip to be told there is no run.
* a cold weather turn — the RPC, plus the weather cache lookup. The lookup is
  the thing that saves the provider HTTP call, so it is a read that buys a much
  more expensive one.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from sarjy.config import Settings
from sarjy.container import Container
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.contexts.assessment.infrastructure.offline_interpreter import OfflineInterpreter
from sarjy.contexts.assessment.infrastructure.offline_narrator import OfflineNarrator
from sarjy.contexts.conversation.application.ports import (
    FunctionCall,
    LLMEvent,
    LLMFinished,
    LLMFunctionCall,
    LLMRequest,
    LLMText,
)
from sarjy.contexts.conversation.domain.message import Message
from sarjy.contexts.conversation.domain.session import Session
from sarjy.contexts.conversation.domain.turn import TurnInput
from sarjy.contexts.guardrails.infrastructure.offline_classifier import OfflineClassifier
from sarjy.contexts.weather.infrastructure.mock_provider import MockProvider
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import MessageId, RunId, SessionId, UserId

pytestmark = pytest.mark.integration

DEF_ID = "ocean_mini_ipip"

# The exact statements a turn is allowed to run, normalised the way `ReadCounter`
# records them. Spelled out rather than substring-matched so that a query which
# changes shape has to be looked at rather than silently still passing.
RPC = "select public.load_turn_context($1,$2,$3)"
WEATHER_CACHE = "select payload, fetched_at from weather_cache where cache_key=$1"
# Truncated at 90 characters, the way `ReadCounter` records it.
RUN_AGGREGATE = (
    "select id, user_id, definition_id, definition_version, status, current_item, skips_used, r"
)
INSTRUMENT = "select definition from workflow_definitions where id = $1 and active"


class ScriptedLLM:
    """A `LLMPort` that replays a fixed script per model round."""

    def __init__(self, *rounds: list[LLMEvent]) -> None:
        self.rounds = [list(r) for r in rounds]
        self.requests: list[LLMRequest] = []

    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        self.requests.append(req)
        for e in self.rounds.pop(0):
            yield e

    async def generate_json(self, req: LLMRequest, schema: Any) -> Any:
        raise NotImplementedError


def _say(text: str) -> list[LLMEvent]:
    return [LLMText(text), LLMFinished("stop")]


class ReadCounter:
    """Wraps a `Database` and records the statements it runs."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.reads: list[str] = []
        self.writes: list[str] = []
        for name in ("fetch", "fetchrow", "fetchval"):
            setattr(db, name, self._wrap(getattr(db, name), self.reads))
        db.execute = self._wrap(db.execute, self.writes)  # type: ignore[method-assign]

    @staticmethod
    def _wrap(fn: Any, log: list[str]) -> Any:
        async def wrapped(q: str, *args: Any) -> Any:
            log.append(" ".join(q.split())[:90])
            return await fn(q, *args)

        return wrapped


@pytest.fixture
async def container() -> AsyncIterator[Container]:
    # `weather_provider="mock"`: the weather case must reach the cache without
    # reaching the internet, and the mock provider answers any city offline.
    c = Container.build(Settings(weather_provider="mock"), connect_db=True)  # type: ignore[call-arg]
    # No live Gemini from an integration test: the Layer-3 classifier is only
    # consulted for an `uncertain` verdict, which none of these fixtures is, but
    # a test must not depend on that. The assessment interpreter/narrator are
    # swapped for the same reason — the OCEAN case runs a real turn through them.
    c.rebuild_guards(classifier=OfflineClassifier())
    c.interpreter, c.narrator = OfflineInterpreter(), OfflineNarrator()
    c.rebuild_assessment()
    c.weather_provider = MockProvider(c.clock, temp_c=18.0)
    c.rebuild_weather()
    c.rebuild_run_turn()
    await c.startup()
    yield c
    await c.shutdown()


async def _user(c: Container) -> UserId:
    u = uuid.uuid4()
    await c.db.execute("insert into auth.users (id,email) values ($1,$2)", u, f"{u}@x.test")
    return UserId(u)


@dataclass
class Case:
    """One turn shape, the reads it is allowed, and what it takes to set up."""

    id: str
    text: str
    rounds: list[list[LLMEvent]]
    expected_reads: list[str]
    resume: bool = False
    open_run: bool = False


CASES = [
    Case(
        id="new_session",
        text="hello there",
        rounds=[_say("Hello there.")],
        expected_reads=[RPC],
    ),
    Case(
        id="resumed_session",
        text="hello there",
        rounds=[_say("Hello there.")],
        expected_reads=[RPC],
        resume=True,
    ),
    Case(
        id="weather_cold_cache",
        text="what's the weather in Tokyo",
        rounds=[
            [LLMFunctionCall(FunctionCall("get_weather", {"location": "Tokyo"})), LLMFinished("s")],
            _say("It's mild in Tokyo."),
        ],
        # Two, and the second one is the point of L-5: a cache lookup that saves
        # a provider HTTP call is a read worth paying for. What is NOT here is a
        # `memories` read — the tool used to fetch its own copy of the facts the
        # turn had already loaded (I5).
        expected_reads=[RPC, WEATHER_CACHE],
    ),
    Case(
        id="open_ocean_run",
        text="four",
        # The assessment engine takes the whole turn; the model is never asked.
        rounds=[],
        # Two, honestly. `load_turn_context` returns the run's *snapshot* — enough
        # to build the prompt block and to know a run is open — but ADVANCING the
        # run needs the aggregate itself, which the RPC does not return and could
        # not usefully return: it would be read on every chat turn, for the 99%
        # that have no run open, to be told so.
        #
        # A third read, `workflow_definitions`, happens on the first OCEAN turn a
        # process ever serves and never again — `PgInstrumentRepo` caches the
        # definition for the life of the instance. Warmed in `_setup` below,
        # because a per-process cost is not a per-turn one and counting it here
        # would budget for something the second turn of a session never pays.
        expected_reads=[RPC, RUN_AGGREGATE],
        open_run=True,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
async def test_a_turn_runs_exactly_the_reads_it_is_budgeted(
    container: Container, case: Case
) -> None:
    c = container
    user = await _user(c)
    session_id = await _setup(c, case, user)

    c.llm = ScriptedLLM(*case.rounds)
    c.rebuild_run_turn()
    probe = ReadCounter(c.db)

    events = [e async for e in c.run_turn(TurnInput(user, session_id, "probe-1", case.text))]
    await c.bg.drain(timeout=2)

    assert events, "the turn produced no events"
    assert probe.reads == case.expected_reads
    # Writes are unbudgeted but should be exactly the two rows a turn owes,
    # plus the deferred session touch (or the insert of a new session).
    assert sum("into messages" in w for w in probe.writes) == 2
    assert sum("sessions" in w for w in probe.writes) == 1


async def test_the_first_ocean_turn_of_a_process_also_reads_the_definition(
    container: Container,
) -> None:
    # The read the parametrised case above warms away, asserted where it belongs:
    # once per process, not once per turn. Stated rather than hidden, so that if
    # the instrument cache is ever removed this test says which read came back.
    c = container
    user = await _user(c)
    run = WorkflowRun.propose(RunId(uuid.uuid4()), user, DEF_ID, 1, datetime.now(UTC))
    run.confirm(datetime.now(UTC))
    await c.run_repo.save(run)  # type: ignore[union-attr]
    c.llm = ScriptedLLM()
    c.rebuild_run_turn()
    probe = ReadCounter(c.db)

    [e async for e in c.run_turn(TurnInput(user, None, "probe-cold-run", "four"))]
    await c.bg.drain(timeout=2)

    assert probe.reads == [RPC, RUN_AGGREGATE, INSTRUMENT]


async def _setup(c: Container, case: Case, user: UserId) -> SessionId | None:
    """Whatever the case needs in place before the probe goes on."""
    now = datetime.now(UTC)
    if case.open_run:
        run = WorkflowRun.propose(RunId(uuid.uuid4()), user, DEF_ID, 1, now)
        run.confirm(now)
        await c.run_repo.save(run)  # type: ignore[union-attr]
        # Once per process, not once per turn — see the case's comment.
        await c.instrument_repo.get(DEF_ID)  # type: ignore[union-attr]
    if not case.resume:
        return None
    session = Session.start(SessionId(uuid.uuid4()), user, now)
    await c.sessions.save(session)  # type: ignore[union-attr]
    await c.messages.save(  # type: ignore[union-attr]
        Message(MessageId(uuid.uuid4()), session.id, user, "user", "hi", now)
    )
    return session.id


async def test_a_resumed_turn_hands_the_model_its_own_history(container: Container) -> None:
    # The session and its history came out of the same read the budget above
    # counts — the point of folding four round trips into one is that the turn
    # still knows everything it knew before.
    c = container
    user = await _user(c)
    now = datetime.now(UTC)
    session = Session.start(SessionId(uuid.uuid4()), user, now)
    await c.sessions.save(session)  # type: ignore[union-attr]
    await c.messages.save(  # type: ignore[union-attr]
        Message(MessageId(uuid.uuid4()), session.id, user, "user", "hi", now)
    )
    llm = ScriptedLLM(_say("Hello there."))
    c.llm = llm
    c.rebuild_run_turn()

    [e async for e in c.run_turn(TurnInput(user, session.id, "probe-hist", "hello there"))]
    await c.bg.drain(timeout=2)

    assert any(m.text == "<user>hi</user>" for m in llm.requests[0].messages)
