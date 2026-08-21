import uuid
from datetime import UTC, datetime

from sarjy.config import Settings
from sarjy.contexts.conversation.application.ports import (
    FunctionCall,
    LLMFinished,
    LLMFunctionCall,
    LLMText,
)
from sarjy.contexts.conversation.application.prompt_builder import PromptBuilder
from sarjy.contexts.conversation.application.run_turn import RunTurn
from sarjy.contexts.conversation.application.tool_router import ToolRouter
from sarjy.contexts.conversation.domain.turn import TurnInput
from sarjy.contexts.conversation.infrastructure.memory_repos import (
    InMemoryContextLoader,
    MemMessages,
    MemSessions,
)
from sarjy.contexts.conversation.infrastructure.noop_guards import (
    AllowAllInputGuard,
    NoActiveRun,
    NoFacts,
    PassOutputGuard,
)
from sarjy.contexts.weather.application.intent import is_weather_question
from sarjy.shared.clock import FakeClock
from sarjy.shared.ids import UserId
from tests.unit.conversation.test_run_turn import FakeLLM, Weather, _turn


def _rt(llm: FakeLLM) -> RunTurn:
    tools = ToolRouter()
    tools.register(Weather())
    messages, runs = MemMessages(), NoActiveRun()
    return RunTurn(
        llm=llm,
        prompt_builder=PromptBuilder(),
        tools=tools,
        input_guard=AllowAllInputGuard(),
        output_guard=PassOutputGuard(),
        context=InMemoryContextLoader(NoFacts(), messages, runs),
        active_run=runs,
        sessions=MemSessions(),
        messages=messages,
        clock=FakeClock(datetime(2026, 8, 21, tzinfo=UTC)),
        settings=Settings(),
        force_tool_when=lambda t: "get_weather" if is_weather_question(t) else None,
    )


async def test_weather_question_forces_tool_on_first_hop_only() -> None:
    llm = FakeLLM(
        [
            [
                LLMFunctionCall(FunctionCall("get_weather", {"location": "Tokyo"})),
                LLMFinished("stop"),
            ],
            [LLMText("It's 22 degrees in Tokyo."), LLMFinished("stop")],
        ]
    )
    _ = await _turn(
        _rt(llm), TurnInput(UserId(uuid.uuid4()), None, "t1", "what's the weather in tokyo")
    )
    assert llm.requests[0].force_tool == "get_weather"
    assert llm.requests[1].force_tool is None


async def test_non_weather_question_does_not_force() -> None:
    llm = FakeLLM([[LLMText("Hi!"), LLMFinished("stop")]])
    _ = await _turn(_rt(llm), TurnInput(UserId(uuid.uuid4()), None, "t2", "hello there"))
    assert llm.requests[0].force_tool is None


async def test_forced_tool_not_registered_is_not_forced() -> None:
    llm = FakeLLM([[LLMText("Hi!"), LLMFinished("stop")]])
    messages, runs = MemMessages(), NoActiveRun()
    rt = RunTurn(
        llm=llm,
        prompt_builder=PromptBuilder(),
        tools=ToolRouter(),
        input_guard=AllowAllInputGuard(),
        output_guard=PassOutputGuard(),
        context=InMemoryContextLoader(NoFacts(), messages, runs),
        active_run=runs,
        sessions=MemSessions(),
        messages=messages,
        clock=FakeClock(datetime(2026, 8, 21, tzinfo=UTC)),
        settings=Settings(),
        force_tool_when=lambda t: "get_weather" if is_weather_question(t) else None,
    )
    _ = await _turn(rt, TurnInput(UserId(uuid.uuid4()), None, "t3", "what's the weather in tokyo"))
    assert llm.requests[0].force_tool is None
