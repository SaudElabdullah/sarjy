import asyncio
import uuid
from typing import Any, ClassVar

import pytest

from sarjy.contexts.conversation.application.ports import Fact, FunctionCall, ToolResult
from sarjy.contexts.conversation.application.tool_router import ToolRouter
from sarjy.shared.ids import UserId


class Echo:
    name: ClassVar[str] = "echo"
    mutating: ClassVar[bool] = False
    declaration: ClassVar[dict] = {
        "name": "echo",
        "description": "echo",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
    }

    async def invoke(  # type: ignore[type-arg]
        self, user_id: UserId, args: dict, facts: list[Fact] | None = None
    ) -> ToolResult:
        return ToolResult(ok=True, data={"x": args["x"]})


class ExceptionTool:
    name: ClassVar[str] = "exception_tool"
    mutating: ClassVar[bool] = False
    declaration: ClassVar[dict] = {
        "name": "exception_tool",
        "description": "raises an exception",
        "parameters": {"type": "object", "properties": {}},
    }

    async def invoke(  # type: ignore[type-arg]
        self, user_id: UserId, args: dict, facts: list[Fact] | None = None
    ) -> ToolResult:
        raise ValueError("Something went wrong")


class SlowTool:
    name: ClassVar[str] = "slow_tool"
    mutating: ClassVar[bool] = False
    declaration: ClassVar[dict] = {
        "name": "slow_tool",
        "description": "slow tool that times out",
        "parameters": {"type": "object", "properties": {}},
    }

    async def invoke(  # type: ignore[type-arg]
        self, user_id: UserId, args: dict, facts: list[Fact] | None = None
    ) -> ToolResult:
        await asyncio.sleep(1.0)  # Sleep longer than timeout
        return ToolResult(ok=True, data={})


async def test_router_dispatches_and_lists_declarations() -> None:
    r = ToolRouter()
    r.register(Echo())
    assert [d["name"] for d in r.declarations()] == ["echo"]
    res = await r.invoke(UserId(uuid.uuid4()), FunctionCall("echo", {"x": "hi"}))
    assert res.ok
    assert res.data["x"] == "hi"
    assert res.latency_ms >= 0
    assert "_latency_ms" not in res.data


async def test_unknown_tool_is_safe() -> None:
    res = await ToolRouter().invoke(UserId(uuid.uuid4()), FunctionCall("nope", {}))
    assert not res.ok and res.data["error"] == "unknown_tool"


async def test_tool_exception() -> None:
    r = ToolRouter()
    r.register(ExceptionTool())
    res = await r.invoke(UserId(uuid.uuid4()), FunctionCall("exception_tool", {}))
    assert not res.ok
    assert res.data["error"] == "exception"


async def test_tool_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sarjy.contexts.conversation.application.tool_router.TOOL_TIMEOUT_S", 0.01)
    r = ToolRouter()
    r.register(SlowTool())
    res = await r.invoke(UserId(uuid.uuid4()), FunctionCall("slow_tool", {}))
    assert not res.ok
    assert res.data["error"] == "timeout"


class SharedDictTool:
    """Returns the *same* dict object every call, as a tool with a cached payload would."""

    name: ClassVar[str] = "shared"
    mutating: ClassVar[bool] = False
    declaration: ClassVar[dict] = {
        "name": "shared",
        "description": "shared payload",
        "parameters": {"type": "object", "properties": {}},
    }

    def __init__(self) -> None:
        self.payload: dict[str, Any] = {"temp_c": 22}

    async def invoke(  # type: ignore[type-arg]
        self, user_id: UserId, args: dict, facts: list[Fact] | None = None
    ) -> ToolResult:
        return ToolResult(ok=True, data=self.payload)


async def test_latency_is_reported_without_mutating_the_tool_payload() -> None:
    r = ToolRouter()
    tool = SharedDictTool()
    r.register(tool)
    res = await r.invoke(UserId(uuid.uuid4()), FunctionCall("shared", {}))
    assert res.latency_ms >= 0
    assert tool.payload == {"temp_c": 22}  # the tool's own dict is untouched
    assert res.data == {"temp_c": 22}
