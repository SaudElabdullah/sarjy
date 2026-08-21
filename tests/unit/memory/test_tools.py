import uuid
from datetime import UTC, datetime

from sarjy.contexts.conversation.application.ports import ToolPort
from sarjy.contexts.memory.application.forget import ForgetFact
from sarjy.contexts.memory.application.ports import ScreenVerdict
from sarjy.contexts.memory.application.recall import RecallFacts
from sarjy.contexts.memory.application.remember import RememberFact
from sarjy.contexts.memory.application.tools import ForgetTool, RecallTool, RememberTool
from sarjy.contexts.memory.infrastructure.in_memory_repo import InMemoryMemoryRepo
from sarjy.shared.clock import FakeClock
from sarjy.shared.ids import MessageId, UserId


class _AllowAllScreen:
    """`ValueScreenPort` double: these tests are about tool wiring, not
    screening (Phase 8 T6b fix round 1, Minor 3: `screen` is a required
    `RememberFact` constructor arg now)."""

    def screen(
        self, text: str, *, user_id: UserId | None = None, message_id: MessageId | None = None
    ) -> ScreenVerdict:
        return ScreenVerdict(allowed=True)


def _tools():  # type: ignore[no-untyped-def]
    repo, clock = InMemoryMemoryRepo(), FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    return (
        RememberTool(RememberFact(repo, clock, screen=_AllowAllScreen())),
        ForgetTool(ForgetFact(repo, clock)),
        RecallTool(RecallFacts(repo)),
    )


def test_tools_satisfy_tool_port() -> None:
    r, f, c = _tools()
    remember_port: ToolPort = r
    forget_port: ToolPort = f
    recall_port: ToolPort = c
    for tool in (remember_port, forget_port, recall_port):
        assert tool.declaration["name"] == tool.name


def test_declarations_match_prd() -> None:
    r, f, c = _tools()
    assert (r.name, f.name, c.name) == ("remember", "forget", "recall")
    assert r.declaration["parameters"]["required"] == ["key", "value"]
    assert f.declaration["parameters"]["required"] == ["key"]
    assert set(r.declaration["parameters"]["properties"]["kind"]["enum"]) == {
        "fact",
        "preference",
        "person",
        "place",
        "note",
    }
    desc = c.declaration["description"].lower()
    assert "never guess" in desc or "don't have it stored" in desc


async def test_remember_forget_recall_roundtrip() -> None:
    r, f, c = _tools()
    u = UserId(uuid.uuid4())
    res = await r.invoke(u, {"key": "favorite color", "value": "teal"})
    assert res.ok and res.data["status"] == "created" and res.data["key"] == "favorite_color"
    res = await c.invoke(u, {})
    assert res.ok and res.data["count"] == 1 and res.data["facts"][0]["value"] == "teal"
    res = await f.invoke(u, {"key": "favorite_color"})
    assert res.ok and res.data["status"] == "forgotten"
    res = await c.invoke(u, {"query": "color"})
    assert res.ok and res.data["count"] == 0 and res.data["facts"] == []


async def test_remember_pii_rejected_with_spoken_error() -> None:
    r, _, _ = _tools()
    res = await r.invoke(UserId(uuid.uuid4()), {"key": "ssn", "value": "123-45-6789"})
    assert not res.ok and res.data["status"] == "rejected"
    assert res.spoken_error and res.spoken_error.startswith("I won't store that")


async def test_remember_missing_args_is_safe() -> None:
    r, _, _ = _tools()
    res = await r.invoke(UserId(uuid.uuid4()), {"key": "x"})
    assert not res.ok and res.data["status"] == "rejected"


async def test_remember_invalid_kind_falls_back_to_fact() -> None:
    r, _, c = _tools()
    u = UserId(uuid.uuid4())
    res = await r.invoke(u, {"key": "snack", "value": "olives", "kind": "banana"})
    assert res.ok and res.data["status"] == "created"
    recalled = await c.invoke(u, {"query": "snack"})
    assert recalled.data["facts"][0]["kind"] == "fact"


async def test_remember_on_existing_key_returns_updated() -> None:
    r, _, _ = _tools()
    u = UserId(uuid.uuid4())
    first = await r.invoke(u, {"key": "favorite_color", "value": "teal"})
    assert first.ok and first.data["status"] == "created"
    second = await r.invoke(u, {"key": "favorite_color", "value": "navy"})
    assert second.ok and second.data["status"] == "updated" and second.data["value"] == "navy"


async def test_forget_on_key_that_never_existed_is_domain_not_found() -> None:
    _, f, _ = _tools()
    u = UserId(uuid.uuid4())
    # A real, non-empty key that was simply never remembered — distinct from the
    # tool's own "missing key" argument-validation branch above.
    res = await f.invoke(u, {"key": "shoe_size"})
    assert res.ok and res.data["status"] == "not_found"
