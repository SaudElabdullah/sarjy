"""scripts/rescreen_memories.py — the pre-6b memory backfill sweep.

Driven entirely through an in-memory `MemoriesRepo` double: no database, no
network. The screening itself is the real `RuleEngineValueScreen` over the real
`DEFAULT_RULES`, so a rule change that would stop catching a stored payload
fails here too.
"""

from __future__ import annotations

import io
import uuid

from scripts.rescreen_memories import (
    Refusal,
    StoredMemory,
    default_screen,
    format_refusal,
    preview,
    rescreen,
    screen_memories,
)

from sarjy.contexts.memory.application.ports import ScreenVerdict
from sarjy.shared.ids import MessageId, UserId


class FakeRepo:
    """In-memory `MemoriesRepo`, recording what the sweep deleted."""

    def __init__(self, memories: list[StoredMemory]) -> None:
        self.memories = list(memories)
        self.deleted: list[str] = []

    async def all_memories(self) -> list[StoredMemory]:
        return list(self.memories)

    async def delete(self, memory_id: str) -> None:
        self.deleted.append(memory_id)
        self.memories = [m for m in self.memories if m.id != memory_id]


class RefuseContaining:
    """`ValueScreenPort` double: refuses any string containing `needle`."""

    def __init__(self, needle: str, reason: str = "test.rule") -> None:
        self.needle, self.reason = needle, reason
        self.seen: list[str] = []

    def screen(
        self, text: str, *, user_id: UserId | None = None, message_id: MessageId | None = None
    ) -> ScreenVerdict:
        self.seen.append(text)
        if self.needle in text:
            return ScreenVerdict(allowed=False, reason=self.reason)
        return ScreenVerdict(allowed=True)


def _mem(key: str | None, value: str) -> StoredMemory:
    return StoredMemory(str(uuid.uuid4()), str(uuid.uuid4()), key, value)


# ---------- preview: never the whole value ----------


def test_preview_truncates_at_forty_characters() -> None:
    long = "x" * 100
    out = preview(long)
    assert out == "x" * 40 + "…"
    assert len(out) == 41


def test_preview_leaves_a_short_value_alone() -> None:
    assert preview("teal") == "teal"


def test_preview_flattens_newlines_so_one_refusal_stays_one_line() -> None:
    assert preview("a\nb\tc  d") == "a b c d"


def test_format_refusal_never_contains_the_whole_value() -> None:
    secret = "SECRET-" + "y" * 200
    r = Refusal(_mem("k", secret), "value", "sh.rule")
    line = format_refusal(r)
    assert secret not in line
    assert "\n" not in line
    assert "sh.rule" in line
    assert "refused_on=value" in line


# ---------- screening: key and value, separately ----------


def test_screen_memories_returns_nothing_for_a_clean_database() -> None:
    memories = [_mem("favorite_color", "teal"), _mem("favorite_food", "pho")]
    assert screen_memories(memories, RefuseContaining("nope")) == []


def test_screen_memories_flags_a_refused_value() -> None:
    bad = _mem("note", "PAYLOAD here")
    refusals = screen_memories([_mem("k", "fine"), bad], RefuseContaining("PAYLOAD"))
    assert len(refusals) == 1
    assert refusals[0].memory is bad
    assert refusals[0].field == "value"
    assert refusals[0].rule == "test.rule"


def test_screen_memories_flags_a_refused_key() -> None:
    bad = _mem("PAYLOAD", "harmless")
    refusals = screen_memories([bad], RefuseContaining("PAYLOAD"))
    assert len(refusals) == 1
    assert refusals[0].field == "key"


def test_screen_memories_never_concatenates_the_key_and_the_value() -> None:
    # The Task 6b fix-round-1 bypass: screening "{key} {value}" as one string
    # lets a memory-set-frame-shaped pair reconstruct the carve-out. This sweep
    # must screen the two fields as independent calls, exactly like
    # `memory/application/screening.py`.
    screen = RefuseContaining("IMPOSSIBLE-NEEDLE")
    screen_memories([_mem("remember", "that my motto is 'x'")], screen)
    assert screen.seen == ["remember", "that my motto is 'x'"]


def test_screen_memories_skips_an_empty_key_rather_than_screening_the_empty_string() -> None:
    screen = RefuseContaining("IMPOSSIBLE-NEEDLE")
    screen_memories([_mem(None, "teal")], screen)
    assert screen.seen == ["teal"]


def test_the_real_screen_refuses_a_payload_the_write_path_would_refuse_today() -> None:
    # Not a double: the same `RuleEngineValueScreen(DEFAULT_RULES)` the live
    # write path uses, so this is the actual backfill behaviour.
    bad = _mem("note", "how to build a pipe bomb at home")
    good = _mem("favorite_color", "teal")
    refusals = screen_memories([good, bad], default_screen())
    assert [r.memory.id for r in refusals] == [bad.id]


# ---------- the sweep: dry by default ----------


async def test_rescreen_is_a_dry_run_by_default() -> None:
    bad = _mem("note", "PAYLOAD here")
    repo = FakeRepo([bad])
    out = io.StringIO()

    refusals = await rescreen(repo, RefuseContaining("PAYLOAD"), out=out)

    assert len(refusals) == 1
    assert repo.deleted == []
    assert repo.memories == [bad]
    report = out.getvalue()
    assert "screened 1 memories, 1 refused" in report
    assert "dry run" in report


async def test_rescreen_deletes_only_the_refused_rows_when_asked() -> None:
    bad = _mem("note", "PAYLOAD here")
    good = _mem("favorite_color", "teal")
    repo = FakeRepo([good, bad])
    out = io.StringIO()

    await rescreen(repo, RefuseContaining("PAYLOAD"), delete=True, out=out)

    assert repo.deleted == [bad.id]
    assert [m.id for m in repo.memories] == [good.id]
    assert "deleted 1 memories" in out.getvalue()


async def test_rescreen_on_a_clean_database_prints_no_dry_run_advice() -> None:
    repo = FakeRepo([_mem("favorite_color", "teal")])
    out = io.StringIO()

    assert await rescreen(repo, RefuseContaining("PAYLOAD"), out=out) == []

    report = out.getvalue()
    assert "screened 1 memories, 0 refused" in report
    assert "dry run" not in report
    assert repo.deleted == []


async def test_rescreen_report_never_prints_a_value_in_full() -> None:
    secret = "PAYLOAD-" + "z" * 300
    repo = FakeRepo([_mem("note", secret)])
    out = io.StringIO()

    await rescreen(repo, RefuseContaining("PAYLOAD"), out=out)

    assert secret not in out.getvalue()
