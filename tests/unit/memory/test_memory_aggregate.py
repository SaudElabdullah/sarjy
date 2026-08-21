import uuid
from datetime import UTC, datetime, timedelta

import pytest

from sarjy.contexts.memory.domain.memory import Memory, MemoryChanged
from sarjy.shared.errors import ValidationError
from sarjy.shared.ids import MemoryId, UserId

T0 = datetime(2026, 8, 21, tzinfo=UTC)


def _mem(value: str = "teal") -> Memory:
    return Memory.create(
        MemoryId(uuid.uuid4()), UserId(uuid.uuid4()), "favorite_color", value, "fact", now=T0
    )


def test_create_records_event_and_is_live() -> None:
    m = _mem()
    assert m.is_live and m.value == "teal"
    evs = m.pull_events()
    assert len(evs) == 1 and isinstance(evs[0], MemoryChanged)
    assert evs[0].action == "create" and evs[0].new_value == "teal" and evs[0].old_value is None
    assert m.pull_events() == []  # drained


def test_update_records_old_and_new() -> None:
    m = _mem()
    m.pull_events()
    m.update("navy", now=T0 + timedelta(days=1))
    ev = m.pull_events()[0]
    assert ev.action == "update" and ev.old_value == "teal" and ev.new_value == "navy"
    assert m.updated_at == T0 + timedelta(days=1)


def test_forget_soft_deletes() -> None:
    m = _mem()
    m.pull_events()
    m.forget(now=T0 + timedelta(hours=1))
    assert not m.is_live and m.deleted_at == T0 + timedelta(hours=1)
    assert m.pull_events()[0].action == "delete"


def test_create_rejects_empty_or_long_value() -> None:
    with pytest.raises(ValidationError):
        Memory.create(MemoryId(uuid.uuid4()), UserId(uuid.uuid4()), "k", "   ", "fact", now=T0)
    with pytest.raises(ValidationError):
        Memory.create(MemoryId(uuid.uuid4()), UserId(uuid.uuid4()), "k", "x" * 201, "fact", now=T0)


def test_update_on_deleted_raises() -> None:
    m = _mem()
    m.forget(now=T0)
    with pytest.raises(ValidationError):
        m.update("x", now=T0)
