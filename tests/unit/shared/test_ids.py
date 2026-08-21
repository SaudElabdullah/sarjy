import uuid

from sarjy.shared.ids import SessionId, UserId, new_id, parse_id


def test_new_id_is_uuid4() -> None:
    sid = new_id(SessionId)
    assert uuid.UUID(str(sid)).version == 4


def test_parse_id_roundtrip() -> None:
    raw = "0b2a4b1e-5a1c-4d7a-9b6e-2f1d3c4b5a60"
    uid = parse_id(UserId, raw)
    assert str(uid) == raw


def test_parse_id_rejects_garbage() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_id(UserId, "nope")
