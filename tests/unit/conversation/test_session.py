import uuid
from datetime import UTC, datetime, timedelta

from sarjy.contexts.conversation.domain.session import Session
from sarjy.shared.ids import SessionId, UserId


def _s(t: datetime) -> Session:
    return Session.start(SessionId(uuid.uuid4()), UserId(uuid.uuid4()), now=t)


def test_session_expires_after_30_minutes() -> None:
    t0 = datetime(2026, 8, 21, tzinfo=UTC)
    s = _s(t0)
    assert not s.is_expired(t0 + timedelta(minutes=29))
    assert s.is_expired(t0 + timedelta(minutes=31))


def test_touch_extends() -> None:
    t0 = datetime(2026, 8, 21, tzinfo=UTC)
    s = _s(t0)
    s.touch(t0 + timedelta(minutes=20))
    assert not s.is_expired(t0 + timedelta(minutes=45))
