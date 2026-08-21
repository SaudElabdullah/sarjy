from datetime import UTC, datetime, timedelta

from sarjy.shared.clock import FakeClock, SystemClock


def test_system_clock_is_utc() -> None:
    assert SystemClock().now().tzinfo == UTC


def test_fake_clock_advances() -> None:
    c = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    c.advance(timedelta(seconds=5))
    assert c.now() == datetime(2026, 8, 21, 0, 0, 5, tzinfo=UTC)
