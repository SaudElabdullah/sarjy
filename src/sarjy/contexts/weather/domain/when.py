from __future__ import annotations

from datetime import date, timedelta

MAX_DAYS_AHEAD = 7

_TODAY_TOKENS = ("now", "today", "")


def parse_when(when: str, today: date, slack_days: int = 0) -> date | None:
    """Validate a `when` token — input validation only, never day resolution.

    The day a request actually lands on is resolved in the *location's* calendar
    by `index_for_when`, against the provider's own day list. This function only
    rejects tokens that are malformed or plainly out of range, judged against
    whatever calendar the caller has to hand (server UTC). `slack_days` widens
    the window at both ends so a location that is a day ahead of or behind the
    server is never refused a date its own calendar still considers in range.
    """
    w = (when or "now").strip().lower()
    if w in _TODAY_TOKENS:
        return today
    if w == "tomorrow":
        return today + timedelta(days=1)
    try:
        d = date.fromisoformat(w)
    except ValueError:
        return None
    if (today - d).days > slack_days or (d - today).days > MAX_DAYS_AHEAD + slack_days:
        return None
    return d


def index_for_when(when: str, days: list[date]) -> int | None:
    """Index of the requested day in a provider's own local-calendar day list.

    `days[0]` is *the location's* today whatever the server's UTC date happens
    to be, so relative tokens are offsets from index 0 and an ISO date is looked
    up in the list itself. Returns None when the day is not covered.
    """
    if not days:
        return None
    w = (when or "now").strip().lower()
    if w in _TODAY_TOKENS:
        return 0
    if w == "tomorrow":
        return 1 if len(days) > 1 else None
    try:
        d = date.fromisoformat(w)
    except ValueError:
        return None
    return days.index(d) if d in days else None


def day_label(when: str) -> str:
    """How the requested day is said out loud.

    Derived from the request token rather than from the resolved date: the
    caller asked for "tomorrow" in their own calendar, and that is the word
    that will make sense back to them whatever date it landed on.
    """
    w = (when or "now").strip().lower()
    if w in _TODAY_TOKENS:
        return "right now"
    if w == "tomorrow":
        return "tomorrow"
    try:
        return f"on {date.fromisoformat(w).strftime('%A')}"
    except ValueError:
        return "right now"
