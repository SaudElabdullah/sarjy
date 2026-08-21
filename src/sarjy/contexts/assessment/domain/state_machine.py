from __future__ import annotations

from enum import StrEnum


class Status(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    PAUSED = "paused"
    SCORING = "scoring"
    COMPLETE = "complete"
    ABANDONED = "abandoned"


# (current status, action) -> next status. Anything absent is illegal.
TRANSITIONS: dict[tuple[Status, str], Status] = {
    (Status.PROPOSED, "confirm"): Status.ACTIVE,
    (Status.PROPOSED, "quit"): Status.ABANDONED,
    (Status.ACTIVE, "answer"): Status.ACTIVE,
    (Status.ACTIVE, "back"): Status.ACTIVE,
    (Status.ACTIVE, "pause"): Status.PAUSED,
    (Status.ACTIVE, "quit"): Status.ABANDONED,
    (Status.ACTIVE, "score"): Status.SCORING,
    (Status.PAUSED, "resume"): Status.ACTIVE,
    (Status.PAUSED, "quit"): Status.ABANDONED,
    # A run should never rest in SCORING (the turn handler scores, narrates and
    # completes in one save), but one left there by an older build or a
    # half-landed write has to be recoverable rather than terminal. "back"
    # reopens it — the way out when an answer row turns out to be missing and
    # the item has to be asked again — and "quit" abandons it, so a user who
    # asks to stop is not told the run cannot be stopped.
    (Status.SCORING, "back"): Status.ACTIVE,
    (Status.SCORING, "quit"): Status.ABANDONED,
    (Status.SCORING, "finish"): Status.COMPLETE,
}


def next_status(current: Status, action: str) -> Status | None:
    return TRANSITIONS.get((current, action))
