"""One read for everything a turn needs to start (PRD L-7).

Before this, `RunTurn` opened a turn with three sequential awaits — history,
facts, active run — each a full round trip to Postgres before the first token
could even be requested from the model. They are independent reads of the same
user's rows, so the round trips were the cost, not the queries. `load_turn_
context` (the RPC) answers all of them at once, and this module is the seam
between that one JSON blob and the domain objects `RunTurn` works in.

`context_from_rpc` is deliberately a free function rather than a method on the
Postgres adapter: it is pure mapping, it is where every "what does this key
mean" decision lives, and it is the part worth testing without a database.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from sarjy.contexts.conversation.application.ports import ActiveRunPort, ActiveRunSnapshot, Fact
from sarjy.contexts.conversation.domain.message import Message
from sarjy.contexts.conversation.domain.session import Session
from sarjy.shared.ids import MessageId, SessionId, UserId, new_id

# The Big Five's own trait vocabulary, keyed by the codes the instrument scores
# under. It is here rather than behind the assessment port because the RPC's
# `last_results` row carries scores and bands but no instrument — and because
# O/C/E/A/N are the published names of the model itself, not this instrument's
# private labels. A second instrument scoring different traits would be the
# moment to move this behind `ActiveRunPort` alongside `snapshot_from_row`.
TRAIT_NAMES = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}
# The top of the response scale. Declared as a speakable figure because every
# line about a score says "out of five" — the same reasoning (and the same
# number) as `handle_turn.results_numbers`.
SCALE_TOP = 5.0


# What makes the user's OWN turn a question about their results, and so worth
# spending prompt budget (and arming a decimal-tight grounding check) on. The
# block is not free: it is text in every prompt and a check on every sentence,
# so a finished run does not get to colour a conversation about the weather —
# it is injected when, and only when, the user brings it up.
_RESULTS_QUESTION = re.compile(
    r"\b(test|scores?|scored|trait|traits|personality|big five|results?"
    r"|openness|conscientiousness|extraversion|agreeableness|neuroticism)\b",
    re.I,
)


def asks_about_results(text: str) -> bool:
    """Is the user's turn about their assessment results?"""
    return _RESULTS_QUESTION.search(text) is not None


# The five trait names, as a pattern, built from the same table the results
# block is written from — so a sixth trait could never be speakable in one place
# and unrecognised in the other.
_TRAIT_MENTION = re.compile(
    r"\b(" + "|".join(n.lower() for n in TRAIT_NAMES.values()) + r")\b", re.I
)


def mentions_trait(text: str) -> bool:
    """Does this text name one of the Big Five traits?

    Used on the LAST assistant turn, not on the user's: a follow-up that carries
    no vocabulary of its own ("and the other four?") is about the results
    because the sentence before it was. See `RunTurn._results_in_play`.
    """
    return _TRAIT_MENTION.search(text) is not None


@dataclass(frozen=True, slots=True)
class LastResults:
    """The user's finished Big Five run, in the two forms a turn needs it.

    `prompt_block` grounds the model: the scores are stated for it, so a
    follow-up ("how did I score on openness?") is answered from the row rather
    than from a transcript that has already scrolled out of the history window.
    `grounding_numbers` grounds the *output guard*: it is the set of figures the
    reply is entitled to speak, so a fabricated "you scored 4.9" is cut on the
    way out (P-9/P-11). Both or neither — a block without the numbers would
    have the guard cutting Sarjy's own correct answer.
    """

    prompt_block: str
    grounding_numbers: tuple[float, ...]


@dataclass(slots=True)
class TurnContext:
    facts: list[Fact]
    history: list[Message]
    workflow: ActiveRunSnapshot | None
    profile: dict[str, Any] = field(default_factory=dict)
    # Present only when a run has completed AND none is open — see the RPC.
    last_results: LastResults | None = None
    # The session row for the id the caller asked to resume, unvalidated:
    # ownership and expiry are the caller's to decide (a foreign or stale id
    # starts a NEW session rather than erroring), so the loader just reports
    # what is stored.
    session: Session | None = None
    # Whether the loader looked the session up at all. Without this, `session is
    # None` is ambiguous — "no such row" and "this loader doesn't do sessions"
    # want opposite things from `RunTurn`: start fresh, or fall back to
    # `SessionRepo.get`. The RPC always answers; the in-memory loader never does.
    session_loaded: bool = False


class ContextLoaderPort(Protocol):
    async def load(
        self, user_id: UserId, session_id: SessionId, history_limit: int
    ) -> TurnContext: ...


class LastResultsPort(Protocol):
    async def latest_results(self, user_id: UserId) -> dict[str, Any] | None:
        """The latest completed run as `{results, narrative, completed_at}`.

        The same shape the RPC's `last_results` key carries, so both loaders
        feed `last_results_from_row` and there is one interpretation of it.
        Callers only ask when no run is open.
        """
        ...


def last_results_from_row(row: dict[str, Any] | None) -> LastResults | None:
    """Build the grounded results block from a completed run's row.

    Returns `None` for anything that cannot ground an answer — a missing row, a
    run whose `results` never landed, or a report where no trait scored (fewer
    than three answers each). A block naming no numbers would tell the model
    results exist while giving it nothing to quote, which is worse than silence.
    """
    if not row:
        return None
    results = row.get("results") or {}
    bands = results.get("bands") or {}
    parts: list[str] = []
    numbers: list[float] = [SCALE_TOP]
    for code, name in TRAIT_NAMES.items():
        raw = results.get(code)
        if not isinstance(raw, int | float) or isinstance(raw, bool):
            continue
        score = float(raw)
        band = bands.get(code)
        parts.append(f"{name} {score:.1f} ({band})" if band else f"{name} {score:.1f}")
        # The integer a score rounds to is speakable too ("2.8, so about three"),
        # exactly as `results_numbers` declares it for the results reply itself.
        numbers.extend((score, float(round(score))))
    if not parts:
        return None
    block = (
        "The user's latest Big Five results: "
        + ", ".join(parts)
        + ", each out of five. Say only these numbers — do not re-score, "
        "estimate, or invent any other figure."
    )
    return LastResults(prompt_block=block, grounding_numbers=tuple(dict.fromkeys(numbers)))


def session_from_rpc(row: dict[str, Any] | None, session_id: SessionId) -> Session | None:
    """Rebuild the `sessions` row the RPC returned. `None` when there is none."""
    if not row:
        return None
    return Session(
        id=session_id,
        user_id=UserId(uuid.UUID(row["user_id"])),
        started_at=_ts(row["started_at"]),
        last_active_at=_ts(row["last_active_at"]),
        summary=row.get("summary"),
    )


def _ts(v: Any) -> datetime:
    """A timestamptz out of jsonb, which renders it as an ISO string.

    Postgres writes `+00:00` for UTC, which `fromisoformat` reads natively; the
    result is always tz-aware, because a naive one compared against `Clock.now()`
    raises — and the comparison it feeds is the session expiry check.
    """
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(v))


def context_from_rpc(
    raw: dict[str, Any], user_id: UserId, session_id: SessionId, runs: ActiveRunPort
) -> TurnContext:
    facts = [Fact(m["k"], m["v"], m["kind"]) for m in raw.get("memories") or []]
    # The RPC returns no message ids or timestamps: nothing downstream of here
    # reads them (history is prompt material, not something we write back), and
    # shipping them would be paying for columns to throw away. Fresh ids and a
    # single `now` keep `Message` honest about being a value object here.
    now = datetime.now(UTC)
    history = [
        Message(
            id=new_id(MessageId),
            session_id=session_id,
            user_id=user_id,
            role=h["role"],
            content=h["content"],
            created_at=now,
            client_turn_id=h.get("client_turn_id"),
            guard_decision=h.get("guard_decision"),
        )
        for h in raw.get("history") or []
    ]
    wf = runs.snapshot_from_row(raw["workflow"]) if raw.get("workflow") else None
    return TurnContext(
        facts=facts,
        history=history,
        workflow=wf,
        profile=raw.get("profile") or {},
        last_results=last_results_from_row(raw.get("last_results")),
        session=session_from_rpc(raw.get("session"), session_id),
        session_loaded=True,
    )
