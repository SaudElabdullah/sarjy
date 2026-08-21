"""Application-layer ports for the assessment context.

The use cases below depend only on these Protocols; infrastructure (Postgres
repos, the Gemini interpreter/narrator) imports this module, never the other
way round.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.scoring import ScoreReport
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.shared.ids import RunId, UserId

Control = Literal["repeat", "skip", "back", "explain", "pause", "quit", "off_topic"]


@dataclass(frozen=True, slots=True)
class Interpretation:
    """What the user's utterance meant for the item currently being asked.

    Exactly one of `value` / `control` is normally set. Both `None` means the
    utterance was an answer attempt nobody could read — the caller re-asks with
    the scale hint rather than guessing.
    """

    value: int | None
    confidence: float
    control: Control | None


class AnswerInterpreterPort(Protocol):
    async def interpret(
        self, item_text: str, scale_labels: list[str], user_text: str
    ) -> Interpretation: ...


class NarratorPort(Protocol):
    async def narrate(self, report: ScoreReport) -> str: ...


class RunRepo(Protocol):
    async def get_open(self, user_id: UserId) -> WorkflowRun | None:
        """The newest run that is still open: proposed, active, scoring or paused.

        `scoring` counts as open even though nothing should persist a run in it
        — see `pg_run_repo.OPEN_STATUSES`.
        """
        ...

    async def get(self, run_id: RunId, user_id: UserId) -> WorkflowRun | None:
        """One run by id — and only if `user_id` owns it.

        The owner is part of the lookup, not a check the caller is trusted to
        remember: a run id that leaks (a log line, a stale client) must not be
        a way to read someone else's answers.
        """
        ...

    async def latest_complete(self, user_id: UserId) -> WorkflowRun | None: ...

    async def save(self, run: WorkflowRun) -> None: ...

    async def save_answer(
        self, run_id: RunId, item_no: int, raw_text: str, value: int | None, confidence: float
    ) -> None: ...

    async def answers(self, run_id: RunId) -> dict[int, int | None]: ...


class InstrumentRepo(Protocol):
    async def get(self, id: str) -> Instrument: ...

    def cached(self, id: str) -> Instrument | None:
        """The instrument last fetched by `get`, if any — sync, in-process.

        Lets `ActiveRunAdapter.snapshot_from_row` (Phase 7's single-RPC context
        loader) read the instrument's `total_items` without a second DB
        round-trip. Populated as a side effect of `get`; empty until something
        has actually called it.
        """
        ...
