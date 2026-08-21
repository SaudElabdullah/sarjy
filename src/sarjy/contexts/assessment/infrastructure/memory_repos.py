"""In-memory assessment repositories — used by unit tests and by the test
container. Every read hands back a deep copy, so a caller mutating what it got
cannot change stored state without going through `save`, the same way a
Postgres-backed repo behaves.
"""

from __future__ import annotations

import copy

from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.shared.ids import RunId, UserId

# Mirrors `pg_run_repo.OPEN_STATUSES` — `scoring` included, so a run stranded
# mid-scoring is found and finished rather than orphaned (I1/I2).
OPEN = {Status.PROPOSED, Status.ACTIVE, Status.PAUSED, Status.SCORING}


class MemRunRepo:
    def __init__(self) -> None:
        self.runs: dict[RunId, WorkflowRun] = {}
        self.answers_by_run: dict[RunId, dict[int, tuple[str, int | None, float]]] = {}

    async def get_open(self, user_id: UserId) -> WorkflowRun | None:
        c = [r for r in self.runs.values() if r.user_id == user_id and r.status in OPEN]
        return copy.deepcopy(max(c, key=lambda r: r.updated_at)) if c else None

    async def get(self, run_id: RunId, user_id: UserId) -> WorkflowRun | None:
        r = self.runs.get(run_id)
        # Ownership is part of the lookup, as it is in SQL — see `PgRunRepo.get`.
        return copy.deepcopy(r) if r is not None and r.user_id == user_id else None

    async def latest_complete(self, user_id: UserId) -> WorkflowRun | None:
        c = [r for r in self.runs.values() if r.user_id == user_id and r.status is Status.COMPLETE]
        return copy.deepcopy(max(c, key=lambda r: r.completed_at or r.updated_at)) if c else None

    async def save(self, run: WorkflowRun) -> None:
        self.runs[run.id] = copy.deepcopy(run)

    async def save_answer(
        self, run_id: RunId, item_no: int, raw_text: str, value: int | None, confidence: float
    ) -> None:
        self.answers_by_run.setdefault(run_id, {})[item_no] = (raw_text, value, confidence)

    async def answers(self, run_id: RunId) -> dict[int, int | None]:
        return {k: v[1] for k, v in self.answers_by_run.get(run_id, {}).items()}


class MemInstrumentRepo:
    def __init__(self, items: dict[str, Instrument]) -> None:
        self.items = items
        self._cache: dict[str, Instrument] = {}

    async def get(self, id: str) -> Instrument:
        ins = self.items[id]
        self._cache[id] = ins
        return ins

    def cached(self, id: str) -> Instrument | None:
        return self._cache.get(id)
