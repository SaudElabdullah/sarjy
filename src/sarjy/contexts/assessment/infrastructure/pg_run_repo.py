"""Postgres-backed `RunRepo`.

Two rules shape everything here. First, a run row is written whole on every
save: the aggregate is the authority on its own state, and a partial update is
how a paused run comes back thinking it is active. Second, every read filters
on `user_id` in SQL — not in Python — so one user's run can never surface in
another user's turn even if a run id leaks. `save` carries the same predicate
into its upsert: an insert that collides with an id owned by somebody else
updates nothing rather than overwriting their run.

`workflow_answers` is a separate table keyed `(run_id, item_no)`, upserted, so
going back and re-answering item seven overwrites item seven rather than
appending a second row that would quietly double its weight in scoring.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import RunId, UserId

_COLS = (
    "id, user_id, definition_id, definition_version, status, current_item, skips_used, "
    "results, narrative, started_at, updated_at, completed_at, pending_confirmation, resume_hint"
)
# `scoring` is open: a run should never be persisted in it (the turn handler
# scores, narrates and completes in one save), but a row left behind by an
# older build or a save that half-landed has to be findable, or the user is
# holding twenty answers nothing will ever finish (I1/I2).
OPEN_STATUSES = ("proposed", "active", "paused", "scoring")


def _j(v: Any) -> Any:
    """asyncpg hands back jsonb as text unless a codec is registered."""
    return json.loads(v) if isinstance(v, str) else v


def _dump(v: Any) -> str | None:
    return json.dumps(v) if v is not None else None


def _row(r: asyncpg.Record) -> WorkflowRun:
    return WorkflowRun(
        id=RunId(r["id"]),
        user_id=UserId(r["user_id"]),
        definition_id=r["definition_id"],
        definition_version=r["definition_version"],
        status=Status(r["status"]),
        current_item=r["current_item"],
        skips_used=r["skips_used"],
        results=_j(r["results"]),
        narrative=r["narrative"],
        started_at=r["started_at"],
        updated_at=r["updated_at"],
        completed_at=r["completed_at"],
        pending_confirmation=_j(r["pending_confirmation"]),
        resume_hint=r["resume_hint"],
    )


class PgRunRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def get_open(self, user_id: UserId) -> WorkflowRun | None:
        r = await self.db.fetchrow(
            f"select {_COLS} from workflow_runs "  # noqa: S608 — _COLS is a module constant
            "where user_id = $1 and status = any($2::workflow_status[]) "
            "order by updated_at desc limit 1",
            user_id,
            list(OPEN_STATUSES),
        )
        return _row(r) if r is not None else None

    async def get(self, run_id: RunId, user_id: UserId) -> WorkflowRun | None:
        r = await self.db.fetchrow(
            f"select {_COLS} from workflow_runs where id = $1 and user_id = $2",  # noqa: S608
            run_id,
            user_id,
        )
        return _row(r) if r is not None else None

    async def latest_complete(self, user_id: UserId) -> WorkflowRun | None:
        r = await self.db.fetchrow(
            f"select {_COLS} from workflow_runs "  # noqa: S608
            "where user_id = $1 and status = 'complete' "
            "order by completed_at desc nulls last limit 1",
            user_id,
        )
        return _row(r) if r is not None else None

    async def save(self, run: WorkflowRun) -> None:
        await self.db.execute(
            """insert into workflow_runs
                 (id, user_id, definition_id, definition_version, status, current_item,
                  skips_used, results, narrative, started_at, updated_at, completed_at,
                  pending_confirmation, resume_hint)
               values ($1, $2, $3, $4, $5::workflow_status, $6, $7, $8::jsonb, $9, $10, $11, $12,
                       $13::jsonb, $14)
               on conflict (id) do update set
                 status = excluded.status,
                 current_item = excluded.current_item,
                 skips_used = excluded.skips_used,
                 results = excluded.results,
                 narrative = excluded.narrative,
                 updated_at = excluded.updated_at,
                 completed_at = excluded.completed_at,
                 pending_confirmation = excluded.pending_confirmation,
                 resume_hint = excluded.resume_hint
               where workflow_runs.user_id = excluded.user_id""",
            run.id,
            run.user_id,
            run.definition_id,
            run.definition_version,
            run.status.value,
            run.current_item,
            run.skips_used,
            _dump(run.results),
            run.narrative,
            run.started_at,
            run.updated_at,
            run.completed_at,
            _dump(run.pending_confirmation),
            run.resume_hint,
        )

    async def save_answer(
        self, run_id: RunId, item_no: int, raw_text: str, value: int | None, confidence: float
    ) -> None:
        await self.db.execute(
            """insert into workflow_answers (run_id, item_no, raw_text, value, confidence)
               values ($1, $2, $3, $4, $5)
               on conflict (run_id, item_no) do update set
                 raw_text = excluded.raw_text,
                 value = excluded.value,
                 confidence = excluded.confidence,
                 answered_at = now()""",
            run_id,
            item_no,
            raw_text,
            value,
            confidence,
        )

    async def answers(self, run_id: RunId) -> dict[int, int | None]:
        rows = await self.db.fetch(
            "select item_no, value from workflow_answers where run_id = $1 order by item_no",
            run_id,
        )
        return {r["item_no"]: r["value"] for r in rows}
