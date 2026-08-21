"""`GET /workflow/latest` — the assessment status/results the web UI polls
after every turn's `DoneEvent`.

Read-only: scoring already happens inside `HandleAssessmentTurn` mid-turn
(`WorkflowRun.begin_scoring` / `finish_scoring`). This endpoint just reports
whatever the repos already hold, for the user's own run — the two reads it
uses (`get_open`, `latest_complete`) are both scoped by `user_id` in SQL, so a
run can never surface for anyone but the user who owns it.

A run stranded in `scoring` (I1/I2) is reported as it is: `status="scoring"`
with `results` still null. Inventing a `complete` for it, or 404ing over it,
would both be lies — the run exists, the answers are safe, and the next turn
the user takes finishes it. The web client already treats any status other
than `active`/`complete` as "show nothing".
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from sarjy.contexts.assessment.application.handle_turn import DISCLAIMER
from sarjy.interfaces.http.auth import CurrentUserDep

router = APIRouter(prefix="/workflow", tags=["workflow"])


class WorkflowLatest(BaseModel):
    status: str
    run_id: str
    definition_id: str
    current_item: int
    total_items: int
    results: dict[str, Any] | None
    narrative: str | None
    bands: dict[str, Any] | None
    disclaimer: str


@router.get("/latest", response_model=WorkflowLatest)
async def latest(user: CurrentUserDep, request: Request) -> WorkflowLatest:
    c = request.app.state.container
    run = await c.run_repo.get_open(user.user_id) or await c.run_repo.latest_complete(user.user_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no_run")
    ins = await c.instrument_repo.get(run.definition_id)
    total = ins.total_items
    bands = run.results.get("bands") if run.results else None
    return WorkflowLatest(
        status=run.status.value,
        run_id=str(run.id),
        definition_id=run.definition_id,
        # After the last answer `current_item` is total + 1; the client shows a
        # position, not a cursor, so it is clamped the same way `workflow_dict`
        # (the mid-turn payload) clamps it.
        current_item=min(run.current_item, total),
        total_items=total,
        results=run.results,
        narrative=run.narrative,
        bands=bands,
        disclaimer=DISCLAIMER,
    )
