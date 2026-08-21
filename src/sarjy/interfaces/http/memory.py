"""`GET /memory`, `PATCH /memory/{id}`, `DELETE /memory/{id}` — memory REST endpoints (PRD §9.2)."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from sarjy.interfaces.http.auth import CurrentUserDep
from sarjy.shared.ids import MemoryId

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryOut(BaseModel):
    id: uuid.UUID
    key: str | None
    value: str
    kind: str
    updated_at: datetime


class MemoryPatch(BaseModel):
    value: str = Field(min_length=1, max_length=200)


@router.get("", response_model=list[MemoryOut])
async def list_memories(user: CurrentUserDep, request: Request) -> list[MemoryOut]:
    repo = request.app.state.container.memory_repo
    mems = await repo.list_live(user.user_id, limit=200)
    return [
        MemoryOut(id=m.id, key=m.key, value=m.value, kind=m.kind, updated_at=m.updated_at)
        for m in mems
    ]


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(memory_id: uuid.UUID, user: CurrentUserDep, request: Request) -> Response:
    forget = request.app.state.container.forget_fact
    out = await forget.by_id(user.user_id, MemoryId(memory_id))
    if out.status == "not_found":
        raise HTTPException(status_code=404, detail="memory not found")
    return Response(status_code=204)


@router.patch("/{memory_id}", response_model=MemoryOut)
async def patch_memory(
    memory_id: uuid.UUID, body: MemoryPatch, user: CurrentUserDep, request: Request
) -> MemoryOut:
    # Routed through `EditFact` (not a direct repo write) since fix round 1,
    # Critical 1: PATCH used to write straight through `upsert_with_history`
    # after only the PII filter, so it bypassed the same guardrail
    # rule-engine screening the `remember` tool enforces. `EditFact` applies
    # the identical PII filter + `ValueScreenPort` screening `RememberFact`
    # does.
    c = request.app.state.container
    out = await c.edit_fact(user.user_id, MemoryId(memory_id), body.value)
    if out.status == "not_found":
        raise HTTPException(status_code=404, detail="memory not found")
    if out.status == "rejected":
        raise HTTPException(status_code=422, detail=f"I won't store that — {out.reason}.")
    m = out.memory
    assert m is not None
    return MemoryOut(id=m.id, key=m.key, value=m.value, kind=m.kind, updated_at=m.updated_at)
