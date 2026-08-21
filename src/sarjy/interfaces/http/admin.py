"""`GET /admin/latency` — internal latency/guard/funnel dashboard (PRD §13)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from sarjy.interfaces.http.auth import CurrentUserDep

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/latency")
async def latency(user: CurrentUserDep, request: Request) -> dict[str, list[dict[str, Any]]]:
    settings = request.app.state.settings
    if user.user_id not in settings.admin_user_id_set:
        raise HTTPException(status_code=403, detail="admin only")
    repo = request.app.state.container.admin_repo
    return {
        "latency_daily": await repo.latency_daily(),
        "latency_by_browser": await repo.latency_by_browser(),
        "guard_daily": await repo.guard_daily(),
        "ocean_funnel": await repo.ocean_funnel(),
    }
