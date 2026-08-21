"""`DELETE /account`, `GET /account/export` — account deletion/export (PRD §11)."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from sarjy.interfaces.http.auth import CurrentUserDep
from sarjy.observability.logging import get_logger

router = APIRouter(tags=["account"])
log = get_logger(__name__)


@router.delete("/account", status_code=204)
async def delete_account(user: CurrentUserDep, request: Request) -> Response:
    s = request.app.state.settings
    # Deleting the auth user cascades to every table via FK (PRD §8). Use the Admin API
    # so auth internals (sessions, identities, etc.) stay consistent. The service-role
    # key goes out over TLS to Supabase's own Admin API and nowhere else — never logged,
    # not even on failure (log the exception type/status only, never headers or bodies).
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.delete(
                f"{s.supabase_url}/auth/v1/admin/users/{user.user_id}",
                headers={
                    "apikey": s.supabase_service_role_key.get_secret_value(),
                    "Authorization": f"Bearer {s.supabase_service_role_key.get_secret_value()}",
                },
            )
    except httpx.HTTPError as e:
        log.warning(
            "account_delete_admin_api_unreachable",
            user_id=str(user.user_id),
            error_type=type(e).__name__,
        )
        raise HTTPException(status_code=502, detail="deletion failed") from e
    if r.status_code == 404:
        # The auth user is already gone (a retried request, or deleted out-of-band) —
        # deletion is idempotent from the caller's point of view.
        return Response(status_code=204)
    if r.status_code not in (200, 204):
        log.warning(
            "account_delete_admin_api_error",
            user_id=str(user.user_id),
            status_code=r.status_code,
        )
        raise HTTPException(status_code=502, detail="deletion failed")
    return Response(status_code=204)


@router.get("/account/export")
async def export_account(user: CurrentUserDep) -> Response:
    return Response(status_code=501, content="export is planned for a later release")
