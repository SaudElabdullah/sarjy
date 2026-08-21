"""`POST /internal/audit/run` — Layer 7 async audit sampling trigger (PRD §Layer 7).

Called by pg_cron every 10 minutes via pg_net (the `audit` job, `supabase/
migrations/20260821000800_audit_sample.sql`), and safe to call manually for a
backfill or an ops check. Not a user-facing endpoint — no `CurrentUserDep`,
no Supabase JWT — so it is deliberately NOT registered on any router that
carries that dependency; auth here is a single shared secret compared with
`hmac.compare_digest` (never a plain `==`, which short-circuits on the first
mismatched byte and turns "does this token match" into a timing side-channel
an attacker can probe one byte at a time).

`hmac.compare_digest` takes `str` operands as ASCII-only — it raises
`TypeError` on anything else, which for a `str` argument coming straight off
an HTTP header (Starlette decodes header bytes permissively; nothing upstream
of this handler validates they're ASCII) turns a byte >127 in `X-Internal-
Token` into an unhandled 500 rather than the 401 a wrong token gets. `_eq`
below moves the comparison into `bytes` first — `str.encode` never raises for
any `str`, `surrogateescape` included only so that stays true even for the
one input shape (lone surrogate codepoints) plain UTF-8 encoding rejects —
so every input this endpoint can receive compares cleanly, matching or not.

An unset `Settings.internal_token` is a 503, not a 404/401: this endpoint is
meant to exist in every deployment that runs the audit cron, so a request
arriving with nothing configured to check it against is a misconfiguration
worth being loud about (503, the kind of response an uptime check pages on),
not something that should look like "endpoint doesn't exist" (404) or "wrong
caller" (401) — both of which a monitor would shrug off as expected traffic
noise from an unconfigured environment.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Header, HTTPException, Request

router = APIRouter(prefix="/internal", tags=["internal"])


def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(
        a.encode("utf-8", "surrogateescape"), b.encode("utf-8", "surrogateescape")
    )


@router.post("/audit/run")
async def run_audit(
    request: Request, x_internal_token: str | None = Header(default=None)
) -> dict[str, int]:
    settings = request.app.state.settings
    token = settings.internal_token
    if token is None:
        raise HTTPException(status_code=503, detail="internal audit endpoint not configured")
    if x_internal_token is None or not _eq(x_internal_token, token.get_secret_value()):
        raise HTTPException(status_code=401, detail="invalid internal token")

    worker = request.app.state.container.audit_worker
    if worker is None:
        # connect_db=False (no database, so no audit_queue to read/write) —
        # not a token problem, hence a separate 503 rather than folding this
        # into the check above.
        raise HTTPException(status_code=503, detail="audit worker not configured")

    result = await worker.run_once()
    return {"processed": result.processed, "flagged": result.flagged}
