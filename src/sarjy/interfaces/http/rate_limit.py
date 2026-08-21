"""Per-endpoint rate-limit dependencies.

One factory, three namespaces. The limiter counts into buckets keyed by
`(user_id, "<namespace>:<window>", window_start)`, so each endpoint spends its
own allowance rather than eating into a shared one — see
`PgRateLimiter.hit`. The limits themselves are identical per namespace; what is
being separated is the budgets, not their size.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request

from sarjy.interfaces.http.auth import CurrentUserDep


def _limiter_for(window_ns: str) -> Callable[[CurrentUserDep, Request], Awaitable[None]]:
    """A FastAPI dependency that charges one hit to `window_ns`.

    A factory because a dependency's signature is its injection contract:
    FastAPI resolves every parameter, so the namespace cannot be one of them.
    Closing over it is what lets three endpoints share one implementation.
    """

    async def dependency(user: CurrentUserDep, request: Request) -> None:
        rl = request.app.state.container.rate_limiter
        if rl is None:
            return
        res = await rl.hit(user.user_id, user.is_anonymous, window_ns=window_ns)
        if not res.allowed:
            raise HTTPException(
                status_code=429,
                detail="rate_limited",
                headers={"Retry-After": str(res.retry_after_s)},
            )

    return dependency


# `/chat` — the endpoint the configured limits were sized against, and the name
# every existing caller of `rate_limited` still means.
rate_limited = _limiter_for("chat")
# `/telemetry` — a voice turn posts its marks, so sharing `chat`'s budget would
# make every turn cost two hits of an allowance sized for one.
rate_limited_telemetry = _limiter_for("tele")
# `/chat/confirm` — cheap, but it writes rows and inserts into an in-process
# store, so it is no longer something to leave unmetered.
rate_limited_confirm = _limiter_for("confirm")
