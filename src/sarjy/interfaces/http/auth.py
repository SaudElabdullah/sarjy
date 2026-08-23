from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, cast

import httpx
import jwt
from fastapi import Depends, Header, HTTPException, Request

from sarjy.shared.ids import UserId

_INVALID = HTTPException(status_code=401, detail="invalid or expired token")
_DECODE_OPTS = cast("Any", {"require": ["exp", "sub"]})


@dataclass(frozen=True, slots=True)
class CurrentUser:
    user_id: UserId
    is_anonymous: bool


def _to_user(claims: dict[str, Any]) -> CurrentUser:
    try:
        uid = UserId(uuid.UUID(claims["sub"]))
    except (KeyError, ValueError) as e:
        raise _INVALID from e
    return CurrentUser(user_id=uid, is_anonymous=bool(claims.get("is_anonymous", False)))


def decode_supabase_jwt(token: str, secret: str) -> CurrentUser:
    """HS256 path: legacy shared-secret projects and the local Supabase stack."""
    try:
        claims = jwt.decode(
            token, secret, algorithms=["HS256"], audience="authenticated", options=_DECODE_OPTS
        )
    except jwt.PyJWTError as e:
        raise _INVALID from e
    return _to_user(claims)


def decode_with_key(token: str, key: jwt.PyJWK) -> CurrentUser:
    """Asymmetric path: tokens signed with one of the project's JWKS keys."""
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=["ES256", "RS256"],
            audience="authenticated",
            options=_DECODE_OPTS,
        )
    except jwt.PyJWTError as e:
        raise _INVALID from e
    return _to_user(claims)


class JwksCache:
    """Caches `<supabase_url>/auth/v1/.well-known/jwks.json`.

    Supabase projects created since late 2025 sign access tokens with an
    asymmetric key (ES256) instead of the shared HS256 secret; the key set is
    public and rotates rarely. A miss on `kid` triggers one refetch, throttled
    so a flood of bad tokens cannot turn into a flood of JWKS requests.
    """

    MIN_REFETCH_S = 60.0

    def __init__(self, supabase_url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._url = supabase_url.rstrip("/") + "/auth/v1/.well-known/jwks.json"
        self._client = client
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at = 0.0
        self._lock = asyncio.Lock()

    async def key(self, kid: str) -> jwt.PyJWK | None:
        if kid in self._keys:
            return self._keys[kid]
        async with self._lock:
            if kid in self._keys:
                return self._keys[kid]
            if time.monotonic() - self._fetched_at < self.MIN_REFETCH_S:
                return None
            await self._refresh()
        return self._keys.get(kid)

    async def _refresh(self) -> None:
        self._fetched_at = time.monotonic()
        try:
            client = self._client or httpx.AsyncClient(timeout=3.0)
            try:
                r = await client.get(self._url)
                r.raise_for_status()
                body = r.json()
            finally:
                if client is not self._client:
                    await client.aclose()
            keys = {k.key_id: k for k in jwt.PyJWKSet.from_dict(body).keys if k.key_id}
        except (httpx.HTTPError, ValueError, jwt.PyJWTError):
            return  # keep whatever we had; the caller answers 401 for this token
        self._keys = keys


async def verify_token(token: str, secret: str, jwks: JwksCache) -> CurrentUser:
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as e:
        raise _INVALID from e
    alg = header.get("alg")
    if alg == "HS256":
        return decode_supabase_jwt(token, secret)
    kid = header.get("kid")
    if alg in ("ES256", "RS256") and isinstance(kid, str):
        key = await jwks.key(kid)
        if key is not None:
            return decode_with_key(token, key)
    raise _INVALID


def _jwks_for(request: Request) -> JwksCache:
    cache: JwksCache | None = getattr(request.app.state, "jwks", None)
    if cache is None:
        cache = JwksCache(request.app.state.settings.supabase_url)
        request.app.state.jwks = cache
    return cache


async def current_user(
    request: Request, authorization: Annotated[str | None, Header()] = None
) -> CurrentUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    secret: str = request.app.state.settings.supabase_jwt_secret.get_secret_value()
    return await verify_token(authorization[7:], secret, _jwks_for(request))


CurrentUserDep = Annotated[CurrentUser, Depends(current_user)]
