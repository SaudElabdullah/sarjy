from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, Request

from sarjy.shared.ids import UserId


@dataclass(frozen=True, slots=True)
class CurrentUser:
    user_id: UserId
    is_anonymous: bool


def decode_supabase_jwt(token: str, secret: str) -> CurrentUser:
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="authenticated",
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail="invalid or expired token") from e
    try:
        uid = UserId(uuid.UUID(claims["sub"]))
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=401, detail="invalid or expired token") from e
    return CurrentUser(user_id=uid, is_anonymous=bool(claims.get("is_anonymous", False)))


def current_user(
    request: Request, authorization: Annotated[str | None, Header()] = None
) -> CurrentUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    secret: str = request.app.state.settings.supabase_jwt_secret.get_secret_value()
    return decode_supabase_jwt(authorization[7:], secret)


CurrentUserDep = Annotated[CurrentUser, Depends(current_user)]
