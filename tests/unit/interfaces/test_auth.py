import time
import uuid

import jwt
import pytest
from fastapi import HTTPException

from sarjy.interfaces.http.auth import decode_supabase_jwt

SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"  # noqa: S105


def _token(**over):  # type: ignore[no-untyped-def]
    claims = {
        "sub": str(uuid.uuid4()),
        "aud": "authenticated",
        "role": "authenticated",
        "exp": int(time.time()) + 3600,
        "is_anonymous": False,
    }
    claims.update(over)
    return jwt.encode(claims, SECRET, algorithm="HS256")


def test_decode_valid_token() -> None:
    u = decode_supabase_jwt(_token(), SECRET)
    assert not u.is_anonymous


def test_decode_anonymous_flag() -> None:
    assert decode_supabase_jwt(_token(is_anonymous=True), SECRET).is_anonymous


def test_rejects_expired() -> None:
    with pytest.raises(HTTPException) as e:
        decode_supabase_jwt(_token(exp=int(time.time()) - 10), SECRET)
    assert e.value.status_code == 401


def test_rejects_wrong_audience() -> None:
    with pytest.raises(HTTPException):
        decode_supabase_jwt(_token(aud="anon"), SECRET)


def test_rejects_missing_exp() -> None:
    # Create token without exp claim
    claims = {
        "sub": str(uuid.uuid4()),
        "aud": "authenticated",
        "role": "authenticated",
        "is_anonymous": False,
    }
    token = jwt.encode(claims, SECRET, algorithm="HS256")
    with pytest.raises(HTTPException) as e:
        decode_supabase_jwt(token, SECRET)
    assert e.value.status_code == 401
