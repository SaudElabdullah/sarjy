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


# ---------- asymmetric (ES256 / JWKS) path ----------
import asyncio  # noqa: E402
import json  # noqa: E402

import httpx  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from sarjy.interfaces.http.auth import JwksCache, verify_token  # noqa: E402

_PRIV = ec.generate_private_key(ec.SECP256R1())
_PEM = _PRIV.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
)
_KID = "kid-1"
_JWKS = {
    "keys": [
        {
            **jwt.algorithms.ECAlgorithm.to_jwk(_PRIV.public_key(), as_dict=True),
            "kid": _KID,
            "alg": "ES256",
            "use": "sig",
        }
    ]
}


def _es_token(kid: str = _KID, **over):  # type: ignore[no-untyped-def]
    claims = {"sub": str(uuid.uuid4()), "aud": "authenticated", "exp": int(time.time()) + 3600}
    claims.update(over)
    return jwt.encode(claims, _PEM, algorithm="ES256", headers={"kid": kid})


def _cache(jwks: dict = _JWKS, status: int = 200) -> tuple[JwksCache, list[int]]:  # type: ignore[type-arg]
    calls: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, content=json.dumps(jwks))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return JwksCache("https://proj.supabase.co", client=client), calls


def test_es256_token_verified_via_jwks() -> None:
    cache, calls = _cache()
    u = asyncio.run(verify_token(_es_token(is_anonymous=True), SECRET, cache))
    assert u.is_anonymous and len(calls) == 1
    asyncio.run(verify_token(_es_token(), SECRET, cache))
    assert len(calls) == 1  # cached


def test_hs256_still_accepted_without_touching_jwks() -> None:
    cache, calls = _cache()
    asyncio.run(verify_token(_token(), SECRET, cache))
    assert calls == []


def test_unknown_kid_is_401_and_refetch_throttled() -> None:
    cache, calls = _cache()
    for _ in range(3):
        with pytest.raises(HTTPException):
            asyncio.run(verify_token(_es_token(kid="nope"), SECRET, cache))
    assert len(calls) == 1


def test_es256_wrong_key_rejected() -> None:
    other = ec.generate_private_key(ec.SECP256R1())
    jwks = {
        "keys": [
            {
                **jwt.algorithms.ECAlgorithm.to_jwk(other.public_key(), as_dict=True),
                "kid": _KID,
                "alg": "ES256",
            }
        ]
    }
    cache, _ = _cache(jwks)
    with pytest.raises(HTTPException):
        asyncio.run(verify_token(_es_token(), SECRET, cache))


def test_jwks_endpoint_down_is_401_not_500() -> None:
    cache, _ = _cache(status=503)
    with pytest.raises(HTTPException) as e:
        asyncio.run(verify_token(_es_token(), SECRET, cache))
    assert e.value.status_code == 401


def test_alg_none_rejected() -> None:
    tok = jwt.encode(
        {"sub": str(uuid.uuid4()), "aud": "authenticated", "exp": int(time.time()) + 60},
        key=None,
        algorithm="none",
    )
    cache, _ = _cache()
    with pytest.raises(HTTPException):
        asyncio.run(verify_token(tok, SECRET, cache))
