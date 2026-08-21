from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from sarjy.config import Settings, get_settings
from sarjy.container import Container
from sarjy.interfaces.http import (
    account,
    admin,
    chat,
    health,
    internal,
    memory,
    telemetry,
    web,
    workflow,
)
from sarjy.interfaces.http.security import SecurityHeadersMiddleware, UnhandledErrorMiddleware
from sarjy.observability.logging import configure_logging


async def _validation_exception_handler(_request: Request, exc: Exception) -> Response:
    # FastAPI's default handler renders errors through Starlette's JSONResponse, which
    # calls json.dumps(..., allow_nan=False) — standards-compliant, but it means a client
    # payload that failed validation *because* it was NaN/Infinity (e.g. a `Marks` field
    # with `allow_inf_nan=False`) crashes the error response itself with an unhandled
    # ValueError (a 500), since pydantic's error detail echoes the raw invalid input.
    # allow_nan=True here just re-enables the standard library's non-standard-but-widely-
    # accepted NaN/Infinity literals for this one response, so the error path itself can
    # never 500.
    # `exc` is typed `Exception` (rather than `RequestValidationError`) only to match
    # Starlette's `add_exception_handler` signature — it is always a `RequestValidationError`
    # in practice, since that is the only class this handler is registered for below.
    assert isinstance(exc, RequestValidationError)
    body = json.dumps({"detail": jsonable_encoder(exc.errors())}, allow_nan=True)
    return Response(content=body, status_code=422, media_type="application/json")


async def _unhandled_exception_handler(_request: Request, _exc: Exception) -> Response:
    return Response(
        content=json.dumps({"detail": "internal error"}),
        status_code=500,
        media_type="application/json",
    )


def create_app(settings: Settings | None = None, connect_db: bool = True) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    container = Container.build(settings, connect_db=connect_db)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await container.startup()
        yield
        await container.shutdown()

    app = FastAPI(title="Sarjy", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.container = container
    # Middleware order matters here. `add_middleware` prepends, so the LAST one
    # added is the outermost: the stack below runs SecurityHeaders → CORS →
    # UnhandledError → routes. `UnhandledErrorMiddleware` has to be innermost so
    # the JSON 500 it produces is an ordinary response the two outer layers then
    # decorate — an exception that escaped it would reach Starlette's
    # `ServerErrorMiddleware` outside the whole user stack, and go out with no
    # CSP and no HSTS (I3). See the class docstring in security.py for why an
    # `add_exception_handler(Exception, ...)` registration cannot do this job.
    app.add_middleware(UnhandledErrorMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.add_middleware(
        SecurityHeadersMiddleware,
        supabase_origin=settings.supabase_url,
        turnstile_site_key=settings.turnstile_site_key,
    )
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    # Belt and braces for the narrow case `UnhandledErrorMiddleware` cannot
    # reach: an exception raised by the CORS or header middleware themselves,
    # above it in the stack. Starlette hands the `Exception` key to
    # `ServerErrorMiddleware`, so this response gets no security headers — but
    # it is at least the same JSON shape a client sees for every other 500,
    # rather than Starlette's plain-text default.
    app.add_exception_handler(Exception, _unhandled_exception_handler)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(memory.router)
    app.include_router(workflow.router)
    app.include_router(telemetry.router)
    app.include_router(admin.router)
    app.include_router(account.router)
    app.include_router(internal.router)
    app.include_router(web.router)
    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "interfaces" / "web" / "static")),
        name="static",
    )
    return app
