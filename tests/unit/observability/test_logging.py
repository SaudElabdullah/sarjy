from __future__ import annotations

import logging

import httpx
import pytest
import respx

from sarjy.observability.logging import configure_logging

_SECRET = "s3cr3t-owm-key"  # noqa: S105 — a fake key, the point is that it must not be logged


@respx.mock
async def test_httpx_request_logging_never_reaches_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """httpx logs the full request line, query string and all, at INFO. The OWM
    fallback puts its API key there, so at INFO the key would be logged on every
    single weather request."""
    configure_logging("INFO")
    respx.get("https://api.openweathermap.org/data/3.0/onecall").mock(
        return_value=httpx.Response(200, json={})
    )
    with caplog.at_level(logging.INFO):
        async with httpx.AsyncClient() as client:
            await client.get(
                "https://api.openweathermap.org/data/3.0/onecall",
                params={"lat": 1.0, "lon": 2.0, "appid": _SECRET},
            )

    assert not any(_SECRET in r.getMessage() for r in caplog.records)
    assert not [r for r in caplog.records if r.name.startswith(("httpx", "httpcore"))]


def test_configure_logging_pins_the_http_loggers_to_warning() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
