import uuid

import pytest
from pydantic import ValidationError

from sarjy.config import Settings


def test_settings_defaults(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    s = Settings()
    assert s.gemini_chat_model == "gemini-2.5-flash"
    assert s.history_limit == 12
    assert s.guard_mode == "enforce"
    assert s.weather_provider == "open-meteo"


def test_cors_origins_parsed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com")
    assert Settings().cors_origin_list == ["http://a.com", "http://b.com"]


def test_settings_repr_hides_secrets(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "top-secret-service-role-value")
    s = Settings()
    assert "top-secret-service-role-value" not in repr(s)


def test_cors_origins_rejects_wildcard(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(ValidationError):
        Settings()


def test_admin_user_ids_parsed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    admin = "b3e1d6a0-2f2a-4d3a-9b1a-4e6c9a2f1a11"
    monkeypatch.setenv("ADMIN_USER_IDS", f" {admin} , ")
    s = Settings()
    assert s.admin_user_id_set == {uuid.UUID(admin)}


def test_admin_user_ids_rejects_bad_uuid(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("ADMIN_USER_IDS", "not-a-uuid")
    with pytest.raises(ValidationError):
        Settings()
