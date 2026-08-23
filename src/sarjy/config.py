from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    gemini_api_key: SecretStr
    gemini_chat_model: str = "gemini-3.6-flash"
    gemini_guard_model: str = "gemini-3.5-flash-lite"
    gemini_embed_model: str = "gemini-embedding-001"

    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: SecretStr
    supabase_jwt_secret: SecretStr
    database_url: SecretStr
    database_url_direct: SecretStr

    # Weather
    weather_provider: Literal["open-meteo", "owm", "mock"] = "open-meteo"
    owm_api_key: SecretStr | None = None

    # Behaviour
    guard_mode: Literal["enforce", "shadow"] = "enforce"
    history_limit: int = 12
    speculative_enabled: bool = False
    audio_mode: Literal["webspeech", "gemini-tts"] = "webspeech"

    # App
    app_env: Literal["dev", "test", "staging", "prod"] = "dev"
    app_base_url: str = "http://localhost:8000"
    # Absolute URL of this app (e.g. https://sarjy-prod.fly.dev), emitted as
    # `apiBase` by GET /config.js. Empty by default (relative fetches), which is
    # correct whenever the page and the API are served from the same origin. Set
    # this when the static client is published elsewhere (Supabase Storage — see
    # scripts/upload_static.py) so the client's fetches reach this app absolutely.
    public_api_base: str = ""
    cors_origins: str = "http://localhost:8000"
    log_level: str = "INFO"
    sentry_dsn: SecretStr | None = None
    turnstile_site_key: str | None = None
    # Shared secret for `POST /internal/audit/run` (Phase 8 Task 4, PRD Layer 7):
    # compared with `hmac.compare_digest`, never logged. Unset means the endpoint
    # is not configured for this deployment — see `interfaces/http/internal.py`.
    internal_token: SecretStr | None = None

    # Admin (PRD §13 internal latency/guard/funnel dashboard)
    admin_user_ids: str = ""

    # Limits (PRD Layer 0/1)
    max_utterance_chars: int = 600
    rate_limit_per_10min: int = 60
    rate_limit_per_day: int = 500

    # Gemini generation (PRD C-3)
    chat_temperature: float = Field(0.6, ge=0, le=2)
    chat_max_output_tokens: int = 300
    gemini_first_token_timeout_s: float = 8.0
    gemini_total_timeout_s: float = 25.0

    @field_validator("cors_origins", mode="after")
    @classmethod
    def _no_wildcard_cors(cls, v: str) -> str:
        if any(o.strip() == "*" for o in v.split(",")):
            raise ValueError("cors_origins must not contain '*' (Bearer tokens, not cookies)")
        return v

    @field_validator("admin_user_ids", mode="after")
    @classmethod
    def _admin_user_ids_are_uuids(cls, v: str) -> str:
        for u in v.split(","):
            u = u.strip()
            if not u:
                continue
            try:
                uuid.UUID(u)
            except ValueError as e:
                raise ValueError(f"admin_user_ids: {u!r} is not a valid UUID") from e
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def admin_user_id_set(self) -> set[uuid.UUID]:
        return {uuid.UUID(u.strip()) for u in self.admin_user_ids.split(",") if u.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
