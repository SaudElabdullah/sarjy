import os

import pytest

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("SUPABASE_URL", "http://localhost:54321")
os.environ.setdefault("SUPABASE_ANON_KEY", "anon")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "service")
os.environ.setdefault(
    "SUPABASE_JWT_SECRET", "super-secret-jwt-token-with-at-least-32-characters-long"
)
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
os.environ.setdefault(
    "DATABASE_URL_DIRECT", "postgresql://postgres:postgres@localhost:54322/postgres"
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
