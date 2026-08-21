import asyncio
import os

import asyncpg
import pytest

DB = os.environ["DATABASE_URL_DIRECT"]


async def _delete_x_test_users() -> None:
    # Deletes rows in auth.users created by the integration tests (emails like
    # "<uuid>@x.test"). FK cascades (on delete cascade) remove the dependent
    # public.memories, public.sessions, public.workflow_runs and public.profiles
    # rows for those users.
    conn = await asyncpg.connect(DB)
    try:
        await conn.execute("delete from auth.users where email like '%@x.test'")
    finally:
        await conn.close()


@pytest.fixture(scope="session", autouse=True)
def cleanup_x_test_users() -> None:
    """Session-scoped, autouse cleanup of integration-test users.

    Plain sync fixture (not an async pytest-asyncio fixture) so it is
    unaffected by asyncio_mode = "auto" function-scoped event loops; it
    drives its own event loop via asyncio.run() at setup and teardown.
    """
    # Clean up any leftovers from earlier/crashed runs before this session's
    # tests create new @x.test users.
    asyncio.run(_delete_x_test_users())
    yield
    # Clean up everything this session created.
    asyncio.run(_delete_x_test_users())
