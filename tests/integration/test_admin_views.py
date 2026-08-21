"""`v_latency_daily` / `v_latency_by_browser` / `v_guard_daily` / `v_ocean_funnel` —
views created by `supabase/migrations/20260821000600_latency_views.sql` (PRD §13).
"""

import os

import asyncpg
import pytest

pytestmark = pytest.mark.integration

DB = os.environ["DATABASE_URL_DIRECT"]


async def _conn() -> asyncpg.Connection:
    return await asyncpg.connect(DB)


async def test_latency_daily_view_runs() -> None:
    # Not asserting on row count: `telemetry_turns` rows outlive the user that created them
    # (its FK is `on delete set null`, not cascade — see 20260821000450_ops_tables_lockdown.sql),
    # so other integration tests in the same run can leave rows behind. The view running at
    # all is what this test is for.
    c = await _conn()
    try:
        await c.fetch("select * from public.v_latency_daily")
    finally:
        await c.close()


async def test_latency_by_browser_view_runs() -> None:
    c = await _conn()
    try:
        await c.fetch("select * from public.v_latency_by_browser")
    finally:
        await c.close()


async def test_guard_daily_view_runs() -> None:
    c = await _conn()
    try:
        await c.fetch("select * from public.v_guard_daily")
    finally:
        await c.close()


async def test_ocean_funnel_view_runs() -> None:
    c = await _conn()
    try:
        await c.fetch("select * from public.v_ocean_funnel")
    finally:
        await c.close()


# The columns each view is expected to expose, in order. The four tests above
# only prove the SQL parses; a dashboard reads these views by NAME, so a rename
# or a dropped column breaks a chart while leaving `select *` perfectly happy.
# Asserted as an exact ordered tuple, because that is what the view's own
# `select` list is and what the reader downstream is written against.
_VIEW_COLUMNS = {
    "v_latency_daily": (
        "day",
        "mode",
        "turns",
        "ttfa_p50",
        "ttfa_p95",
        "first_byte_p50",
        "gemini_first_token_p50",
        "turn_p50",
    ),
    "v_latency_by_browser": ("browser", "turns", "ttfa_p50", "ttfa_p95"),
    "v_guard_daily": ("day", "layer", "kind", "action", "n"),
    "v_ocean_funnel": ("day", "proposed", "started", "completed"),
}

_VIEWS = tuple(_VIEW_COLUMNS)


@pytest.mark.parametrize("view", _VIEWS)
async def test_views_expose_the_columns_their_readers_name(view: str) -> None:
    c = await _conn()
    try:
        rows = await c.fetch(
            """select column_name from information_schema.columns
               where table_schema = 'public' and table_name = $1
               order by ordinal_position""",
            view,
        )
    finally:
        await c.close()
    assert tuple(r["column_name"] for r in rows) == _VIEW_COLUMNS[view]


@pytest.mark.parametrize("role", ["anon", "authenticated"])
async def test_views_are_revoked_from_anon_and_authenticated(role: str) -> None:
    c = await _conn()
    try:
        for view in _VIEWS:
            async with c.transaction():
                await c.execute(f"set role {role}")
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    # `view` is one of the four fixed names in `_VIEWS` above, not
                    # user input.
                    await c.fetch(f"select * from public.{view}")  # noqa: S608
            await c.execute("reset role")
    finally:
        await c.close()
