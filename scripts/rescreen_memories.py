#!/usr/bin/env python3
"""Re-screen already-stored memories against the Layer-2 rule engine.

Phase 8 Task 6b added guardrail screening to the memory write path
(`RememberFact` via the `remember` tool, `EditFact` via `PATCH /memory/{id}` —
see `src/sarjy/contexts/memory/application/screening.py`). Anything stored
*before* that landed never went through it, so a value the write path would
refuse today can still be sitting in `memories` — and every recall re-injects
it into the prompt. This script is the backfill: it reads every row, screens
the key and the value the same way the write path does, and reports (or, with
`--delete`, removes) the ones that would now be refused.

Usage:
    uv run python scripts/rescreen_memories.py              # dry run (default)
    uv run python scripts/rescreen_memories.py --delete     # remove refused rows

Reads `DATABASE_URL_DIRECT` (the service-role, non-pooled connection string —
`memories` is RLS'd per user, and this is a cross-user sweep). Nothing else is
read from the environment and nothing is written to it.

**Refused values are never printed in full.** The whole point of a refusal is
that the text is unsafe to keep re-injecting; dumping it into a terminal (and
from there into CI logs, a scrollback buffer, a pasted ticket) would undo that.
Output is `user_id`, the key, the rule that fired, and the first
`_PREVIEW_CHARS` characters of the value, ellipsised — enough to recognise a
false positive, not enough to be a copy of the fact.

`--delete` removes the `memories` row *and* its `memories_history` rows, for
the same reason `ForgetFact` does (the history table stores `old_value`/
`new_value` verbatim, so deleting only the memory leaves the refused text one
table over). This is a hard delete, not the soft delete a user-initiated
forget does: a row that should never have been stored should not linger as a
`deleted_at`-stamped tombstone with its value intact.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

import asyncpg

# Run as a script (`uv run python scripts/rescreen_memories.py`) the repo's
# `src/` is not necessarily importable; mirror smoke.py's sys.path handling.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from sarjy.contexts.guardrails.domain.rules import DEFAULT_RULES, RuleEngine  # noqa: E402
from sarjy.contexts.guardrails.infrastructure.value_screen import (  # noqa: E402
    RuleEngineValueScreen,
)
from sarjy.contexts.memory.application.ports import ValueScreenPort  # noqa: E402

_PREVIEW_CHARS = 40


@dataclass(frozen=True, slots=True)
class StoredMemory:
    """One `memories` row, as much of it as screening and reporting need."""

    id: str
    user_id: str
    key: str | None
    value: str


@dataclass(frozen=True, slots=True)
class Refusal:
    memory: StoredMemory
    field: str  # "key" or "value" — which of the two the screen refused
    rule: str | None


class MemoriesRepo(Protocol):
    """The two operations this script needs, so it can be driven by a double."""

    async def all_memories(self) -> list[StoredMemory]: ...
    async def delete(self, memory_id: str) -> None: ...


def preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    """First `limit` characters, ellipsised, newlines flattened.

    Never the whole value — see the module docstring. Newlines are collapsed so
    a multi-line value cannot break the one-refusal-per-line output format (or
    smuggle terminal escapes past a reader skimming the report).
    """
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def screen_memories(memories: Sequence[StoredMemory], screen: ValueScreenPort) -> list[Refusal]:
    """Screen each memory's key and value SEPARATELY, exactly as the write path does.

    Never concatenated: a key that normalises to "note"/"remember"/"save"/
    "store" paired with a `"that my X is '<payload>'"` value reconstructs the
    rule engine's memory-set frame at screening time and smuggles an
    uncertain-severity payload past the screen — the Phase 8 Task 6b fix-round-1
    bypass. Screening the two fields independently here means this sweep and
    the write path (`memory/application/screening.py`) refuse the same set of
    rows; a backfill that was more permissive than the live path would report
    a clean database that is not.
    """
    refusals: list[Refusal] = []
    for m in memories:
        for field, text in (("key", m.key or ""), ("value", m.value)):
            if not text:
                continue
            verdict = screen.screen(text)
            if not verdict.allowed:
                refusals.append(Refusal(m, field, verdict.reason))
                break  # one refusal per row is enough to condemn it
    return refusals


def format_refusal(r: Refusal) -> str:
    return (
        f"{r.memory.user_id}  key={r.memory.key!r}  refused_on={r.field}  "
        f"rule={r.rule}  value={preview(r.memory.value)!r}"
    )


def default_screen() -> ValueScreenPort:
    """The same adapter `Container.rebuild_memory` wires behind `ValueScreenPort`.

    No `GuardEventRepo`/`BackgroundTasks`: a backfill sweep is not a user write,
    and writing thousands of `guardrail_events` rows attributed to no message
    would be noise in the audit trail rather than signal.
    """
    return RuleEngineValueScreen(RuleEngine(DEFAULT_RULES))


class PgMemoriesRepo:
    """`MemoriesRepo` over asyncpg, service-role connection (no RLS in the way)."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def _connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(self._dsn)

    async def all_memories(self) -> list[StoredMemory]:
        conn = await self._connect()
        try:
            # Soft-deleted rows included on purpose: `deleted_at` stops a row
            # being recalled, it does not remove the stored text, and a refused
            # value is not something to keep just because it is tombstoned.
            rows = await conn.fetch("select id,user_id,key,value from public.memories order by id")
        finally:
            await conn.close()
        return [StoredMemory(str(r["id"]), str(r["user_id"]), r["key"], r["value"]) for r in rows]

    async def delete(self, memory_id: str) -> None:
        conn = await self._connect()
        try:
            async with conn.transaction():
                await conn.execute(
                    "delete from public.memories_history where memory_id = $1::uuid", memory_id
                )
                await conn.execute("delete from public.memories where id = $1::uuid", memory_id)
        finally:
            await conn.close()


async def rescreen(
    repo: MemoriesRepo,
    screen: ValueScreenPort,
    *,
    delete: bool = False,
    out: TextIO | None = None,
) -> list[Refusal]:
    """Screen everything in `repo`, print a report, optionally delete the refusals."""
    stream: TextIO = out if out is not None else sys.stdout
    memories = await repo.all_memories()
    refusals = screen_memories(memories, screen)
    print(f"screened {len(memories)} memories, {len(refusals)} refused", file=stream)
    for r in refusals:
        print(format_refusal(r), file=stream)
    if not refusals:
        return refusals
    if delete:
        for r in refusals:
            await repo.delete(r.memory.id)
        print(f"deleted {len(refusals)} memories and their history rows", file=stream)
    else:
        print("dry run — re-run with --delete to remove them", file=stream)
    return refusals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the refused memories (and their history rows). Off by default.",
    )
    args = parser.parse_args(argv)

    dsn = os.environ.get("DATABASE_URL_DIRECT")
    if not dsn:
        print("DATABASE_URL_DIRECT is not set", file=sys.stderr)
        return 2

    asyncio.run(rescreen(PgMemoriesRepo(dsn), default_screen(), delete=args.delete))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
