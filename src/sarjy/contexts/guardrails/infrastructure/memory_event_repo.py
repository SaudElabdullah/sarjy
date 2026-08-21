"""In-memory `GuardEventRepo` — the no-Postgres counterpart to `PgGuardEventRepo`.

Used by the `Container` whenever there is no database to write to (local
runs with `connect_db=False`, `use_in_memory_repos()` in tests), so the
guards keep their real behaviour instead of being swapped for no-ops: a
guard that silently stops recording is a guard nobody can verify.

`rows` is kept so a test can assert on what was recorded; it is bounded by
`max_rows` (oldest dropped first) so a long-running process using this repo
cannot grow without limit.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from sarjy.shared.ids import MessageId, UserId


class MemGuardEvents:
    def __init__(self, max_rows: int = 1000) -> None:
        self._rows: deque[dict[str, Any]] = deque(maxlen=max_rows)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    async def record(
        self,
        *,
        user_id: UserId | None,
        message_id: MessageId | None,
        layer: int,
        kind: str,
        action: str,
        severity: int,
        detail: dict[str, Any],
    ) -> None:
        self._rows.append(
            {
                "user_id": user_id,
                "message_id": message_id,
                "layer": layer,
                "kind": kind,
                "action": action,
                "severity": severity,
                "detail": detail,
            }
        )
