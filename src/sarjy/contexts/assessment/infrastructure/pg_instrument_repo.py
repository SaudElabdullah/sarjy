"""Postgres-backed `InstrumentRepo`.

A definition is versioned, immutable content that every single turn of an open
run needs, so it is cached in-process behind a short TTL: without it a
twenty-item questionnaire costs twenty identical `select`s. The TTL (rather
than a permanent cache) is what lets a corrected item text reach a long-lived
process without a redeploy.

`cached()` is the sync half of the same store — see `InstrumentRepo.cached`.
"""

from __future__ import annotations

import json
import time

from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.errors import NotFound

TTL_S = 600.0


class PgInstrumentRepo:
    def __init__(self, db: Database, ttl_s: float = TTL_S) -> None:
        self.db = db
        self._ttl = ttl_s
        self._cache: dict[str, tuple[float, Instrument]] = {}

    async def get(self, id: str) -> Instrument:
        hit = self.cached(id)
        if hit is not None:
            return hit
        row = await self.db.fetchrow(
            "select definition from workflow_definitions where id = $1 and active", id
        )
        if row is None:
            raise NotFound(f"workflow_definition {id}")
        raw = row["definition"]
        ins = Instrument.from_definition(json.loads(raw) if isinstance(raw, str) else raw)
        self._cache[id] = (time.monotonic(), ins)
        return ins

    def cached(self, id: str) -> Instrument | None:
        hit = self._cache.get(id)
        if hit is None or time.monotonic() - hit[0] >= self._ttl:
            return None
        return hit[1]
