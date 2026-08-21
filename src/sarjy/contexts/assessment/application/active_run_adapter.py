"""`ActiveRunAdapter` — the conversation context's `ActiveRunPort` implementation.

`RunTurn` depends only on `ActiveRunPort` (in the conversation context); this
adapter is the assessment context's side of that seam. It builds the PRD §10
block 8 prompt text from a `WorkflowRun`'s status and hands workflow turns off
to `HandleAssessmentTurn` unchanged.

`get_open` also returns a run stranded in `scoring` (I1/I2), so every status it
can hand back needs a block here — `RunTurn` asks for the snapshot before it
asks the handler to take the turn, and a missing entry would take the whole
turn down rather than degrade it.
"""

from __future__ import annotations

import uuid
from typing import Any

from sarjy.contexts.assessment.application.handle_turn import HandleAssessmentTurn
from sarjy.contexts.assessment.application.ports import InstrumentRepo, RunRepo
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.conversation.application.ports import ActiveRunSnapshot, AssessmentReply
from sarjy.shared.ids import RunId, UserId

_BLOCK: dict[Status, str] = {
    Status.PROPOSED: (
        "Active: Big Five personality test has been offered and is awaiting the user's yes/no. "
        "If they ask something else, answer briefly, then ask if they'd like to start."
    ),
    Status.ACTIVE: (
        "Active: Big Five test, item {item} of {total}. The user's answers are handled by the "
        "test engine. If they ask something else, answer it briefly and do not invent test "
        "items or scores."
    ),
    Status.SCORING: (
        "Active: Big Five test, all {total} items are answered and the test engine is "
        "producing the results itself. Do not state, guess or summarise any score."
    ),
    Status.PAUSED: (
        "Paused: Big Five test at item {item} of {total}. If the user wants to continue, call "
        "workflow_control with action resume. If the user wants to stop, confirm once, then "
        "call workflow_control with action quit."
    ),
}


# Used only when `snapshot_from_row` runs before anything has fetched the
# instrument, so there is nothing to derive a real count from. It is the item
# count of the one definition v1 seeds (`supabase/mini_ipip.json`); a second
# instrument with a different length would be wrong here for exactly one turn,
# until `InstrumentRepo.get` fills the cache. The number is never spoken — it
# only sizes the prompt block and clamps the item counter.
FALLBACK_TOTAL_ITEMS = 20


class ActiveRunAdapter:
    def __init__(
        self, runs: RunRepo, instruments: InstrumentRepo, handle_turn: HandleAssessmentTurn
    ) -> None:
        self.runs = runs
        self.instruments = instruments
        self.handle = handle_turn

    async def active_run(self, user_id: UserId) -> ActiveRunSnapshot | None:
        run = await self.runs.get_open(user_id)
        if run is None:
            return None
        ins = await self.instruments.get(run.definition_id)
        item = min(run.current_item, ins.total_items)
        block = _BLOCK[run.status].format(item=item, total=ins.total_items)
        if run.resume_hint and run.status is Status.ACTIVE:
            block += (
                f' After answering, end with exactly: "Ready to continue? We were on item {item}."'
            )
        return ActiveRunSnapshot(
            run_id=run.id,
            definition_id=run.definition_id,
            status=run.status.value,
            current_item=item,
            total_items=ins.total_items,
            prompt_block=block,
        )

    async def handle_turn(self, user_id: UserId, text: str) -> AssessmentReply | None:
        return await self.handle.execute(user_id, text)

    async def latest_results(self, user_id: UserId) -> dict[str, Any] | None:
        """The finished run a follow-up question can be grounded against.

        Deliberately the same three keys the RPC's `last_results` returns, so
        the in-memory and Postgres loaders hand `last_results_from_row` the same
        shape. No open-run check here: the caller only asks once it knows there
        is none (the RPC decides that in SQL, for the same reason).
        """
        run = await self.runs.latest_complete(user_id)
        if run is None or not run.results:
            return None
        return {
            "results": run.results,
            "narrative": run.narrative,
            "completed_at": run.completed_at,
        }

    def snapshot_from_row(self, row: dict[str, Any]) -> ActiveRunSnapshot | None:
        """Build a snapshot from the `workflow` JSON returned by the
        load_turn_context RPC (Phase 7), without an extra DB round-trip.

        The instrument's item count comes from the row itself when the RPC
        carries one, else from `self.instruments.cached` — a sync accessor
        backed by the in-process cache that a real `InstrumentRepo` (e.g.
        `PgInstrumentRepo`) fills the first time `get` is called for that
        definition — and only then from `FALLBACK_TOTAL_ITEMS`.
        """
        if not row or row.get("status") not in ("proposed", "active", "paused", "scoring"):
            return None
        status = Status(row["status"])
        ins = self.instruments.cached(row["definition_id"])
        total = int(row.get("total_items") or 0) or (
            ins.total_items if ins else FALLBACK_TOTAL_ITEMS
        )
        item = min(int(row.get("current_item", 1)), total)
        block = _BLOCK[status].format(item=item, total=total)
        if row.get("resume_hint") and status is Status.ACTIVE:
            block += (
                f' After answering, end with exactly: "Ready to continue? We were on item {item}."'
            )
        return ActiveRunSnapshot(
            run_id=RunId(uuid.UUID(row["id"])),
            definition_id=row["definition_id"],
            status=status.value,
            current_item=item,
            total_items=total,
            prompt_block=block,
        )
