"""Explicit run control — what the `workflow_control` tool calls.

Resuming a paused run is deliberately not something the turn handler infers
from free text: `HandleAssessmentTurn` declines every turn while a run is
paused, so the only way back in is through here.

A run stranded in SCORING is the one status this use case cannot answer on its
own: "resume" would ask for an item past the end of the instrument, and until
the state machine grew a `quit` edge for it, "quit" raised. Both are now
handled — resume hands the run to the turn handler's recovery, which is the
only code that knows how to finish or reopen it, and quit says plainly that
the scoring will not be finished.
"""

from __future__ import annotations

from typing import Literal

from sarjy.contexts.assessment.application.handle_turn import (
    HandleAssessmentTurn,
    item_sentence,
    nav_numbers,
    workflow_dict,
)
from sarjy.contexts.assessment.application.ports import InstrumentRepo, RunRepo
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.conversation.application.ports import AssessmentReply
from sarjy.shared.clock import Clock
from sarjy.shared.ids import UserId

NO_RUN = {"status": "none", "item": 0, "total": 0, "run_id": ""}


class ControlRun:
    def __init__(
        self,
        runs: RunRepo,
        instruments: InstrumentRepo,
        clock: Clock,
        handle: HandleAssessmentTurn,
    ) -> None:
        self.runs = runs
        self.instruments = instruments
        self.clock = clock
        self.handle = handle

    async def execute(self, user_id: UserId, action: Literal["resume", "quit"]) -> AssessmentReply:
        run = await self.runs.get_open(user_id)
        if run is None:
            return AssessmentReply(
                ["There's no personality test in progress.", "Want to start it?"], dict(NO_RUN)
            )
        ins = await self.instruments.get(run.definition_id)
        total = ins.total_items
        now = self.clock.now()
        if action == "quit":
            # Legal from SCORING too, so a stranded run can still be stopped.
            stopping_mid_score = run.status is Status.SCORING
            run.quit(now)
            await self.runs.save(run)
            lead = (
                "Okay, I won't finish scoring the test."
                if stopping_mid_score
                else "Okay, I've stopped the test."
            )
            return AssessmentReply(
                [lead, "We can start fresh any time."],
                workflow_dict(run, total),
            )
        if run.status is Status.SCORING or run.is_finished_answering(total):
            # Every item is answered; there is nothing to resume, only results
            # to produce (or one missing answer to ask for). The turn handler
            # owns that recovery, so this defers to it rather than growing a
            # second copy that could drift.
            recovered = await self.handle.execute(user_id, "resume")
            if recovered is not None:
                return recovered
        if run.status is Status.PAUSED:
            run.resume(now)
            await self.runs.save(run)
            return AssessmentReply(
                [f"Picking up at item {run.current_item}.", item_sentence(ins, run.current_item)],
                workflow_dict(run, total),
                nav_numbers(run, total),
            )
        if run.status is Status.PROPOSED:
            return AssessmentReply(
                ["Say yes when you're ready to start."], workflow_dict(run, total)
            )
        return AssessmentReply(
            [item_sentence(ins, run.current_item)],
            workflow_dict(run, total),
            nav_numbers(run, total),
        )
