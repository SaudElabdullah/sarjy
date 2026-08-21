"""Offer the assessment — or, if one is already open, offer to pick it up.

A second run for a user who already has one is the failure this use case
exists to prevent: two open runs means `get_open` starts choosing between
them, and the user's answers land in whichever one it picked.
"""

from __future__ import annotations

from sarjy.contexts.assessment.application.handle_turn import (
    DISCLAIMER,
    nav_numbers,
    scale_numbers,
    workflow_dict,
)
from sarjy.contexts.assessment.application.ports import InstrumentRepo, RunRepo
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.contexts.conversation.application.ports import AssessmentReply
from sarjy.shared.clock import Clock
from sarjy.shared.ids import RunId, UserId, new_id
from sarjy.shared.text import to_speech


def intro(ins: Instrument) -> str:
    """The offer, with the instrument's own numbers in it.

    It used to hardcode "twenty" and "about five minutes". The first is a lie
    the moment a second instrument is seeded (PRD P-12: adding a workflow is
    meant to be a definition row), and it is a lie the output guard cannot
    catch, because the reply declares `total` and the sentence says twenty.
    The second was a promise nobody measured, so it is now the honest vaguer
    one. `to_speech` supplies the spoken word for the count, the same
    conversion the sentence would get on its way to the speaker anyway.
    """
    count = to_speech(str(ins.total_items))
    top = to_speech(str(len(ins.scale_labels)))
    return (
        f"Sure — this is the Big Five, {count} quick statements about you, a few minutes. "
        f"For each one, tell me how accurate it is on a one-to-{top} scale."
    )


class StartRun:
    def __init__(self, runs: RunRepo, instruments: InstrumentRepo, clock: Clock) -> None:
        self.runs = runs
        self.instruments = instruments
        self.clock = clock

    async def execute(
        self, user_id: UserId, definition_id: str = "ocean_mini_ipip"
    ) -> AssessmentReply:
        ins = await self.instruments.get(definition_id)
        total = ins.total_items
        existing = await self.runs.get_open(user_id)
        if existing is not None:
            nav = nav_numbers(existing, total)
            item = min(existing.current_item, total)
            if existing.status is Status.PAUSED:
                return AssessmentReply(
                    [
                        f"You have a paused test at item {item} of {total}.",
                        "Say continue to pick it up, or quit to start over.",
                    ],
                    workflow_dict(existing, total),
                    nav,
                )
            if existing.status in (Status.ACTIVE, Status.SCORING):
                # SCORING here means a stranded run (I1/I2); the next turn the
                # user takes finishes it, so "we're already on item N" is both
                # true and the right nudge.
                return AssessmentReply(
                    [f"We're already on item {item}.", "Ready to continue?"],
                    workflow_dict(existing, total),
                    nav,
                )
            return AssessmentReply(
                ["I'd already offered the test — ready to start?"], workflow_dict(existing, total)
            )
        run = WorkflowRun.propose(new_id(RunId), user_id, ins.id, ins.version, self.clock.now())
        await self.runs.save(run)
        return AssessmentReply(
            [intro(ins), DISCLAIMER, "Ready?"],
            workflow_dict(run, total),
            # The intro speaks the length of the test and both ends of the scale.
            (*scale_numbers(ins), float(total)),
        )
