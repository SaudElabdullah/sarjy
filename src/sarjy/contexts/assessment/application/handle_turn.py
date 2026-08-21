"""The assessment turn handler — the engine that owns a user's utterance while
a run is open.

`RunTurn` calls `execute` before it builds any prompt: a reply means the
workflow answered the turn itself (the sentences are spoken verbatim, guarded
but never re-generated), and `None` means "not mine" — the turn falls through
to ordinary chat.

Scoring is the tail of the last turn, not a state the run rests in: the report
and the narrative are produced first and `begin_scoring`/`finish_scoring` are
applied together, in one save. A run persisted in `scoring` and then lost to a
crash is a run holding twenty answers that nothing will ever finish (I1/I2);
`execute` still knows how to pick one up, but nothing here creates one.

Every reply carries `grounding_numbers` (PRD G-6): the output guard's numeric
check is armed whenever that tuple is non-empty, so a reply must declare the
figures it is entitled to say. An item prompt says "Seven: ..." out of twenty,
so it declares its own counters; the results reply declares the scores. Getting
this wrong is not a cosmetic bug — an undeclared number gets the sentence cut.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from num2words import num2words

from sarjy.contexts.assessment.application.ports import (
    AnswerInterpreterPort,
    InstrumentRepo,
    Interpretation,
    NarratorPort,
    RunRepo,
)
from sarjy.contexts.assessment.domain.explanations import EXPLANATIONS
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.scoring import ScoreReport, score
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.domain.workflow_run import (
    MAX_SKIPS,
    IllegalTransition,
    TooManySkips,
    WorkflowRun,
)
from sarjy.contexts.conversation.application.ports import AssessmentReply
from sarjy.observability.logging import get_logger
from sarjy.shared.clock import Clock
from sarjy.shared.ids import UserId

log = get_logger(__name__)

DISCLAIMER = (
    "This is a well-known research questionnaire for self-reflection, "
    "not a clinical or diagnostic tool."
)
SCALE_HINT = (
    "Tell me how accurate that is on a scale of one to five, "
    "one meaning very inaccurate and five very accurate."
)
CONFIRM_THRESHOLD = 0.7
PENDING_QUIT = "quit"

_YES = re.compile(
    r"^\s*(y|yes|yep|yeah|yup|sure|ok|okay|ready|let'?s go|go ahead|correct|right"
    r"|that'?s right|continue)\b",
    re.I,
)
_NO = re.compile(r"^\s*(n|no|nope|nah|not now|later|stop|cancel|wrong|incorrect)\b", re.I)


# --- reply plumbing -------------------------------------------------------


def workflow_dict(run: WorkflowRun, total: int) -> dict[str, Any]:
    return {
        "status": run.status.value,
        # After the last answer `current_item` is total + 1; the client shows a
        # position, not a cursor, so it is clamped.
        "item": min(run.current_item, total),
        "total": total,
        "run_id": str(run.id),
    }


def _reply(
    sentences: list[str], run: WorkflowRun, total: int, numbers: Iterable[float] = ()
) -> AssessmentReply:
    return AssessmentReply(sentences, workflow_dict(run, total), tuple(dict.fromkeys(numbers)))


def nav_numbers(run: WorkflowRun, total: int) -> tuple[float, ...]:
    """The counters a navigational reply is allowed to speak: this item, of N."""
    return (float(min(run.current_item, total)), float(total))


def scale_numbers(ins: Instrument) -> tuple[float, ...]:
    """The ends of the response scale — spoken by `SCALE_HINT` ("one to five")."""
    return (1.0, float(len(ins.scale_labels)))


def results_numbers(ins: Instrument, report: ScoreReport) -> tuple[float, ...]:
    """Every figure the results reply may say: each trait score at one decimal
    place, the integer it rounds to (the narrator may speak it either way), and
    the top of the scale, because each line says "... out of five". Item and
    total counters are deliberately absent — a results reply that quotes a
    question number is reciting something it did not compute.
    """
    out: list[float] = [float(len(ins.scale_labels))]
    for t in report.traits:
        if t.score is not None:
            out.extend((float(t.score), float(round(t.score))))
    return tuple(dict.fromkeys(out))


# --- sentence helpers -----------------------------------------------------


# The instrument's own label for the midpoint is the bare word "Neither",
# which lands as an unfinished sentence when it is read out: "I'll put that as
# a three — neither." Spelled out, the confirmation says what the 3 means.
LABEL_SPEECH = {"neither": "neither accurate nor inaccurate"}


def confirm_sentence(ins: Instrument, value: int) -> str:
    """Read a provisional answer back to the user for a yes/no.

    The value is spoken as a word, because that is how the sentence will be
    heard either way and a lone digit mid-sentence reads like a label. It is
    still declared as a figure by the caller — `extract_numbers` reads "three"
    exactly as it reads "3", so the guard is armed the same way.
    """
    label = ins.scale_labels[value - 1].lower()
    return f"I'll put that as a {num2words(value)} — {LABEL_SPEECH.get(label, label)}. Right?"


def item_sentence(ins: Instrument, no: int) -> str:
    word = str(num2words(no)).capitalize()
    return f"{word}: {ins.item(no).text} How accurate is that for you?"


def results_sentences(ins: Instrument, report: ScoreReport, narrative: str) -> list[str]:
    out = ["Here are your results."]
    for t in report.traits:
        if t.score is None:
            out.append(f"{t.name}: not enough answers to score.")
        else:
            out.append(f"{t.name}: {t.score:.1f} out of five, which is {t.band}.")
    out += [s.strip() for s in re.split(r"(?<=[.!?])\s+", narrative.strip()) if s.strip()]
    out.append(DISCLAIMER)
    return out


class HandleAssessmentTurn:
    def __init__(
        self,
        runs: RunRepo,
        instruments: InstrumentRepo,
        interpreter: AnswerInterpreterPort,
        narrator: NarratorPort,
        clock: Clock,
    ) -> None:
        self.runs = runs
        self.instruments = instruments
        self.interpreter = interpreter
        self.narrator = narrator
        self.clock = clock

    async def execute(self, user_id: UserId, text: str) -> AssessmentReply | None:
        run = await self.runs.get_open(user_id)
        if run is None or run.status is Status.PAUSED:
            # A paused run is resumed only through the explicit workflow_control
            # tool, so ordinary chat stays ordinary chat.
            return None
        ins = await self.instruments.get(run.definition_id)
        total = ins.total_items
        now = self.clock.now()

        if run.status is Status.SCORING or (
            run.status is Status.ACTIVE and run.is_finished_answering(total)
        ):
            # I1/I2 recovery, and it is idempotent: the previous turn answered
            # the last item but its results never reached the user — the save
            # failed, the process died, or an older build parked the run in
            # `scoring`. Every answer is on disk either way, so the only thing
            # missing is the results reply. Producing it here beats asking for
            # item twenty-one (which does not exist) or wedging the run.
            return await self._after_advance(run, ins, now)

        if run.status is Status.PROPOSED:
            if _YES.search(text):
                run.confirm(now)
                await self.runs.save(run)
                return _reply([item_sentence(ins, 1)], run, total, nav_numbers(run, total))
            if _NO.search(text):
                run.quit(now)
                await self.runs.save(run)
                return _reply(["No problem — we can do it another time."], run, total)
            return None  # unrelated chat while the offer is open: normal path answers it

        # ---- ACTIVE ----
        if run.pending_confirmation is not None:
            return await self._resolve_pending(run, ins, text, now)

        item = ins.item(run.current_item)
        interp = await self.interpreter.interpret(item.text, ins.scale_labels, text)
        if interp.control is not None:
            return await self._control(run, ins, interp, now)
        if interp.value is None:
            return self._reask(run, ins)
        if interp.confidence < CONFIRM_THRESHOLD:
            run.set_pending(run.current_item, interp.value, text)
            await self.runs.save(run)
            return _reply(
                [confirm_sentence(ins, interp.value)],
                run,
                total,
                (float(interp.value), *nav_numbers(run, total)),
            )
        return await self._record(run, ins, interp.value, text, interp.confidence, now)

    # ------------------------------------------------------------------
    def _reask(self, run: WorkflowRun, ins: Instrument) -> AssessmentReply:
        total = ins.total_items
        return _reply(
            [SCALE_HINT, item_sentence(ins, run.current_item)],
            run,
            total,
            (*scale_numbers(ins), *nav_numbers(run, total)),
        )

    async def _resolve_pending(
        self, run: WorkflowRun, ins: Instrument, text: str, now: datetime
    ) -> AssessmentReply:
        pending = run.pending_confirmation or {}
        total = ins.total_items
        if pending.get("kind") == PENDING_QUIT:
            run.clear_pending()
            if _YES.search(text):
                run.quit(now)
                await self.runs.save(run)
                return _reply(
                    ["Okay, I've stopped the test.", "We can start fresh any time."], run, total
                )
            await self.runs.save(run)
            return _reply(
                ["Great, let's keep going.", item_sentence(ins, run.current_item)],
                run,
                total,
                nav_numbers(run, total),
            )
        if _YES.search(text):
            return await self._record(
                run, ins, int(pending["value"]), str(pending["raw_text"]), 1.0, now
            )
        run.clear_pending()
        await self.runs.save(run)
        return self._reask(run, ins)

    async def _control(
        self, run: WorkflowRun, ins: Instrument, interp: Interpretation, now: datetime
    ) -> AssessmentReply | None:
        total = ins.total_items
        c = interp.control
        if c == "repeat":
            return _reply(
                [item_sentence(ins, run.current_item)], run, total, nav_numbers(run, total)
            )
        if c == "explain":
            return _reply(
                [
                    EXPLANATIONS.get(
                        run.current_item, "It's asking how well that statement describes you."
                    ),
                    "How accurate is that for you?",
                ],
                run,
                total,
                nav_numbers(run, total),
            )
        if c == "back":
            if run.current_item <= 1:
                return _reply(
                    ["We're on the first one already.", item_sentence(ins, 1)],
                    run,
                    total,
                    nav_numbers(run, total),
                )
            run.back(now)
            await self.runs.save(run)
            return _reply(
                ["Okay, going back.", item_sentence(ins, run.current_item)],
                run,
                total,
                nav_numbers(run, total),
            )
        if c == "skip":
            item_no = run.current_item
            try:
                run.record_answer(item_no, None, "skip", 1.0, total, now)
            except TooManySkips:
                return _reply(
                    [
                        f"You've used your {num2words(MAX_SKIPS)} skips, "
                        "so I'd like your best guess here.",
                        item_sentence(ins, run.current_item),
                    ],
                    run,
                    total,
                    (float(MAX_SKIPS), *nav_numbers(run, total)),
                )
            await self.runs.save_answer(run.id, item_no, "skip", None, 1.0)
            await self.runs.save(run)
            return await self._after_advance(run, ins, now, prefix="Skipped.")
        if c == "pause":
            run.pause(now)
            await self.runs.save(run)
            return _reply(
                [
                    f"Paused at item {run.current_item}.",
                    "Say continue the personality test whenever you're ready.",
                ],
                run,
                total,
                nav_numbers(run, total),
            )
        if c == "quit":
            run.pending_confirmation = {"kind": PENDING_QUIT}
            await self.runs.save(run)
            return _reply(
                [
                    "Are you sure you want to stop the test?",
                    "Your answers so far will be discarded.",
                ],
                run,
                total,
            )
        # off_topic: the workflow declines the turn so ordinary chat answers it,
        # and flags the run so the next prompt block carries the resume nudge.
        run.resume_hint = True
        await self.runs.save(run)
        return None

    async def _record(
        self, run: WorkflowRun, ins: Instrument, value: int, raw: str, conf: float, now: datetime
    ) -> AssessmentReply:
        item_no = run.current_item
        try:
            run.record_answer(item_no, value, raw, conf, ins.total_items, now)
        except (IllegalTransition, TooManySkips):
            # Same shape as the skip path below: a value the aggregate refuses
            # is a spoken re-ask, not a five hundred. The interpreter is a
            # model, and a model can hand back a 0 or a 7 whatever the schema
            # says; nothing is written and the item is simply asked again.
            log.warning("answer_rejected", item_no=item_no)
            return self._reask(run, ins)
        await self.runs.save_answer(run.id, item_no, raw, value, conf)
        await self.runs.save(run)
        return await self._after_advance(run, ins, now)

    async def _after_advance(
        self, run: WorkflowRun, ins: Instrument, now: datetime, prefix: str | None = None
    ) -> AssessmentReply:
        total = ins.total_items
        # A run being recovered out of SCORING has no items left to ask, whatever
        # its cursor says, so it skips straight to the checks below — which it
        # must not skip: a stranded run is exactly the one whose answer rows are
        # most likely to be incomplete, and scoring nineteen answers as if they
        # were twenty publishes a trait mean computed from a hole.
        if run.status is not Status.SCORING and not run.is_finished_answering(total):
            sentences = [item_sentence(ins, run.current_item)]
            if prefix:
                sentences.insert(0, prefix)
            return _reply(sentences, run, total, nav_numbers(run, total))
        answers = await self.runs.answers(run.id)
        missing = run.missing_answers(answers, total)
        if missing:
            # The cursor says every item is done but an answer row is not there
            # — a save that failed, or a row lost between the two writes. So we
            # walk the cursor back to the gap and ask that item again (which
            # reopens a SCORING run: see the state machine's "back" edge).
            # Asked before scoring, not caught after, so a doomed run costs
            # neither a score nor a narration call.
            return await self._reask_missing(run, ins, missing[0], now)
        return await self._finish(run, ins, answers, now)

    async def _finish(
        self,
        run: WorkflowRun,
        ins: Instrument,
        answers: dict[int, int | None],
        now: datetime,
    ) -> AssessmentReply:
        """Score, narrate and complete the run — in that order, in one save.

        The order is the point (I1/I2). The report and the narrative are both
        in hand before the aggregate moves at all, and then both transitions
        land in a single write: there is no window in which the database holds
        a run that has left ACTIVE and not yet reached COMPLETE. `narrate`
        falls back to a deterministic template rather than raising, so the only
        way this ends without a result is the save itself failing — and that
        leaves the stored run exactly as it was, ACTIVE with every answer
        recorded, which `execute` picks up and finishes on the next turn.
        """
        total = ins.total_items
        report = score(ins, answers)
        narrative = await self.narrator.narrate(report)
        if run.status is not Status.SCORING:  # already there if this is a recovery
            run.begin_scoring(now, answers, total)
        run.finish_scoring(report.as_dict(), narrative, self.clock.now())
        await self.runs.save(run)
        return _reply(
            results_sentences(ins, report, narrative), run, total, results_numbers(ins, report)
        )

    async def _reask_missing(
        self,
        run: WorkflowRun,
        ins: Instrument,
        first_missing: int,
        now: datetime,
    ) -> AssessmentReply:
        total = ins.total_items
        # Always at least one step: any run that reaches here has a cursor past
        # the last item, and `first_missing` is at most the last item. The first
        # step is also what returns a SCORING run to ACTIVE.
        while run.current_item > first_missing:
            run.back(now)
        await self.runs.save(run)
        return _reply(
            # No numeral in the lead-in: only the item counters are declared.
            [
                "I seem to be missing an answer — let's fill that in.",
                item_sentence(ins, first_missing),
            ],
            run,
            total,
            nav_numbers(run, total),
        )
