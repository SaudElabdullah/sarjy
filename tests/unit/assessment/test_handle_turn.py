from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest

from sarjy.contexts.assessment.application.control_run import ControlRun
from sarjy.contexts.assessment.application.handle_turn import DISCLAIMER, HandleAssessmentTurn
from sarjy.contexts.assessment.application.ports import Interpretation
from sarjy.contexts.assessment.application.start_run import StartRun
from sarjy.contexts.assessment.domain.explanations import EXPLANATIONS
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.scoring import ScoreReport
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.domain.workflow_run import IllegalTransition, WorkflowRun
from sarjy.contexts.assessment.infrastructure.memory_repos import MemInstrumentRepo, MemRunRepo
from sarjy.contexts.conversation.application.ports import AssessmentReply
from sarjy.contexts.guardrails.domain.grounding import extract_numbers, ungrounded_numbers
from sarjy.shared.clock import FakeClock
from sarjy.shared.ids import RunId, UserId

SEED = json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text())
INS = Instrument.from_definition(SEED)
U = UserId(uuid.uuid4())


class ScriptedInterpreter:
    """Maps exact user text -> Interpretation; anything unknown is a confident 3."""

    TABLE: ClassVar[dict[str, Interpretation]] = {
        "nah": Interpretation(2, 0.9, None),
        "yeah totally": Interpretation(5, 0.95, None),
        "four": Interpretation(4, 1.0, None),
        "three": Interpretation(3, 1.0, None),
        "sort of": Interpretation(4, 0.55, None),
        "what does that mean?": Interpretation(None, 1.0, "explain"),
        "go back": Interpretation(None, 1.0, "back"),
        "repeat": Interpretation(None, 1.0, "repeat"),
        "skip": Interpretation(None, 1.0, "skip"),
        "let's stop for now": Interpretation(None, 1.0, "pause"),
        "quit the test": Interpretation(None, 1.0, "quit"),
        "what's the weather in rome": Interpretation(None, 1.0, "off_topic"),
        "mumble": Interpretation(None, 0.2, None),
    }

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def interpret(
        self, item_text: str, scale_labels: list[str], user_text: str
    ) -> Interpretation:
        self.calls.append(user_text)
        return self.TABLE.get(user_text.lower(), Interpretation(3, 0.9, None))


class FakeNarrator:
    async def narrate(self, report: ScoreReport) -> str:
        return "You come across as balanced. " * 2


def _sut() -> tuple[HandleAssessmentTurn, StartRun, ControlRun, MemRunRepo]:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    handle = HandleAssessmentTurn(runs, ins, ScriptedInterpreter(), FakeNarrator(), clock)
    return handle, StartRun(runs, ins, clock), ControlRun(runs, ins, clock, handle), runs


async def test_no_run_returns_none() -> None:
    h, _, _, _ = _sut()
    assert await h.execute(U, "hello") is None


async def test_start_gives_intro_with_disclaimer_and_proposed_status() -> None:
    _, start, _, runs = _sut()
    reply = await start.execute(U)
    assert any(DISCLAIMER in s for s in reply.sentences)
    assert reply.workflow["status"] == "proposed" and reply.workflow["total"] == 20
    open_run = await runs.get_open(U)
    assert open_run is not None and open_run.status is Status.PROPOSED


async def test_start_twice_offers_to_resume_instead_of_a_second_run() -> None:
    h, start, _, runs = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    reply = await start.execute(U)
    assert reply.workflow["status"] == "active" and reply.workflow["item"] == 1
    assert len(runs.runs) == 1


async def test_confirm_reads_first_item() -> None:
    h, start, _, _ = _sut()
    await start.execute(U)
    reply = await h.execute(U, "yep")
    assert reply is not None and reply.sentences[0].startswith("One: I am the life of the party.")
    assert reply.workflow == {
        "status": "active",
        "item": 1,
        "total": 20,
        "run_id": reply.workflow["run_id"],
    }


async def test_decline_abandons() -> None:
    h, start, _, runs = _sut()
    await start.execute(U)
    reply = await h.execute(U, "no thanks")
    assert reply is not None and reply.workflow["status"] == "abandoned"
    assert await runs.get_open(U) is None


async def test_answer_flow_with_controls() -> None:
    h, start, _, runs = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    r1 = await h.execute(U, "nah")  # item 1 = 2 -> reads item 2
    assert r1 is not None and r1.sentences[0].startswith("Two:")
    r2 = await h.execute(U, "what does that mean?")  # explain item 2, re-ask
    assert r2 is not None and EXPLANATIONS[2] in r2.sentences[0] and r2.workflow["item"] == 2
    await h.execute(U, "yeah totally")  # item 2 = 5 -> item 3
    r3 = await h.execute(U, "go back")  # back to item 2
    assert r3 is not None and r3.workflow["item"] == 2 and r3.sentences[-1].startswith("Two:")
    await h.execute(U, "four")  # item 2 = 4 (overwrites) -> item 3
    r4 = await h.execute(U, "repeat")
    assert r4 is not None and r4.sentences[0].startswith("Three:")
    run = await runs.get_open(U)
    assert run is not None
    assert await runs.answers(run.id) == {1: 2, 2: 4}


async def test_back_on_first_item_stays_put() -> None:
    h, start, _, _ = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    r = await h.execute(U, "go back")
    assert r is not None and r.workflow["item"] == 1 and r.sentences[-1].startswith("One:")


async def test_low_confidence_asks_for_confirmation_then_records() -> None:
    h, start, _, runs = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    r = await h.execute(U, "sort of")
    assert r is not None and r.workflow["item"] == 1
    assert r.sentences[0] == "I'll put that as a four — moderately accurate. Right?"
    assert 4.0 in r.grounding_numbers
    r2 = await h.execute(U, "yes")
    assert r2 is not None and r2.workflow["item"] == 2
    run = await runs.get_open(U)
    assert run is not None and await runs.answers(run.id) == {1: 4}


async def test_low_confidence_rejected_reasks() -> None:
    h, start, _, _ = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    await h.execute(U, "sort of")
    r = await h.execute(U, "no")
    assert r is not None and r.workflow["item"] == 1 and "one to five" in r.sentences[0].lower()


async def test_unintelligible_reasks_with_scale_hint() -> None:
    h, start, _, _ = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    r = await h.execute(U, "mumble")
    assert r is not None and "one to five" in r.sentences[0].lower() and r.workflow["item"] == 1
    # The hint itself says "one ... five", so the guard has to be told those.
    assert {1.0, 5.0} <= set(r.grounding_numbers)


async def test_skip_limit_message() -> None:
    h, start, _, _ = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    await h.execute(U, "skip")
    await h.execute(U, "skip")
    r = await h.execute(U, "skip")
    assert r is not None and "two skips" in r.sentences[0].lower() and r.workflow["item"] == 3
    assert 2.0 in r.grounding_numbers  # "two skips"


async def test_skips_are_recorded_as_none() -> None:
    h, start, _, runs = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    r = await h.execute(U, "skip")
    assert r is not None and r.workflow["item"] == 2
    run = await runs.get_open(U)
    assert run is not None and await runs.answers(run.id) == {1: None}


async def test_pause_and_resume_across_calls() -> None:
    h, start, control, _ = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    await h.execute(U, "three")
    r = await h.execute(U, "let's stop for now")
    assert r is not None and r.workflow["status"] == "paused"
    assert r.grounding_numbers == (2.0, 20.0)
    assert await h.execute(U, "hello again") is None  # paused run does not intercept chat
    r2 = await control.execute(U, "resume")
    assert r2.workflow["status"] == "active" and r2.sentences[-1].startswith("Two:")
    assert r2.grounding_numbers == (2.0, 20.0)


async def test_control_quit_and_no_run() -> None:
    h, start, control, runs = _sut()
    assert (await control.execute(U, "resume")).workflow["status"] == "none"
    await start.execute(U)
    await h.execute(U, "yes")
    r = await control.execute(U, "quit")
    assert r.workflow["status"] == "abandoned" and await runs.get_open(U) is None


async def test_off_topic_returns_none_and_sets_resume_hint() -> None:
    h, start, _, runs = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    assert await h.execute(U, "what's the weather in rome") is None
    run = await runs.get_open(U)
    assert run is not None and run.resume_hint is True and run.status is Status.ACTIVE


async def test_quit_requires_confirmation() -> None:
    h, start, _, runs = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    r = await h.execute(U, "quit the test")
    assert r is not None and "sure" in r.sentences[0].lower() and r.workflow["status"] == "active"
    r2 = await h.execute(U, "yes")
    assert r2 is not None and r2.workflow["status"] == "abandoned"
    assert await runs.get_open(U) is None


async def test_quit_declined_keeps_going() -> None:
    h, start, _, _ = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    await h.execute(U, "quit the test")
    r = await h.execute(U, "no")
    assert r is not None and r.workflow["status"] == "active" and r.sentences[-1].startswith("One:")


async def test_item_prompt_grounds_its_own_counters() -> None:
    h, start, _, _ = _sut()
    await start.execute(U)
    r0 = await h.execute(U, "yes")
    assert r0 is not None and r0.grounding_numbers == (1.0, 20.0)
    r1 = await h.execute(U, "four")
    assert r1 is not None and r1.grounding_numbers == (2.0, 20.0)
    r2 = await h.execute(U, "what does that mean?")
    assert r2 is not None and r2.grounding_numbers == (2.0, 20.0)


async def test_full_run_scores_and_completes() -> None:
    h, start, _, runs = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    reply = None
    for _ in range(20):
        reply = await h.execute(U, "four")
    assert reply is not None and reply.workflow["status"] == "complete"
    joined = " ".join(reply.sentences)
    assert "Extraversion" in joined and DISCLAIMER in joined
    done = await runs.latest_complete(U)
    assert done is not None and done.results is not None
    assert done.results["E"] == 3.0 and done.narrative  # 4 and 6-4=2 -> mean 3.0
    assert await h.execute(U, "hi") is None


async def test_results_reply_grounds_every_score_and_not_the_counters() -> None:
    h, start, _, _ = _sut()
    await start.execute(U)
    await h.execute(U, "yes")
    reply = None
    for _ in range(20):
        reply = await h.execute(U, "four")
    assert reply is not None
    nums = set(reply.grounding_numbers)
    # Four traits score 3.0; Openness has three reversed items and scores 2.5.
    assert {3.0, 2.5, 2.0} <= nums  # each score plus its integer rounding
    assert 5.0 in nums  # "... out of five"
    assert 20.0 not in nums  # item/total counters are not part of a results reply


def _assert_speakable(reply: AssessmentReply | None) -> None:
    """Every figure in the reply survives the real output-guard check (PRD G-6).

    An empty `grounding_numbers` switches that check off, which is only honest
    for a reply that states no figure at all — so that case is asserted too.
    """
    assert reply is not None
    allowed = list(reply.grounding_numbers)
    for s in reply.sentences:
        if allowed:
            assert ungrounded_numbers(s, allowed) == [], (s, allowed)
        else:
            assert extract_numbers(s) == [], s


async def test_every_reply_is_grounded_for_the_output_guard() -> None:
    h, start, control, _ = _sut()
    _assert_speakable(await start.execute(U))  # intro: 20 statements, one-to-five
    _assert_speakable(await h.execute(U, "yes"))  # "One: ..."
    _assert_speakable(await h.execute(U, "mumble"))  # scale hint + re-ask
    _assert_speakable(await h.execute(U, "sort of"))  # "I'll put that as a 4"
    _assert_speakable(await h.execute(U, "no"))  # rejected -> re-ask
    _assert_speakable(await h.execute(U, "go back"))  # "the first one"
    _assert_speakable(await h.execute(U, "four"))  # "Two: ..."
    _assert_speakable(await h.execute(U, "what does that mean?"))  # explanation
    _assert_speakable(await h.execute(U, "skip"))  # "Skipped." + next item
    _assert_speakable(await h.execute(U, "skip"))
    _assert_speakable(await h.execute(U, "skip"))  # "two skips" limit
    _assert_speakable(await h.execute(U, "quit the test"))  # confirmation
    _assert_speakable(await h.execute(U, "no"))  # declined -> keep going
    _assert_speakable(await start.execute(U))  # "We're already on item 4."
    _assert_speakable(await h.execute(U, "let's stop for now"))  # "Paused at item 4."
    _assert_speakable(await start.execute(U))  # paused resume offer
    _assert_speakable(await control.execute(U, "resume"))  # "Picking up at item 4."
    reply = None
    while reply is None or reply.workflow["status"] != "complete":
        reply = await h.execute(U, "four")
        _assert_speakable(reply)  # ends on the results reply: five scores
    assert "Extraversion: 3.0 out of five" in " ".join(reply.sentences)


class LosesAnswerRepo(MemRunRepo):
    """A run repo that forgets one answer row — a save that failed, or a row
    lost between the two writes that record an answer."""

    def __init__(self, drop: int) -> None:
        super().__init__()
        self.drop: int | None = drop

    async def answers(self, run_id: uuid.UUID) -> dict[int, int | None]:  # type: ignore[override]
        rows = await super().answers(run_id)  # type: ignore[arg-type]
        if self.drop is not None:
            rows.pop(self.drop, None)
        return rows


def _sut_with(runs: MemRunRepo) -> tuple[HandleAssessmentTurn, StartRun]:
    ins = MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    return (
        HandleAssessmentTurn(runs, ins, ScriptedInterpreter(), FakeNarrator(), clock),
        StartRun(runs, ins, clock),
    )


async def test_a_missing_answer_reasks_that_item_instead_of_wedging_the_run() -> None:
    runs = LosesAnswerRepo(drop=7)
    h, start = _sut_with(runs)
    u = UserId(uuid.uuid4())
    await start.execute(u)
    await h.execute(u, "yes")

    reply = None
    for _ in range(20):
        reply = await h.execute(u, "four")

    # Scoring a run with a hole would publish a trait mean computed from
    # nineteen answers as if it were twenty.
    assert reply is not None and reply.workflow["status"] == "active"
    assert reply.workflow["item"] == 7
    assert INS.item(7).text in " ".join(reply.sentences)
    _assert_speakable(reply)
    open_run = await runs.get_open(u)
    assert open_run is not None and open_run.current_item == 7

    # And the run is not stuck: once the row is readable again it finishes.
    runs.drop = None
    for _ in range(14):
        reply = await h.execute(u, "four")
    assert reply is not None and reply.workflow["status"] == "complete"


# --- I1/I2: scoring is never a state the run is left in --------------------


class FailsTheFinalSaveRepo(MemRunRepo):
    """Loses the write that completes the run — the crash I1 is about."""

    def __init__(self) -> None:
        super().__init__()
        self.failing = True

    async def save(self, run: WorkflowRun) -> None:
        if self.failing and run.status is Status.COMPLETE:
            raise RuntimeError("the save that completes the run did not land")
        await super().save(run)


async def test_a_lost_completing_save_leaves_an_active_run_the_next_turn_finishes() -> None:
    runs = FailsTheFinalSaveRepo()
    h, start = _sut_with(runs)
    u = UserId(uuid.uuid4())
    await start.execute(u)
    await h.execute(u, "yes")
    for _ in range(19):
        await h.execute(u, "four")

    with pytest.raises(RuntimeError):
        await h.execute(u, "four")  # the twentieth answer: scores, then the save fails

    # Nothing half-written: the run is still ACTIVE, with all twenty answers.
    stranded = await runs.get_open(u)
    assert stranded is not None
    assert stranded.status is Status.ACTIVE and stranded.current_item == 21
    assert len(await runs.answers(stranded.id)) == 20
    assert stranded.results is None and stranded.narrative is None

    # And the very next turn produces the results the user never heard.
    runs.failing = False
    reply = await h.execute(u, "so what did I get?")
    assert reply is not None and reply.workflow["status"] == "complete"
    assert "Extraversion: 3.0 out of five" in " ".join(reply.sentences)
    _assert_speakable(reply)
    done = await runs.latest_complete(u)
    assert done is not None and done.results is not None and done.narrative


async def _seed_scoring_run(runs: MemRunRepo, u: UserId) -> WorkflowRun:
    """A run parked in SCORING — what an older build persisted mid-turn."""
    now = datetime(2026, 8, 21, tzinfo=UTC)
    run = WorkflowRun.propose(RunId(uuid.uuid4()), u, INS.id, INS.version, now)
    run.confirm(now)
    for n in range(1, 21):
        run.record_answer(n, 4, "four", 1.0, 20, now)
        await runs.save_answer(run.id, n, "four", 4, 1.0)
    run.begin_scoring(now, await runs.answers(run.id), 20)
    await runs.save(run)
    return run


async def test_a_run_stranded_in_scoring_is_open_and_the_next_turn_completes_it() -> None:
    runs = MemRunRepo()
    h, _ = _sut_with(runs)
    u = UserId(uuid.uuid4())
    seeded = await _seed_scoring_run(runs, u)

    # It has to be findable at all, or nothing will ever finish it.
    open_run = await runs.get_open(u)
    assert open_run is not None and open_run.id == seeded.id
    assert open_run.status is Status.SCORING

    reply = await h.execute(u, "hello?")
    assert reply is not None and reply.workflow["status"] == "complete"
    assert reply.workflow["run_id"] == str(seeded.id)
    _assert_speakable(reply)
    done = await runs.latest_complete(u)
    assert done is not None and done.results is not None and done.results["E"] == 3.0


async def test_finishing_a_stranded_run_is_idempotent() -> None:
    runs = MemRunRepo()
    h, _ = _sut_with(runs)
    u = UserId(uuid.uuid4())
    await _seed_scoring_run(runs, u)
    first = await h.execute(u, "hello?")
    assert first is not None
    # The run is COMPLETE now, so it is no longer open and ordinary chat is
    # ordinary chat again — replaying the recovery cannot double-score it.
    assert await h.execute(u, "hello again") is None
    assert len(runs.runs) == 1


class CountingNarrator(FakeNarrator):
    def __init__(self) -> None:
        self.calls = 0

    async def narrate(self, report: ScoreReport) -> str:
        self.calls += 1
        return await super().narrate(report)


async def test_a_run_with_a_gap_is_never_scored_or_narrated() -> None:
    # I1's ordering has a second benefit: the missing-answer check runs before
    # the expensive tail, so a doomed run costs no narration call at all.
    runs = LosesAnswerRepo(drop=7)
    ins = MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    narrator = CountingNarrator()
    h = HandleAssessmentTurn(runs, ins, ScriptedInterpreter(), narrator, clock)
    start = StartRun(runs, ins, clock)
    u = UserId(uuid.uuid4())
    await start.execute(u)
    await h.execute(u, "yes")
    for _ in range(20):
        reply = await h.execute(u, "four")
    assert reply is not None and reply.workflow["item"] == 7
    assert narrator.calls == 0


# --- minors ----------------------------------------------------------------


async def test_the_midpoint_confirmation_says_what_a_three_means() -> None:
    # The instrument's own label for 3 is the bare word "Neither", which reads
    # as an unfinished sentence out loud.
    h, start, _, _ = _sut()
    u = UserId(uuid.uuid4())
    await start.execute(u)
    await h.execute(u, "yes")

    class Unsure(ScriptedInterpreter):
        async def interpret(self, item_text, scale_labels, user_text):  # type: ignore[no-untyped-def]
            return Interpretation(3, 0.5, None)

    h.interpreter = Unsure()
    r = await h.execute(u, "kind of, i guess")
    assert r is not None
    assert r.sentences[0] == ("I'll put that as a three — neither accurate nor inaccurate. Right?")
    assert 3.0 in r.grounding_numbers  # the spoken "three" is still declared
    _assert_speakable(r)


async def test_an_out_of_range_interpreted_value_reasks_instead_of_raising() -> None:
    h, start, _, runs = _sut()
    u = UserId(uuid.uuid4())
    await start.execute(u)
    await h.execute(u, "yes")

    class OffScale(ScriptedInterpreter):
        async def interpret(self, item_text, scale_labels, user_text):  # type: ignore[no-untyped-def]
            return Interpretation(7, 1.0, None)  # a model can do this whatever the schema says

    h.interpreter = OffScale()
    r = await h.execute(u, "eleven out of ten")
    assert r is not None and "one to five" in r.sentences[0].lower()
    assert r.workflow["item"] == 1 and r.workflow["status"] == "active"
    run = await runs.get_open(u)
    assert run is not None and await runs.answers(run.id) == {}  # nothing written
    _assert_speakable(r)


async def test_the_intro_speaks_the_instrument_s_own_length() -> None:
    short = Instrument.from_definition({**SEED, "items": SEED["items"][:10]})
    runs = MemRunRepo()
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    start = StartRun(runs, MemInstrumentRepo({short.id: short}), clock)
    reply = await start.execute(UserId(uuid.uuid4()), short.id)
    assert "ten quick statements" in reply.sentences[0]
    assert "twenty" not in reply.sentences[0]
    assert "five minutes" not in reply.sentences[0]  # a promise nobody measured
    assert 10.0 in reply.grounding_numbers
    _assert_speakable(reply)


async def test_a_run_is_only_readable_by_id_by_the_user_who_owns_it() -> None:
    runs = MemRunRepo()
    _, start = _sut_with(runs)
    owner, other = UserId(uuid.uuid4()), UserId(uuid.uuid4())
    await start.execute(owner)
    run = await runs.get_open(owner)
    assert run is not None

    assert (await runs.get(run.id, owner)) is not None
    assert (await runs.get(run.id, other)) is None


# --- a stranded run is still not scored from a hole -------------------------


async def _seed_scoring_run_missing(runs: MemRunRepo, u: UserId, drop: int) -> WorkflowRun:
    """A SCORING run whose answer row for `drop` never made it to storage."""
    run = await _seed_scoring_run(runs, u)
    del runs.answers_by_run[run.id][drop]
    return run


async def test_a_stranded_scoring_run_with_a_gap_reasks_instead_of_scoring_it() -> None:
    runs = MemRunRepo()
    h, _ = _sut_with(runs)
    u = UserId(uuid.uuid4())
    seeded = await _seed_scoring_run_missing(runs, u, drop=7)

    reply = await h.execute(u, "so what did I get?")
    # Scoring nineteen answers as if they were twenty would publish a trait
    # mean computed from a hole and call it a result.
    assert reply is not None and reply.workflow["status"] == "active"
    assert reply.workflow["item"] == 7
    assert INS.item(7).text in " ".join(reply.sentences)
    _assert_speakable(reply)
    reopened = await runs.get_open(u)
    assert reopened is not None and reopened.status is Status.ACTIVE
    assert reopened.id == seeded.id and reopened.results is None

    # And it finishes normally once the missing answer is given.
    for _ in range(14):
        reply = await h.execute(u, "four")
    assert reply is not None and reply.workflow["status"] == "complete"
    done = await runs.latest_complete(u)
    assert done is not None and done.results is not None and done.results["E"] == 3.0


async def test_going_back_out_of_scoring_is_the_only_reopening_route() -> None:
    # PAUSED and PROPOSED still cannot step the cursor back; the new edge is
    # SCORING -> ACTIVE and nothing else.
    now = datetime(2026, 8, 21, tzinfo=UTC)
    run = WorkflowRun.propose(RunId(uuid.uuid4()), U, INS.id, INS.version, now)
    with pytest.raises(IllegalTransition):
        run.back(now)
    run.confirm(now)
    run.record_answer(1, 4, "four", 1.0, 20, now)
    run.pause(now)
    with pytest.raises(IllegalTransition):
        run.back(now)


# --- explicit control of a stranded run ------------------------------------


async def test_control_quit_stops_a_run_stranded_in_scoring() -> None:
    runs = MemRunRepo()
    h, _ = _sut_with(runs)
    ins = MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    control = ControlRun(runs, ins, clock, h)
    u = UserId(uuid.uuid4())
    await _seed_scoring_run(runs, u)

    r = await control.execute(u, "quit")
    assert r.sentences[0] == "Okay, I won't finish scoring the test."
    assert r.workflow["status"] == "abandoned"
    assert await runs.get_open(u) is None


async def test_control_resume_finishes_a_run_stranded_in_scoring() -> None:
    runs = MemRunRepo()
    h, _ = _sut_with(runs)
    ins = MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    control = ControlRun(runs, ins, clock, h)
    u = UserId(uuid.uuid4())
    await _seed_scoring_run(runs, u)

    r = await control.execute(u, "resume")
    assert r.workflow["status"] == "complete"
    assert "Extraversion: 3.0 out of five" in " ".join(r.sentences)


async def test_control_resume_on_a_stranded_run_with_a_gap_asks_for_the_answer() -> None:
    runs = MemRunRepo()
    h, _ = _sut_with(runs)
    ins = MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    control = ControlRun(runs, ins, clock, h)
    u = UserId(uuid.uuid4())
    await _seed_scoring_run_missing(runs, u, drop=12)

    r = await control.execute(u, "resume")
    assert r.workflow["status"] == "active" and r.workflow["item"] == 12
    assert INS.item(12).text in " ".join(r.sentences)
