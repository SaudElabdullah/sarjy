import uuid
from datetime import UTC, datetime

import pytest

from sarjy.contexts.assessment.domain.events import AnswerRecorded, RunCompleted, RunConfirmed
from sarjy.contexts.assessment.domain.workflow_run import (
    IllegalTransition,
    Status,
    TooManySkips,
    WorkflowRun,
)
from sarjy.shared.ids import RunId, UserId

NOW = datetime(2026, 8, 21, tzinfo=UTC)
TOTAL = 20


def _run() -> WorkflowRun:
    return WorkflowRun.propose(RunId(uuid.uuid4()), UserId(uuid.uuid4()), "ocean_mini_ipip", 1, NOW)


def test_propose_then_confirm() -> None:
    r = _run()
    assert r.status is Status.PROPOSED and r.current_item == 1
    r.confirm(NOW)
    assert r.status is Status.ACTIVE and isinstance(r.events[-1], RunConfirmed)


def test_decline_proposed_abandons() -> None:
    r = _run()
    r.quit(NOW)
    assert r.status is Status.ABANDONED


def test_record_answer_advances_and_emits() -> None:
    r = _run()
    r.confirm(NOW)
    ev = r.record_answer(1, 4, "yeah mostly", 0.9, TOTAL, NOW)
    assert isinstance(ev, AnswerRecorded) and r.current_item == 2


def test_record_answer_wrong_item_is_illegal() -> None:
    r = _run()
    r.confirm(NOW)
    with pytest.raises(IllegalTransition):
        r.record_answer(3, 4, "x", 0.9, TOTAL, NOW)


def test_skip_limit() -> None:
    r = _run()
    r.confirm(NOW)
    r.record_answer(1, None, "skip", 1.0, TOTAL, NOW)
    r.record_answer(2, None, "skip", 1.0, TOTAL, NOW)
    with pytest.raises(TooManySkips):
        r.record_answer(3, None, "skip", 1.0, TOTAL, NOW)
    assert r.skips_used == 2 and r.current_item == 3


def test_back_decrements_but_not_below_one() -> None:
    r = _run()
    r.confirm(NOW)
    r.record_answer(1, 3, "three", 1.0, TOTAL, NOW)
    r.back(NOW)
    assert r.current_item == 1
    with pytest.raises(IllegalTransition):
        r.back(NOW)


def test_pause_resume() -> None:
    r = _run()
    r.confirm(NOW)
    r.pause(NOW)
    assert r.status is Status.PAUSED
    with pytest.raises(IllegalTransition):
        r.record_answer(1, 3, "x", 1.0, TOTAL, NOW)
    r.resume(NOW)
    assert r.status is Status.ACTIVE


def test_last_answer_moves_to_scoring_then_complete() -> None:
    r = _run()
    r.confirm(NOW)
    for n in range(1, 21):
        r.record_answer(n, 3, "three", 1.0, TOTAL, NOW)
    assert r.is_finished_answering(TOTAL) and r.status is Status.ACTIVE
    r.begin_scoring(NOW, dict.fromkeys(range(1, 21), 3), TOTAL)
    assert r.status is Status.SCORING
    r.finish_scoring({"O": 3.0}, "narrative", NOW)
    assert (
        r.status is Status.COMPLETE
        and r.completed_at == NOW
        and isinstance(r.events[-1], RunCompleted)
    )


def test_complete_is_terminal() -> None:
    r = _run()
    r.confirm(NOW)
    for n in range(1, 21):
        r.record_answer(n, 3, "three", 1.0, TOTAL, NOW)
    r.begin_scoring(NOW, dict.fromkeys(range(1, 21), 3), TOTAL)
    r.finish_scoring({}, "", NOW)
    for op in (
        lambda: r.pause(NOW),
        lambda: r.resume(NOW),
        lambda: r.confirm(NOW),
        lambda: r.quit(NOW),
    ):
        with pytest.raises(IllegalTransition):
            op()


def test_pending_confirmation_roundtrip() -> None:
    r = _run()
    r.confirm(NOW)
    r.set_pending(1, 4, "sort of")
    assert r.pending_confirmation == {"item_no": 1, "value": 4, "raw_text": "sort of"}
    r.clear_pending()
    assert r.pending_confirmation is None


def test_begin_scoring_refuses_an_incomplete_run() -> None:
    r = _run()
    r.confirm(NOW)
    for n in range(1, 20):
        r.record_answer(n, 3, "three", 1.0, TOTAL, NOW)
    # Item 20 was never answered: scoring it would silently drop a trait item.
    with pytest.raises(IllegalTransition):
        r.begin_scoring(NOW, dict.fromkeys(range(1, 20), 3), TOTAL)
    assert r.status is Status.ACTIVE


def test_begin_scoring_counts_a_skip_as_recorded() -> None:
    r = _run()
    r.confirm(NOW)
    r.record_answer(1, None, "skip", 1.0, TOTAL, NOW)
    for n in range(2, 21):
        r.record_answer(n, 3, "three", 1.0, TOTAL, NOW)
    answers: dict[int, int | None] = dict.fromkeys(range(2, 21), 3)
    answers[1] = None
    r.begin_scoring(NOW, answers, TOTAL)
    assert r.status is Status.SCORING
