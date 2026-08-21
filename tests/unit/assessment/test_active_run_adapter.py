import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sarjy.contexts.assessment.application.active_run_adapter import (
    FALLBACK_TOTAL_ITEMS,
    ActiveRunAdapter,
)
from sarjy.contexts.assessment.application.handle_turn import HandleAssessmentTurn
from sarjy.contexts.assessment.application.start_run import StartRun
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.infrastructure.memory_repos import MemInstrumentRepo, MemRunRepo
from sarjy.contexts.conversation.application.ports import ActiveRunPort
from sarjy.shared.clock import FakeClock
from sarjy.shared.ids import UserId
from tests.unit.assessment.test_handle_turn import (
    FakeNarrator,
    ScriptedInterpreter,
    _seed_scoring_run,
)

SEED = json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text())
INS = Instrument.from_definition(SEED)
U = UserId(uuid.uuid4())


def _sut():  # type: ignore[no-untyped-def]
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    h = HandleAssessmentTurn(runs, ins, ScriptedInterpreter(), FakeNarrator(), clock)
    return ActiveRunAdapter(runs, ins, h), StartRun(runs, ins, clock), runs


def test_adapter_satisfies_active_run_port() -> None:
    a, _, _ = _sut()
    port: ActiveRunPort = a
    assert port is a


async def test_snapshot_none_without_run() -> None:
    a, _, _ = _sut()
    assert await a.active_run(U) is None


async def test_snapshot_block_and_resume_hint() -> None:
    a, start, _ = _sut()
    await start.execute(U)
    await a.handle_turn(U, "yes")
    snap = await a.active_run(U)
    assert snap is not None and snap.current_item == 1 and snap.total_items == 20
    assert "item 1 of 20" in snap.prompt_block
    assert "Ready to continue" not in snap.prompt_block
    assert await a.handle_turn(U, "what's the weather in rome") is None
    snap2 = await a.active_run(U)
    assert snap2 is not None and "Ready to continue? We were on item 1." in snap2.prompt_block


async def test_proposed_block_wording() -> None:
    a, start, _ = _sut()
    await start.execute(U)
    snap = await a.active_run(U)
    assert snap is not None and snap.status == "proposed"
    assert "awaiting the user's yes/no" in snap.prompt_block


async def test_paused_block_instructs_confirm_then_workflow_control() -> None:
    a, start, _ = _sut()
    await start.execute(U)
    await a.handle_turn(U, "yes")
    await a.handle_turn(U, "let's stop for now")
    snap = await a.active_run(U)
    assert snap is not None and snap.status == "paused"
    assert "confirm once, then call workflow_control with action quit" in snap.prompt_block
    assert "workflow_control" in snap.prompt_block and "resume" in snap.prompt_block


def test_snapshot_from_row_builds_block_without_db() -> None:
    a, _, _ = _sut()
    row = {
        "id": str(uuid.uuid4()),
        "definition_id": INS.id,
        "status": "active",
        "current_item": 7,
        "resume_hint": False,
    }
    # cached() is empty until the instrument has actually been fetched via `get`.
    snap = a.snapshot_from_row(row)
    assert snap is not None and snap.current_item == 7 and "item 7 of 20" in snap.prompt_block


async def test_snapshot_from_row_uses_cached_instrument_total() -> None:
    a, _, _ = _sut()
    # populate the in-process cache the way PgInstrumentRepo would (Phase 7): a
    # prior `get` call.
    await a.instruments.get(INS.id)
    row = {
        "id": str(uuid.uuid4()),
        "definition_id": INS.id,
        "status": "active",
        "current_item": 25,
        "resume_hint": True,
    }
    snap = a.snapshot_from_row(row)
    assert snap is not None
    assert snap.current_item == 20  # clamped to total_items
    assert "item 20 of 20" in snap.prompt_block
    assert 'Ready to continue? We were on item 20."' in snap.prompt_block


def test_snapshot_from_row_none_for_closed_status() -> None:
    a, _, _ = _sut()
    assert a.snapshot_from_row({"status": "complete"}) is None
    assert a.snapshot_from_row({}) is None


async def test_a_stranded_scoring_run_still_produces_a_prompt_block() -> None:
    # `RunTurn` asks for the snapshot before it hands the turn to the engine,
    # so a status with no block here would take the whole turn down (I1/I2).
    a, _, runs = _sut()
    u = UserId(uuid.uuid4())
    await _seed_scoring_run(runs, u)
    snap = await a.active_run(u)
    assert snap is not None and snap.status == "scoring"
    assert snap.current_item == 20 and snap.total_items == 20
    assert "all 20 items are answered" in snap.prompt_block
    assert "Do not state, guess or summarise any score." in snap.prompt_block


def test_snapshot_from_row_reads_a_scoring_row_too() -> None:
    a, _, _ = _sut()
    row = {
        "id": str(uuid.uuid4()),
        "definition_id": INS.id,
        "status": "scoring",
        "current_item": 21,
        "resume_hint": False,
    }
    snap = a.snapshot_from_row(row)
    assert snap is not None and snap.status == "scoring" and snap.current_item == 20


def test_snapshot_from_row_prefers_the_row_s_own_item_count() -> None:
    # Nothing has been fetched, so `cached` is empty — but the row knows.
    a, _, _ = _sut()
    row = {
        "id": str(uuid.uuid4()),
        "definition_id": "some_other_instrument",
        "status": "active",
        "current_item": 9,
        "total_items": 30,
        "resume_hint": False,
    }
    snap = a.snapshot_from_row(row)
    assert snap is not None and snap.total_items == 30
    assert "item 9 of 30" in snap.prompt_block


def test_snapshot_from_row_falls_back_only_when_nothing_else_knows() -> None:
    a, _, _ = _sut()
    row = {
        "id": str(uuid.uuid4()),
        "definition_id": "some_other_instrument",
        "status": "active",
        "current_item": 3,
        "resume_hint": False,
    }
    snap = a.snapshot_from_row(row)
    assert snap is not None and snap.total_items == FALLBACK_TOTAL_ITEMS


async def test_a_cached_instrument_beats_the_fallback() -> None:
    a, start, _ = _sut()
    await start.execute(U)  # warms `MemInstrumentRepo.cached`
    row = {
        "id": str(uuid.uuid4()),
        "definition_id": INS.id,
        "status": "active",
        "current_item": 3,
        "resume_hint": False,
    }
    snap = a.snapshot_from_row(row)
    assert snap is not None and snap.total_items == INS.total_items
