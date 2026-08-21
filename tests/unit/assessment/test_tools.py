import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sarjy.contexts.assessment.application.control_run import ControlRun
from sarjy.contexts.assessment.application.handle_turn import HandleAssessmentTurn
from sarjy.contexts.assessment.application.start_run import StartRun
from sarjy.contexts.assessment.application.tools import StartWorkflowTool, WorkflowControlTool
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.infrastructure.memory_repos import MemInstrumentRepo, MemRunRepo
from sarjy.contexts.conversation.application.ports import ToolPort
from sarjy.shared.clock import FakeClock
from sarjy.shared.ids import UserId
from tests.unit.assessment.test_handle_turn import FakeNarrator, ScriptedInterpreter

SEED = json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text())
INS = Instrument.from_definition(SEED)
U = UserId(uuid.uuid4())


def _control(runs: MemRunRepo, ins: MemInstrumentRepo, clock: FakeClock) -> ControlRun:
    """`ControlRun` with the turn handler it defers stranded runs to."""
    handle = HandleAssessmentTurn(runs, ins, ScriptedInterpreter(), FakeNarrator(), clock)
    return ControlRun(runs, ins, clock, handle)


def test_tools_satisfy_tool_port() -> None:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    start_tool: ToolPort = StartWorkflowTool(StartRun(runs, ins, clock))
    control_tool: ToolPort = WorkflowControlTool(_control(runs, ins, clock))
    assert start_tool.name == "start_workflow"
    assert control_tool.name == "workflow_control"


async def test_start_tool_declaration_matches_prd() -> None:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    t = StartWorkflowTool(StartRun(runs, ins, clock))
    assert t.name == "start_workflow"
    assert t.declaration["name"] == "start_workflow"
    assert t.declaration["parameters"]["properties"]["workflow_id"]["enum"] == ["ocean_mini_ipip"]
    assert t.declaration["parameters"]["required"] == ["workflow_id"]


async def test_start_tool_returns_direct_sentences() -> None:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    t = StartWorkflowTool(StartRun(runs, ins, clock))
    res = await t.invoke(U, {"workflow_id": "ocean_mini_ipip"})
    assert res.ok and res.direct_sentences and res.direct_sentences[-1] == "Ready?"
    assert res.data["workflow"]["status"] == "proposed"
    assert res.grounding_numbers  # the intro states the scale/length numbers


async def test_start_tool_rejects_unknown_workflow() -> None:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    t = StartWorkflowTool(StartRun(runs, ins, FakeClock(datetime(2026, 8, 21, tzinfo=UTC))))
    res = await t.invoke(U, {"workflow_id": "nope"})
    assert not res.ok and res.spoken_error


async def test_control_tool_declaration_matches_prd() -> None:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    t = WorkflowControlTool(_control(runs, ins, FakeClock(datetime(2026, 8, 21, tzinfo=UTC))))
    assert t.name == "workflow_control"
    assert t.declaration["parameters"]["properties"]["action"]["enum"] == ["resume", "quit"]
    assert t.declaration["parameters"]["required"] == ["action"]


async def test_control_tool_quit_without_run() -> None:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    t = WorkflowControlTool(_control(runs, ins, FakeClock(datetime(2026, 8, 21, tzinfo=UTC))))
    res = await t.invoke(U, {"action": "quit"})
    assert res.ok
    assert res.direct_sentences and "no personality test" in res.direct_sentences[0].lower()


async def test_control_tool_resume_after_start_and_pause() -> None:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    start = StartWorkflowTool(StartRun(runs, ins, clock))
    control = WorkflowControlTool(_control(runs, ins, clock))
    await start.invoke(U, {"workflow_id": "ocean_mini_ipip"})
    run = await runs.get_open(U)
    assert run is not None
    run.confirm(clock.now())
    run.pause(clock.now())
    await runs.save(run)
    res = await control.invoke(U, {"action": "resume"})
    assert res.ok and res.direct_sentences and res.data["workflow"]["status"] == "active"


async def test_control_tool_rejects_bad_action() -> None:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    t = WorkflowControlTool(_control(runs, ins, FakeClock(datetime(2026, 8, 21, tzinfo=UTC))))
    res = await t.invoke(U, {"action": "nope"})
    assert not res.ok and res.spoken_error
