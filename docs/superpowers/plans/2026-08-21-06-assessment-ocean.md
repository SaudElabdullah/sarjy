# Phase 6 — Assessment (Big Five / OCEAN) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A user can say "give me a personality test", answer 20 Mini-IPIP items by voice using natural language, pause/resume across sessions, and receive deterministic OCEAN scores with a short LLM-written narrative — with the LLM never touching a number.

**Architecture:** The Assessment context owns a server-side state machine (`WorkflowRun` aggregate) persisted in Postgres. The LLM is used for exactly two narrow jobs through ports: interpreting a spoken answer into `{value, confidence, control}` (strict JSON schema, temperature 0) and writing a 4-sentence narrative from already-computed scores. The context plugs into Conversation via `ActiveRunPort` (intercepts turns while a run is active) and two `ToolPort`s (`start_workflow`, `workflow_control`). Tool replies that must be spoken verbatim use a new `ToolResult.direct_sentences` channel so `RunTurn` speaks them without a second LLM hop.

**Tech Stack:** Python dataclasses/enums, pydantic (interpreter schema), google-genai via `LLMPort`, asyncpg, Jinja2 + one small `ocean.js` module.

**Spec:** `PRD.md` §7.5 (P-1…P-12, state diagram), §9.3 (`start_workflow`, `workflow_control`), §10 block 8, Appendix B (Mini-IPIP seed). Master plan §3.

## Global Constraints

- Instrument = 20-item Mini-IPIP, 5-point scale, 4 items per trait, reverse-keyed items use `6 - v` (Appendix B).
- Scoring is deterministic Python; the LLM never produces or alters numbers (P-8). Narrative is post-checked for every score number.
- Max 2 skips per run; a trait with < 3 answered items reports "not enough answers" (P-5).
- Confidence < 0.7 with a value → confirm before recording (P-4).
- Answer interpreter: `temperature 0`, JSON schema, guard model (`gemini_guard_model`). Narrator: `thinking_budget 1024`, `temperature 0.7`, `max_output_tokens 400`, chat model.
- Disclaimer spoken at start and shown on the results card (P-10).
- Every answer is persisted immediately; state survives reload/new session (P-6).
- Domain layer imports nothing outside `sarjy.contexts.assessment.domain` and `sarjy.shared`.

---

## File structure created in this phase

```
src/sarjy/contexts/assessment/__init__.py
src/sarjy/contexts/assessment/domain/{__init__,instrument,scoring,state_machine,workflow_run,explanations,events}.py
src/sarjy/contexts/assessment/application/{__init__,ports,start_run,handle_turn,control_run,active_run_adapter,tools}.py
src/sarjy/contexts/assessment/infrastructure/{__init__,pg_run_repo,pg_instrument_repo,gemini_interpreter,gemini_narrator,memory_repos}.py
src/sarjy/contexts/assessment/infrastructure/prompts/{interpreter.md,narrative.md}
src/sarjy/interfaces/http/workflow.py
src/sarjy/interfaces/web/static/ocean.js
supabase/migrations/20260821000500_workflow_pending.sql
supabase/seed.sql
tests/unit/assessment/{test_instrument,test_scoring,test_workflow_run,test_handle_turn,test_active_run_adapter,test_tools,test_gemini_interpreter,test_gemini_narrator}.py
tests/unit/conversation/test_run_turn_direct_sentences.py
tests/integration/test_assessment_repos.py
tests/evals/ocean.jsonl
```

Modified: `src/sarjy/contexts/conversation/application/ports.py` (`ToolResult.direct_sentences`), `src/sarjy/contexts/conversation/application/run_turn.py`, `src/sarjy/container.py`, `src/sarjy/main.py`, `src/sarjy/interfaces/web/templates/index.html`, `tests/evals/run_evals.py`.

---

### Task 1: Domain — Instrument, scoring, state machine, WorkflowRun aggregate

**Files:**
- Create: `domain/__init__.py`, `domain/instrument.py`, `domain/scoring.py`, `domain/state_machine.py`, `domain/workflow_run.py`, `domain/events.py`, `domain/explanations.py`
- Test: `tests/unit/assessment/test_instrument.py`, `test_scoring.py`, `test_workflow_run.py`

**Interfaces:**
- `Item(no:int, trait:str, reverse:bool, text:str)`; `Instrument(id, version, scale_labels: list[str], traits: dict[str,str], bands: dict[str,tuple[float,float]], items: list[Item])` with `Instrument.from_definition(d: dict) -> Instrument`, `.total_items`, `.item(no) -> Item`, `.items_for_trait(code) -> list[Item]`.
- `TraitScore(code, name, score: float|None, band: str|None, answered: int)`, `ScoreReport(traits: list[TraitScore], answered: int, skipped: int)` with `.as_dict()` → `{"O":3.8,...,"bands":{...},"answered":..,"skipped":..}`; `score(instrument, answers: dict[int,int|None]) -> ScoreReport`; `band_for(instrument, value) -> str`.
- `Status` enum; `TRANSITIONS: dict[tuple[Status,str], Status]`; `IllegalTransition`, `TooManySkips` (subclass `DomainError`).
- `WorkflowRun(id, user_id, definition_id, definition_version, status, current_item, skips_used, results, narrative, started_at, updated_at, completed_at, pending_confirmation: dict|None, resume_hint: bool)` with `WorkflowRun.propose(...)`, `confirm(now)`, `record_answer(item_no, value, raw_text, confidence, total_items, now) -> AnswerRecorded`, `back(now)`, `pause(now)`, `resume(now)`, `quit(now)`, `begin_scoring(now)`, `finish_scoring(results, narrative, now)`, `set_pending(item_no, value, raw_text)`, `clear_pending()`, `.events: list[DomainEvent]`, `.is_finished_answering(total_items)`.
- `EXPLANATIONS: dict[int, str]` (20 entries).

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/assessment/test_instrument.py
from sarjy.contexts.assessment.domain.instrument import Instrument

DEF = {
    "id": "ocean_mini_ipip", "version": 1,
    "scale": {"min": 1, "max": 5, "labels": ["Very inaccurate", "Moderately inaccurate", "Neither", "Moderately accurate", "Very accurate"]},
    "traits": {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion", "A": "Agreeableness", "N": "Neuroticism"},
    "bands": {"low": [1, 2.4], "moderate": [2.5, 3.5], "high": [3.6, 5]},
    "items": [
        {"no": 1, "trait": "E", "reverse": False, "text": "I am the life of the party."},
        {"no": 2, "trait": "A", "reverse": False, "text": "I sympathize with others' feelings."},
        {"no": 6, "trait": "E", "reverse": True, "text": "I don't talk a lot."},
    ],
    "scoring": "mean",
}


def test_from_definition_parses_items_and_bands() -> None:
    ins = Instrument.from_definition(DEF)
    assert ins.id == "ocean_mini_ipip" and ins.total_items == 3
    assert ins.item(6).reverse is True and ins.item(1).trait == "E"
    assert ins.bands["moderate"] == (2.5, 3.5)
    assert [i.no for i in ins.items_for_trait("E")] == [1, 6]
    assert ins.scale_labels[4] == "Very accurate"
```

```python
# tests/unit/assessment/test_scoring.py
import json
from pathlib import Path

import pytest

from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.scoring import band_for, score

SEED = json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text())
INS = Instrument.from_definition(SEED)


def test_all_fives_gives_high_on_non_reversed_and_low_on_reversed_traits() -> None:
    # Every trait has 2 normal + 2 reversed items; 5 → 5 and 6-5=1 → mean 3.0 for all
    r = score(INS, {n: 5 for n in range(1, 21)})
    assert all(t.score == 3.0 and t.band == "moderate" for t in r.traits)


def test_hand_computed_extraversion() -> None:
    # E items: 1 (+), 6 (-), 11 (+), 16 (-). Answers 5,1,4,2 -> 5,5,4,4 -> mean 4.5 high
    answers = {n: 3 for n in range(1, 21)}
    answers.update({1: 5, 6: 1, 11: 4, 16: 2})
    e = next(t for t in score(INS, answers).traits if t.code == "E")
    assert e.score == 4.5 and e.band == "high"


def test_rounding_one_decimal() -> None:
    # A items 2(+),7(-),12(+),17(-): 4,2,4,3 -> 4,4,4,3 -> 3.75 -> 3.8
    answers = {n: 3 for n in range(1, 21)}
    answers.update({2: 4, 7: 2, 12: 4, 17: 3})
    a = next(t for t in score(INS, answers).traits if t.code == "A")
    assert a.score == 3.8


def test_skips_with_three_answered_still_scores() -> None:
    answers = {n: 3 for n in range(1, 21)}
    answers[5] = None  # O item skipped; O has 4 items → 3 answered
    r = score(INS, answers)
    o = next(t for t in r.traits if t.code == "O")
    assert o.score == 3.0 and o.answered == 3 and r.skipped == 1


def test_two_skips_in_one_trait_gives_none() -> None:
    answers = {n: 3 for n in range(1, 21)}
    answers[5] = None
    answers[10] = None
    o = next(t for t in score(INS, answers).traits if t.code == "O")
    assert o.score is None and o.band is None and o.answered == 2


@pytest.mark.parametrize("v,b", [(1.0, "low"), (2.4, "low"), (2.5, "moderate"), (3.5, "moderate"), (3.6, "high"), (5.0, "high"), (2.45, "moderate")])
def test_bands(v: float, b: str) -> None:
    assert band_for(INS, v) == b


def test_as_dict_shape() -> None:
    d = score(INS, {n: 4 for n in range(1, 21)}).as_dict()
    assert set(d) == {"O", "C", "E", "A", "N", "bands", "answered", "skipped"}
```

```python
# tests/unit/assessment/test_workflow_run.py
import uuid
from datetime import UTC, datetime

import pytest

from sarjy.contexts.assessment.domain.events import AnswerRecorded, RunCompleted, RunConfirmed
from sarjy.contexts.assessment.domain.workflow_run import IllegalTransition, Status, TooManySkips, WorkflowRun
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
    r = _run(); r.confirm(NOW)
    ev = r.record_answer(1, 4, "yeah mostly", 0.9, TOTAL, NOW)
    assert isinstance(ev, AnswerRecorded) and r.current_item == 2


def test_record_answer_wrong_item_is_illegal() -> None:
    r = _run(); r.confirm(NOW)
    with pytest.raises(IllegalTransition):
        r.record_answer(3, 4, "x", 0.9, TOTAL, NOW)


def test_skip_limit() -> None:
    r = _run(); r.confirm(NOW)
    r.record_answer(1, None, "skip", 1.0, TOTAL, NOW)
    r.record_answer(2, None, "skip", 1.0, TOTAL, NOW)
    with pytest.raises(TooManySkips):
        r.record_answer(3, None, "skip", 1.0, TOTAL, NOW)
    assert r.skips_used == 2 and r.current_item == 3


def test_back_decrements_but_not_below_one() -> None:
    r = _run(); r.confirm(NOW)
    r.record_answer(1, 3, "three", 1.0, TOTAL, NOW)
    r.back(NOW)
    assert r.current_item == 1
    with pytest.raises(IllegalTransition):
        r.back(NOW)


def test_pause_resume() -> None:
    r = _run(); r.confirm(NOW)
    r.pause(NOW)
    assert r.status is Status.PAUSED
    with pytest.raises(IllegalTransition):
        r.record_answer(1, 3, "x", 1.0, TOTAL, NOW)
    r.resume(NOW)
    assert r.status is Status.ACTIVE


def test_last_answer_moves_to_scoring_then_complete() -> None:
    r = _run(); r.confirm(NOW)
    for n in range(1, 21):
        r.record_answer(n, 3, "three", 1.0, TOTAL, NOW)
    assert r.is_finished_answering(TOTAL) and r.status is Status.ACTIVE
    r.begin_scoring(NOW)
    assert r.status is Status.SCORING
    r.finish_scoring({"O": 3.0}, "narrative", NOW)
    assert r.status is Status.COMPLETE and r.completed_at == NOW and isinstance(r.events[-1], RunCompleted)


def test_complete_is_terminal() -> None:
    r = _run(); r.confirm(NOW)
    for n in range(1, 21):
        r.record_answer(n, 3, "three", 1.0, TOTAL, NOW)
    r.begin_scoring(NOW); r.finish_scoring({}, "", NOW)
    for op in (lambda: r.pause(NOW), lambda: r.resume(NOW), lambda: r.confirm(NOW), lambda: r.quit(NOW)):
        with pytest.raises(IllegalTransition):
            op()


def test_pending_confirmation_roundtrip() -> None:
    r = _run(); r.confirm(NOW)
    r.set_pending(1, 4, "sort of")
    assert r.pending_confirmation == {"item_no": 1, "value": 4, "raw_text": "sort of"}
    r.clear_pending()
    assert r.pending_confirmation is None
```

Also create `supabase/mini_ipip.json` (the Appendix B JSON, used by both tests and `seed.sql` — written in Task 4 Step 1; for this task create it now with the full content from Task 4 Step 1).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/assessment -q` — Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# src/sarjy/contexts/assessment/domain/instrument.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Item:
    no: int
    trait: str
    reverse: bool
    text: str


@dataclass(frozen=True, slots=True)
class Instrument:
    id: str
    version: int
    scale_labels: list[str]
    traits: dict[str, str]
    bands: dict[str, tuple[float, float]]
    items: list[Item]

    @classmethod
    def from_definition(cls, d: dict[str, Any]) -> Instrument:
        items = [Item(int(i["no"]), str(i["trait"]), bool(i["reverse"]), str(i["text"])) for i in d["items"]]
        items.sort(key=lambda i: i.no)
        bands = {k: (float(v[0]), float(v[1])) for k, v in d["bands"].items()}
        return cls(id=str(d["id"]), version=int(d["version"]), scale_labels=list(d["scale"]["labels"]),
                   traits=dict(d["traits"]), bands=bands, items=items)

    @property
    def total_items(self) -> int:
        return len(self.items)

    def item(self, no: int) -> Item:
        for i in self.items:
            if i.no == no:
                return i
        raise KeyError(no)

    def items_for_trait(self, code: str) -> list[Item]:
        return [i for i in self.items if i.trait == code]
```

```python
# src/sarjy/contexts/assessment/domain/scoring.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sarjy.contexts.assessment.domain.instrument import Instrument

MIN_ANSWERED_PER_TRAIT = 3
SCALE_MAX_PLUS_ONE = 6  # reversal: 6 - v


@dataclass(frozen=True, slots=True)
class TraitScore:
    code: str
    name: str
    score: float | None
    band: str | None
    answered: int


@dataclass(frozen=True, slots=True)
class ScoreReport:
    traits: list[TraitScore]
    answered: int
    skipped: int

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {t.code: t.score for t in self.traits}
        d["bands"] = {t.code: t.band for t in self.traits}
        d["answered"] = self.answered
        d["skipped"] = self.skipped
        return d

    def trait(self, code: str) -> TraitScore:
        return next(t for t in self.traits if t.code == code)


def band_for(instrument: Instrument, value: float) -> str:
    # Bands are contiguous at 1 dp; a value between two bands (e.g. 2.45) rounds first.
    v = round(value, 1)
    for name, (lo, hi) in instrument.bands.items():
        if lo <= v <= hi:
            return name
    # fallback: nearest band by lower bound
    return min(instrument.bands.items(), key=lambda kv: abs(kv[1][0] - v))[0]


def score(instrument: Instrument, answers: dict[int, int | None]) -> ScoreReport:
    traits: list[TraitScore] = []
    answered_total = sum(1 for v in answers.values() if v is not None)
    skipped_total = sum(1 for v in answers.values() if v is None)
    for code, name in instrument.traits.items():
        vals: list[int] = []
        for item in instrument.items_for_trait(code):
            v = answers.get(item.no)
            if v is None:
                continue
            vals.append(SCALE_MAX_PLUS_ONE - v if item.reverse else v)
        if len(vals) < MIN_ANSWERED_PER_TRAIT:
            traits.append(TraitScore(code, name, None, None, len(vals)))
            continue
        mean = round(sum(vals) / len(vals), 1)
        traits.append(TraitScore(code, name, mean, band_for(instrument, mean), len(vals)))
    return ScoreReport(traits=traits, answered=answered_total, skipped=skipped_total)
```

```python
# src/sarjy/contexts/assessment/domain/state_machine.py
from __future__ import annotations

from enum import Enum


class Status(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    PAUSED = "paused"
    SCORING = "scoring"
    COMPLETE = "complete"
    ABANDONED = "abandoned"


# (current status, action) -> next status. Anything absent is illegal.
TRANSITIONS: dict[tuple[Status, str], Status] = {
    (Status.PROPOSED, "confirm"): Status.ACTIVE,
    (Status.PROPOSED, "quit"): Status.ABANDONED,
    (Status.ACTIVE, "answer"): Status.ACTIVE,
    (Status.ACTIVE, "back"): Status.ACTIVE,
    (Status.ACTIVE, "pause"): Status.PAUSED,
    (Status.ACTIVE, "quit"): Status.ABANDONED,
    (Status.ACTIVE, "score"): Status.SCORING,
    (Status.PAUSED, "resume"): Status.ACTIVE,
    (Status.PAUSED, "quit"): Status.ABANDONED,
    (Status.SCORING, "finish"): Status.COMPLETE,
}


def next_status(current: Status, action: str) -> Status | None:
    return TRANSITIONS.get((current, action))
```

```python
# src/sarjy/contexts/assessment/domain/events.py
from __future__ import annotations

from dataclasses import dataclass

from sarjy.shared.events import DomainEvent
from sarjy.shared.ids import RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class RunProposed(DomainEvent):
    run_id: RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfirmed(DomainEvent):
    run_id: RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerRecorded(DomainEvent):
    run_id: RunId
    item_no: int
    value: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunPaused(DomainEvent):
    run_id: RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class RunResumed(DomainEvent):
    run_id: RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class RunAbandoned(DomainEvent):
    run_id: RunId


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCompleted(DomainEvent):
    run_id: RunId
```

```python
# src/sarjy/contexts/assessment/domain/workflow_run.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sarjy.contexts.assessment.domain.events import (
    AnswerRecorded, RunAbandoned, RunCompleted, RunConfirmed, RunPaused, RunProposed, RunResumed,
)
from sarjy.contexts.assessment.domain.state_machine import Status, next_status
from sarjy.shared.errors import DomainError
from sarjy.shared.events import DomainEvent
from sarjy.shared.ids import RunId, UserId

MAX_SKIPS = 2


class IllegalTransition(DomainError):
    pass


class TooManySkips(DomainError):
    pass


@dataclass(slots=True)
class WorkflowRun:
    id: RunId
    user_id: UserId
    definition_id: str
    definition_version: int
    status: Status
    current_item: int
    skips_used: int
    results: dict[str, Any] | None
    narrative: str | None
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    pending_confirmation: dict[str, Any] | None = None
    resume_hint: bool = False
    events: list[DomainEvent] = field(default_factory=list)

    # -- factories ---------------------------------------------------------
    @classmethod
    def propose(cls, id: RunId, user_id: UserId, definition_id: str, version: int, now: datetime) -> WorkflowRun:
        r = cls(id=id, user_id=user_id, definition_id=definition_id, definition_version=version, status=Status.PROPOSED,
                current_item=1, skips_used=0, results=None, narrative=None, started_at=now, updated_at=now)
        r.events.append(RunProposed(run_id=id))
        return r

    # -- transitions -------------------------------------------------------
    def _go(self, action: str, now: datetime) -> None:
        nxt = next_status(self.status, action)
        if nxt is None:
            raise IllegalTransition(f"{self.status.value} -> {action}")
        self.status = nxt
        self.updated_at = now

    def confirm(self, now: datetime) -> None:
        self._go("confirm", now)
        self.events.append(RunConfirmed(run_id=self.id))

    def record_answer(self, item_no: int, value: int | None, raw_text: str, confidence: float,
                      total_items: int, now: datetime) -> AnswerRecorded:
        if self.status is not Status.ACTIVE:
            raise IllegalTransition(f"{self.status.value} -> answer")
        if item_no != self.current_item:
            raise IllegalTransition(f"expected item {self.current_item}, got {item_no}")
        if value is None:
            if self.skips_used >= MAX_SKIPS:
                raise TooManySkips(f"max {MAX_SKIPS} skips")
            self.skips_used += 1
        elif not 1 <= value <= 5:
            raise IllegalTransition(f"value out of range: {value}")
        self._go("answer", now)
        self.pending_confirmation = None
        self.resume_hint = False
        if self.current_item < total_items + 1:
            self.current_item += 1
        ev = AnswerRecorded(run_id=self.id, item_no=item_no, value=value)
        self.events.append(ev)
        return ev

    def back(self, now: datetime) -> None:
        if self.status is not Status.ACTIVE or self.current_item <= 1:
            raise IllegalTransition("cannot go back")
        self._go("back", now)
        self.current_item -= 1
        self.pending_confirmation = None

    def pause(self, now: datetime) -> None:
        self._go("pause", now)
        self.pending_confirmation = None
        self.events.append(RunPaused(run_id=self.id))

    def resume(self, now: datetime) -> None:
        self._go("resume", now)
        self.resume_hint = False
        self.events.append(RunResumed(run_id=self.id))

    def quit(self, now: datetime) -> None:
        self._go("quit", now)
        self.events.append(RunAbandoned(run_id=self.id))

    def begin_scoring(self, now: datetime) -> None:
        self._go("score", now)

    def finish_scoring(self, results: dict[str, Any], narrative: str, now: datetime) -> None:
        self._go("finish", now)
        self.results, self.narrative, self.completed_at = results, narrative, now
        self.events.append(RunCompleted(run_id=self.id))

    # -- helpers -----------------------------------------------------------
    def is_finished_answering(self, total_items: int) -> bool:
        return self.current_item > total_items

    def set_pending(self, item_no: int, value: int, raw_text: str) -> None:
        self.pending_confirmation = {"item_no": item_no, "value": value, "raw_text": raw_text}

    def clear_pending(self) -> None:
        self.pending_confirmation = None
```

```python
# src/sarjy/contexts/assessment/domain/explanations.py
"""One-sentence paraphrases for the `explain` control (PRD P-5)."""

EXPLANATIONS: dict[int, str] = {
    1: "It's asking whether you're usually the one bringing energy and fun to a group.",
    2: "It's asking whether you tend to feel for people when they're going through something.",
    3: "It's asking whether you handle tasks promptly instead of putting them off.",
    4: "It's asking whether your mood changes often and quickly.",
    5: "It's asking whether you daydream or picture things vividly in your mind.",
    6: "It's asking whether you're generally quiet and say little in conversation.",
    7: "It's asking whether other people's troubles don't really grab your attention.",
    8: "It's asking whether you tend to leave things lying around instead of putting them away.",
    9: "It's asking whether you feel calm and at ease most of the time.",
    10: "It's asking whether theoretical or abstract ideas don't interest you much.",
    11: "It's asking whether you chat with lots of different people at social events.",
    12: "It's asking whether you pick up on and share what other people are feeling.",
    13: "It's asking whether you prefer things to be tidy and organised.",
    14: "It's asking whether small things can upset you easily.",
    15: "It's asking whether abstract or theoretical ideas are hard for you to follow.",
    16: "It's asking whether you prefer to stay out of the spotlight.",
    17: "It's asking whether you're not especially curious about other people.",
    18: "It's asking whether things tend to go wrong or get messy when you handle them.",
    19: "It's asking whether you rarely feel down or sad.",
    20: "It's asking whether you don't think of yourself as very imaginative.",
}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/assessment -q && uv run mypy` — Expected: all passed; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(assessment): instrument, deterministic scoring, workflow state machine"
```

---

### Task 2: Application — ports, `StartRun`, `HandleAssessmentTurn`, `ControlRun`

**Files:**
- Create: `application/__init__.py`, `application/ports.py`, `application/start_run.py`, `application/handle_turn.py`, `application/control_run.py`, `infrastructure/memory_repos.py` (in-memory repos used by tests and by Container test helper)
- Test: `tests/unit/assessment/test_handle_turn.py`

**Interfaces:**
- `ports.py`:
  ```python
  Control = Literal["repeat","skip","back","explain","pause","quit","off_topic"]
  @dataclass(frozen=True) class Interpretation: value: int|None; confidence: float; control: Control|None
  class AnswerInterpreterPort(Protocol): async def interpret(self, item_text: str, scale_labels: list[str], user_text: str) -> Interpretation
  class NarratorPort(Protocol): async def narrate(self, report: ScoreReport) -> str
  class RunRepo(Protocol):
      async def get_open(self, user_id) -> WorkflowRun|None          # proposed/active/paused, newest
      async def get(self, run_id) -> WorkflowRun|None
      async def latest_complete(self, user_id) -> WorkflowRun|None
      async def save(self, run) -> None
      async def save_answer(self, run_id, item_no, raw_text, value, confidence) -> None
      async def answers(self, run_id) -> dict[int, int|None]
  class InstrumentRepo(Protocol): async def get(self, id: str) -> Instrument
  ```
- `StartRun(runs, instruments, clock).execute(user_id, definition_id="ocean_mini_ipip") -> AssessmentReply` — if an open run exists returns resume offer instead of creating another.
- `HandleAssessmentTurn(runs, instruments, interpreter, narrator, clock).execute(user_id, text) -> AssessmentReply | None`.
- `ControlRun(runs, instruments, clock).execute(user_id, action: Literal["resume","quit"]) -> AssessmentReply`.
- Shared sentence helpers in `handle_turn.py`: `item_sentence(ins, no) -> str` = `f"{no}: {text} How accurate is that for you?"` (first item says "One: …" using num2words ordinal-free words), `DISCLAIMER`, `results_sentences(ins, report, narrative) -> list[str]`.
- `AssessmentReply` and `workflow` dict = `{"status": str, "item": int, "total": int, "run_id": str}`.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/assessment/test_handle_turn.py
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sarjy.contexts.assessment.application.control_run import ControlRun
from sarjy.contexts.assessment.application.handle_turn import DISCLAIMER, HandleAssessmentTurn
from sarjy.contexts.assessment.application.ports import Interpretation
from sarjy.contexts.assessment.application.start_run import StartRun
from sarjy.contexts.assessment.domain.explanations import EXPLANATIONS
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.scoring import ScoreReport
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.infrastructure.memory_repos import MemInstrumentRepo, MemRunRepo
from sarjy.shared.clock import FakeClock
from sarjy.shared.ids import UserId

SEED = json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text())
INS = Instrument.from_definition(SEED)
U = UserId(uuid.uuid4())


class ScriptedInterpreter:
    """Maps exact user text → Interpretation; anything unknown is a confident 3."""

    TABLE: dict[str, Interpretation] = {
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

    async def interpret(self, item_text: str, scale_labels: list[str], user_text: str) -> Interpretation:
        self.calls.append(user_text)
        return self.TABLE.get(user_text.lower(), Interpretation(3, 0.9, None))


class FakeNarrator:
    async def narrate(self, report: ScoreReport) -> str:
        return "You come across as balanced. " * 2


def _sut() -> tuple[HandleAssessmentTurn, StartRun, ControlRun, MemRunRepo]:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    return (HandleAssessmentTurn(runs, ins, ScriptedInterpreter(), FakeNarrator(), clock),
            StartRun(runs, ins, clock), ControlRun(runs, ins, clock), runs)


async def test_no_run_returns_none() -> None:
    h, _, _, _ = _sut()
    assert await h.execute(U, "hello") is None


async def test_start_gives_intro_with_disclaimer_and_proposed_status() -> None:
    _, start, _, runs = _sut()
    reply = await start.execute(U)
    assert any(DISCLAIMER in s for s in reply.sentences)
    assert reply.workflow["status"] == "proposed" and reply.workflow["total"] == 20
    assert (await runs.get_open(U)).status is Status.PROPOSED  # type: ignore[union-attr]


async def test_confirm_reads_first_item() -> None:
    h, start, _, _ = _sut()
    await start.execute(U)
    reply = await h.execute(U, "yep")
    assert reply is not None and reply.sentences[0].startswith("One: I am the life of the party.")
    assert reply.workflow == {"status": "active", "item": 1, "total": 20, "run_id": reply.workflow["run_id"]}


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
    r1 = await h.execute(U, "nah")                      # item 1 = 2 → reads item 2
    assert r1.sentences[0].startswith("Two:")            # type: ignore[union-attr]
    r2 = await h.execute(U, "what does that mean?")      # explain item 2, re-ask
    assert EXPLANATIONS[2] in r2.sentences[0] and r2.workflow["item"] == 2  # type: ignore[union-attr]
    await h.execute(U, "yeah totally")                   # item 2 = 5 → item 3
    r3 = await h.execute(U, "go back")                   # back to item 2
    assert r3.workflow["item"] == 2 and r3.sentences[0].startswith("Two:")  # type: ignore[union-attr]
    await h.execute(U, "four")                           # item 2 = 4 (overwrites) → item 3
    r4 = await h.execute(U, "repeat")
    assert r4.sentences[0].startswith("Three:")          # type: ignore[union-attr]
    run = await runs.get_open(U)
    answers = await runs.answers(run.id)                 # type: ignore[union-attr]
    assert answers == {1: 2, 2: 4}


async def test_low_confidence_asks_for_confirmation_then_records() -> None:
    h, start, _, runs = _sut()
    await start.execute(U); await h.execute(U, "yes")
    r = await h.execute(U, "sort of")
    assert r is not None and "I'll put that as a 4" in r.sentences[0] and r.workflow["item"] == 1
    r2 = await h.execute(U, "yes")
    assert r2 is not None and r2.workflow["item"] == 2
    run = await runs.get_open(U)
    assert (await runs.answers(run.id)) == {1: 4}        # type: ignore[union-attr]


async def test_low_confidence_rejected_reasks() -> None:
    h, start, _, _ = _sut()
    await start.execute(U); await h.execute(U, "yes")
    await h.execute(U, "sort of")
    r = await h.execute(U, "no")
    assert r is not None and r.workflow["item"] == 1 and "one to five" in r.sentences[0].lower()


async def test_unintelligible_reasks_with_scale_hint() -> None:
    h, start, _, _ = _sut()
    await start.execute(U); await h.execute(U, "yes")
    r = await h.execute(U, "mumble")
    assert r is not None and "one to five" in r.sentences[0].lower() and r.workflow["item"] == 1


async def test_skip_limit_message() -> None:
    h, start, _, _ = _sut()
    await start.execute(U); await h.execute(U, "yes")
    await h.execute(U, "skip"); await h.execute(U, "skip")
    r = await h.execute(U, "skip")
    assert r is not None and "two skips" in r.sentences[0].lower() and r.workflow["item"] == 3


async def test_pause_and_resume_across_calls() -> None:
    h, start, control, runs = _sut()
    await start.execute(U); await h.execute(U, "yes"); await h.execute(U, "three")
    r = await h.execute(U, "let's stop for now")
    assert r is not None and r.workflow["status"] == "paused"
    assert await h.execute(U, "hello again") is None      # paused run does not intercept chat
    r2 = await control.execute(U, "resume")
    assert r2.workflow["status"] == "active" and r2.sentences[-1].startswith("Two:")


async def test_off_topic_returns_none_and_sets_resume_hint() -> None:
    h, start, _, runs = _sut()
    await start.execute(U); await h.execute(U, "yes")
    assert await h.execute(U, "what's the weather in rome") is None
    run = await runs.get_open(U)
    assert run is not None and run.resume_hint is True and run.status is Status.ACTIVE


async def test_quit_requires_confirmation() -> None:
    h, start, _, runs = _sut()
    await start.execute(U); await h.execute(U, "yes")
    r = await h.execute(U, "quit the test")
    assert r is not None and "sure" in r.sentences[0].lower() and r.workflow["status"] == "active"
    r2 = await h.execute(U, "yes")
    assert r2 is not None and r2.workflow["status"] == "abandoned" and await runs.get_open(U) is None


async def test_full_run_scores_and_completes() -> None:
    h, start, _, runs = _sut()
    await start.execute(U); await h.execute(U, "yes")
    reply = None
    for _ in range(20):
        reply = await h.execute(U, "four")
    assert reply is not None and reply.workflow["status"] == "complete"
    joined = " ".join(reply.sentences)
    assert "Extraversion" in joined and DISCLAIMER in joined
    done = await runs.latest_complete(U)
    assert done is not None and done.results["E"] == 3.0 and done.narrative  # 4 and 6-4=2 → mean 3.0
    assert await h.execute(U, "hi") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/assessment/test_handle_turn.py -q` — Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# src/sarjy/contexts/assessment/application/ports.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.scoring import ScoreReport
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.shared.ids import RunId, UserId

Control = Literal["repeat", "skip", "back", "explain", "pause", "quit", "off_topic"]


@dataclass(frozen=True, slots=True)
class Interpretation:
    value: int | None
    confidence: float
    control: Control | None


class AnswerInterpreterPort(Protocol):
    async def interpret(self, item_text: str, scale_labels: list[str], user_text: str) -> Interpretation: ...


class NarratorPort(Protocol):
    async def narrate(self, report: ScoreReport) -> str: ...


class RunRepo(Protocol):
    async def get_open(self, user_id: UserId) -> WorkflowRun | None: ...
    async def get(self, run_id: RunId) -> WorkflowRun | None: ...
    async def latest_complete(self, user_id: UserId) -> WorkflowRun | None: ...
    async def save(self, run: WorkflowRun) -> None: ...
    async def save_answer(self, run_id: RunId, item_no: int, raw_text: str, value: int | None, confidence: float) -> None: ...
    async def answers(self, run_id: RunId) -> dict[int, int | None]: ...


class InstrumentRepo(Protocol):
    async def get(self, id: str) -> Instrument: ...
```

```python
# src/sarjy/contexts/assessment/infrastructure/memory_repos.py
from __future__ import annotations

import copy

from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.shared.ids import RunId, UserId

OPEN = {Status.PROPOSED, Status.ACTIVE, Status.PAUSED}


class MemRunRepo:
    def __init__(self) -> None:
        self.runs: dict[RunId, WorkflowRun] = {}
        self.answers_by_run: dict[RunId, dict[int, tuple[str, int | None, float]]] = {}

    async def get_open(self, user_id: UserId) -> WorkflowRun | None:
        c = [r for r in self.runs.values() if r.user_id == user_id and r.status in OPEN]
        return copy.deepcopy(max(c, key=lambda r: r.updated_at)) if c else None

    async def get(self, run_id: RunId) -> WorkflowRun | None:
        r = self.runs.get(run_id)
        return copy.deepcopy(r) if r else None

    async def latest_complete(self, user_id: UserId) -> WorkflowRun | None:
        c = [r for r in self.runs.values() if r.user_id == user_id and r.status is Status.COMPLETE]
        return copy.deepcopy(max(c, key=lambda r: r.completed_at or r.updated_at)) if c else None

    async def save(self, run: WorkflowRun) -> None:
        self.runs[run.id] = copy.deepcopy(run)

    async def save_answer(self, run_id: RunId, item_no: int, raw_text: str, value: int | None, confidence: float) -> None:
        self.answers_by_run.setdefault(run_id, {})[item_no] = (raw_text, value, confidence)

    async def answers(self, run_id: RunId) -> dict[int, int | None]:
        return {k: v[1] for k, v in self.answers_by_run.get(run_id, {}).items()}


class MemInstrumentRepo:
    def __init__(self, items: dict[str, Instrument]) -> None:
        self.items = items

    async def get(self, id: str) -> Instrument:
        return self.items[id]
```

```python
# src/sarjy/contexts/assessment/application/start_run.py
from __future__ import annotations

from sarjy.contexts.assessment.application.handle_turn import DISCLAIMER, workflow_dict
from sarjy.contexts.assessment.application.ports import InstrumentRepo, RunRepo
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.contexts.conversation.application.ports import AssessmentReply
from sarjy.shared.clock import Clock
from sarjy.shared.ids import RunId, UserId, new_id

INTRO = ("Sure — this is the Big Five, twenty quick statements about you, about five minutes. "
         "For each one, tell me how accurate it is on a one-to-five scale.")


class StartRun:
    def __init__(self, runs: RunRepo, instruments: InstrumentRepo, clock: Clock) -> None:
        self.runs, self.instruments, self.clock = runs, instruments, clock

    async def execute(self, user_id: UserId, definition_id: str = "ocean_mini_ipip") -> AssessmentReply:
        ins = await self.instruments.get(definition_id)
        existing = await self.runs.get_open(user_id)
        if existing is not None:
            if existing.status is Status.PAUSED:
                return AssessmentReply(
                    [f"You have a paused test at item {existing.current_item} of {ins.total_items}. Say continue to pick it up, or quit to start over."],
                    workflow_dict(existing, ins.total_items))
            if existing.status is Status.ACTIVE:
                return AssessmentReply([f"We're already on item {existing.current_item}. Ready to continue?"],
                                       workflow_dict(existing, ins.total_items))
            return AssessmentReply(["I'd already offered the test — ready to start?"], workflow_dict(existing, ins.total_items))
        run = WorkflowRun.propose(new_id(RunId), user_id, ins.id, ins.version, self.clock.now())
        await self.runs.save(run)
        return AssessmentReply([INTRO, DISCLAIMER, "Ready?"], workflow_dict(run, ins.total_items))
```

```python
# src/sarjy/contexts/assessment/application/control_run.py
from __future__ import annotations

from typing import Literal

from sarjy.contexts.assessment.application.handle_turn import item_sentence, workflow_dict
from sarjy.contexts.assessment.application.ports import InstrumentRepo, RunRepo
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.conversation.application.ports import AssessmentReply
from sarjy.shared.clock import Clock
from sarjy.shared.ids import UserId


class ControlRun:
    def __init__(self, runs: RunRepo, instruments: InstrumentRepo, clock: Clock) -> None:
        self.runs, self.instruments, self.clock = runs, instruments, clock

    async def execute(self, user_id: UserId, action: Literal["resume", "quit"]) -> AssessmentReply:
        run = await self.runs.get_open(user_id)
        if run is None:
            return AssessmentReply(["There's no personality test in progress. Want to start one?"],
                                   {"status": "none", "item": 0, "total": 0, "run_id": ""})
        ins = await self.instruments.get(run.definition_id)
        now = self.clock.now()
        if action == "quit":
            run.quit(now)
            await self.runs.save(run)
            return AssessmentReply(["Okay, I've stopped the test. We can start fresh any time."], workflow_dict(run, ins.total_items))
        if run.status is Status.PAUSED:
            run.resume(now)
            await self.runs.save(run)
            return AssessmentReply([f"Picking up at item {run.current_item}.", item_sentence(ins, run.current_item)],
                                   workflow_dict(run, ins.total_items))
        if run.status is Status.PROPOSED:
            return AssessmentReply(["Say yes when you're ready to start."], workflow_dict(run, ins.total_items))
        return AssessmentReply([item_sentence(ins, run.current_item)], workflow_dict(run, ins.total_items))
```

```python
# src/sarjy/contexts/assessment/application/handle_turn.py
from __future__ import annotations

import re
from typing import Any

from num2words import num2words

from sarjy.contexts.assessment.application.ports import AnswerInterpreterPort, InstrumentRepo, Interpretation, NarratorPort, RunRepo
from sarjy.contexts.assessment.domain.explanations import EXPLANATIONS
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.scoring import ScoreReport, score
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.domain.workflow_run import TooManySkips, WorkflowRun
from sarjy.contexts.conversation.application.ports import AssessmentReply
from sarjy.shared.clock import Clock
from sarjy.shared.ids import UserId

DISCLAIMER = "This is a well-known research questionnaire for self-reflection, not a clinical or diagnostic tool."
SCALE_HINT = "Tell me how accurate that is on a scale of one to five, one meaning very inaccurate and five very accurate."
CONFIRM_THRESHOLD = 0.7
PENDING_QUIT = "__quit__"

_YES = re.compile(r"^\s*(y|yes|yep|yeah|yup|sure|ok|okay|ready|let'?s go|go ahead|correct|right|that'?s right|continue)\b", re.I)
_NO = re.compile(r"^\s*(n|no|nope|nah|not now|later|stop|cancel|wrong|incorrect)\b", re.I)


def workflow_dict(run: WorkflowRun, total: int) -> dict[str, Any]:
    return {"status": run.status.value, "item": min(run.current_item, total), "total": total, "run_id": str(run.id)}


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
    def __init__(self, runs: RunRepo, instruments: InstrumentRepo, interpreter: AnswerInterpreterPort,
                 narrator: NarratorPort, clock: Clock) -> None:
        self.runs, self.instruments, self.interpreter, self.narrator, self.clock = runs, instruments, interpreter, narrator, clock

    async def execute(self, user_id: UserId, text: str) -> AssessmentReply | None:
        run = await self.runs.get_open(user_id)
        if run is None or run.status is Status.PAUSED:
            return None  # paused runs are resumed only via explicit control (tool)
        ins = await self.instruments.get(run.definition_id)
        total = ins.total_items
        now = self.clock.now()

        if run.status is Status.PROPOSED:
            if _YES.search(text):
                run.confirm(now)
                await self.runs.save(run)
                return AssessmentReply([item_sentence(ins, 1)], workflow_dict(run, total))
            if _NO.search(text):
                run.quit(now)
                await self.runs.save(run)
                return AssessmentReply(["No problem — we can do it another time."], workflow_dict(run, total))
            return None  # unrelated chat while proposed; let normal path answer

        # ---- ACTIVE ----
        if run.pending_confirmation is not None:
            return await self._resolve_pending(run, ins, text, now)

        item = ins.item(run.current_item)
        interp = await self.interpreter.interpret(item.text, ins.scale_labels, text)
        if interp.control is not None:
            return await self._control(run, ins, interp, now)
        if interp.value is None:
            return AssessmentReply([SCALE_HINT, item_sentence(ins, run.current_item)], workflow_dict(run, total))
        if interp.confidence < CONFIRM_THRESHOLD:
            run.set_pending(run.current_item, interp.value, text)
            await self.runs.save(run)
            label = ins.scale_labels[interp.value - 1].lower()
            return AssessmentReply([f"I'll put that as a {interp.value} — {label}. Right?"], workflow_dict(run, total))
        return await self._record(run, ins, interp.value, text, interp.confidence, now)

    # ------------------------------------------------------------------
    async def _resolve_pending(self, run: WorkflowRun, ins: Instrument, text: str, now: Any) -> AssessmentReply:
        pending = run.pending_confirmation or {}
        total = ins.total_items
        if pending.get("item_no") == PENDING_QUIT:
            run.clear_pending()
            if _YES.search(text):
                run.quit(now)
                await self.runs.save(run)
                return AssessmentReply(["Okay, I've stopped the test. We can start fresh any time."], workflow_dict(run, total))
            await self.runs.save(run)
            return AssessmentReply(["Great, let's keep going.", item_sentence(ins, run.current_item)], workflow_dict(run, total))
        if _YES.search(text):
            return await self._record(run, ins, int(pending["value"]), str(pending["raw_text"]), 1.0, now)
        run.clear_pending()
        await self.runs.save(run)
        return AssessmentReply([SCALE_HINT, item_sentence(ins, run.current_item)], workflow_dict(run, total))

    async def _control(self, run: WorkflowRun, ins: Instrument, interp: Interpretation, now: Any) -> AssessmentReply | None:
        total = ins.total_items
        c = interp.control
        if c == "repeat":
            return AssessmentReply([item_sentence(ins, run.current_item)], workflow_dict(run, total))
        if c == "explain":
            return AssessmentReply([EXPLANATIONS.get(run.current_item, "It's asking how well that statement describes you."),
                                    "How accurate is that for you?"], workflow_dict(run, total))
        if c == "back":
            if run.current_item <= 1:
                return AssessmentReply(["We're on the first one already.", item_sentence(ins, 1)], workflow_dict(run, total))
            run.back(now)
            await self.runs.save(run)
            return AssessmentReply(["Okay, going back.", item_sentence(ins, run.current_item)], workflow_dict(run, total))
        if c == "skip":
            try:
                run.record_answer(run.current_item, None, "skip", 1.0, total, now)
            except TooManySkips:
                return AssessmentReply(["You've used your two skips — give this one your best guess.",
                                        item_sentence(ins, run.current_item)], workflow_dict(run, total))
            await self.runs.save_answer(run.id, run.current_item - 1, "skip", None, 1.0)
            await self.runs.save(run)
            return await self._after_advance(run, ins, now, prefix="Skipped.")
        if c == "pause":
            run.pause(now)
            await self.runs.save(run)
            return AssessmentReply([f"Paused at item {run.current_item}. Say continue the personality test whenever you're ready."],
                                   workflow_dict(run, total))
        if c == "quit":
            run.pending_confirmation = {"item_no": PENDING_QUIT}
            await self.runs.save(run)
            return AssessmentReply(["Are you sure you want to stop the test? Your answers so far will be discarded."],
                                   workflow_dict(run, total))
        # off_topic
        run.resume_hint = True
        await self.runs.save(run)
        return None

    async def _record(self, run: WorkflowRun, ins: Instrument, value: int, raw: str, conf: float, now: Any) -> AssessmentReply:
        item_no = run.current_item
        run.record_answer(item_no, value, raw, conf, ins.total_items, now)
        await self.runs.save_answer(run.id, item_no, raw, value, conf)
        await self.runs.save(run)
        return await self._after_advance(run, ins, now)

    async def _after_advance(self, run: WorkflowRun, ins: Instrument, now: Any, prefix: str | None = None) -> AssessmentReply:
        total = ins.total_items
        if not run.is_finished_answering(total):
            s = item_sentence(ins, run.current_item)
            return AssessmentReply([f"{prefix} {s}"] if prefix else [s], workflow_dict(run, total))
        run.begin_scoring(now)
        await self.runs.save(run)
        report = score(ins, await self.runs.answers(run.id))
        narrative = await self.narrator.narrate(report)
        run.finish_scoring(report.as_dict(), narrative, self.clock.now())
        await self.runs.save(run)
        return AssessmentReply(results_sentences(ins, report, narrative), workflow_dict(run, total))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/assessment -q && uv run mypy` — Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(assessment): start/handle/control use cases with in-memory repos"
```

---

### Task 3: `ActiveRunAdapter`, workflow tools, and `ToolResult.direct_sentences`

**Files:**
- Create: `application/active_run_adapter.py`, `application/tools.py`
- Modify: `src/sarjy/contexts/conversation/application/ports.py` (`ToolResult`), `src/sarjy/contexts/conversation/application/run_turn.py`
- Test: `tests/unit/assessment/test_active_run_adapter.py`, `tests/unit/assessment/test_tools.py`, `tests/unit/conversation/test_run_turn_direct_sentences.py`

**Interfaces:**
- `ActiveRunAdapter(runs, instruments, handle_turn)` implements `ActiveRunPort`: `active_run(user_id) -> ActiveRunSnapshot|None` (for PROPOSED/ACTIVE/PAUSED runs; `prompt_block` per PRD §10 block 8, plus the resume hint sentence when `run.resume_hint`); `handle_turn(user_id, text)` delegates.
- `StartWorkflowTool(start_run)` and `WorkflowControlTool(control_run)` implement `ToolPort`; both return `ToolResult(ok=True, data={"sentences": [...], "workflow": {...}}, direct_sentences=[...])`.
- `ToolResult` gains `direct_sentences: list[str] | None = None`. `RunTurn`: after a tool returns `direct_sentences`, yield each as `SentenceEvent`, set `DoneEvent.workflow = res.data.get("workflow")`, and **end the turn without another LLM hop**.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/assessment/test_active_run_adapter.py
import json, uuid
from datetime import UTC, datetime
from pathlib import Path

from sarjy.contexts.assessment.application.active_run_adapter import ActiveRunAdapter
from sarjy.contexts.assessment.application.handle_turn import HandleAssessmentTurn
from sarjy.contexts.assessment.application.start_run import StartRun
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.infrastructure.memory_repos import MemInstrumentRepo, MemRunRepo
from sarjy.shared.clock import FakeClock
from sarjy.shared.ids import UserId
from tests.unit.assessment.test_handle_turn import FakeNarrator, ScriptedInterpreter

INS = Instrument.from_definition(json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text()))
U = UserId(uuid.uuid4())


def _sut():  # type: ignore[no-untyped-def]
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    h = HandleAssessmentTurn(runs, ins, ScriptedInterpreter(), FakeNarrator(), clock)
    return ActiveRunAdapter(runs, ins, h), StartRun(runs, ins, clock), runs


async def test_snapshot_none_without_run() -> None:
    a, _, _ = _sut()
    assert await a.active_run(U) is None


async def test_snapshot_block_and_resume_hint() -> None:
    a, start, _ = _sut()
    await start.execute(U); await a.handle_turn(U, "yes")
    snap = await a.active_run(U)
    assert snap is not None and snap.current_item == 1 and snap.total_items == 20 and "item 1 of 20" in snap.prompt_block
    assert "Ready to continue" not in snap.prompt_block
    assert await a.handle_turn(U, "what's the weather in rome") is None
    snap2 = await a.active_run(U)
    assert snap2 is not None and "Ready to continue? We were on item 1." in snap2.prompt_block
```

```python
# tests/unit/assessment/test_tools.py
import json, uuid
from datetime import UTC, datetime
from pathlib import Path

from sarjy.contexts.assessment.application.control_run import ControlRun
from sarjy.contexts.assessment.application.start_run import StartRun
from sarjy.contexts.assessment.application.tools import StartWorkflowTool, WorkflowControlTool
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.infrastructure.memory_repos import MemInstrumentRepo, MemRunRepo
from sarjy.shared.clock import FakeClock
from sarjy.shared.ids import UserId

INS = Instrument.from_definition(json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text()))
U = UserId(uuid.uuid4())


async def test_start_tool_returns_direct_sentences() -> None:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    t = StartWorkflowTool(StartRun(runs, ins, clock))
    assert t.name == "start_workflow" and t.declaration["parameters"]["properties"]["workflow_id"]["enum"] == ["ocean_mini_ipip"]
    res = await t.invoke(U, {"workflow_id": "ocean_mini_ipip"})
    assert res.ok and res.direct_sentences and res.direct_sentences[-1] == "Ready?" and res.data["workflow"]["status"] == "proposed"


async def test_start_tool_rejects_unknown_workflow() -> None:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    t = StartWorkflowTool(StartRun(runs, ins, FakeClock(datetime(2026, 8, 21, tzinfo=UTC))))
    res = await t.invoke(U, {"workflow_id": "nope"})
    assert not res.ok and res.spoken_error


async def test_control_tool_quit_without_run() -> None:
    runs, ins = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    t = WorkflowControlTool(ControlRun(runs, ins, FakeClock(datetime(2026, 8, 21, tzinfo=UTC))))
    res = await t.invoke(U, {"action": "quit"})
    assert res.ok and "no personality test" in res.direct_sentences[0].lower()  # type: ignore[index]
```

```python
# tests/unit/conversation/test_run_turn_direct_sentences.py
import uuid

from sarjy.contexts.conversation.application.ports import FunctionCall, LLMFinished, LLMFunctionCall, ToolResult
from sarjy.contexts.conversation.application.tool_router import ToolRouter
from sarjy.contexts.conversation.domain.events import DoneEvent, SentenceEvent
from sarjy.contexts.conversation.domain.turn import TurnInput
from sarjy.shared.ids import UserId
from tests.unit.conversation.test_run_turn import FakeLLM, _make


class DirectTool:
    name = "start_workflow"
    declaration = {"name": "start_workflow", "description": "d", "parameters": {"type": "object", "properties": {"workflow_id": {"type": "string"}}}}

    async def invoke(self, user_id: UserId, args: dict) -> ToolResult:  # type: ignore[type-arg]
        return ToolResult(ok=True, data={"workflow": {"status": "proposed", "item": 1, "total": 20}}, direct_sentences=["Intro.", "Ready?"])


async def test_direct_sentences_end_turn_without_second_llm_hop() -> None:
    llm = FakeLLM([[LLMFunctionCall(FunctionCall("start_workflow", {"workflow_id": "ocean_mini_ipip"})), LLMFinished("stop")]])
    tools = ToolRouter(); tools.register(DirectTool())
    rt, msgs = _make(llm, tools)
    events = [e async for e in rt(TurnInput(UserId(uuid.uuid4()), None, "t-ds", "give me a personality test"))]
    sents = [e.sentence.text for e in events if isinstance(e, SentenceEvent)]
    assert sents == ["Intro.", "Ready?"]
    assert len(llm.requests) == 1                       # no second hop
    done = events[-1]
    assert isinstance(done, DoneEvent) and done.workflow == {"status": "proposed", "item": 1, "total": 20}
    assert msgs.items[-1].content == "Intro. Ready?"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/assessment/test_active_run_adapter.py tests/unit/assessment/test_tools.py tests/unit/conversation/test_run_turn_direct_sentences.py -q` — Expected: ImportError / TypeError on `direct_sentences`.

- [ ] **Step 3: Implement**

Edit `src/sarjy/contexts/conversation/application/ports.py` — `ToolResult`:

```python
@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    data: dict[str, Any]
    grounding_numbers: tuple[float, ...] = ()
    spoken_error: str | None = None
    spoken_summary: str | None = None          # added in Phase 5
    direct_sentences: list[str] | None = None  # Phase 6: speak verbatim, end turn
```

Edit `src/sarjy/contexts/conversation/application/run_turn.py` inside the tool-hop loop, immediately after `yield ToolStatusEvent(pending_call.name, "end", ok=res.ok)`:

```python
            if res.direct_sentences is not None:
                for s_ in res.direct_sentences:
                    yield SentenceEvent(Sentence(idx, s_, to_speech(s_))); sentences.append(s_); idx += 1
                direct_workflow = res.data.get("workflow")
                break
```

and declare `direct_workflow: dict[str, Any] | None = None` before the loop, then change the final `yield DoneEvent(mid, t.as_dict())` to `yield DoneEvent(mid, t.as_dict(), workflow=direct_workflow)`. Because we `break` before appending `function_call`/`function_response` messages, no second LLM hop occurs.

```python
# src/sarjy/contexts/assessment/application/active_run_adapter.py
from __future__ import annotations

from sarjy.contexts.assessment.application.handle_turn import HandleAssessmentTurn
from sarjy.contexts.assessment.application.ports import InstrumentRepo, RunRepo
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.conversation.application.ports import ActiveRunSnapshot, AssessmentReply
from sarjy.shared.ids import UserId

_BLOCK = {
    Status.PROPOSED: "Active: Big Five personality test has been offered and is awaiting the user's yes/no. "
                     "If they ask something else, answer briefly, then ask if they'd like to start.",
    Status.ACTIVE: "Active: Big Five test, item {item} of {total}. The user's answers are handled by the test engine. "
                   "If they ask something else, answer it briefly and do not invent test items or scores.",
    Status.PAUSED: "Paused: Big Five test at item {item} of {total}. If the user wants to continue, call workflow_control "
                   "with action resume; if they want to stop, call it with action quit.",
}


class ActiveRunAdapter:
    def __init__(self, runs: RunRepo, instruments: InstrumentRepo, handle_turn: HandleAssessmentTurn) -> None:
        self.runs, self.instruments, self.handle = runs, instruments, handle_turn

    async def active_run(self, user_id: UserId) -> ActiveRunSnapshot | None:
        run = await self.runs.get_open(user_id)
        if run is None:
            return None
        ins = await self.instruments.get(run.definition_id)
        item = min(run.current_item, ins.total_items)
        block = _BLOCK[run.status].format(item=item, total=ins.total_items)
        if run.resume_hint and run.status is Status.ACTIVE:
            block += f' After answering, end with exactly: "Ready to continue? We were on item {item}."'
        return ActiveRunSnapshot(run_id=run.id, definition_id=run.definition_id, status=run.status.value,
                                 current_item=item, total_items=ins.total_items, prompt_block=block)

    async def handle_turn(self, user_id: UserId, text: str) -> AssessmentReply | None:
        return await self.handle.execute(user_id, text)

    def snapshot_from_row(self, row: dict[str, Any]) -> ActiveRunSnapshot | None:
        """Build a snapshot from the `workflow` JSON returned by the load_turn_context RPC (Phase 7).

        Avoids a second DB round-trip: the instrument size is read from the cached definition
        (`self.instruments.cached(definition_id)`, a sync accessor backed by the in-process cache
        that `PgInstrumentRepo` fills on first `get`). Falls back to 20 items if not cached yet.
        """
        if not row or row.get("status") not in ("proposed", "active", "paused"):
            return None
        status = Status(row["status"])
        ins = self.instruments.cached(row["definition_id"])
        total = ins.total_items if ins else 20
        item = min(int(row.get("current_item", 1)), total)
        block = _BLOCK[status].format(item=item, total=total)
        if row.get("resume_hint") and status is Status.ACTIVE:
            block += f' After answering, end with exactly: "Ready to continue? We were on item {item}."'
        return ActiveRunSnapshot(run_id=RunId(uuid.UUID(row["id"])), definition_id=row["definition_id"], status=status.value,
                                 current_item=item, total_items=total, prompt_block=block)
```

Add to `InstrumentRepo` port and both implementations: `def cached(self, id: str) -> Instrument | None` (in-memory dict populated by `get`). Add `import uuid` and `from sarjy.shared.ids import RunId` to the adapter module. `NoActiveRun` (Phase 2) gets `def snapshot_from_row(self, row): return None`. Unit test:

```python
def test_snapshot_from_row_builds_block_without_db() -> None:
    adapter = ActiveRunAdapter(MemRunRepo(), MemInstrumentRepo(INSTRUMENT), handle_turn=None)  # type: ignore[arg-type]
    snap = adapter.snapshot_from_row({"id": str(uuid.uuid4()), "definition_id": "ocean_mini_ipip", "status": "active", "current_item": 7, "resume_hint": False})
    assert snap is not None and snap.current_item == 7 and "item 7 of 20" in snap.prompt_block
    assert adapter.snapshot_from_row({"status": "complete"}) is None
```

```python
# src/sarjy/contexts/assessment/application/tools.py
from __future__ import annotations

from typing import Any

from sarjy.contexts.assessment.application.control_run import ControlRun
from sarjy.contexts.assessment.application.start_run import StartRun
from sarjy.contexts.conversation.application.ports import ToolResult
from sarjy.shared.ids import UserId

KNOWN_WORKFLOWS = ("ocean_mini_ipip",)


class StartWorkflowTool:
    name = "start_workflow"
    declaration: dict[str, Any] = {
        "name": "start_workflow",
        "description": "Begin a multistep flow. Currently only 'ocean_mini_ipip' (Big Five personality test). Call when the user asks for a personality test.",
        "parameters": {"type": "object", "properties": {"workflow_id": {"type": "string", "enum": ["ocean_mini_ipip"]}},
                       "required": ["workflow_id"]},
    }

    def __init__(self, start_run: StartRun) -> None:
        self.start_run = start_run

    async def invoke(self, user_id: UserId, args: dict[str, Any]) -> ToolResult:
        wid = str(args.get("workflow_id", ""))
        if wid not in KNOWN_WORKFLOWS:
            return ToolResult(ok=False, data={"error": "unknown_workflow"}, spoken_error="I only have the Big Five personality test right now.")
        reply = await self.start_run.execute(user_id, wid)
        return ToolResult(ok=True, data={"sentences": reply.sentences, "workflow": reply.workflow}, direct_sentences=list(reply.sentences))


class WorkflowControlTool:
    name = "workflow_control"
    declaration: dict[str, Any] = {
        "name": "workflow_control",
        "description": "Control an active workflow: continue after a pause, or quit.",
        "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["resume", "quit"]}}, "required": ["action"]},
    }

    def __init__(self, control_run: ControlRun) -> None:
        self.control_run = control_run

    async def invoke(self, user_id: UserId, args: dict[str, Any]) -> ToolResult:
        action = args.get("action")
        if action not in ("resume", "quit"):
            return ToolResult(ok=False, data={"error": "bad_action"}, spoken_error="I can continue or quit the test — which one?")
        reply = await self.control_run.execute(user_id, action)  # type: ignore[arg-type]
        return ToolResult(ok=True, data={"sentences": reply.sentences, "workflow": reply.workflow}, direct_sentences=list(reply.sentences))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit -q && uv run mypy` — Expected: all passed (including the Phase 2/5 RunTurn tests, unchanged behaviour when `direct_sentences is None`).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(assessment): ActiveRunAdapter, workflow tools, direct_sentences tool channel"
```

---

### Task 4: Infrastructure — Postgres repos, Gemini interpreter & narrator, seed & migration

**Files:**
- Create: `supabase/mini_ipip.json`, `supabase/seed.sql`, `supabase/migrations/20260821000500_workflow_pending.sql`, `infrastructure/pg_run_repo.py`, `infrastructure/pg_instrument_repo.py`, `infrastructure/gemini_interpreter.py`, `infrastructure/gemini_narrator.py`, `infrastructure/prompts/interpreter.md`, `infrastructure/prompts/narrative.md`
- Test: `tests/unit/assessment/test_gemini_interpreter.py`, `test_gemini_narrator.py`, `tests/integration/test_assessment_repos.py`

**Interfaces:**
- `PgRunRepo(db)` implements `RunRepo`; `PgInstrumentRepo(db)` implements `InstrumentRepo` (caches definitions in-process for 10 min).
- `GeminiAnswerInterpreter(llm: LLMPort)` implements `AnswerInterpreterPort` via `generate_json(..., InterpretationOut)`.
- `GeminiNarrator(llm: LLMPort)` implements `NarratorPort`; `fallback_narrative(report) -> str` deterministic.

- [ ] **Step 1: Seed JSON, seed.sql, migration**

`supabase/mini_ipip.json` (Appendix B, verbatim):
```json
{
  "id": "ocean_mini_ipip",
  "version": 1,
  "scale": {"min": 1, "max": 5, "labels": ["Very inaccurate", "Moderately inaccurate", "Neither", "Moderately accurate", "Very accurate"]},
  "traits": {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion", "A": "Agreeableness", "N": "Neuroticism"},
  "bands": {"low": [1, 2.4], "moderate": [2.5, 3.5], "high": [3.6, 5]},
  "items": [
    {"no": 1, "trait": "E", "reverse": false, "text": "I am the life of the party."},
    {"no": 2, "trait": "A", "reverse": false, "text": "I sympathize with others' feelings."},
    {"no": 3, "trait": "C", "reverse": false, "text": "I get chores done right away."},
    {"no": 4, "trait": "N", "reverse": false, "text": "I have frequent mood swings."},
    {"no": 5, "trait": "O", "reverse": false, "text": "I have a vivid imagination."},
    {"no": 6, "trait": "E", "reverse": true, "text": "I don't talk a lot."},
    {"no": 7, "trait": "A", "reverse": true, "text": "I am not interested in other people's problems."},
    {"no": 8, "trait": "C", "reverse": true, "text": "I often forget to put things back in their proper place."},
    {"no": 9, "trait": "N", "reverse": true, "text": "I am relaxed most of the time."},
    {"no": 10, "trait": "O", "reverse": true, "text": "I am not interested in abstract ideas."},
    {"no": 11, "trait": "E", "reverse": false, "text": "I talk to a lot of different people at parties."},
    {"no": 12, "trait": "A", "reverse": false, "text": "I feel others' emotions."},
    {"no": 13, "trait": "C", "reverse": false, "text": "I like order."},
    {"no": 14, "trait": "N", "reverse": false, "text": "I get upset easily."},
    {"no": 15, "trait": "O", "reverse": true, "text": "I have difficulty understanding abstract ideas."},
    {"no": 16, "trait": "E", "reverse": true, "text": "I keep in the background."},
    {"no": 17, "trait": "A", "reverse": true, "text": "I am not really interested in others."},
    {"no": 18, "trait": "C", "reverse": true, "text": "I make a mess of things."},
    {"no": 19, "trait": "N", "reverse": true, "text": "I seldom feel blue."},
    {"no": 20, "trait": "O", "reverse": true, "text": "I do not have a good imagination."}
  ],
  "scoring": "trait = mean(value if !reverse else 6 - value) over answered items; require >= 3 answered per trait"
}
```

`supabase/seed.sql`:
```sql
insert into public.workflow_definitions (id, version, definition, active)
values ('ocean_mini_ipip', 1, $json$
{"id":"ocean_mini_ipip","version":1,
 "scale":{"min":1,"max":5,"labels":["Very inaccurate","Moderately inaccurate","Neither","Moderately accurate","Very accurate"]},
 "traits":{"O":"Openness","C":"Conscientiousness","E":"Extraversion","A":"Agreeableness","N":"Neuroticism"},
 "bands":{"low":[1,2.4],"moderate":[2.5,3.5],"high":[3.6,5]},
 "items":[
  {"no":1,"trait":"E","reverse":false,"text":"I am the life of the party."},
  {"no":2,"trait":"A","reverse":false,"text":"I sympathize with others' feelings."},
  {"no":3,"trait":"C","reverse":false,"text":"I get chores done right away."},
  {"no":4,"trait":"N","reverse":false,"text":"I have frequent mood swings."},
  {"no":5,"trait":"O","reverse":false,"text":"I have a vivid imagination."},
  {"no":6,"trait":"E","reverse":true,"text":"I don't talk a lot."},
  {"no":7,"trait":"A","reverse":true,"text":"I am not interested in other people's problems."},
  {"no":8,"trait":"C","reverse":true,"text":"I often forget to put things back in their proper place."},
  {"no":9,"trait":"N","reverse":true,"text":"I am relaxed most of the time."},
  {"no":10,"trait":"O","reverse":true,"text":"I am not interested in abstract ideas."},
  {"no":11,"trait":"E","reverse":false,"text":"I talk to a lot of different people at parties."},
  {"no":12,"trait":"A","reverse":false,"text":"I feel others' emotions."},
  {"no":13,"trait":"C","reverse":false,"text":"I like order."},
  {"no":14,"trait":"N","reverse":false,"text":"I get upset easily."},
  {"no":15,"trait":"O","reverse":true,"text":"I have difficulty understanding abstract ideas."},
  {"no":16,"trait":"E","reverse":true,"text":"I keep in the background."},
  {"no":17,"trait":"A","reverse":true,"text":"I am not really interested in others."},
  {"no":18,"trait":"C","reverse":true,"text":"I make a mess of things."},
  {"no":19,"trait":"N","reverse":true,"text":"I seldom feel blue."},
  {"no":20,"trait":"O","reverse":true,"text":"I do not have a good imagination."}
 ],
 "scoring":"trait = mean(value if !reverse else 6 - value) over answered items; require >= 3 answered per trait"}
$json$::jsonb, true)
on conflict (id) do update set version = excluded.version, definition = excluded.definition, active = excluded.active;
```

`supabase/migrations/20260821000500_workflow_pending.sql`:
```sql
alter table public.workflow_runs
  add column if not exists pending_confirmation jsonb,
  add column if not exists resume_hint boolean not null default false;
```

Run: `supabase db reset` (applies migrations + `seed.sql`). Expected: no errors; `select id from workflow_definitions` returns `ocean_mini_ipip`.

- [ ] **Step 2: Prompts**

`infrastructure/prompts/interpreter.md`:
```
You convert a spoken answer to a personality-questionnaire item into structured data. The scale is 1–5:
1 = Very inaccurate, 2 = Moderately inaccurate, 3 = Neither accurate nor inaccurate, 4 = Moderately accurate, 5 = Very accurate.
You receive the item statement and the user's words. Return JSON {"value": 1-5 or null, "confidence": 0.0-1.0, "control": one of repeat, skip, back, explain, pause, quit, off_topic, or null}.
Rules:
- If the user gives a number word or digit 1–5, value is that number, confidence 1.0.
- If the user expresses agreement/disagreement in natural language, map its strength: strong agreement → 5, mild → 4, neutral/unsure → 3 (confidence ≤ 0.6), mild disagreement → 2, strong → 1.
- If the user asks to hear it again → control repeat. Asks to skip → skip. Asks to go back → back. Asks what it means → explain. Wants to stop for now → pause. Wants to quit/abandon → quit.
- If the words are unrelated to the item (a different question or request, e.g. weather, what time is it) → control off_topic.
- If you cannot tell, value null, confidence below 0.3, control null.
Examples:
"totally me" → {"value":5,"confidence":0.95,"control":null}
"yeah pretty much" → {"value":4,"confidence":0.9,"control":null}
"sort of" → {"value":3,"confidence":0.6,"control":null}
"not really" → {"value":2,"confidence":0.9,"control":null}
"not at all" → {"value":1,"confidence":0.95,"control":null}
"three" → {"value":3,"confidence":1.0,"control":null}
"repeat that" → {"value":null,"confidence":1.0,"control":"repeat"}
"skip" → {"value":null,"confidence":1.0,"control":"skip"}
"go back" → {"value":null,"confidence":1.0,"control":"back"}
"what does that mean" → {"value":null,"confidence":1.0,"control":"explain"}
"let's stop for now" → {"value":null,"confidence":1.0,"control":"pause"}
"what's the weather" → {"value":null,"confidence":1.0,"control":"off_topic"}
```

`infrastructure/prompts/narrative.md`:
```
You write a short, warm, non-judgmental reflection on a person's Big Five results for a voice assistant to read aloud.
You receive five trait scores (1–5, one decimal) with bands (low/moderate/high). Some may be "not scored".
Write exactly four sentences, plain prose, no lists, no markdown, no emojis. Mention the two or three most notable traits by name with their exact score as written (for example "3.8"); never round, change, or invent numbers; never diagnose, never use words like disorder, problem, or bad. Close with one encouraging sentence. Do not mention that this is not clinical — the assistant adds that separately.
```

- [ ] **Step 3: Write failing tests**

```python
# tests/unit/assessment/test_gemini_interpreter.py
from sarjy.contexts.assessment.infrastructure.gemini_interpreter import GeminiAnswerInterpreter, InterpretationOut


class FakeLLM:
    def __init__(self) -> None:
        self.req = None

    async def generate_json(self, req, schema):  # type: ignore[no-untyped-def]
        self.req = req
        assert schema is InterpretationOut
        return InterpretationOut(value=4, confidence=0.9, control=None)

    def stream(self, req):  # type: ignore[no-untyped-def]
        raise NotImplementedError


async def test_interpreter_maps_and_uses_temperature_zero() -> None:
    llm = FakeLLM()
    out = await GeminiAnswerInterpreter(llm).interpret("I like order.", ["a", "b", "c", "d", "e"], "yeah mostly")  # type: ignore[arg-type]
    assert out.value == 4 and out.control is None
    assert llm.req.temperature == 0 and "I like order." in llm.req.messages[-1].text  # type: ignore[union-attr]


async def test_interpreter_value_out_of_range_becomes_none() -> None:
    class Bad(FakeLLM):
        async def generate_json(self, req, schema):  # type: ignore[no-untyped-def]
            return InterpretationOut(value=9, confidence=0.9, control=None)
    out = await GeminiAnswerInterpreter(Bad()).interpret("x", [], "y")  # type: ignore[arg-type]
    assert out.value is None and out.confidence == 0.0
```

```python
# tests/unit/assessment/test_gemini_narrator.py
import json
from pathlib import Path
from types import SimpleNamespace

from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.scoring import score
from sarjy.contexts.assessment.infrastructure.gemini_narrator import GeminiNarrator, fallback_narrative

INS = Instrument.from_definition(json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text()))
answers = {n: 3 for n in range(1, 21)}
answers.update({1: 5, 6: 1, 11: 4, 16: 2})  # E = 4.5
REPORT = score(INS, answers)


class TextLLM:
    def __init__(self, texts: list[str]) -> None:
        self.texts, self.calls = texts, 0

    async def generate_json(self, req, schema):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def stream(self, req):  # type: ignore[no-untyped-def]
        from sarjy.contexts.conversation.application.ports import LLMFinished, LLMText
        self.calls += 1
        yield LLMText(self.texts.pop(0)); yield LLMFinished("stop")


async def test_narrative_accepted_when_numbers_match() -> None:
    llm = TextLLM(["Your Extraversion of 4.5 stands out. You enjoy company. Others sit at 3.0. Keep being you."])
    n = await GeminiNarrator(llm).narrate(REPORT)  # type: ignore[arg-type]
    assert "4.5" in n and llm.calls == 1


async def test_narrative_regenerates_once_then_falls_back() -> None:
    llm = TextLLM(["Your Extraversion of 4.7 stands out.", "Extraversion is 4.9, wow."])
    n = await GeminiNarrator(llm).narrate(REPORT)  # type: ignore[arg-type]
    assert llm.calls == 2 and n == fallback_narrative(REPORT) and "4.5" in n
```

```python
# tests/integration/test_assessment_repos.py
import os, uuid
from datetime import UTC, datetime

import pytest

from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.contexts.assessment.infrastructure.pg_instrument_repo import PgInstrumentRepo
from sarjy.contexts.assessment.infrastructure.pg_run_repo import PgRunRepo
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import RunId, UserId

pytestmark = pytest.mark.integration


async def test_run_repo_roundtrip_and_answers() -> None:
    db = Database(os.environ["DATABASE_URL_DIRECT"]); await db.connect()
    u = UserId(uuid.uuid4())
    await db.execute("insert into auth.users (id,email) values ($1,$2)", u, f"{u}@x.test")
    ins = await PgInstrumentRepo(db).get("ocean_mini_ipip")
    assert ins.total_items == 20
    repo = PgRunRepo(db)
    now = datetime.now(UTC)
    run = WorkflowRun.propose(RunId(uuid.uuid4()), u, ins.id, ins.version, now)
    await repo.save(run)
    run.confirm(now); run.set_pending(1, 4, "sort of"); await repo.save(run)
    loaded = await repo.get_open(u)
    assert loaded is not None and loaded.status is Status.ACTIVE and loaded.pending_confirmation == {"item_no": 1, "value": 4, "raw_text": "sort of"}
    run.record_answer(1, 4, "sort of", 1.0, 20, now)
    await repo.save_answer(run.id, 1, "sort of", 4, 1.0)
    await repo.save_answer(run.id, 1, "four", 4, 1.0)  # overwrite after 'back'
    await repo.save(run)
    assert await repo.answers(run.id) == {1: 4}
    await db.close()
```

- [ ] **Step 4: Implement**

```python
# src/sarjy/contexts/assessment/infrastructure/pg_instrument_repo.py
from __future__ import annotations

import json
import time

from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.errors import NotFound

_TTL_S = 600


class PgInstrumentRepo:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._cache: dict[str, tuple[float, Instrument]] = {}

    async def get(self, id: str) -> Instrument:
        hit = self._cache.get(id)
        if hit and time.monotonic() - hit[0] < _TTL_S:
            return hit[1]
        row = await self.db.fetchrow("select definition from workflow_definitions where id=$1 and active", id)
        if row is None:
            raise NotFound(f"workflow_definition {id}")
        d = row["definition"]
        ins = Instrument.from_definition(json.loads(d) if isinstance(d, str) else d)
        self._cache[id] = (time.monotonic(), ins)
        return ins
```

```python
# src/sarjy/contexts/assessment/infrastructure/pg_run_repo.py
from __future__ import annotations

import json
from typing import Any

import asyncpg

from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.ids import RunId, UserId

_COLS = "id,user_id,definition_id,definition_version,status,current_item,skips_used,results,narrative,started_at,updated_at,completed_at,pending_confirmation,resume_hint"


def _j(v: Any) -> Any:
    return json.loads(v) if isinstance(v, str) else v


def _row(r: asyncpg.Record) -> WorkflowRun:
    return WorkflowRun(id=RunId(r["id"]), user_id=UserId(r["user_id"]), definition_id=r["definition_id"],
                       definition_version=r["definition_version"], status=Status(r["status"]), current_item=r["current_item"],
                       skips_used=r["skips_used"], results=_j(r["results"]), narrative=r["narrative"], started_at=r["started_at"],
                       updated_at=r["updated_at"], completed_at=r["completed_at"], pending_confirmation=_j(r["pending_confirmation"]),
                       resume_hint=r["resume_hint"])


class PgRunRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def get_open(self, user_id: UserId) -> WorkflowRun | None:
        r = await self.db.fetchrow(
            f"select {_COLS} from workflow_runs where user_id=$1 and status in ('proposed','active','paused') order by updated_at desc limit 1", user_id)
        return _row(r) if r else None

    async def get(self, run_id: RunId) -> WorkflowRun | None:
        r = await self.db.fetchrow(f"select {_COLS} from workflow_runs where id=$1", run_id)
        return _row(r) if r else None

    async def latest_complete(self, user_id: UserId) -> WorkflowRun | None:
        r = await self.db.fetchrow(
            f"select {_COLS} from workflow_runs where user_id=$1 and status='complete' order by completed_at desc limit 1", user_id)
        return _row(r) if r else None

    async def save(self, run: WorkflowRun) -> None:
        await self.db.execute(
            """insert into workflow_runs (id,user_id,definition_id,definition_version,status,current_item,skips_used,results,narrative,
                                          started_at,updated_at,completed_at,pending_confirmation,resume_hint)
               values ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9,$10,$11,$12,$13::jsonb,$14)
               on conflict (id) do update set status=excluded.status, current_item=excluded.current_item, skips_used=excluded.skips_used,
                 results=excluded.results, narrative=excluded.narrative, updated_at=excluded.updated_at, completed_at=excluded.completed_at,
                 pending_confirmation=excluded.pending_confirmation, resume_hint=excluded.resume_hint""",
            run.id, run.user_id, run.definition_id, run.definition_version, run.status.value, run.current_item, run.skips_used,
            json.dumps(run.results) if run.results is not None else None, run.narrative, run.started_at, run.updated_at,
            run.completed_at, json.dumps(run.pending_confirmation) if run.pending_confirmation is not None else None, run.resume_hint)

    async def save_answer(self, run_id: RunId, item_no: int, raw_text: str, value: int | None, confidence: float) -> None:
        await self.db.execute(
            """insert into workflow_answers (run_id,item_no,raw_text,value,confidence) values ($1,$2,$3,$4,$5)
               on conflict (run_id,item_no) do update set raw_text=excluded.raw_text, value=excluded.value,
                 confidence=excluded.confidence, answered_at=now()""",
            run_id, item_no, raw_text, value, confidence)

    async def answers(self, run_id: RunId) -> dict[int, int | None]:
        rows = await self.db.fetch("select item_no, value from workflow_answers where run_id=$1", run_id)
        return {r["item_no"]: r["value"] for r in rows}
```

```python
# src/sarjy/contexts/assessment/infrastructure/gemini_interpreter.py
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from sarjy.contexts.assessment.application.ports import Control, Interpretation
from sarjy.contexts.conversation.application.ports import LLMMessage, LLMPort, LLMRequest

_PROMPT = (Path(__file__).parent / "prompts" / "interpreter.md").read_text(encoding="utf-8")


class InterpretationOut(BaseModel):
    value: int | None = None
    confidence: float = 0.0
    control: Control | None = None


class GeminiAnswerInterpreter:
    def __init__(self, llm: LLMPort) -> None:
        self.llm = llm

    async def interpret(self, item_text: str, scale_labels: list[str], user_text: str) -> Interpretation:
        user = f"Item: {item_text}\nUser said: {user_text}"
        req = LLMRequest(system=_PROMPT, messages=[LLMMessage(role="user", text=user)], tools=[], temperature=0.0, max_output_tokens=60)
        out = await self.llm.generate_json(req, InterpretationOut)
        value = out.value if out.value is not None and 1 <= out.value <= 5 else None
        conf = max(0.0, min(1.0, out.confidence)) if value is not None or out.control else 0.0
        return Interpretation(value=value, confidence=conf, control=out.control)
```

```python
# src/sarjy/contexts/assessment/infrastructure/gemini_narrator.py
from __future__ import annotations

import re
from pathlib import Path

from sarjy.contexts.assessment.domain.scoring import ScoreReport
from sarjy.contexts.conversation.application.ports import LLMMessage, LLMPort, LLMRequest, LLMText

_PROMPT = (Path(__file__).parent / "prompts" / "narrative.md").read_text(encoding="utf-8")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def fallback_narrative(report: ScoreReport) -> str:
    scored = [t for t in report.traits if t.score is not None]
    if not scored:
        return "There weren't enough answers to describe a pattern this time. Feel free to try again whenever you like."
    top = sorted(scored, key=lambda t: abs(t.score - 3.0), reverse=True)[:2]  # type: ignore[operator]
    parts = [f"Your most distinctive trait is {top[0].name} at {top[0].score:.1f}, which sits in the {top[0].band} range."]
    if len(top) > 1:
        parts.append(f"{top[1].name} also stands out at {top[1].score:.1f}, in the {top[1].band} range.")
    parts.append("The rest of your scores sit closer to the middle, which is very common.")
    parts.append("These are tendencies, not labels, and you get to decide what to do with them.")
    return " ".join(parts)


class GeminiNarrator:
    def __init__(self, llm: LLMPort) -> None:
        self.llm = llm

    async def narrate(self, report: ScoreReport) -> str:
        lines = [f"{t.name}: {t.score:.1f} ({t.band})" if t.score is not None else f"{t.name}: not scored" for t in report.traits]
        req = LLMRequest(system=_PROMPT, messages=[LLMMessage(role="user", text="\n".join(lines))], tools=[],
                         temperature=0.7, max_output_tokens=400, thinking_budget=1024)
        for _ in range(2):
            text = "".join([ev.text async for ev in self.llm.stream(req) if isinstance(ev, LLMText)]).strip()
            if self._numbers_ok(text, report):
                return text
        return fallback_narrative(report)

    @staticmethod
    def _numbers_ok(text: str, report: ScoreReport) -> bool:
        allowed = {f"{t.score:.1f}" for t in report.traits if t.score is not None}
        found = set(_NUM.findall(text))
        return bool(found) and all(n in allowed or n in {"1", "5", "one", "five"} for n in found)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/assessment -q && make test-integration && uv run mypy` — Expected: all passed.

- [ ] **Step 6: Commit**

```bash
supabase db push
git add -A && git commit -m "feat(assessment): postgres repos, gemini interpreter and narrator, mini-ipip seed"
```

---

### Task 5: Web — `/workflow/latest` endpoint, item scale UI, results card

**Files:**
- Create: `src/sarjy/interfaces/http/workflow.py`, `src/sarjy/interfaces/web/static/ocean.js`
- Modify: `src/sarjy/interfaces/web/templates/index.html`, `src/sarjy/interfaces/web/static/app.css`, `src/sarjy/main.py`
- Test: `tests/unit/interfaces/test_workflow_http.py`

**Interfaces:**
- `GET /workflow/latest` → `WorkflowLatest{run_id, status, completed_at, results: {O,C,E,A,N,bands,...}, narrative, traits: [{code,name,score,band}]}` or 404 `{"detail":"no_completed_run"}`.
- `Container.assessment_runs: RunRepo`, `Container.assessment_instruments: InstrumentRepo` (set in Task 6; the router reads them from `request.app.state.container`).
- `ocean.js` listens for a custom DOM event `sarjy:done` dispatched by `voice.js` with `detail = data` of the `done` SSE event (add one line to `voice.js` in the `done` case: `document.dispatchEvent(new CustomEvent("sarjy:done", {detail: data}))`). It also listens for `sarjy:sentence` (same one-liner in the `sentence` case) to detect item prompts.

- [ ] **Step 1: Failing test**

```python
# tests/unit/interfaces/test_workflow_http.py
import json, time, uuid
from datetime import UTC, datetime
from pathlib import Path

import jwt
from fastapi.testclient import TestClient

from sarjy.config import Settings
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.contexts.assessment.infrastructure.memory_repos import MemInstrumentRepo, MemRunRepo
from sarjy.main import create_app
from sarjy.shared.ids import RunId, UserId

SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"
INS = Instrument.from_definition(json.loads((Path(__file__).parents[2] / "supabase" / "mini_ipip.json").read_text()))


def _tok(u: uuid.UUID) -> str:
    return jwt.encode({"sub": str(u), "aud": "authenticated", "exp": int(time.time()) + 60}, SECRET, algorithm="HS256")


def test_latest_404_then_200() -> None:
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    c.assessment_runs, c.assessment_instruments = MemRunRepo(), MemInstrumentRepo({INS.id: INS})
    u = uuid.uuid4()
    with TestClient(app) as client:
        assert client.get("/workflow/latest", headers={"Authorization": f"Bearer {_tok(u)}"}).status_code == 404
        now = datetime(2026, 8, 21, tzinfo=UTC)
        run = WorkflowRun.propose(RunId(uuid.uuid4()), UserId(u), INS.id, 1, now)
        run.confirm(now)
        for n in range(1, 21):
            run.record_answer(n, 4, "four", 1.0, 20, now)
        run.begin_scoring(now)
        run.finish_scoring({"O": 3.0, "C": 3.0, "E": 3.0, "A": 3.0, "N": 3.0, "bands": {k: "moderate" for k in "OCEAN"}, "answered": 20, "skipped": 0}, "Nice.", now)
        import asyncio; asyncio.get_event_loop().run_until_complete(c.assessment_runs.save(run))
        r = client.get("/workflow/latest", headers={"Authorization": f"Bearer {_tok(u)}"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete" and body["traits"][2] == {"code": "E", "name": "Extraversion", "score": 3.0, "band": "moderate"}
```

- [ ] **Step 2: Run → fails.** **Step 3: Implement**

```python
# src/sarjy/interfaces/http/workflow.py
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from sarjy.interfaces.http.auth import CurrentUserDep

router = APIRouter(prefix="/workflow")


class TraitOut(BaseModel):
    code: str
    name: str
    score: float | None
    band: str | None


class WorkflowLatest(BaseModel):
    run_id: str
    status: str
    completed_at: datetime | None
    results: dict[str, Any]
    narrative: str | None
    traits: list[TraitOut]


@router.get("/latest", response_model=WorkflowLatest)
async def latest(user: CurrentUserDep, request: Request) -> WorkflowLatest:
    c = request.app.state.container
    run = await c.assessment_runs.latest_complete(user.user_id)
    if run is None or run.results is None:
        raise HTTPException(status_code=404, detail="no_completed_run")
    ins = await c.assessment_instruments.get(run.definition_id)
    bands = run.results.get("bands", {})
    traits = [TraitOut(code=code, name=name, score=run.results.get(code), band=bands.get(code)) for code, name in ins.traits.items()]
    return WorkflowLatest(run_id=str(run.id), status=run.status.value, completed_at=run.completed_at,
                          results=run.results, narrative=run.narrative, traits=traits)
```

`main.py`: `app.include_router(workflow.router)`.

`index.html` additions (inside `<main>`, after `#transcript`):
```html
  <section id="ocean-item" hidden>
    <p id="ocean-item-text"></p>
    <div class="scale" role="group" aria-label="Answer from one to five">
      <button data-v="1">1</button><button data-v="2">2</button><button data-v="3">3</button><button data-v="4">4</button><button data-v="5">5</button>
    </div>
    <small>1 = very inaccurate · 5 = very accurate</small>
  </section>
  <section id="ocean-results" hidden>
    <h2>Your Big Five</h2>
    <div id="ocean-bars"></div>
    <p id="ocean-narrative"></p>
    <p class="disclaimer">This is a well-known research questionnaire for self-reflection, not a clinical or diagnostic tool.</p>
  </section>
```
and `<script type="module" src="/static/ocean.js"></script>` after `voice.js`.

`voice.js` — two one-line additions in `handle()`: in `case "sentence":` append `document.dispatchEvent(new CustomEvent("sarjy:sentence", { detail: data }));` and in `case "done":` append `document.dispatchEvent(new CustomEvent("sarjy:done", { detail: data }));`. Also expose `window.sarjySend = (t) => { bubble("user", t); controller.marks.speech_end = performance.now(); controller.send(t, "text"); };`.

```javascript
// src/sarjy/interfaces/web/static/ocean.js
const $ = (id) => document.getElementById(id);
const itemBox = $("ocean-item"), itemText = $("ocean-item-text"), results = $("ocean-results"), bars = $("ocean-bars"), narrative = $("ocean-narrative");
const ITEM_RE = /^(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty): (.+?) How accurate is that for you\?$/;

document.addEventListener("sarjy:sentence", (e) => {
  const m = ITEM_RE.exec(e.detail.text || "");
  if (m) { itemText.textContent = m[2]; itemBox.hidden = false; results.hidden = true; }
});

document.addEventListener("sarjy:done", async (e) => {
  const wf = e.detail.workflow;
  if (!wf) return;
  if (wf.status !== "active") itemBox.hidden = true;
  if (wf.status === "complete") await renderResults();
});

itemBox.querySelectorAll("button[data-v]").forEach((b) => b.addEventListener("click", () => window.sarjySend(b.dataset.v)));

async function renderResults() {
  const { data: { session } } = await window.sb.auth.getSession();
  const r = await fetch("/workflow/latest", { headers: { Authorization: `Bearer ${session.access_token}` } });
  if (!r.ok) return;
  const body = await r.json();
  bars.innerHTML = "";
  for (const t of body.traits) {
    const row = document.createElement("div"); row.className = "bar-row";
    const pct = t.score == null ? 0 : ((t.score - 1) / 4) * 100;
    row.innerHTML = `<span class="bar-label">${t.name}</span><div class="bar"><div class="bar-fill" style="width:${pct}%"></div></div><span class="bar-val">${t.score == null ? "n/a" : `${t.score.toFixed(1)} · ${t.band}`}</span>`;
    bars.append(row);
  }
  narrative.textContent = body.narrative || "";
  results.hidden = false;
}
```
(`voice.js` must expose its client as `window.sb = sb;` — add that line after creating the client.)

`app.css` additions: `.scale button` large tappable squares (min 44px), `.bar-row` grid `140px 1fr 120px`, `.bar` 10px rounded track, `.bar-fill` accent colour, `.disclaimer` muted small text.

- [ ] **Step 4: Run** `uv run pytest tests/unit/interfaces -q` → pass; manual: start test in browser, confirm scale buttons submit numbers and results card renders after item 20.
- [ ] **Step 5: Commit** `feat(web): OCEAN item scale and results card, /workflow/latest`.

---

### Task 6: Container wiring and OCEAN eval suite

**Files:**
- Modify: `src/sarjy/container.py`
- Create: `tests/evals/ocean.jsonl`; modify `tests/evals/run_evals.py`

- [ ] **Step 1: Container additions** (in `__post_init__`, after guardrails wiring, before the final `rebuild_run_turn()`):

```python
        # --- Assessment context
        self.assessment_runs = PgRunRepo(self.db) if self.connect_db else MemRunRepo()
        self.assessment_instruments = PgInstrumentRepo(self.db) if self.connect_db else MemInstrumentRepo({})
        interp_llm = GeminiLLM(self.settings.gemini_api_key, self.settings.gemini_guard_model, first_token_timeout_s=2.0)
        interpreter = GeminiAnswerInterpreter(interp_llm)
        narrator = GeminiNarrator(self.llm)
        handle = HandleAssessmentTurn(self.assessment_runs, self.assessment_instruments, interpreter, narrator, self.clock)
        start = StartRun(self.assessment_runs, self.assessment_instruments, self.clock)
        control = ControlRun(self.assessment_runs, self.assessment_instruments, self.clock)
        self.active_run = ActiveRunAdapter(self.assessment_runs, self.assessment_instruments, handle)
        self.tools.register(StartWorkflowTool(start))
        self.tools.register(WorkflowControlTool(control))
        self.rebuild_run_turn()
```
Add `assessment_runs: RunRepo | None = None` and `assessment_instruments: InstrumentRepo | None = None` dataclass fields so tests can override before `__post_init__` runs (or set after and call `rebuild_run_turn()`).

- [ ] **Step 2: Eval file** `tests/evals/ocean.jsonl` — 10 rows, each `{"id":"oc-01","answers":[...20 strings...],"expected":{"O":..,"C":..,"E":..,"A":..,"N":..}}`. Compute expectations by hand with the reversal rule; include these three explicitly (the remaining seven vary phrasings of 1–5 and one includes two skips):

```json
{"id":"oc-01","answers":["four","four","four","four","four","four","four","four","four","four","four","four","four","four","four","four","four","four","four","four"],"expected":{"O":3.0,"C":3.0,"E":3.0,"A":3.0,"N":3.0}}
{"id":"oc-02","answers":["yeah totally","five","five","one","five","not at all","one","one","five","one","five","five","five","one","one","one","one","one","five","one"],"expected":{"O":5.0,"C":5.0,"E":5.0,"A":5.0,"N":1.0}}
{"id":"oc-03","answers":["three","three","three","three","skip","three","three","three","three","skip","three","three","three","three","three","three","three","three","three","three"],"expected":{"O":null,"C":3.0,"E":3.0,"A":3.0,"N":3.0}}
```
(oc-02 check, E items 1,6,11,16 = 5, 1→5, 5, 1→5 → 5.0; N items 4,9,14,19 = 1, 5→1, 1, 5→1 → 1.0; oc-03: O items 5 and 10 skipped → 2 answered → null.)

- [ ] **Step 3: Runner extension** — in `run_evals.py` add `run_ocean_suite()`: for each row, new user JWT, send "give me a personality test", then "yes", then each answer in order via `/chat`; after the last, read `GET /workflow/latest` and assert `results[code] == expected[code]` for all five; pass criterion 100 %. Append to the summary table and exit code.

- [ ] **Step 4: Run** `uv run pytest -q && make evals` — Expected: unit green; OCEAN suite 10/10 (interpreter variance on "yeah totally" etc. is tolerated only if the final score matches; if a phrasing flakes, tighten the few-shots in `interpreter.md`).
- [ ] **Step 5: Commit** `feat: wire assessment context; ocean eval suite (10/10)`.

---

### Task 7: Deploy and manual acceptance test P

- [ ] `supabase db push` (migration 000500 + seed on staging: run `psql "$DATABASE_URL_DIRECT" -f supabase/seed.sql`), `fly deploy`.
- [ ] Run PRD §7.5 acceptance test by voice on staging: start the test, answer 7 items with "nah", "yeah totally", "four", "what does that mean?", "go back", "three"; say "let's stop"; close the tab; open a new tab; say "continue the personality test" → resumes at the right item; finish; verify the five scores against `workflow_answers` by hand.
- [ ] Commit `chore: phase 6 staging deploy`.

---

## Phase 6 self-review

- Spec coverage: P-1 ✔ (StartRun intro + confirm), P-2 ✔ (item read verbatim + scale UI), P-3 ✔ (interpreter schema + few-shots), P-4 ✔ (pending confirmation, re-ask with hint), P-5 ✔ (all six controls, skip limit, quit confirmation), P-6 ✔ (immediate `save_answer`, `get_open` resume, proactive offer in `StartRun` when paused), P-7 ✔ (off_topic → None + `resume_hint` in prompt block), P-8 ✔ (`score()` deterministic; narrator numeric post-check + fallback), P-9 ✔ (results sentences + card; follow-up Q&A is grounded because results are in `workflow_runs` and the prompt block can be extended in Phase 7 to include them for COMPLETE runs — noted as P1), P-10 ✔, P-11 ✔ (`latest_complete`, old runs retained), P-12 ✔ (engine is generic over `Instrument`; adding a workflow = new definition row + interpreter prompt). §9.3 declarations verbatim ✔. Appendix B ✔.
- Type consistency: `AssessmentReply(sentences, workflow)` and `ActiveRunSnapshot(run_id, definition_id, status, current_item, total_items, prompt_block)` match Phase 2 ports; `ToolResult.direct_sentences` added alongside Phase 5's `spoken_summary`; `RunRepo.get_open` is the name used everywhere (the directive's `get_active_or_paused` was renamed to `get_open` because PROPOSED runs must also be found — consistent across ports, memory repo, Pg repo, and tests).
- Boundary note: the Assessment context imports `AssessmentReply`, `ActiveRunSnapshot`, `ToolResult`, `LLMPort` from Conversation's application ports — the one sanctioned cross-context import direction (master plan §3.1).
- No placeholders.
