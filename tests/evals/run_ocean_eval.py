"""Runs tests/evals/ocean.jsonl — the Big Five (Mini-IPIP) assessment eval.

Each row is one scripted run: 20 items' worth of `answers` (the ground-truth
value 1-5, or `null` for a skip) paired with `phrasings` (what a user might
actually say for that answer — the text an interpreter has to read). The gate
is 100%: the engine must reproduce every row's `expected` trait scores
exactly (PRD Task 6 Step 4). Those expectations are engine-verified
(round-half-even) rather than hand-computed — `score()` rounds with Python's
`round`, so a mean of 2.25 is 2.2, and a hand calculation that rounds half up
disagrees with the engine on every such row. See
`sarjy.contexts.assessment.domain.scoring`.

Two modes, only one of which needs a model:

* ``--offline`` — drives `HandleAssessmentTurn` in-process with a
  `ScriptedInterpreter` that returns each row's `answers` values directly,
  never touching the `phrasings` text at all. This proves the scoring/state
  machine (confirm -> twenty answers -> score -> complete) end to end without
  Gemini, and is what `tests/evals/run_evals.py --offline --suites ocean`
  delegates to. No API key needed.
* default (live) — the real `GeminiAnswerInterpreter`, fed each row's
  `phrasings` one at a time through the same `HandleAssessmentTurn`/`StartRun`
  pair, in-process (`Container.build(..., connect_db=False)`, so no Postgres
  needed either — just a real `GEMINI_API_KEY`). Parked: not run in CI or
  `make check` (pytest's `evals` marker / `addopts` skip it) pending a key;
  see `run_weather_eval.py`'s docstring for the same arrangement.

`--dry-run` validates tests/evals/ocean.jsonl (each row has 20 `answers`, 20
`phrasings`, and an `expected` dict with exactly the five OCEAN keys) without
hitting the network or importing Gemini — the CI-safe smoke check, also called
directly by tests/unit/evals/test_ocean_jsonl.py.

Usage:
    uv run python tests/evals/run_ocean_eval.py --dry-run
    uv run python tests/evals/run_ocean_eval.py --offline
    uv run python tests/evals/run_ocean_eval.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

_JSONL_PATH = Path(__file__).with_name("ocean.jsonl")
_MINI_IPIP_PATH = Path(__file__).parents[2] / "supabase" / "mini_ipip.json"
TRAITS = ("O", "C", "E", "A", "N")


def _load_cases(path: Path) -> list[dict]:  # type: ignore[type-arg]
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate(path: Path) -> int:
    """Validate an ocean-eval JSONL file's shape without any network access.

    Importable and side-effect-free (no I/O happens at module import time —
    only when this is called), so the unit suite can call it directly instead
    of shelling out to this script. Returns 0 on success, 1 on the first
    validation failure (also printed).
    """
    cases = _load_cases(path)
    for case in cases:
        cid = case.get("id")
        if not isinstance(cid, str) or not cid:
            print(f"invalid case (missing string 'id'): {case!r}")
            return 1
        answers = case.get("answers")
        if not isinstance(answers, list) or len(answers) != 20:
            print(f"case {cid!r}: 'answers' must be a list of 20 entries")
            return 1
        for v in answers:
            if v is not None and (not isinstance(v, int) or not 1 <= v <= 5):
                print(f"case {cid!r}: answer {v!r} must be null or an int 1-5")
                return 1
        phrasings = case.get("phrasings")
        if not isinstance(phrasings, list) or len(phrasings) != 20:
            print(f"case {cid!r}: 'phrasings' must be a list of 20 entries")
            return 1
        if not all(isinstance(p, str) and p.strip() for p in phrasings):
            print(f"case {cid!r}: every phrasing must be a non-empty string")
            return 1
        expected = case.get("expected")
        if not isinstance(expected, dict) or set(expected) != set(TRAITS):
            print(f"case {cid!r}: 'expected' must have exactly the keys {TRAITS}")
            return 1
        for k in TRAITS:
            v = expected[k]
            if v is not None and not isinstance(v, int | float):
                print(f"case {cid!r}: expected[{k!r}]={v!r} must be null or a number")
                return 1
    print(f"dry-run OK: {len(cases)} scripts parsed from {Path(path).name}")
    return 0


def dry_run() -> int:
    """Validate tests/evals/ocean.jsonl. Thin CLI wrapper around `validate`."""
    return validate(_JSONL_PATH)


def _scores_match(got: dict, expected: dict) -> str | None:  # type: ignore[type-arg]
    """`None` when every trait matches; otherwise a description of the first miss.

    Compared with a tolerance rather than `==`: `score()` rounds each trait mean
    to one decimal place, but that mean is still a `float`, so an exact
    string/repr match would be one float-representation bug away from a false
    failure that has nothing to do with the scoring logic this eval exists to
    check.
    """
    for k in TRAITS:
        g, e = got.get(k), expected.get(k)
        if (g is None) != (e is None):
            return f"{k}: got {g!r}, expected {e!r}"
        if g is not None and e is not None and abs(float(g) - float(e)) > 1e-9:
            return f"{k}: got {g!r}, expected {e!r}"
    return None


# ----------------------------------------------------------------- offline mode
class ScriptedInterpreter:
    """Returns a row's `answers` values in order, ignoring the text it is handed.

    The offline eval's whole point is to measure the scoring/state machine
    without Gemini in the loop, so this never reads `user_text` at all — it is
    handed the *ground truth* value for "whatever this row's phrasing turns
    out to mean", not asked to re-derive it.
    """

    def __init__(self, answers: list[int | None]) -> None:
        self._answers = list(answers)
        self._i = 0

    async def interpret(self, item_text: str, scale_labels: list[str], user_text: str):  # type: ignore[no-untyped-def]
        from sarjy.contexts.assessment.application.ports import Interpretation

        v = self._answers[self._i]
        self._i += 1
        if v is None:
            return Interpretation(value=None, confidence=1.0, control="skip")
        return Interpretation(value=v, confidence=1.0, control=None)


async def _run_case_offline(case: dict, ins) -> tuple[bool, str]:  # type: ignore[no-untyped-def,type-arg]
    from sarjy.contexts.assessment.application.handle_turn import HandleAssessmentTurn
    from sarjy.contexts.assessment.application.start_run import StartRun
    from sarjy.contexts.assessment.infrastructure.memory_repos import MemInstrumentRepo, MemRunRepo
    from sarjy.contexts.assessment.infrastructure.offline_narrator import OfflineNarrator
    from sarjy.shared.clock import FakeClock
    from sarjy.shared.ids import UserId

    runs, instruments = MemRunRepo(), MemInstrumentRepo({ins.id: ins})
    clock = FakeClock(datetime(2026, 8, 21, tzinfo=UTC))
    handle = HandleAssessmentTurn(
        runs, instruments, ScriptedInterpreter(case["answers"]), OfflineNarrator(), clock
    )
    start = StartRun(runs, instruments, clock)
    user = UserId(uuid.uuid4())

    await start.execute(user)
    reply = await handle.execute(user, "yes")
    if reply is None or reply.workflow["status"] != "active":
        return False, f"confirm did not start the run: {reply!r}"
    for phrasing in case["phrasings"]:
        reply = await handle.execute(user, phrasing)
        if reply is None:
            return False, f"turn {phrasing!r} fell through to ordinary chat"

    run = await runs.get_open(user) or await runs.latest_complete(user)
    if run is None or run.status.value != "complete":
        status = run.status.value if run else "none"
        return False, f"run did not complete: status={status}"
    mismatch = _scores_match(run.results or {}, case["expected"])
    if mismatch:
        return False, mismatch
    return True, "ok"


def _load_instrument():  # type: ignore[no-untyped-def]
    from sarjy.contexts.assessment.domain.instrument import Instrument

    return Instrument.from_definition(json.loads(_MINI_IPIP_PATH.read_text(encoding="utf-8")))


async def _run_all_offline(cases: list[dict], ins) -> list[tuple[bool, str]]:  # type: ignore[type-arg,no-untyped-def]
    return await asyncio.gather(*(_run_case_offline(c, ins) for c in cases))


def run_offline() -> int:
    ins = _load_instrument()
    cases = _load_cases(_JSONL_PATH)
    results = asyncio.run(_run_all_offline(cases, ins))
    fails = [(c["id"], why) for c, (ok, why) in zip(cases, results, strict=True) if not ok]
    for cid, why in fails:
        print(f"FAIL {cid}: {why}")
    print(f"ocean evals (offline): {len(cases) - len(fails)}/{len(cases)} passed")
    return 1 if fails else 0


# -------------------------------------------------------------------- live mode
async def _run_case_live(case: dict) -> tuple[bool, str]:
    from sarjy.config import Settings
    from sarjy.container import Container

    settings = Settings(weather_provider="mock")  # type: ignore[call-arg]
    c = Container.build(settings, connect_db=False)
    from sarjy.contexts.assessment.application.start_run import StartRun
    from sarjy.shared.ids import UserId

    start = StartRun(c.run_repo, c.instrument_repo, c.clock)
    user = UserId(uuid.uuid4())

    await start.execute(user)
    reply = await c.active_run.handle_turn(user, "yes")
    if reply is None or reply.workflow["status"] != "active":
        return False, f"confirm did not start the run: {reply!r}"
    for phrasing in case["phrasings"]:
        reply = await c.active_run.handle_turn(user, phrasing)
        if reply is None:
            return False, f"turn {phrasing!r} fell through to ordinary chat"

    run = await c.run_repo.get_open(user) or await c.run_repo.latest_complete(user)
    if run is None or run.status.value != "complete":
        status = run.status.value if run else "none"
        return False, f"run did not complete: status={status}"
    mismatch = _scores_match(run.results or {}, case["expected"])
    if mismatch:
        return False, mismatch
    return True, "ok"


async def _main_async_live() -> int:
    cases = _load_cases(_JSONL_PATH)
    results = []
    # Sequential, not gathered: each case makes ~20 real Gemini calls, and
    # running ten of them concurrently would just trade wall-clock time for
    # rate-limit errors that have nothing to do with interpreter accuracy.
    for case in cases:
        results.append(await _run_case_live(case))
    fails = [(c["id"], why) for c, (ok, why) in zip(cases, results, strict=True) if not ok]
    for cid, why in fails:
        print(f"FAIL {cid}: {why}")
    print(f"ocean evals (live): {len(cases) - len(fails)}/{len(cases)} passed")
    return 1 if fails else 0


def main() -> int:
    if "--dry-run" in sys.argv:
        return dry_run()
    if "--offline" in sys.argv:
        return run_offline()
    return asyncio.run(_main_async_live())


if __name__ == "__main__":
    sys.exit(main())
