"""Runs tests/evals/weather.jsonl against `RunTurn` with the mock weather provider
and the real Gemini model.

Usage: uv run python tests/evals/run_weather_eval.py

Each case runs in-process against a fresh `Container` (in-memory repos, no
Postgres, `MockProvider` standing in for the network weather providers) so
this needs a real `GEMINI_API_KEY` but no running app / local Supabase — see
Task 6 Step 4 (gate is 100%, PRD §14.3). Not run in CI or `make check`
(pytest's `evals` marker / `addopts` in pyproject.toml skip it); Phase 8 adds
it to the nightly job.

`--dry-run` validates tests/evals/weather.jsonl (parses, each case has an
`id`/`user`, `expect_tool_call` is a bool, any `expect_*` list field really is
a list) without hitting the network or importing Gemini. This is the CI-safe
smoke check for the eval file while the real run stays parked pending an API
key — see tests/unit/evals/test_weather_jsonl.py.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

_JSONL_PATH = Path(__file__).with_name("weather.jsonl")


def _load_cases(path: Path) -> list[dict]:  # type: ignore[type-arg]
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate(path: Path) -> int:
    """Validate a weather-eval JSONL file's shape without any network access.

    Importable and side-effect-free (no I/O happens at module import time —
    only when this is called), so the unit suite can call it directly instead
    of shelling out to this script. Returns 0 on success, 1 on the first
    validation failure (also printed).
    """
    cases = _load_cases(path)
    for case in cases:
        if "id" not in case or "user" not in case:
            print(f"invalid case (missing id/user): {case!r}")
            return 1
        if "expect_tool_call" not in case or not isinstance(case["expect_tool_call"], bool):
            print(f"case {case['id']!r}: 'expect_tool_call' must be a bool")
            return 1
        for key in ("expect_contains_any", "expect_not_contains"):
            if key in case and not isinstance(case[key], list):
                print(f"case {case['id']!r}: {key!r} must be a list, got {case[key]!r}")
                return 1
        if "setup_facts" in case and not isinstance(case["setup_facts"], dict):
            print(f"case {case['id']!r}: 'setup_facts' must be a dict, got {case['setup_facts']!r}")
            return 1
    print(f"dry-run OK: {len(cases)} scripts parsed from {Path(path).name}")
    return 0


def dry_run() -> int:
    """Validate tests/evals/weather.jsonl. Thin CLI wrapper around `validate`."""
    return validate(_JSONL_PATH)


NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _spoken_numbers(text: str, words_to_num: dict[str, float]) -> set[float]:
    found = {float(n) for n in NUM.findall(text)}
    low = text.lower()
    for words, n in words_to_num.items():
        if words and words in low:
            found.add(n)
    return found


async def _run_case(case: dict, words_to_num: dict[str, float]) -> tuple[bool, str]:  # type: ignore[type-arg]
    from sarjy.config import Settings
    from sarjy.container import Container
    from sarjy.contexts.conversation.application.ports import Fact
    from sarjy.contexts.conversation.domain.events import SentenceEvent, ToolStatusEvent
    from sarjy.contexts.conversation.domain.turn import TurnInput
    from sarjy.contexts.conversation.infrastructure.memory_repos import MemMessages, MemSessions
    from sarjy.contexts.weather.infrastructure.mock_provider import MockProvider
    from sarjy.shared.clock import FakeClock
    from sarjy.shared.ids import UserId

    class StaticFacts:
        def __init__(self, d: dict[str, str]) -> None:
            self.d = d

        async def snapshot(self, user_id: UserId) -> list[Fact]:
            return [Fact(k, v, "fact") for k, v in self.d.items()]

    settings = Settings(weather_provider="mock")  # type: ignore[call-arg]
    clock = FakeClock(datetime(2026, 8, 21, 12, tzinfo=UTC))
    c = Container.build(settings, connect_db=False)
    c.clock = clock
    c.sessions, c.messages = MemSessions(), MemMessages()
    c.facts = StaticFacts(case.get("setup_facts", {}))
    c.tools._tools.clear()
    c.weather_provider = MockProvider(clock, fail=bool(case.get("provider_fail")))
    c.weather_cache = None
    c.rebuild_weather()
    c.rebuild_run_turn()

    text, tool_called, tool_numbers = "", False, set()
    async for ev in c.run_turn(
        TurnInput(UserId(uuid.uuid4()), None, str(uuid.uuid4()), case["user"])
    ):
        if isinstance(ev, ToolStatusEvent) and ev.tool == "get_weather" and ev.state == "start":
            tool_called = True
        if isinstance(ev, SentenceEvent):
            text += " " + ev.sentence.text
    # The session touch is spawned on the container's background tasks (L-7);
    # draining keeps a case's writes inside the case that made them rather than
    # leaking into whatever the next one measures.
    await c.bg.drain()
    for tc in c.messages.tool_calls:  # type: ignore[union-attr]
        res = tc[4]
        tool_numbers |= {float(v) for v in res.values() if isinstance(v, int | float)}
    text = text.strip()

    if tool_called != case["expect_tool_call"]:
        return False, f"tool_call={tool_called} expected {case['expect_tool_call']} :: {text}"
    if case.get("expect_numbers_subset_of_tool_result") and tool_called:
        extra = {
            n
            for n in _spoken_numbers(text, words_to_num)
            if not any(abs(n - t) <= 1.0 for t in tool_numbers)
        }
        if extra:
            return False, f"ungrounded numbers {extra} :: {text}"
    any_ = case.get("expect_contains_any")
    if any_ and not any(a.lower() in text.lower() for a in any_):
        return False, f"missing any of {any_} :: {text}"
    for bad in case.get("expect_not_contains", []):
        if bad.lower() in text.lower():
            return False, f"contains forbidden {bad!r} :: {text}"
    return True, text


async def _main_async() -> int:
    from sarjy.shared.text import to_speech

    words_to_num = {to_speech(str(n)): float(n) for n in range(-40, 121)}
    cases = _load_cases(_JSONL_PATH)
    results = await asyncio.gather(*(_run_case(c, words_to_num) for c in cases))
    fails = [(c["id"], why) for c, (ok, why) in zip(cases, results, strict=True) if not ok]
    for cid, why in fails:
        print(f"FAIL {cid}: {why}")
    print(f"weather evals: {len(cases) - len(fails)}/{len(cases)} passed")
    return 1 if fails else 0


def main() -> int:
    if "--dry-run" in sys.argv:
        return dry_run()
    return asyncio.run(_main_async())


if __name__ == "__main__":
    sys.exit(main())
