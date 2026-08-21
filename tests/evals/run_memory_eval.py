"""Runs tests/evals/memory.jsonl against the live app (real Gemini, local Supabase).

Usage: APP_URL=http://localhost:8000 uv run python tests/evals/run_memory_eval.py

Requires a real Gemini API key wired into the running app and a local Supabase
instance (`make run` in another shell, `SUPABASE_URL`/`SUPABASE_ANON_KEY` set)
so it is not run in CI or `make check` — see PRD Task 7 Step 5.

`--dry-run` validates tests/evals/memory.jsonl (parses, each turn has a `user`
key, and any `expect_*` field is a list) without hitting the network, and
prints the script count. This is the CI-safe smoke check for the eval file
while the real run stays parked pending an API key.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

_JSONL_PATH = Path(__file__).with_name("memory.jsonl")


def _load_cases(path: Path) -> list[dict]:  # type: ignore[type-arg]
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate(path: Path) -> int:
    """Validate a memory-eval JSONL file's shape without any network access.

    Importable and side-effect-free (no I/O happens at module import time —
    only when this is called), so the unit suite can call it directly instead
    of shelling out to this script. Returns 0 on success, 1 on the first
    validation failure (also printed).
    """
    cases = _load_cases(path)
    for case in cases:
        if "id" not in case or "turns" not in case:
            print(f"invalid case (missing id/turns): {case!r}")
            return 1
        for turn in case["turns"]:
            if "user" not in turn:
                print(f"case {case['id']!r}: turn missing 'user': {turn!r}")
                return 1
            for key in ("expect_contains", "expect_not_contains", "expect_contains_any"):
                if key in turn and not isinstance(turn[key], list):
                    print(f"case {case['id']!r}: {key!r} must be a list, got {turn[key]!r}")
                    return 1
    print(f"dry-run OK: {len(cases)} scripts parsed from {Path(path).name}")
    return 0


def dry_run() -> int:
    """Validate tests/evals/memory.jsonl. Thin CLI wrapper around `validate`."""
    return validate(_JSONL_PATH)


def main() -> int:
    if "--dry-run" in sys.argv:
        return dry_run()

    import httpx

    app = os.environ.get("APP_URL", "http://localhost:8000")
    supabase_url = os.environ["SUPABASE_URL"]
    anon = os.environ["SUPABASE_ANON_KEY"]
    gate = 0.95

    def anon_token() -> str:
        r = httpx.post(
            f"{supabase_url}/auth/v1/signup",
            headers={"apikey": anon, "Content-Type": "application/json"},
            json={},
        )
        r.raise_for_status()
        return str(r.json()["access_token"])

    def chat(token: str, session_id: str | None, text: str) -> tuple[str, str | None]:
        body = {
            "session_id": session_id,
            "client_turn_id": str(uuid.uuid4()),
            "text": text,
            "input_mode": "text",
        }
        sentences: list[str] = []
        sid = session_id
        with httpx.stream(
            "POST",
            f"{app}/chat",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        ) as r:
            r.raise_for_status()
            for block in r.iter_text():
                for raw in block.split("\n\n"):
                    if not raw.strip():
                        continue
                    ev = raw.split("\n", 1)[0].removeprefix("event: ").strip()
                    data = json.loads(raw.split("data: ", 1)[1]) if "data: " in raw else {}
                    if ev == "session":
                        sid = data["session_id"]
                    elif ev == "sentence":
                        sentences.append(data["text"])
        return " ".join(sentences), sid

    def check(reply: str, turn: dict) -> list[str]:  # type: ignore[type-arg]
        low = reply.lower()
        fails = [f"missing {s!r}" for s in turn.get("expect_contains", []) if s.lower() not in low]
        fails += [
            f"forbidden {s!r}" for s in turn.get("expect_not_contains", []) if s.lower() in low
        ]
        anys = turn.get("expect_contains_any", [])
        if anys and not any(s.lower() in low for s in anys):
            fails.append(f"none of {anys!r}")
        return fails

    cases = _load_cases(_JSONL_PATH)
    passed = 0
    for case in cases:
        tokens = [anon_token() for _ in range(case.get("users", 1))]
        sessions: list[str | None] = [None] * len(tokens)
        failures: list[str] = []
        for turn in case["turns"]:
            who = turn.get("as", 0)
            if turn.get("new_session"):
                sessions[who] = None
            reply, sessions[who] = chat(tokens[who], sessions[who], turn["user"])
            f = check(reply, turn)
            if f:
                failures.append(f"  turn {turn['user']!r} -> {reply!r}: {f}")
        status = "PASS" if not failures else "FAIL"
        passed += status == "PASS"
        print(f"[{status}] {case['id']}")
        for line in failures:
            print(line)
    rate = passed / len(cases)
    print(f"\nmemory eval: {passed}/{len(cases)} = {rate:.0%} (gate {gate:.0%})")
    return 0 if rate >= gate else 1


if __name__ == "__main__":
    sys.exit(main())
