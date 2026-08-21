#!/usr/bin/env python3
"""Post-deploy smoke test (Phase 8 Task 5): health, anonymous sign-up, one `/chat` turn.

Usage:
    uv run python scripts/smoke.py <base_url> <supabase_url> <anon_key>
    uv run python scripts/smoke.py <base_url> --health-only

Exercises the same three things a human would check right after a deploy:

1. `GET /healthz` returns 200 — the app is up and answering.
2. An anonymous user can sign up against the target Supabase project (run
   against the real staging/prod project as part of the manual release flow
   (`make release-*`), so a broken Auth config — anonymous sign-ins disabled,
   wrong keys — fails here rather than silently in production).
3. One `/chat` turn ("say hi") produces a `sentence` SSE event within 5s and a
   `done` event within 20s — the full request path (auth, guard, LLM,
   text-to-speech-ready sentence splitting) runs end to end.

`--health-only` skips steps 2 and 3 (and drops the `supabase_url`/`anon_key`
requirement), so this script can be run against a local `uvicorn` with no
`GEMINI_API_KEY` and no Supabase project configured. The full run needs both
a real Gemini key and a Supabase project with anonymous sign-ins enabled,
neither of which is available in every environment this script runs in — see
`.superpowers/sdd/2026-08-21-08-deployment-hardening/task-5-report.md`.

Exits non-zero with a one-line reason on any failure — that's what fails this
step of the manual release flow and blocks `make release-*` from proceeding
to the next command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

# Run as a script (`uv run python scripts/smoke.py`), this file's own
# directory is already sys.path[0]; imported as a module (the unit suite) it
# is not, so this mirrors tests/evals/run_evals.py's sys.path handling.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _supabase_auth import SignupError, anon_signup  # noqa: E402

SENTENCE_DEADLINE_S = 5.0
DONE_DEADLINE_S = 20.0


class SmokeError(RuntimeError):
    """A smoke check failed; `str(exc)` is the one-line reason to print."""


def parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse SSE text — `event: ...\\ndata: ...` blocks separated by a blank
    line — into `(event, data)` pairs, in order.

    Works equally on a full canned stream (unit tests) and on a lone block
    sliced out of a live response as it arrives (`stream_chat` below): either
    way it's just "however much SSE text you have so far". A block with no
    `data:` line is skipped rather than erroring, matching the encoder in
    `sarjy.interfaces.http.sse`, which never emits one, but a stray keep-alive
    comment line would otherwise crash this.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.split("\n\n"):
        block = block.strip("\n")
        if not block.strip():
            continue
        name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if not data_lines:
            continue
        events.append((name, json.loads("\n".join(data_lines))))
    return events


def check_health(base_url: str, *, timeout: float = 10.0) -> None:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/healthz", timeout=timeout)
    except httpx.HTTPError as e:
        raise SmokeError(f"GET /healthz unreachable: {e}") from e
    if r.status_code != 200:
        raise SmokeError(f"GET /healthz returned {r.status_code}: {r.text[:200]}")


def get_token(supabase_url: str, anon_key: str) -> str:
    try:
        return anon_signup(supabase_url, anon_key)
    except (httpx.HTTPError, SignupError) as e:
        raise SmokeError(f"anonymous sign-up failed: {e}") from e


def stream_chat(
    base_url: str,
    token: str,
    text: str,
    *,
    sentence_deadline_s: float = SENTENCE_DEADLINE_S,
    done_deadline_s: float = DONE_DEADLINE_S,
) -> tuple[float, float]:
    """POST one `/chat` turn and time the `sentence` / `done` SSE events.

    Returns `(seconds_to_first_sentence, seconds_to_done)`. Raises
    `SmokeError` the moment either deadline is missed, or if the stream ends
    with one of the two events never seen.
    """
    body = {
        "session_id": None,
        "client_turn_id": str(uuid.uuid4()),
        "text": text,
        "input_mode": "text",
    }
    start = time.monotonic()
    sentence_at: float | None = None
    done_at: float | None = None
    buf = ""
    try:
        with httpx.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=done_deadline_s + 5,
        ) as r:
            if r.status_code != 200:
                r.read()
                raise SmokeError(f"POST /chat returned {r.status_code}: {r.text[:200]}")
            for chunk in r.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    block, buf = buf.split("\n\n", 1)
                    for name, _data in parse_sse(block + "\n\n"):
                        if name == "sentence" and sentence_at is None:
                            sentence_at = time.monotonic() - start
                        elif name == "done":
                            done_at = time.monotonic() - start
                elapsed = time.monotonic() - start
                if sentence_at is None and elapsed > sentence_deadline_s:
                    raise SmokeError(
                        f"no 'sentence' event within {sentence_deadline_s:.0f}s "
                        f"(elapsed {elapsed:.1f}s)"
                    )
                if elapsed > done_deadline_s:
                    raise SmokeError(
                        f"no 'done' event within {done_deadline_s:.0f}s (elapsed {elapsed:.1f}s)"
                    )
                if done_at is not None:
                    break
    except httpx.HTTPError as e:
        raise SmokeError(f"POST /chat failed: {e}") from e
    if sentence_at is None:
        raise SmokeError(
            f"stream ended with no 'sentence' event (elapsed {time.monotonic() - start:.1f}s)"
        )
    if done_at is None:
        raise SmokeError(f"stream ended with no 'done' event within {done_deadline_s:.0f}s")
    return sentence_at, done_at


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("base_url", help="deployed app base URL, e.g. https://sarjy-staging.fly.dev")
    p.add_argument("supabase_url", nargs="?", default=None)
    p.add_argument("anon_key", nargs="?", default=None)
    p.add_argument(
        "--health-only",
        action="store_true",
        help="only check /healthz; skip sign-up and /chat (no Supabase project/Gemini key needed)",
    )
    args = p.parse_args(argv)

    if not args.health_only and (not args.supabase_url or not args.anon_key):
        p.error("supabase_url and anon_key are required unless --health-only")

    try:
        check_health(args.base_url)
        print(f"OK  GET /healthz ({args.base_url})")
        if args.health_only:
            print("OK  --health-only: skipping sign-up and /chat")
            return 0
        assert args.supabase_url is not None and args.anon_key is not None
        token = get_token(args.supabase_url, args.anon_key)
        print("OK  anonymous sign-up")
        sentence_s, done_s = stream_chat(args.base_url, token, "say hi")
        print(f"OK  /chat: sentence at {sentence_s:.2f}s, done at {done_s:.2f}s")
    except SmokeError as e:
        print(f"SMOKE FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
