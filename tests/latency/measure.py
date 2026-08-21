"""Real-browser TTFA latency harness (Phase 7 Task 7, PRD acceptance test L).

Drives an actual Chromium tab against a deployed Sarjy page with Playwright,
using Chromium's fake-audio-capture device to "speak" a fixed WAV file into
getUserMedia instead of a real microphone. Each turn: click the mic, wait for
the turn to finish (state returns to "idle" and the last audio mark lands),
then read `window.__sarjy.marks`/`serverTimings` back out of the page — the
same object `flushTelemetry()` POSTs to `/telemetry` (see
`src/sarjy/interfaces/web/static/voice.js`). `window.__sarjy` only exists
when the page is loaded with `?debug=1` (opt-in, so production visitors never
get a debug surface).

Usage:
    uv run python tests/latency/measure.py https://sarjy-staging.fly.dev --n 50

Requires: a deployed page with a real Gemini key behind it (this harness
exercises the full stack, not a fake), and `uv run playwright install
chromium` having been run once. See `tests/latency/README.md` for the full
run procedure, the WAV fixtures this expects in `tests/latency/fixtures/`,
and why no run is checked into this commit.

NOT run as part of `make check` / CI — it drives a real browser against a
real deployment and costs real Gemini tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# Five short prompts spanning the tool paths that dominate first-token latency
# differently (grounded tool call, memory read/write, plain chat) — see
# tests/latency/README.md for how to record these.
DEFAULT_PROMPTS = [
    "weather_paris.wav",
    "weather_tokyo.wav",
    "remember_fact.wav",
    "recall_fact.wav",
    "small_talk.wav",
]
# Stage name -> (mark key, baseline key). Matches `_d()` in
# src/sarjy/interfaces/http/telemetry.py — same deltas, computed client-side here.
STAGES: list[tuple[str, str]] = [
    ("t_request_ms", "request_sent"),
    ("t_first_byte_ms", "first_byte"),
    ("t_first_sentence_ms", "first_sentence"),
    ("ttfa_ms", "first_audio"),
    ("t_last_audio_ms", "last_audio"),
]
CHROMIUM_MEDIA_ARGS = [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
]


def _delta(mark: float | None, baseline: float) -> float | None:
    return None if mark is None else mark - baseline


async def _run_turn(page: Any, timeout_s: float) -> dict[str, Any]:
    """Click the mic, wait for the turn to finish, and read the marks back."""
    await page.click("#mic")
    await page.wait_for_function(
        "() => window.__sarjy && window.__sarjy.state === 'idle' "
        "&& window.__sarjy.marks && window.__sarjy.marks.last_audio",
        timeout=timeout_s * 1000,
    )
    raw: dict[str, Any] = await page.evaluate(
        "() => ({ marks: window.__sarjy.marks, server: window.__sarjy.serverTimings })"
    )
    marks = raw["marks"]
    baseline = marks["speech_end"]
    raw["derived"] = {name: _delta(marks.get(key), baseline) for name, key in STAGES}
    return raw


async def measure(
    url: str,
    n: int,
    prompts: list[Path],
    timeout_s: float,
    headless: bool,
    out_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run `n` turns split evenly across `prompts`.

    Chromium's fake-audio-capture WAV is a *launch* argument, not something
    swappable mid-session, so each prompt gets its own browser launch (and a
    fresh page/session) rather than trying to rotate files within one tab.

    Each turn runs in its own try/except: a single stuck or crashed turn (a
    navigation error, a timeout waiting for `state === 'idle'`) is recorded as
    an error entry rather than aborting every turn still queued behind it —
    50 real turns against a live Gemini deployment is long enough that one bad
    turn losing the whole run would be expensive to just re-run from scratch.
    When `out_path` is given, each turn's result (success or error) is
    appended to it as one JSON line the moment that turn completes, so a run
    that dies partway through — network drop, Ctrl-C, an unhandled exception
    outside this loop — still leaves every prior turn's results on disk
    instead of only in the `results` list this function would otherwise lose
    with it.
    """
    results: list[dict[str, Any]] = []
    per_prompt = max(1, n // len(prompts))
    debug_url = url + ("&" if "?" in url else "?") + "debug=1"
    out_fh = out_path.open("a", encoding="utf-8") if out_path else None
    try:
        async with async_playwright() as pw:
            for wav in prompts:
                browser = await pw.chromium.launch(
                    headless=headless,
                    args=[*CHROMIUM_MEDIA_ARGS, f"--use-file-for-fake-audio-capture={wav}"],
                )
                context = await browser.new_context(permissions=["microphone"])
                page = await context.new_page()
                await page.goto(debug_url)
                for _ in range(per_prompt):
                    try:
                        turn = await _run_turn(page, timeout_s)
                        turn["prompt"] = wav.name
                    except Exception as exc:
                        turn = {"prompt": wav.name, "error": f"{type(exc).__name__}: {exc}"}
                        print(f"turn failed ({wav.name}): {exc}", file=sys.stderr)
                    results.append(turn)
                    if out_fh:
                        out_fh.write(json.dumps(turn) + "\n")
                        out_fh.flush()
                await browser.close()
    finally:
        if out_fh:
            out_fh.close()
    return results


def _percentile(values: list[float], pct: int) -> float:
    return (
        statistics.quantiles(values, n=100, method="inclusive")[pct - 1]
        if len(values) > 1
        else values[0]
    )


def report(results: list[dict[str, Any]]) -> None:
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"{len(errors)}/{len(results)} turns failed (see stderr above)")
    print(f"{'stage':<20}{'p50 (ms)':>12}{'p95 (ms)':>12}{'n':>6}")
    for name, _key in STAGES:
        vals = [r["derived"][name] for r in results if r.get("derived", {}).get(name) is not None]
        if not vals:
            continue
        print(
            f"{name:<20}{_percentile(vals, 50):>12.0f}{_percentile(vals, 95):>12.0f}{len(vals):>6}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("url", help="Target URL, e.g. https://sarjy-staging.fly.dev")
    ap.add_argument("--n", type=int, default=50, help="Total turns, split across --fixtures")
    ap.add_argument("--timeout", type=float, default=15.0, help="Per-turn timeout in seconds")
    ap.add_argument("--headed", action="store_true", help="Show the browser window")
    ap.add_argument("--fixtures", type=Path, default=FIXTURES_DIR, help="Directory of WAV prompts")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Append raw per-turn results here as JSON lines, one per turn, written as each "
        "turn completes (so a run that dies partway through isn't lost)",
    )
    args = ap.parse_args()

    prompts = sorted(args.fixtures.glob("*.wav")) or [args.fixtures / p for p in DEFAULT_PROMPTS]
    missing = [p for p in prompts if not p.exists()]
    if missing:
        print("Missing WAV fixtures (see tests/latency/README.md):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(2)

    results = asyncio.run(
        measure(
            args.url, args.n, prompts, args.timeout, headless=not args.headed, out_path=args.out
        )
    )
    report(results)


if __name__ == "__main__":
    main()
