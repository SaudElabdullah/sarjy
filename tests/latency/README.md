# Real-browser latency harness (Phase 7 Task 7, PRD acceptance test L)

`measure.py` drives real Chromium tabs (via Playwright) against a *deployed*
Sarjy page and measures true, end-to-end voice latency: mic click → speech
recognised → `/chat` request → first token → first sentence → first spoken
audio → last spoken audio. It reports p50/p95 per stage, matching the columns
`telemetry_turns` stores (`t_request_ms`, `t_first_byte_ms`,
`t_first_sentence_ms`, `ttfa_ms`, `t_last_audio_ms` — see
`src/sarjy/interfaces/http/telemetry.py`).

This is **not** a synthetic benchmark against a mocked LLM. It needs:

1. A Sarjy instance deployed somewhere reachable (e.g. `fly deploy` to
   staging) with a **real Gemini API key** configured — the whole point is to
   measure real Gemini first-token and TTS-start latency, which a `FakeLLM`
   or offline stand-in can't reproduce.
2. A real microphone path. Chromium supplies this without a physical mic via
   `--use-fake-device-for-media-stream` and
   `--use-file-for-fake-audio-capture=<wav>`, which feed a WAV file into
   `getUserMedia` as if it were spoken live. The browser's `SpeechRecognition`
   (`webkitSpeechRecognition`) then has to actually transcribe it, which in
   turn depends on Chromium's speech backend being reachable — this is the
   other reason a full run needs a real, network-connected browser rather
   than a headless CI sandbox with no outbound access.

## Status: run parked

Neither of the above is available in this environment (no Gemini API key, no
network egress for live speech recognition), so **no run has been executed
and no results are checked in**. What's here:

- `measure.py` — the harness itself, complete and independently verified
  with `uv run python tests/latency/measure.py --help` (argument parsing,
  imports, and the missing-fixtures error path all work) and `node --check`
  on the `voice.js` debug hook it depends on.
- The `?debug=1` hook in `src/sarjy/interfaces/web/static/voice.js`, which
  exposes `window.__sarjy = controller` so the harness can read
  `controller.marks`/`controller.state` between turns without shipping a
  debug surface to every visitor.
- This README.

`playwright` (in `[project.optional-dependencies] dev`, alongside this
project's other dev tools — not a separate `uv` dependency group) and
`uv run playwright install chromium` have both been set up, so the harness is
ready to execute the moment a staging URL with a real Gemini key is
available.

## Running it for real

```bash
# One-time setup (already done in this repo's dev environment):
uv add --optional dev playwright   # lands in [project.optional-dependencies] dev
uv run playwright install chromium

# Record five short WAV prompts into tests/latency/fixtures/ (mono, 16-bit
# PCM; any sample rate Chromium's fake-audio-capture accepts, e.g. 16 kHz).
# On macOS, `say` + `afconvert` works for quick fixtures:
#   say -o /tmp/p.aiff "what's the weather in Paris" && \
#     afconvert /tmp/p.aiff tests/latency/fixtures/weather_paris.wav -d LEI16 -f WAVE
# Expected filenames (or point --fixtures at a directory of your own *.wav):
#   weather_paris.wav   weather_tokyo.wav   remember_fact.wav
#   recall_fact.wav     small_talk.wav

# Run against a deployed instance with a real Gemini key behind it:
uv run python tests/latency/measure.py https://sarjy-staging.fly.dev --n 50
```

Add `--headed` to watch the browser while it runs, `--out results.jsonl` to
keep the raw per-turn marks, and `--timeout <seconds>` to raise the per-turn
wait if a slow deployment is timing out turns rather than just being slow.

`--out` writes one JSON object per line (JSON Lines, not a single JSON array)
and appends as each turn completes — including turns that error out, recorded
as `{"prompt": ..., "error": "..."}` — so a long run that dies partway
through (network drop, Ctrl-C, an unhandled crash) still leaves every prior
turn's results on disk instead of losing the whole run to whatever killed it.

## Reading the output

```
stage                   p50 (ms)    p95 (ms)     n
t_request_ms                  45          80    50
t_first_byte_ms              320         510    50
t_first_sentence_ms          410         640    50
ttfa_ms                      480         720    50
t_last_audio_ms             2100        3400    50
```

Expected after Phase 7 Tasks 3–6: **p50 ≤ 800 ms, p95 ≤ 1500 ms** for `ttfa_ms`
(time from end-of-speech to first spoken audio — the PRD §7.7 budget this
acceptance test gates). If it's over budget, use the per-stage breakdown to
find the dominant stage:

- **`t_first_byte_ms` > 500 ms** → Gemini first-token latency. Check the
  Gemini region/endpoint, `thinking_budget` (Phase 3), and prompt size
  (Task 6 — a static prefix over budget adds directly to this).
- **`ttfa_ms` − `t_first_sentence_ms` > 250 ms** → TTS start latency. Check
  voice selection and whether `tts.prime()` is actually warming up the
  browser's speech engine before the real utterance (see `voice.js`).
- **`t_request_ms` high** → time from end-of-speech to the `/chat` request
  actually being sent; look at STT stabilisation/silence-timer settings
  before blaming the network.

## What each mark means

Captured by `voice.js`'s `controller.marks` (also POSTed to `/telemetry`,
stored in `telemetry_turns`):

| mark            | meaning                                             |
|-----------------|------------------------------------------------------|
| `speech_end`    | baseline — when the user's speech was judged to have ended (0 ms) |
| `request_sent`  | `/chat` request dispatched                          |
| `first_byte`    | first byte of the SSE stream received                |
| `first_sentence`| first complete sentence available to speak           |
| `first_audio`   | TTS actually started speaking (**this is `ttfa_ms`**) |
| `last_audio`    | TTS finished speaking the whole reply                |

### On a tool turn, `ttfa_ms` measures the filler

`first_audio` is set the first time anything is *spoken*, and on a turn that
calls a tool the first thing spoken is usually not the answer. A `tool_status`
start event arms a 700 ms timer (L-4) that speaks a local filler — "let me
check" and friends, from `fillers.js` — if no real sentence has landed by then,
so `ttfa_ms` on those turns is roughly `min(first real sentence, ~700 ms)`
rather than time-to-answer.

That is deliberate: the PRD §7.7 budget is about how long the user waits in
silence, and the filler is what ends the silence. But it means two things when
reading the numbers:

- A tool-heavy sample will show a *tighter, lower* `ttfa_ms` than a chat-only
  one, and improving the tool path will barely move it. Use
  `t_first_sentence_ms` for time-to-real-answer, and `server_timings.t_tool`
  (added in Phase 7) for what the tools themselves cost.
- `ttfa_ms` clustering just under 700 ms is the filler firing, not a coincidence.
  If that is most of the distribution, the tool path is slower than the filler
  and `t_first_sentence_ms` is the number to work on.

Group the run by whether `server_timings` carries a `t_tool` key to tell the two
populations apart.

`server_timings` (also read back and, when `--out` is given, saved alongside
the marks) is the server-side breakdown for the same turn, letting you tell
network/TTS-side latency apart from LLM-side latency within `t_first_byte_ms`.
