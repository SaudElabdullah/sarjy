# WAV fixtures

Expected by `tests/latency/measure.py` (or point `--fixtures` at a different
directory): five short, mono, 16-bit PCM WAV prompts —

- `weather_paris.wav`
- `weather_tokyo.wav`
- `remember_fact.wav`
- `recall_fact.wav`
- `small_talk.wav`

Not checked into this repo — record them locally (see
`tests/latency/README.md` for a quick macOS `say`/`afconvert` recipe) before
running the harness for real.
