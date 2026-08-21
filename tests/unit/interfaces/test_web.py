from fastapi.testclient import TestClient

from sarjy.config import Settings
from sarjy.main import create_app


def test_index_renders_page_with_config() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "/static/voice.js" in body
    assert 'id="memory-panel"' in body
    assert "/static/ocean.js" in body
    assert 'id="ocean-panel"' in body
    assert 'id="delete-account"' in body


def test_index_loads_config_js_and_has_no_inline_sarjy_script() -> None:
    # CSP (security.py) forbids inline <script> — the config that used to be a
    # `<script>window.SARJY = {...}</script>` block now has to come from an
    # external, same-origin script instead.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/").text
    assert '<script src="/config.js"></script>' in body
    assert "window.SARJY" not in body


def test_index_loads_supabase_js_locally_not_from_a_cdn() -> None:
    # supabase-js is vendored (static/supabase.js, see VENDOR.md) so the CSP's
    # script-src can be 'self' only (security.py) — no jsdelivr, no other CDN.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/").text
    assert '<script src="/static/supabase.js"></script>' in body
    assert "cdn.jsdelivr.net" not in body


def test_static_supabase_js_served() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        r = client.get("/static/supabase.js")
    assert r.status_code == 200
    assert "supabase" in r.text


def test_config_js_has_anon_key_not_service_role_key() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        r = client.get("/config.js")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/javascript")
    assert r.headers["cache-control"] == "no-store"
    body = r.text
    assert "window.SARJY" in body
    assert "anon" in body  # SUPABASE_ANON_KEY from test env (see conftest)
    assert "service" not in body  # SUPABASE_SERVICE_ROLE_KEY from test env must never appear
    # L-3: the client is told whether the server will honour a speculative turn.
    # Off by default, and as a JSON literal rather than a bare "False" that would
    # be a ReferenceError the moment voice.js loaded.
    assert '"speculative": false' in body
    # Relative by default: same-origin in local/dev (see the comment on the
    # PUBLIC_API_BASE setting).
    assert '"apiBase": ""' in body


def test_config_js_reports_the_configured_public_api_base() -> None:
    # A Storage-hosted build (scripts/upload_static.py) needs the client's fetches
    # to reach the Fly app absolutely, since the page is served from Storage.
    app = create_app(Settings(public_api_base="https://sarjy-prod.fly.dev"), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/config.js").text
    assert '"apiBase": "https://sarjy-prod.fly.dev"' in body


def test_static_voice_js_served() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        r = client.get("/static/voice.js")
    assert r.status_code == 200


def test_static_memory_js_served() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        r = client.get("/static/memory.js")
    assert r.status_code == 200


def test_static_ocean_js_served() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        r = client.get("/static/ocean.js")
    assert r.status_code == 200


def test_static_app_css_served() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        r = client.get("/static/app.css")
    assert r.status_code == 200


def test_voice_js_has_stream_and_finish_guards() -> None:
    # Regression tripwire (static, not behavioral): guards against the queue-drain race
    # (`streamOpen`), the double-finish STT bug (`finished`), a superseded send()
    # clobbering a newer one's state after a barge-in race (`sendGen`), and a late
    # onend/onerror from a cancelled utterance shifting the wrong entry off the TTS
    # queue (`queue[0] !== u`) reappearing unnoticed.
    # This only proves the identifiers are present, not that the guards behave correctly.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/static/voice.js").text
    assert "streamOpen" in body
    assert "finished" in body
    assert "sendGen" in body
    assert "queue[0] !== u" in body


def test_voice_js_has_telemetry_flush() -> None:
    # Regression tripwire (static, not behavioral): guards against flushTelemetry()
    # regressing to sendBeacon (which cannot set the Authorization header — see the
    # brief), losing its keepalive fetch, or dropping the sendGen guard that stops a
    # superseded send from flushing another turn's stale marks.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/static/voice.js").text
    assert "flushTelemetry" in body
    assert "keepalive: true" in body
    assert "sendBeacon" not in body
    assert "marksGen !== this.sendGen" in body
    # Regression tripwire: guards against a new turn mutating the *shared* `this.marks`
    # object in place (which would leak a superseded turn's first_byte/first_sentence
    # into the next one) rather than replacing it with a fresh per-turn object.
    assert "marks = { speech_end" in body
    assert "/telemetry" in body


def test_voice_js_has_filler_guard() -> None:
    # Regression tripwire (static, not behavioral): guards against the spoken-filler-on-
    # tool-turns feature (L-4) losing its 700ms armed timer (`fillerTimer`) or the local
    # picker it calls (`pickFiller`) — either would mean tool turns go silent again while
    # the tool runs, or (worse) a stale timer from a superseded send speaking into a newer
    # turn. The >=3 count on `clearTimeout(this.fillerTimer)` guards against the timer being
    # cleared on barge-in and "sentence" but *not* on "error"/"done"/send()'s finally — which
    # would let a filler fire after the turn already ended.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/static/voice.js").text
    assert "pickFiller" in body
    assert "fillerTimer" in body
    assert "firstSentenceSeen" in body
    assert body.count("clearTimeout(this.fillerTimer)") >= 3


def test_static_fillers_js_served() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        r = client.get("/static/fillers.js")
    assert r.status_code == 200
    assert "pickFiller" in r.text


def test_config_js_reports_speculation_when_it_is_enabled() -> None:
    app = create_app(Settings(speculative_enabled=True), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/config.js").text
    assert '"speculative": true' in body


def test_voice_js_has_the_speculation_guards() -> None:
    # Regression tripwire (static, not behavioral): guards against the L-3 client
    # losing the server switch it is gated on (`SPECULATIVE_ENABLED` — without it
    # every utterance would be sent twice and persisted twice), the 400ms stable-
    # interim timer that decides when to guess, or the normalised comparison that
    # decides between confirming the answer already spoken and running the real
    # transcript instead.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/static/voice.js").text
    assert "SPECULATIVE_ENABLED" in body
    assert "onStable" in body
    assert "/chat/confirm" in body
    assert "normalise(t) === normalise(this.specText)" in body
    # The wrong-guess path: cancel what is being said, and take the discarded answer
    # off the transcript, before the real turn appends its own.
    # ...and the discard path: called from the wrong-guess branch, the text form and
    # sendText (a guess must not survive the user typing over it), and scoped to the
    # generation it captured so it can never abort a newer turn.
    assert body.count("discardSpeculative()") >= 4
    assert "specGen" in body
    assert "this.specGen === this.sendGen" in body
    assert "specReply" in body


def _confirm_source(body: str) -> str:
    """voice.js's `confirm()`, from its signature to the next method, comments stripped.

    Comments are dropped because the tripwire this feeds is about what the CODE
    does, and the comment explaining the rule names the very identifier the rule
    forbids.
    """
    start = body.index("async confirm(text) {")
    end = body.index("onSpeechQueueDrained(marks", start)
    lines = body[start:end].splitlines()
    return "\n".join(ln for ln in lines if not ln.strip().startswith("//"))


def test_confirm_never_clears_the_done_resolver() -> None:
    # confirm() runs from onFinal, which can fire BEFORE the stream's "done" frame
    # arrives. It captures `specDone` so it can wait for the park before retrying a
    # 409 — but if it also nulls `specDoneResolve`, the "done" handler has nothing
    # left to call, the promise never settles, and the retry never happens: the turn
    # is lost exactly the way C1 was losing them, one layer further out.
    #
    # The resolver is owned by the sites that resolve it. This asserts confirm() is
    # not one of them.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/static/voice.js").text
    src = _confirm_source(body)
    assert "this.specDone" in src, "confirm() should still capture the done promise"
    assert "specDoneResolve" not in src, (
        "confirm() must not touch specDoneResolve — see the TRIPWIRE comment in voice.js"
    )
    # And the sites that DO own it are still there: the done frame and send()'s
    # finally, so the promise settles however the stream ends.
    assert body.count("this.specDoneResolve?.()") >= 2


def test_index_has_no_turnstile_markup_without_a_site_key() -> None:
    # PRD §11 captcha is opt-in: with TURNSTILE_SITE_KEY unset the page must not
    # reference Cloudflare at all (the CSP does not allow it either — see
    # test_security_headers.py) and must carry no widget host to render into.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/").text
    assert "challenges.cloudflare.com" not in body
    assert 'id="turnstile"' not in body


def test_index_renders_the_turnstile_script_and_host_when_a_site_key_is_set() -> None:
    app = create_app(Settings(turnstile_site_key="0x4AAA"), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/").text
    assert (
        '<script src="https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit"'
        " async defer></script>" in body
    )
    assert '<div id="turnstile"></div>' in body
    # render=explicit means the widget is rendered by voice.js, not by api.js
    # auto-scanning the DOM — and still no inline script anywhere.
    assert "<script>" not in body


def test_index_bytes_are_unchanged_when_no_site_key_is_configured() -> None:
    # The `{%- if %}` guards must strip cleanly: an unconfigured deployment gets
    # exactly the page it got before captcha existed, whitespace included.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/").text
    assert '<link rel="stylesheet" href="/static/app.css">\n<!-- Vendored locally' in body
    assert '</details>\n  <footer>\n    <button id="mic"' in body


def test_config_js_reports_the_turnstile_site_key() -> None:
    app = create_app(Settings(turnstile_site_key="0x4AAA"), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/config.js").text
    assert '"turnstileSiteKey": "0x4AAA"' in body


def test_config_js_reports_an_empty_turnstile_site_key_by_default() -> None:
    # Empty string, never null: voice.js reads `window.SARJY.turnstileSiteKey || ""`.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/config.js").text
    assert '"turnstileSiteKey": ""' in body


def test_voice_js_gates_the_captcha_on_the_site_key_and_passes_the_token_to_signin() -> None:
    # Regression tripwire (static, not behavioral): guards against the captcha
    # path losing its site-key gate (which would make every keyless deployment
    # wait on a widget that is not in the page), against the token being dropped
    # from the signInAnonymously call, and against the visible failure notice
    # regressing to a silent one.
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as client:
        body = client.get("/static/voice.js").text
    assert "TURNSTILE_SITE_KEY" in body
    assert 'ts.render("#turnstile"' in body
    assert "{ options: { captchaToken: await captchaToken() } }" in body
    assert "sb.auth.signInAnonymously(credentials)" in body
    assert "Verification failed" in body
