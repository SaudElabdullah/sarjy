// src/sarjy/interfaces/web/static/voice.js
import { pickFiller } from "./fillers.js";
const { supabaseUrl, anonKey, apiBase } = window.SARJY;
// L-3: speculation is a server feature the page is told about, not one the client
// decides to try. With it off, /chat would run and persist the guess as an ordinary
// turn, so a client speculating unilaterally would double every utterance.
const SPECULATIVE_ENABLED = !!window.SARJY.speculative;
// Must stay in step with `normalise()` in speculation.py: the client compares these two
// strings to decide whether to confirm, and the server compares them again to decide
// whether to honour it. A client that is more forgiving than the server just turns a
// confirmation into a 409; one that is stricter throws away a turn already spoken.
const normalise = (t) => t.toLowerCase().replace(/[^a-z0-9 ]+/g, "").replace(/\s+/g, " ").trim();
const sb = supabase.createClient(supabaseUrl, anonKey);
// Exported so other modules (memory.js's "Delete my data") can sign out through the
// same client rather than creating a second one against the same project.
export { sb };
const $ = (id) => document.getElementById(id);
const ui = { state: $("state"), transcript: $("transcript"), mic: $("mic"), form: $("textform"), input: $("textin"), cont: $("continuous"), notice: $("notice") };

// ---------- captcha (PRD §11: anonymous sign-in is captcha-gated) ----------
// Empty unless TURNSTILE_SITE_KEY is set on the server (see config.py / web.py's
// /config.js). With no key nothing below runs, no widget host is in the page
// (index.html renders it only under the same condition), and sign-in is exactly
// what it was before captcha existed.
const TURNSTILE_SITE_KEY = window.SARJY.turnstileSiteKey || "";

// Cloudflare's api.js is loaded `async defer` with `render=explicit`, so
// `window.turnstile` appears at some point after this module runs — hence the
// poll rather than an onload callback: `?onload=name` needs a global that must
// already exist when api.js finishes, and this module (a deferred ES module)
// has no guaranteed ordering against an async script.
function turnstileReady(timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (window.turnstile) return resolve(window.turnstile);
      if (Date.now() > deadline) return reject(new Error("turnstile script did not load"));
      setTimeout(tick, 50);
    };
    tick();
  });
}

// One widget, one token, memoised: `ensureSession()` is called from several
// modules and from every send, and rendering a second widget into the same host
// (or solving a second challenge per page load) is both wasted work and a
// visible flicker. Reset to null on failure so a later attempt can re-render.
let captchaPromise = null;
function captchaToken() {
  captchaPromise ||= (async () => {
    const ts = await turnstileReady();
    return await new Promise((resolve, reject) => {
      ts.render("#turnstile", {
        sitekey: TURNSTILE_SITE_KEY,
        callback: resolve,
        "error-callback": () => reject(new Error("turnstile verification failed")),
        "timeout-callback": () => reject(new Error("turnstile challenge expired")),
      });
    });
  })();
  return captchaPromise;
}

// ---------- auth (anonymous, upgradeable) ----------
export async function ensureSession() {
  let { data: { session } } = await sb.auth.getSession();
  if (session) return session;
  let credentials;
  if (TURNSTILE_SITE_KEY) {
    try {
      credentials = { options: { captchaToken: await captchaToken() } };
    } catch (e) {
      // Nothing here can retry usefully: the widget either failed to load or the
      // challenge itself failed, and both need a fresh page. Say so visibly
      // (a silent failure looks like a dead mic button) and let the caller's
      // own error handling see the rejection.
      captchaPromise = null;
      showNotice("Verification failed — reload the page to try again.");
      throw e;
    }
  }
  let error;
  ({ data: { session }, error } = await sb.auth.signInAnonymously(credentials));
  if (!session) {
    // supabase-js reports failures as a value, not a throw. Surface them: a
    // silent null session would otherwise read as a dead network further down.
    const why = error?.message || "anonymous sign-in returned no session";
    showNotice(`Couldn't sign you in: ${why}`);
    throw new Error(why);
  }
  return session;
}
// Back-compat alias: nothing in this file still calls it, but kept in case any
// non-module inline script relies on the global.
window.sarjyEnsureSession = ensureSession;

// ---------- notice helper (mic errors, capability notices) ----------
let noticeHideTimer = null;
function showNotice(text, autoHideMs) {
  ui.notice.textContent = text;
  ui.notice.hidden = false;
  clearTimeout(noticeHideTimer);
  if (autoHideMs) noticeHideTimer = setTimeout(() => { ui.notice.hidden = true; }, autoHideMs);
}

// ---------- capability detection (V-9) ----------
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
const hasSTT = !!SR, hasTTS = "speechSynthesis" in window;
if (!hasSTT) { ui.mic.hidden = true; showNotice("Voice input isn't supported in this browser — type instead."); }

// ---------- TTS queue (V-4, V-5, V-10) ----------
const tts = {
  queue: [], speaking: false, voice: null, primed: false,
  pickVoice() {
    const prefs = [/Google US English/, /Samantha/, /Aria/, /Natural/, /en-US/];
    const voices = speechSynthesis.getVoices();
    for (const p of prefs) { const v = voices.find(v => p.test(v.name) || p.test(v.lang)); if (v) return v; }
    return voices[0] || null;
  },
  prime() { if (this.primed || !hasTTS) return; speechSynthesis.speak(new SpeechSynthesisUtterance("")); this.primed = true; },
  enqueue(text, onFirstAudio) {
    if (!hasTTS) return;
    const u = new SpeechSynthesisUtterance(text);
    u.voice = this.voice ||= this.pickVoice(); u.rate = 1.05;
    u.onstart = () => { onFirstAudio?.(); };
    // `this.queue[0] !== u` guards the queue against a late event from an utterance
    // that is no longer the head: cancel() (barge-in) empties the queue, but the
    // browser still fires onend/onerror for the utterance it just stopped. Without
    // the guard that stale event shifts the *next* turn's first sentence off the
    // queue unspoken, and every later sentence stays one slot out of step.
    const done = () => { if (this.queue[0] !== u) return; this.queue.shift(); this.speaking = false; this.pump(); };
    u.onend = done; u.onerror = done;
    this.queue.push(u); this.pump();
  },
  pump() { if (this.speaking || !this.queue.length) { if (!this.queue.length) controller.onSpeechQueueDrained(); return; } this.speaking = true; speechSynthesis.speak(this.queue[0]); },
  cancel() { this.queue = []; this.speaking = false; if (hasTTS) speechSynthesis.cancel(); },
};
if (hasTTS) speechSynthesis.onvoiceschanged = () => { tts.voice = tts.pickVoice(); };

// ---------- STT (V-2, V-3) ----------
// `onStable` (optional) fires once the transcript has stopped changing for 400ms and is
// at least three words long — the recogniser has almost certainly heard the whole thing
// and is only waiting to be sure, which is the dead time L-3 spends. It fires at most
// once per utterance, and never replaces onFinal: whatever the recogniser eventually
// commits to is still the transcript of record.
function listenOnce({ onInterim, onFinal, onError, onStable }) {
  const rec = new SR(); rec.lang = "en-US"; rec.interimResults = true; rec.continuous = false; rec.maxAlternatives = 1;
  let finalText = "", silenceTimer = null, lastInterim = "", finished = false;
  let stableTimer = null, stableText = null, stableFired = false;
  // `finished` guards against onresult(isFinal), the silence-timer fallback, and onend
  // all racing to call finish() — without it a single utterance could fire onFinal twice
  // and send two turns for one thing said.
  const finish = () => {
    if (finished) return;
    finished = true;
    clearTimeout(silenceTimer); clearTimeout(stableTimer);
    if (finalText || lastInterim) onFinal(finalText || lastInterim); else onError("no-speech");
    try { rec.stop(); } catch {}
  };
  rec.onresult = (e) => {
    let interim = "";
    for (const r of e.results) { if (r.isFinal) finalText += r[0].transcript; else interim += r[0].transcript; }
    const full = finalText + interim;
    if (onStable && !stableFired && full !== stableText) {
      // Restarted on every change, so the 400ms is 400ms of *silence in the transcript*,
      // not 400ms since the utterance began. Shorter than the 700ms V-3 fallback on
      // purpose: the guess has to be in flight before the recogniser gives up waiting.
      stableText = full; clearTimeout(stableTimer);
      if (full.trim().split(/\s+/).filter(Boolean).length >= 3) {
        stableTimer = setTimeout(() => { stableFired = true; onStable(full); }, 400);
      }
    }
    lastInterim = interim; onInterim(full);
    clearTimeout(silenceTimer); silenceTimer = setTimeout(finish, 700);  // V-3 fallback timer
    if (finalText) finish();
  };
  rec.onerror = (e) => { if (finished) return; finished = true; clearTimeout(silenceTimer); clearTimeout(stableTimer); onError(e.error); };
  rec.onend = () => { clearTimeout(silenceTimer); clearTimeout(stableTimer); if (!finished && !finalText && lastInterim) finish(); };
  rec.start();
  return rec;
}

// ---------- transcript UI (V-12) ----------
function bubble(role, text) { const d = document.createElement("div"); d.className = `bubble ${role}`; d.textContent = text; ui.transcript.append(d); ui.transcript.scrollTop = ui.transcript.scrollHeight; return d; }
function chip(text) { const c = document.createElement("span"); c.className = "chip"; c.textContent = text; ui.transcript.append(c); }

// ---------- turn controller (state machine; V-6 barge-in, V-7 continuous) ----------
const controller = {
  state: "idle", sessionId: sessionStorage.getItem("sarjy.session") || null, rec: null, abort: null, marks: {},
  // streamOpen is true for the lifetime of an in-flight /chat SSE stream (from send() start
  // until its "done"/"error" event is handled or the fetch/read loop otherwise ends). It
  // guards onSpeechQueueDrained() so the brief gap between sentence N ending and sentence
  // N+1 arriving is never mistaken for the assistant finishing its turn.
  streamOpen: false,
  // sendGen is a monotonically increasing generation counter. Each send() call captures the
  // value at the moment it starts (`gen`); every mutation of shared controller state it makes
  // after an `await` is guarded by `this.sendGen !== gen`. This stops a superseded send (one
  // that lost a barge-in race — its fetch/reader was aborted, but its pending await settles
  // *after* a newer send has already started and set streamOpen = true) from clobbering the
  // new send's state when its own catch/finally eventually runs.
  sendGen: 0,
  // Latency telemetry (L-1, L-2): every place that starts a new turn (STT's onFinal, the text
  // form submit, sendText()) replaces `this.marks` with a *fresh* object holding just
  // `speech_end` — never mutates the old one in place. send() captures that object into a local
  // `marks` const right away, so every later write in send()/handle() lands on the object for
  // *this* turn, even if a barge-in replaces `this.marks` again before this turn's async work
  // (or its trailing TTS audio) finishes. Without the fresh-object-per-turn + local-capture
  // combo, a superseded turn's late first_byte/first_sentence writes would land on the next
  // turn's (shared) marks object and produce negative ttfa/etc. deltas.
  // `serverTimings`/`lastMessageId` are captured off the SSE "done" event and reported
  // alongside the marks by flushTelemetry(). `lastMode` mirrors send()'s mode param so
  // onSpeechQueueDrained()'s default-argument flush (see below) can still report voice vs.
  // text without every caller threading it through.
  // marksGen ties "the currently active turn" to `sendGen`, so flushTelemetry() can tell a
  // completed turn's marks from a superseded one's leftovers before reporting them — needed
  // because tts.pump() calls onSpeechQueueDrained() with no marks argument (it has no gen of
  // its own to hand over), so that path falls back to `this.marks`/`this.marksGen`.
  serverTimings: null, lastMessageId: null, lastMode: null, marksGen: -1,
  // firstSentenceSeen/fillerTimer back the spoken filler on tool turns (L-4): a tool_status
  // "start" event arms a 700ms timer that speaks a local filler (fillers.js) if no "sentence"
  // has landed by then. Reset per-turn in send(); cleared on barge-in (bargeIn()), as soon as a
  // real sentence arrives ("sentence"), and whenever the turn ends without one ("error", "done",
  // and send()'s finally, for every other exit path) — otherwise a filler can fire *after* the
  // turn already ended. Every fire also re-checks sendGen so a stale timer from a superseded
  // turn can never speak into the current one.
  firstSentenceSeen: false, fillerTimer: null,
  // Speculation (L-3). `specText` is the interim transcript a turn is already running on and
  // `specTurnId` the id it was sent under; both are cleared at the start of every listen, so
  // a confirmation can only ever belong to the utterance in progress. On a final transcript
  // that normalises to `specText` the answer already being spoken is right and only needs its
  // rows written (/chat/confirm); on anything else the guess was wrong, so the stream is
  // aborted, the audio cancelled, and a normal turn goes out under a NEW id — leaving the
  // orphaned guess to expire server-side, unwritten.
  // `specAbort`/`specGen` are the speculative send's OWN controller and generation, so
  // discarding it aborts that stream and not whichever turn happens to own `this.abort`
  // by the time the discard runs.
  // `specDone` is a promise that settles when the speculative turn's stream has ended —
  // resolved by the "done" frame, and by send()'s finally for every other way a stream can
  // stop. confirm() awaits it before it retries a 409, because a 409 only becomes final
  // once the server has certainly parked (or discarded) the turn, and that happens at the
  // very end of the stream.
  specText: null, specTurnId: null, specReply: null, specAbort: null, specGen: -1,
  specDone: null, specDoneResolve: null,
  set(s) { this.state = s; ui.state.className = `state ${s}`; ui.state.textContent = s[0].toUpperCase() + s.slice(1); },
  async startListening() {
    tts.prime(); this.bargeIn(); this.set("listening");
    ui.notice.hidden = true;
    this.specText = null; this.specTurnId = null; this.specReply = null; this.specAbort = null; this.specGen = -1;
    this.specDoneResolve?.(); this.specDone = null; this.specDoneResolve = null;
    const live = bubble("user", "…");
    this.rec = listenOnce({
      onInterim: (t) => { live.textContent = t; },
      onStable: (t) => {
        if (!SPECULATIVE_ENABLED || this.specText) return;
        this.specText = t;
        // The speech really has ended by now — the transcript stopped moving 400ms ago —
        // so this is the honest zero for the turn's latency, and the one the confirmed
        // turn keeps reporting against (no second `marks` object is minted on confirm).
        this.marks = { speech_end: performance.now() };
        this.send(t, "voice", { speculative: true }).catch(console.error);
      },
      onFinal: (t) => {
        live.textContent = t;
        if (this.specText && this.specTurnId && normalise(t) === normalise(this.specText)) { this.confirm(t).catch(console.error); return; }
        // Either no guess was made, or it was wrong: throw away whatever it was saying and
        // run the real transcript. send() mints a fresh client_turn_id of its own.
        this.discardSpeculative();
        this.marks = { speech_end: performance.now() };
        this.send(t, "voice").catch(console.error);
      },
      onError: (err) => {
        live.remove(); this.set("idle");
        if (err === "no-speech") showNotice("I didn't hear anything", 2500);
        else showNotice(`Mic error: ${err}`);
      },
    });
  },
  bargeIn() { if (this.abort) this.abort.abort(); tts.cancel(); clearTimeout(this.fillerTimer); try { this.rec?.stop(); } catch {} },
  discardSpeculative() {
    // A guess that will not be confirmed: abort its stream, stop the audio it queued, and
    // take the discarded answer off the transcript before the real turn appends its own —
    // two answers to one question, one of them to a question that was never asked, is
    // worse on screen than it is in the database.
    //
    // Deliberately NOT bargeIn(): that aborts `this.abort` and cancels the TTS queue
    // whoever owns them, and by the time a discard runs a newer turn may already be the
    // owner (the user typed while the guess was in flight). Only the captured generation
    // is touched — its own AbortController always, the shared speech queue only while it
    // is still the current turn.
    if (this.specGen === -1 && !this.specReply) return;
    this.specAbort?.abort();
    if (this.specGen === this.sendGen) { tts.cancel(); clearTimeout(this.fillerTimer); }
    this.specReply?.remove();
    this.specText = null; this.specTurnId = null; this.specReply = null; this.specAbort = null; this.specGen = -1;
    this.specDoneResolve?.(); this.specDone = null; this.specDoneResolve = null;
  },
  async confirm(text) {
    // The guess was right: the answer is already streaming (or spoken), and all that is
    // left is telling the server to write the rows it buffered.
    //
    // Nothing here re-runs the turn. Every route to that ends with the same answer spoken
    // a second time, which is a worse thing to do to a listener than losing one exchange
    // from the transcript — and if the confirmation did land and only its response was
    // lost, re-running would duplicate the rows as well. So: one retry for a network
    // failure (the request may simply not have arrived), and one re-POST for a 409.
    //
    // The three statuses (see `/chat/confirm`):
    //   204 — written. Done.
    //   202 — the server has the transcript but the turn had not parked yet; it will write
    //         the rows itself when it does. Doing anything here would race it.
    //   409 — a guess IS parked and it says something else. That verdict is only final once
    //         the turn has actually parked, which is the end of its stream — so wait for
    //         `done` (already past, usually: `await` on a resolved promise is a microtask)
    //         and ask exactly once more. Confirming is idempotent server-side: the parked
    //         entry is consumed by the first call, so the retry can only 202.
    //
    // TRIPWIRE: confirm() must only READ this.specDone and must never touch
    // this.specDoneResolve. It runs from onFinal, which can be BEFORE the stream's
    // "done" frame arrives — and nulling the resolver there means that frame has
    // nothing to call, so `await done` below never wakes and the 409 retry never
    // happens. The resolver sites (the "done" handler, send()'s finally,
    // startListening, discardSpeculative) own the reset. Enforced by
    // test_confirm_never_clears_the_done_resolver in tests/unit/interfaces/test_web.py.
    const id = this.specTurnId, done = this.specDone;
    const body = JSON.stringify({ client_turn_id: id, text });
    this.specText = null; this.specTurnId = null; this.specReply = null; this.specAbort = null; this.specGen = -1;
    const post = async () => {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const { access_token } = await ensureSession();
          const r = await fetch(`${apiBase}/chat/confirm`, { method: "POST", keepalive: true,
            headers: { "Content-Type": "application/json", Authorization: `Bearer ${access_token}` }, body });
          return r.status;
        } catch { /* nothing arrived: try once more, then let it go */ }
      }
      return 0;
    };
    if (await post() !== 409) return;
    await done;
    await post();
  },
  onSpeechQueueDrained(marks = this.marks) {
    if (this.streamOpen || tts.queue.length) return;
    marks.last_audio = performance.now();
    this.flushTelemetry(marks);
    if (this.state === "speaking") {
      this.set("idle");
      if (ui.cont.checked) setTimeout(() => this.startListening(), 250);
    }
  },
  flushTelemetry(marks) {
    // `marks._sent` (rather than clearing `this.marks`) makes this idempotent per-turn: both
    // handle()'s "done" case and send()'s finally can independently call
    // onSpeechQueueDrained(marks) for the same completed turn when the stream closes right
    // after "done", and this must not double-POST.
    if (!marks || !marks.speech_end || marks._sent) return;
    if (this.marksGen !== this.sendGen) return; // superseded: drop stale marks
    const body = JSON.stringify({ message_id: this.lastMessageId, marks, server_timings: this.serverTimings,
      // `ua` is truncated to the 300 characters /telemetry accepts (see ClientInfo): the
      // views only substring-match browser names out of it, and a UA long enough to be
      // refused would cost the whole turn's marks over a field nothing reads in full.
      client_info: { ua: String(navigator.userAgent).slice(0, 300), stt: hasSTT, tts: hasTTS, mode: this.lastMode } });
    marks._sent = true;
    ensureSession().then(({ access_token }) => fetch(`${apiBase}/telemetry`, { method: "POST", keepalive: true,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${access_token}` }, body })).catch(() => {});
  },
  async send(text, mode, opts = {}) {
    const myAbort = new AbortController();
    this.abort = myAbort;
    const gen = ++this.sendGen;
    // Captured before the first await: a discard during the auth round trip still has a
    // controller to abort.
    if (opts.speculative) {
      this.specAbort = myAbort; this.specGen = gen;
      // Armed before the first await so a confirmation that arrives during the auth round
      // trip already has something to wait on. Always resolved (finally, below), never
      // rejected: confirm() only uses it as "the stream is over", and a stream that ended
      // badly is still over.
      this.specDone = new Promise((res) => { this.specDoneResolve = res; });
    }
    // Captured once, up front: every mark this turn records goes through this local, not
    // `this.marks` — see the comment on `marksGen` above.
    const marks = this.marks;
    this.marksGen = gen; this.lastMode = mode; this.firstSentenceSeen = false;
    this.set("thinking"); this.streamOpen = true;
    try {
      const { access_token } = await ensureSession();
      if (this.sendGen !== gen) return; // superseded while awaiting auth
      const turnId = crypto.randomUUID(); const reply = bubble("assistant", ""); let firstAudio = false;
      if (opts.speculative) { this.specTurnId = turnId; this.specReply = reply; }
      marks.request_sent = performance.now();
      let res;
      try {
        res = await fetch(`${apiBase}/chat`, { method: "POST", signal: myAbort.signal,
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${access_token}` },
          body: JSON.stringify({ session_id: this.sessionId, client_turn_id: turnId, text, input_mode: mode, client_ts: Date.now(), speculative: !!opts.speculative }) });
      } catch (e) {
        if (this.sendGen !== gen || e?.name === "AbortError") return;
        this.fail("Sorry, I lost the connection. Try again?");
        return;
      }
      if (this.sendGen !== gen) return; // superseded while the request was in flight
      if (!res.ok) { this.fail(res.status === 429 ? "Give me a moment to catch my breath." : "Something went wrong on my end."); return; }
      if (!res.body) { this.fail("Sorry, I lost the connection. Try again?"); return; }
      const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (this.sendGen !== gen) return; // stale generation: drop the rest of the stream
          if (done) break;
          if (!marks.first_byte) marks.first_byte = performance.now();
          buf += dec.decode(value, { stream: true });
          let idx;
          while ((idx = buf.indexOf("\n\n")) !== -1) {
            const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
            this.handle(raw, reply, gen, marks, () => { if (!firstAudio) { firstAudio = true; marks.first_audio = performance.now(); } });
          }
        }
      } catch (e) {
        // AbortError here means bargeIn() cancelled us on purpose — nothing to report.
        if (this.sendGen !== gen || e?.name === "AbortError") return;
        this.fail("Sorry, I lost the connection. Try again?");
      }
    } catch (e) {
      if (this.sendGen !== gen || e?.name === "AbortError") return;
      this.fail("Sorry, I lost the connection. Try again?");
    } finally {
      // Unconditional, unlike the rest of this block: a superseded speculative send still
      // has to release whatever is waiting on `specDone`, or a confirm() sitting on a 409
      // retry never wakes up.
      if (opts.speculative) { this.specDoneResolve?.(); this.specDoneResolve = null; }
      if (this.sendGen === gen) {
        clearTimeout(this.fillerTimer);
        this.streamOpen = false;
        if (tts.queue.length === 0) this.onSpeechQueueDrained(marks);
      }
    }
  },
  handle(raw, reply, gen, marks, onFirstAudio) {
    if (this.sendGen !== gen) return; // stale frame from a superseded send: drop it
    const ev = /^event: (\w+)/.exec(raw)?.[1];
    let data;
    try { data = JSON.parse(raw.split("data: ")[1] || "{}"); } catch { return; }
    switch (ev) {
      case "session": this.sessionId = data.session_id; sessionStorage.setItem("sarjy.session", data.session_id); break;
      case "tool_status":
        if (data.state === "start") {
          chip(`⚙ ${data.tool}`);
          this.fillerTimer = setTimeout(() => {
            if (this.sendGen !== gen || this.firstSentenceSeen) return; // stale timer or a sentence already spoke
            tts.enqueue(pickFiller(data.tool), onFirstAudio); this.set("speaking");
          }, 700);
        }
        break;
      case "sentence":
        clearTimeout(this.fillerTimer); this.firstSentenceSeen = true;
        if (!marks.first_sentence) marks.first_sentence = performance.now();
        if (this.state !== "speaking") this.set("speaking"); reply.textContent += (reply.textContent ? " " : "") + data.text; tts.enqueue(data.speech || data.text, onFirstAudio); break;
      case "error":
        clearTimeout(this.fillerTimer);
        this.streamOpen = false;
        this.fail(data.message_spoken);
        break;
      case "done":
        clearTimeout(this.fillerTimer);
        marks.done = performance.now();
        this.lastMessageId = data.message_id; this.serverTimings = data.timings;
        this.streamOpen = false;
        // The park has happened server-side by the time this frame exists, so a 409 that
        // confirm() is holding can now be retried against a settled turn.
        this.specDoneResolve?.(); this.specDoneResolve = null;
        document.dispatchEvent(new CustomEvent("sarjy:turn-done", { detail: data }));
        if (tts.queue.length === 0) this.onSpeechQueueDrained(marks);
        break;
      // "token" events are raw model tokens for future UI use — never speak them (PRD §9.1).
      case "token": break;
    }
  },
  fail(msg) {
    bubble("assistant", msg);
    if (hasTTS) { tts.enqueue(msg); this.set("speaking"); }
    else {
      // No TTS to drain the queue and trigger onSpeechQueueDrained() naturally, but we still
      // want continuous mode to auto-relisten — so go through the same "speaking" state and
      // let send()'s finally / the "done" handler run onSpeechQueueDrained()'s idle+relisten
      // logic once streamOpen clears (tts.queue is always empty when !hasTTS).
      this.set("speaking");
    }
  },
};

ui.mic.addEventListener("click", () => {
  if (controller.state === "listening") { controller.bargeIn(); controller.set("idle"); }
  else { controller.startListening(); }
});
ui.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const t = ui.input.value.trim(); if (!t) return;
  // Before bargeIn(): typing while a guess is in flight must discard that guess (its
  // stream, its audio and its bubble) rather than leave it to be confirmed against an
  // utterance the user abandoned — and discarding has to happen while `this.abort` and
  // the speech queue still belong to it.
  controller.discardSpeculative();
  ui.input.value = ""; tts.prime(); controller.bargeIn(); bubble("user", t);
  controller.marks = { speech_end: performance.now() };
  controller.send(t, "text").catch(console.error);
});
// Warm the session (and, when captcha is on, the widget) before the first turn.
// Rejections are already surfaced by ensureSession's showNotice; swallow them here
// so a captcha failure is not also an unhandled promise rejection in the console.
ensureSession().catch(() => {});

// ---------- debug hook (tests/latency/measure.py) ----------
// Opt-in only (?debug=1): exposes the turn controller on `window` so an external
// driver (Playwright) can read `controller.marks`/`controller.state` between turns
// without the app shipping a debug surface to every visitor.
if (new URLSearchParams(location.search).get("debug") === "1") window.__sarjy = controller;

// ---------- programmatic text send (used by ocean.js's item scale) ----------
// Same path as the text form's submit handler, just callable from another module
// instead of needing a synthetic form submission.
export function sendText(text) {
  const t = String(text ?? "").trim();
  if (!t) return;
  controller.discardSpeculative();  // see the text form's handler
  tts.prime(); controller.bargeIn(); bubble("user", t);
  controller.marks = { speech_end: performance.now() };
  controller.send(t, "text").catch(console.error);
}
