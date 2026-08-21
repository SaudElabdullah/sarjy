"""Application service: sentence-level output guard.

`OutputGuard.check_sentence` implements `OutputGuardPort` (see
`sarjy.contexts.conversation.application.ports`) — it is called once per
spoken sentence, synchronously, from the streaming split loop in `RunTurn`.
Because the caller is sync but event logging is async, a would-be-blocking
event write is instead fired as a background task (`asyncio.create_task`)
tracked in `self._pending`; `drain()` lets a caller (tests, or the
`Container` on shutdown) wait for those writes to actually land.

Leak detection runs at two granularities. The per-sentence check (n=8)
catches a sentence that quotes a confidential block outright. The rolling
check (n=6 over the last six sentences, >= 3 shared 6-grams) catches what
that one structurally cannot: a model reciting the prompt a clause at a
time, where no single clause carries eight consecutive matching words but
the passage plainly does. Cut sentences stay in the window — the model is
still reciting whether or not we spoke the last line, and dropping them
would let the window cool down mid-leak.

Verdict priority: prompt leak > persona break > ungrounded number (tool
figures first, then — for sentences that are actually about the user's
assessment results — the scores on file; see `GuardContext.results_numbers`
for why those are two lists and not one). Both a
leak and a persona break are `replace`d with a refusal only for the first
cut of the turn (tracked via `ctx.cut_count`) and `cut` silently after
that — regardless of which category triggered the first cut — so a long
leaked/off-persona passage does not get spoken piecemeal as a string of
refusals. An ungrounded number is always `cut` — the guard has no way to
know the correct unit/wording to substitute, so `RunTurn` is the one that
reacts to the `ungrounded_number` kind, substituting the tool result's own
`spoken_summary` for the sentence it cut (see `RunTurn._speak`).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from sarjy.contexts.conversation.application.ports import GuardContext, SentenceVerdict
from sarjy.contexts.guardrails.application.ports import GuardEventRepo
from sarjy.contexts.guardrails.domain.grounding import mentions_results, ungrounded_numbers
from sarjy.contexts.guardrails.domain.leak_detector import LeakDetector
from sarjy.contexts.guardrails.domain.persona import is_persona_break
from sarjy.contexts.guardrails.domain.templates import refusal_for
from sarjy.observability.logging import get_logger

log = get_logger(__name__)

GuardMode = Literal["enforce", "shadow"]
# Rolling leak window: how many sentences it holds, and the n-gram size and
# threshold used over the joined text. `threshold=2` means "more than 2", i.e.
# three shared 6-grams — enough that an ordinary reply mentioning a couple of
# the prompt's own phrases ("I can't look up news or stocks") stays well clear.
_TAIL_LEN = 6
_WINDOW_N = 6
_WINDOW_THRESHOLD = 2


class OutputGuard:
    def __init__(self, events: GuardEventRepo, mode: GuardMode = "enforce") -> None:
        self.events = events
        self.mode: GuardMode = mode
        self._pending: set[asyncio.Task[None]] = set()

    def check_sentence(self, sentence: str, ctx: GuardContext) -> SentenceVerdict:
        # Append BEFORE evaluating: the rolling check has to see this sentence
        # as part of the window, otherwise the sentence that finally pushes the
        # passage over the threshold is the one sentence not being judged.
        ctx.spoken_tail.append(sentence)
        del ctx.spoken_tail[:-_TAIL_LEN]
        verdict = self._evaluate(sentence, ctx)
        if verdict.action != "pass":
            self._log(ctx, verdict, sentence)
            # I7: `cut_count` advances in shadow mode too. It is what decides
            # whether the NEXT verdict would have been a spoken refusal or a
            # silent cut, so leaving it at zero made shadow record a stream of
            # `replace`s where enforce would have recorded one `replace` and then
            # cuts — the shadow run's whole purpose is to predict the enforce run,
            # and it was predicting a different one.
            ctx.cut_count += 1
            if self.mode == "shadow":
                return SentenceVerdict("pass")
        return verdict

    def _evaluate(self, s: str, ctx: GuardContext) -> SentenceVerdict:
        # Prefer the confidential corpus (policy/grounding/hierarchy/tool-guidance
        # blocks) — it excludes the persona line and capability blurb, which Sarjy
        # is allowed to restate. Fall back to the full system prompt only when no
        # confidential corpus has been supplied.
        corpus = ctx.leak_corpus or ctx.system_prompt
        if LeakDetector(corpus).is_leak(s) or self._window_leak(ctx, corpus):
            if ctx.cut_count == 0:
                return SentenceVerdict("replace", refusal_for("prompt_leak"), "prompt_leak")
            return SentenceVerdict("cut", None, "prompt_leak")
        if is_persona_break(s):
            # Gated the same way as the leak branch above: only the first cut of
            # the turn is spoken as a refusal, everything after is silently cut —
            # so e.g. a leak cut followed by a persona break doesn't produce two
            # refusals back to back.
            if ctx.cut_count == 0:
                return SentenceVerdict("replace", refusal_for("persona_switch"), "persona_break")
            return SentenceVerdict("cut", None, "persona_break")
        if ctx.tool_numbers and ungrounded_numbers(s, ctx.tool_numbers):
            return SentenceVerdict("cut", None, "ungrounded_number")
        # A claim about the user's own results is checked against the scores on
        # file, to the decimal. Two conditions, both required: figures to check
        # against (a finished run this turn was grounded on), and a sentence
        # that is actually about the results. Neither check subsumes the other
        # — see `GuardContext.results_numbers` for why they are not one list.
        if (
            ctx.results_numbers
            and mentions_results(s)
            and ungrounded_numbers(s, ctx.results_numbers, ctx.results_tolerance)
        ):
            return SentenceVerdict("cut", None, "ungrounded_number")
        return SentenceVerdict("pass")

    @staticmethod
    def _window_leak(ctx: GuardContext, corpus: str) -> bool:
        """Does the last handful of sentences, taken together, recite the corpus?

        Only meaningful once there is more than one sentence in the window —
        with one, this is a strictly weaker restatement of the per-sentence
        check that would just lower its effective threshold.
        """
        if len(ctx.spoken_tail) < 2:
            return False
        window = " ".join(ctx.spoken_tail)
        return LeakDetector(corpus, n=_WINDOW_N, threshold=_WINDOW_THRESHOLD).is_leak(window)

    def _log(self, ctx: GuardContext, v: SentenceVerdict, sentence: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (e.g. called from sync test code): nothing to
            # schedule the write on, so drop it rather than raise.
            return
        # I7: shadow rows say what the verdict WOULD have been, not a bare
        # "shadow". `shadow_cut` and `shadow_replace` are directly comparable to
        # the `cut`/`replace` an enforcing run writes, which is what makes a
        # shadow run usable as a rehearsal instead of a headcount.
        action = v.action if self.mode == "enforce" else f"shadow_{v.action}"
        t = loop.create_task(
            self._safe_record(
                user_id=ctx.user_id,
                message_id=ctx.message_id,
                layer=6,
                kind=v.kind or "output",
                action=action,
                severity=3 if v.kind == "prompt_leak" else 2,
                # `speculative` only when true — see `input_guard._spec`.
                detail={"len": len(sentence), **({"speculative": True} if ctx.speculative else {})},
            )
        )
        self._pending.add(t)
        t.add_done_callback(self._pending.discard)

    async def _safe_record(self, **kw: Any) -> None:
        """Record, swallowing failures.

        A raising event write used to reach `drain()` as a bare task exception
        (collected by `return_exceptions=True` and then dropped on the floor,
        which is what "exception was never retrieved" warnings are made of).
        Catching it here means the failure is logged where it happened and
        `drain` has nothing left to swallow.
        """
        try:
            await self.events.record(**kw)
        except Exception:
            log.exception("guard_event_record_failed")

    async def drain(self) -> None:
        """Await all in-flight fire-and-forget event writes.

        Used by tests (to observe events written from a running loop) and by
        the `Container` on shutdown, so no logged verdict is lost. A failure
        in one event write must not prevent draining (or awaiting) the rest,
        so exceptions are collected rather than propagated.
        """
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)
