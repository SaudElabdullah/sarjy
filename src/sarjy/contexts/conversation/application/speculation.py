"""Buffered writes for a turn that was run on a guess (PRD L-3).

Speech recognisers finalise a transcript hundreds of milliseconds after the
speaker has actually stopped: the words are on screen, unchanged, while the
recogniser waits to be sure. L-3 spends that dead time by running the turn on
the *interim* transcript as soon as it stops changing, so the first audio can
be playing before the recogniser has committed to anything.

The gamble is that the interim text is what the final text turns out to be.
Usually it is; when it is not, the turn was wrong and nothing about it may
survive — not the user row, not the tool-call rows, not the assistant row. So a
speculative turn writes NOTHING as it runs. Everything it would have written is
collected into a `PendingPersist` and parked here under the client's turn id,
and only a `POST /chat/confirm` carrying a final transcript that matches the
guess turns it into rows. A guess nobody confirms expires and is dropped: that
is the PRD guarantee, and it is enforced by there being no other way out of
this module.

The two halves of that exchange are not ordered, though, which is what the
early-confirm path below is for. A recogniser that finalises while the model is
still streaming makes the client confirm a turn that has not parked yet: the
confirmation found nothing, and the park that followed a moment later had
nothing to match. Both sides behaved correctly and the turn was lost anyway. So
a confirmation that arrives first is *kept* (`early_confirm`), and the park
reads it (`take_early`) instead of parking: matching text writes the rows on the
spot, and a mismatch drops them. Either way the turn is settled once, by
whichever of the two arrived second.

Matching is on normalised text — lowercase, punctuation stripped, whitespace
collapsed — because "what's the weather in paris" and "Whats the weather in
Paris?" are the same utterance as far as an answer already spoken aloud is
concerned. Anything more than punctuation differing means a different question,
and a different question deserves a different turn.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from sarjy.contexts.conversation.domain.message import Message
from sarjy.contexts.conversation.domain.session import Session

# Everything that is not a lowercase letter, a digit or a space. Applied after
# `.lower()`, so it takes punctuation, newlines and tabs with it; the surviving
# runs of spaces are collapsed afterwards.
_PUNCT = re.compile(r"[^a-z0-9 ]+")

# Kept in step with `normalise()` in voice.js — the client compares the same two
# strings before it decides whether to confirm, and a client that thinks it
# matches while the server disagrees turns a confirmation into a 409 and a
# second, duplicate answer.


def normalise(s: str) -> str:
    """The comparison form of an utterance: lowercase, unpunctuated, single-spaced."""
    return re.sub(r"\s+", " ", _PUNCT.sub("", s.lower())).strip()


@dataclass(slots=True)
class PendingPersist:
    """Every row a speculative turn would have written, in the order to write them.

    `session_touch` is here for the same reason as the rest: a brand-new session
    is normally saved before the user message that points at it, but a
    speculative turn that is never confirmed must not leave an orphan session
    row behind either. Held rather than written, and saved first at confirm time
    so the messages' foreign key has something to land on.

    `tool_calls` are the positional argument tuples for `MessageRepo.
    save_tool_call` rather than a DTO: the repo's signature is the only shape
    they are ever used in, and inventing a record to unpack one call later would
    be ceremony.
    """

    user_msg: Message
    assistant_msg: Message | None = None
    tool_calls: list[tuple[Any, ...]] = field(default_factory=list)
    session_touch: Session | None = None


@dataclass(slots=True)
class _Entry[T]:
    text: str
    pending: T
    at: float


@dataclass(slots=True)
class _Early:
    """A confirmation that arrived before there was anything to confirm (C1)."""

    text: str
    at: float
    owner: str


# `/chat/confirm` is the one endpoint that writes into this process's memory
# without a turn behind it, so it is the one that needs saying no. Per user
# first: 256 is two orders of magnitude more in-flight guesses than a human
# speaking can have, so a caller over it is not a caller whose turns are being
# lost. Then globally, because a limit that is only per-user is no limit at all
# once there are enough users. Both evict oldest-first: the entry most likely to
# be waiting for a turn is the one written last.
_MAX_EARLY_PER_USER = 256
_MAX_EARLY_TOTAL = 10_000
# `_gc` is a sweep of both dicts. It used to run on every single access, which
# was fine when the only way in was a real turn; with `/chat/confirm` able to
# insert directly it becomes O(n) work per request an attacker controls. Expiry
# correctness does not depend on the sweep — every read checks its own entry's
# age — so the sweep is purely about reclaiming memory and can be throttled.
_GC_EVERY = 64


class SpeculativeTurnCache[T]:
    """Turn ids parked with the writes their turn is holding back.

    In-process and per-worker on purpose. The confirmation arrives over a
    separate request, so in theory it could land on a different worker and find
    nothing — the client treats that exactly like a mismatch (409, send a normal
    turn), which is correct and merely slower. A shared store would buy a small
    hit-rate improvement in exchange for putting a network round trip in the
    path of the write it is trying to avoid.

    The SAME caveat applies to the early-confirm half below, and applies to it
    just as literally: `early_confirm` records the final transcript in this
    worker's dict, and `take_early` is read by the turn that is still running in
    this worker. A confirmation routed to worker B while the turn runs on worker
    A leaves B holding an early confirm nobody will ever read and A parking a
    guess nobody will ever confirm — exactly the outcome the parked half already
    has, and no worse: the guess expires unwritten. Nothing here narrows the
    multi-worker gap; it closes the single-worker RACE (a confirmation that
    beats its own turn to the park), which is the one that was losing turns on
    the deployment this actually runs on.

    Generic in the pending payload so the cache can be tested — and reasoned
    about — without dragging a `Message` graph into it.
    """

    def __init__(self, ttl_s: float = 10.0) -> None:
        # 10 seconds: the confirmation is sent by the same client that opened the
        # stream, the moment its recogniser finalises — hundreds of milliseconds
        # later, not seconds. The window is generous enough to absorb a slow
        # recogniser and short enough that an abandoned guess is gone long before
        # the user could have said anything else worth answering.
        #
        # Both halves share it. An early confirm is waiting for a turn that is
        # *mid-flight*, so it needs even less time than a parked guess does; one
        # TTL is one thing to reason about, and the shorter need is covered by
        # the longer window.
        self._ttl = ttl_s
        self._entries: dict[str, _Entry[T]] = {}
        # Insertion-ordered by construction (every dict is), which is what makes
        # "evict the oldest" a scan from the front rather than a sort.
        self._early: dict[str, _Early] = {}
        self._early_per_owner: dict[str, int] = {}
        self._since_gc = 0

    # --- expiry -----------------------------------------------------------
    #
    # Checked per entry at the point of use rather than left to the sweep. The
    # sweep is throttled (see `_GC_EVERY`), so anything that relies on it for
    # CORRECTNESS would start returning entries that have expired — which for
    # `take` means writing a turn whose ten-second window has gone.

    def _expired(self, at: float) -> bool:
        return time.monotonic() - at > self._ttl

    def put(self, client_turn_id: str, text: str, pending: T) -> None:
        self._maybe_gc()
        self._entries[client_turn_id] = _Entry(normalise(text), pending, time.monotonic())

    def has(self, client_turn_id: str) -> bool:
        """Is there a live parked guess under this id?

        `take` cannot answer this: it returns `None` both for "nothing parked"
        and for "parked, but the transcript says something else", and those are
        the two cases `RunTurn.confirm` has to tell apart — the first is a
        confirmation that arrived early (record it), the second is a guess that
        was simply wrong (409).
        """
        e = self._entries.get(client_turn_id)
        return e is not None and not self._expired(e.at)

    def take(self, client_turn_id: str, final_text: str) -> T | None:
        """The parked writes for `client_turn_id`, if `final_text` is what was guessed.

        Removes the entry either way: a turn is confirmable exactly once, and a
        mismatch means the client is about to send a fresh turn under a new id,
        so keeping the guess around would only give it a second chance to be
        written by mistake.
        """
        e = self._entries.pop(client_turn_id, None)
        if e is None or self._expired(e.at) or e.text != normalise(final_text):
            return None
        return e.pending

    def early_confirm(self, client_turn_id: str, final_text: str, *, owner: str) -> None:
        """Remember a confirmation for a turn that has not parked yet (C1).

        The client posts `/chat/confirm` the instant its recogniser finalises,
        which can be *before* the turn it confirms has reached the park at the
        end of its own stream. Dropping that confirmation on the floor cost the
        turn its rows: the park that followed had nothing to match against, and
        the ten-second window then expired against a client that had already
        confirmed and would not confirm again.

        Stored normalised, because that is the only form it is ever compared in.
        Last writer wins: a client that somehow confirms twice means the second
        transcript, and there is exactly one turn per id to apply it to.

        `owner` is passed explicitly rather than parsed back out of the key. The
        key happens to be built from the user id (`_spec_key`), but a cache that
        knows that is a cache that breaks silently the day the key format
        changes — and what it needs the owner FOR is refusing to let one caller
        fill the dict, which is too important to infer.
        """
        self._maybe_gc()
        self._drop_early(client_turn_id)  # last writer wins, and keeps the count honest
        self._early[client_turn_id] = _Early(normalise(final_text), time.monotonic(), owner)
        self._early_per_owner[owner] = self._early_per_owner.get(owner, 0) + 1
        while self._early_per_owner.get(owner, 0) > _MAX_EARLY_PER_USER:
            self._evict_oldest_early(owner)
        while len(self._early) > _MAX_EARLY_TOTAL:
            self._evict_oldest_early(None)

    def take_early(self, client_turn_id: str) -> str | None:
        """The normalised final transcript already confirmed for this id, if any.

        Consumed on read, like `take`: a turn parks exactly once, so a second
        reader would be a bug, and leaving the entry to expire would let a
        re-used turn id pick up a stale confirmation.
        """
        e = self._early.get(client_turn_id)
        self._drop_early(client_turn_id)
        return None if e is None or self._expired(e.at) else e.text

    # --- bookkeeping ------------------------------------------------------

    def _drop_early(self, client_turn_id: str) -> None:
        e = self._early.pop(client_turn_id, None)
        if e is None:
            return
        n = self._early_per_owner.get(e.owner, 0) - 1
        if n > 0:
            self._early_per_owner[e.owner] = n
        else:
            self._early_per_owner.pop(e.owner, None)

    def _evict_oldest_early(self, owner: str | None) -> None:
        """Drop the oldest early confirm, optionally restricted to one owner.

        A linear scan of an insertion-ordered dict rather than a second index to
        keep in step with the first. It is bounded by `_MAX_EARLY_TOTAL` and runs
        at most once per insertion, on a path that is itself rate limited — an
        index would be more code and one more thing to get out of sync than the
        10,000-iteration worst case is worth.
        """
        for k, v in self._early.items():
            if owner is None or v.owner == owner:
                self._drop_early(k)
                return

    def _maybe_gc(self) -> None:
        """Sweep, but not on every access — see `_GC_EVERY`."""
        self._since_gc += 1
        if self._since_gc >= _GC_EVERY or len(self._early) > _MAX_EARLY_TOTAL:
            self._gc()

    def _gc(self) -> None:
        """Drop expired entries from both dicts.

        Memory reclamation only: nothing reads through this for correctness, so
        running it late costs bytes rather than behaviour.
        """
        for k in [k for k, v in self._entries.items() if self._expired(v.at)]:
            del self._entries[k]
        for k in [k for k, v in self._early.items() if self._expired(v.at)]:
            self._drop_early(k)
        self._since_gc = 0

    @property
    def size(self) -> int:
        """Live parked guesses — expired ones do not count, swept or not."""
        return sum(1 for v in self._entries.values() if not self._expired(v.at))

    @property
    def early_size(self) -> int:
        return sum(1 for v in self._early.values() if not self._expired(v.at))
