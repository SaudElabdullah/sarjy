"""Application-layer ports (Protocols) and their DTOs.

This module is the contract every infrastructure adapter (Phase 4+) implements
and every application service (Phase 5+) depends on. The application layer
never imports infrastructure — infrastructure imports this module instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, TypeVar

from pydantic import BaseModel

from sarjy.contexts.conversation.domain.message import Message
from sarjy.contexts.conversation.domain.session import Session
from sarjy.shared.guard import GuardDecision as GuardDecision
from sarjy.shared.ids import MessageId, RunId, SessionId, UserId
from sarjy.shared.text import Sentence as Sentence

T = TypeVar("T", bound=BaseModel)


# --- LLM ---------------------------------------------------------------


@dataclass(frozen=True)
class FunctionCall:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass(frozen=True)
class FunctionResponse:
    name: str
    response: dict[str, Any]
    id: str | None = None


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["user", "model", "tool"]
    text: str | None = None
    function_call: FunctionCall | None = None
    function_response: FunctionResponse | None = None


@dataclass(frozen=True)
class LLMRequest:
    system: str
    messages: list[LLMMessage]
    tools: list[dict[str, Any]]
    temperature: float
    max_output_tokens: int
    thinking_budget: int = 0
    force_tool: str | None = None


@dataclass(frozen=True)
class LLMText:
    text: str


@dataclass(frozen=True)
class LLMFunctionCall:
    call: FunctionCall


@dataclass(frozen=True)
class LLMFinished:
    reason: str


LLMEvent = LLMText | LLMFunctionCall | LLMFinished


class LLMUnavailable(Exception):  # noqa: N818
    """Raised by an LLM adapter when the upstream provider cannot serve the request."""


class LLMTimeout(Exception):  # noqa: N818
    """Raised by an LLM adapter when the upstream provider does not respond in time."""


class LLMPort(Protocol):
    def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]: ...

    async def generate_json(self, req: LLMRequest, schema: type[T]) -> T: ...


# --- Facts / memory ------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    kind: str


class FactSnapshotPort(Protocol):
    async def snapshot(self, user_id: UserId) -> list[Fact]: ...


# --- Tools -----------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: dict[str, Any]
    grounding_numbers: tuple[float, ...] = ()
    spoken_error: str | None = None
    # grounded fallback sentence (Phase 4/5)
    spoken_summary: str | None = None
    # tool-authored reply that ends the turn without another LLM hop (Phase 6)
    direct_sentences: list[str] | None = None
    # measured by ToolRouter; kept out of `data` so it is never sent to the model
    latency_ms: int = 0


class ToolPort(Protocol):
    name: str
    # ClassVar: concrete tools (e.g. RememberTool) declare `declaration` at class
    # level as a fixed, shared value rather than per-instance state — matching that
    # here is what lets mypy accept them as ToolPort implementers (a plain instance
    # attribute is invariant against a ClassVar and would be rejected).
    declaration: ClassVar[dict[str, Any]]
    # Does invoking this tool change anything outside the turn? (L-3.)
    #
    # A speculative turn is a turn run on a guess, and the guess may be wrong —
    # so everything it does has to be undoable by simply not doing it. Its
    # database writes are, because they are buffered. A stored fact, a deleted
    # one, or a workflow advanced to the next question is not: there is no
    # confirmation step that could take it back, and "remember my sister is
    # called Ana" acted on a misheard sentence is a wrong fact the user never
    # said. `RunTurn` promotes a speculative turn to an ordinary one BEFORE
    # invoking a tool that declares this, so the mutation and the rows that
    # explain it are written together, and the guess is never spoken for.
    #
    # Declared here so a new tool has to answer the question. Read-only tools
    # (weather, recall) say False; anything that writes says True.
    mutating: ClassVar[bool]

    # `facts` is the turn's already-loaded fact snapshot, handed down rather
    # than re-read (I5). A tool that needs the user's stored facts — the home
    # city `get_weather` resolves "here" against, the units it prefers — used to
    # fetch its own copy, which over Postgres was a second round trip for rows
    # the turn had loaded milliseconds earlier and was holding in memory. The
    # parameter is optional so a tool can still be driven by a caller that has
    # no snapshot to offer (an eval harness, a unit test), and every tool must
    # accept it whether or not it has a use for one: the router does not know
    # which tools care.
    async def invoke(
        self, user_id: UserId, args: dict[str, Any], facts: list[Fact] | None = None
    ) -> ToolResult: ...


# --- Guards ------------------------------------------------------------
#
# GuardDecision lives in the shared kernel (sarjy.shared.guard) and is
# re-exported here for backward compatibility with existing imports from
# this module.


class InputGuardPort(Protocol):
    async def check(
        self,
        user_id: UserId,
        text: str,
        recent_user_turns: list[str],
        message_id: MessageId | None = None,
        speculative: bool = False,
    ) -> GuardDecision:
        """`recent_user_turns` MAY include `text` as its last element (a caller

        that appends the current message to its own history before calling
        `check`) or MAY NOT (a caller passing only prior turns). Implementations
        that build a classifier context window from `recent_user_turns` must
        handle both shapes without duplicating `text` in that window.

        `message_id` is the id the caller will save this user turn under. It is
        optional only so a caller that has not minted one yet still type-checks;
        pass it whenever it exists, because without it a guardrail_events row
        cannot be joined back to the message that caused it — which is what
        turns the audit trail from a count into an investigation (I4).

        `speculative` says this turn is running on an interim transcript that
        may never be confirmed (L-3). It changes no decision — a guess is
        guarded exactly as strictly as a certainty — but it is stamped on the
        event, because an analyst counting blocks otherwise has no way to tell
        a refusal the user heard from one for a sentence they never finished
        saying.
        """
        ...


@dataclass
class GuardContext:
    system_prompt: str
    tool_numbers: list[float]
    facts: list[Fact]
    cut_count: int = 0
    user_id: UserId | None = None
    # The confidential portion of the system prompt (policy, grounding,
    # instruction-hierarchy, and tool-guidance blocks — see
    # `PromptBuilder.confidential_text`) for the output guard's leak
    # detector to compare against. `RunTurn` sets it on every context it
    # builds; it defaults to empty so a caller constructing a context by
    # hand still gets a valid one — but an empty corpus means the leak
    # check has nothing to match against, so the guard falls back to
    # `system_prompt` (see `OutputGuard._evaluate`).
    leak_corpus: str = ""
    # The user message this turn is answering, so an output-guard verdict can
    # be joined back to the turn that produced it (I4). Set by `RunTurn`.
    message_id: MessageId | None = None
    # Figures a FINISHED assessment run entitles this turn to speak: each trait
    # score, the integer it rounds to, and the top of the scale. Kept apart from
    # `tool_numbers` on purpose. Merging them would have wrecked both checks —
    # the tool check would start accepting five unrelated scores as grounding
    # for a temperature, and the score check would inherit `tool_numbers`'
    # ±1.0 rounding allowance, which on a one-to-five scale accepts every
    # number there is. So this list is checked separately, only against
    # sentences that actually talk about the results (see
    # `mentions_results`), and to the decimal.
    results_numbers: list[float] = field(default_factory=list)
    # Scores are exact numbers read out of a row; the only rounding they are
    # entitled to is the one already declared alongside them.
    results_tolerance: float = 0.05
    # The last few sentences this turn produced, oldest first, cut ones
    # included. `OutputGuard` appends to it and runs a second, shorter-n leak
    # check over the joined window: a model that recites the confidential
    # prompt does it a clause at a time, and no single clause carries enough
    # 8-grams to trip the per-sentence check. See `OutputGuard.check_sentence`.
    spoken_tail: list[str] = field(default_factory=list)
    # This turn is running on an unconfirmed interim transcript (L-3). Stamped
    # on the events the guard writes, for the same reason as
    # `InputGuardPort.check`'s flag; cleared if the turn is promoted mid-flight.
    speculative: bool = False


@dataclass(frozen=True)
class SentenceVerdict:
    action: Literal["pass", "cut", "replace"]
    replacement: str | None = None
    kind: str | None = None


class OutputGuardPort(Protocol):
    def check_sentence(self, sentence: str, ctx: GuardContext) -> SentenceVerdict: ...


class RefusalPort(Protocol):
    def refusal(self, category: str | None) -> str: ...


# --- Active workflow runs -----------------------------------------------


@dataclass(frozen=True)
class ActiveRunSnapshot:
    run_id: RunId
    definition_id: str
    status: str
    current_item: int
    total_items: int
    prompt_block: str


@dataclass(frozen=True)
class AssessmentReply:
    sentences: list[str]
    workflow: dict[str, Any]
    # Numbers this reply is entitled to say out loud — the scores the workflow
    # actually computed. Without them `RunTurn` builds the workflow's guard
    # context with `tool_numbers=[]`, and an empty list switches the grounding
    # check OFF entirely (see `OutputGuard._evaluate`), so a personality result
    # was the one spoken output that could state any figure at all unchecked
    # (I9). Empty by default: a workflow reply with no numbers in it wants the
    # check off, and says so.
    grounding_numbers: tuple[float, ...] = ()


class ActiveRunPort(Protocol):
    async def active_run(self, user_id: UserId) -> ActiveRunSnapshot | None: ...

    async def handle_turn(self, user_id: UserId, text: str) -> AssessmentReply | None: ...

    def snapshot_from_row(self, row: dict[str, Any]) -> ActiveRunSnapshot | None:
        """Used by Phase 7's single-RPC context loader."""
        ...

    async def latest_results(self, user_id: UserId) -> dict[str, Any] | None:
        """The user's latest COMPLETE run as `{results, narrative, completed_at}`.

        The Postgres path never calls this — `load_turn_context` returns the same
        shape as its `last_results` key in the one round trip. It exists for the
        in-memory loader, so a container with no database still grounds follow-up
        questions about a finished test the same way production does. Callers ask
        only when no run is open; `None` when there is nothing completed.
        """
        ...


# --- Repositories --------------------------------------------------------


class SessionRepo(Protocol):
    async def get(self, id: SessionId) -> Session | None: ...

    async def latest_for_user(self, user_id: UserId) -> Session | None: ...

    async def save(self, s: Session) -> None: ...


class MessageRepo(Protocol):
    async def history(
        self, user_id: UserId, session_id: SessionId, limit: int
    ) -> list[Message]: ...

    async def save(self, m: Message) -> None: ...

    async def save_tool_call(
        self,
        message_id: MessageId,
        user_id: UserId,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        status: str,
        latency_ms: int,
    ) -> None: ...
