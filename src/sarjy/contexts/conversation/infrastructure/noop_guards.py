from typing import Any

from sarjy.contexts.conversation.application.ports import (
    ActiveRunSnapshot,
    AssessmentReply,
    Fact,
    GuardContext,
    GuardDecision,
    SentenceVerdict,
)
from sarjy.shared.ids import MessageId, UserId


class AllowAllInputGuard:
    async def check(
        self,
        user_id: UserId,
        text: str,
        recent_user_turns: list[str],
        message_id: MessageId | None = None,
        speculative: bool = False,
    ) -> GuardDecision:
        return GuardDecision(action="allow")


class PassOutputGuard:
    def check_sentence(self, sentence: str, ctx: GuardContext) -> SentenceVerdict:
        return SentenceVerdict(action="pass")


class NoFacts:
    async def snapshot(self, user_id: UserId) -> list[Fact]:
        return []


class NoActiveRun:
    async def active_run(self, user_id: UserId) -> ActiveRunSnapshot | None:
        return None

    async def handle_turn(self, user_id: UserId, text: str) -> AssessmentReply | None:
        return None

    def snapshot_from_row(self, row: dict[str, Any]) -> ActiveRunSnapshot | None:
        return None

    async def latest_results(self, user_id: UserId) -> dict[str, Any] | None:
        return None
