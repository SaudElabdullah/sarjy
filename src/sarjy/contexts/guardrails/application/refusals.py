from __future__ import annotations

import random

from sarjy.contexts.guardrails.domain.templates import refusal_for


class TemplateRefusals:
    """Refusal adapter implementing RefusalPort via template-based responses."""

    def __init__(self, rng: random.Random | None = None) -> None:
        """Initialize adapter with optional random number generator.

        Args:
            rng: Optional RNG for template selection. If None, uses system RNG.
        """
        self.rng = rng

    def refusal(self, category: str | None) -> str:
        """Return a refusal template for the given category.

        Args:
            category: The guardrail category, or None for out_of_scope default.

        Returns:
            A spoken refusal message.
        """
        return refusal_for(category, self.rng)
