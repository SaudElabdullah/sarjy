from __future__ import annotations

from typing import Literal, get_args

Category = Literal[
    "out_of_scope",
    "medical_legal_financial",
    "self_harm",
    "violence_illegal",
    "sexual",
    "hate_harassment",
    "politics_religion",
    "persona_switch",
    "prompt_leak",
    "impersonation",
    "no_tool_for_that",
    "injection",
]
ALL_CATEGORIES: tuple[str, ...] = get_args(Category)
SEVERITY: dict[str, int] = {
    "self_harm": 3,
    "prompt_leak": 3,
    "violence_illegal": 3,
    "sexual": 2,
    "hate_harassment": 2,
    "injection": 2,
    "persona_switch": 2,
    "impersonation": 2,
    "medical_legal_financial": 1,
    "politics_religion": 1,
    "out_of_scope": 1,
    "no_tool_for_that": 1,
}
