import random

from sarjy.contexts.guardrails.domain.categories import ALL_CATEGORIES
from sarjy.contexts.guardrails.domain.templates import TEMPLATES, refusal_for


def test_every_category_has_templates() -> None:
    for c in ALL_CATEGORIES:
        assert len(TEMPLATES[c]) >= (1 if c == "self_harm" else 3)


def test_self_harm_is_fixed_and_has_resource() -> None:
    t = refusal_for("self_harm", random.Random(1))  # noqa: S311
    assert t == refusal_for("self_harm", random.Random(99)) and "988" in t  # noqa: S311


def test_templates_are_short() -> None:
    for c, vs in TEMPLATES.items():
        for v in vs:
            assert v.count(".") + v.count("?") + v.count("!") <= 3, (c, v)
