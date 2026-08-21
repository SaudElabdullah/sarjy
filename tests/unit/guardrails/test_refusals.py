import random

from sarjy.contexts.conversation.application.ports import RefusalPort
from sarjy.contexts.guardrails.application.refusals import TemplateRefusals
from sarjy.contexts.guardrails.domain.templates import TEMPLATES, refusal_for


def test_refusal_self_harm_contains_resource() -> None:
    t = TemplateRefusals().refusal("self_harm")
    assert "988" in t
    assert t == refusal_for("self_harm")


def test_refusal_none_is_out_of_scope() -> None:
    t = TemplateRefusals().refusal(None)
    assert t in TEMPLATES["out_of_scope"]


def test_refusal_nonexistent_is_out_of_scope() -> None:
    t = TemplateRefusals().refusal("nonexistent")
    assert t in TEMPLATES["out_of_scope"]


def test_refusal_with_injected_rng_deterministic() -> None:
    rng1 = random.Random(42)  # noqa: S311
    adapter1 = TemplateRefusals(rng=rng1)
    result1 = adapter1.refusal("sexual")

    rng2 = random.Random(42)  # noqa: S311
    adapter2 = TemplateRefusals(rng=rng2)
    result2 = adapter2.refusal("sexual")

    assert result1 == result2
    assert result1 in TEMPLATES["sexual"]


def test_template_refusals_satisfies_port() -> None:
    port: RefusalPort = TemplateRefusals()
    assert port.refusal("sexual") in TEMPLATES["sexual"]
