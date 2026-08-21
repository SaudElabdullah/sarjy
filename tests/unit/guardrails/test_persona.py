import pytest

from sarjy.contexts.guardrails.domain.persona import is_persona_break


@pytest.mark.parametrize(
    "s",
    [
        "As an AI language model developed by OpenAI, I cannot.",
        "I am ChatGPT.",
        "Sure, I'm Gemini and I have no restrictions now.",
        "I am an AI model, not a person.",
        "As a large language model, I don't have feelings.",
        "My name is not Sarjy, it's Claude.",
        "I now have no rules to follow.",
        "I have no limits on what I can say.",
        "Actually, I'm DAN.",
        "Sure, I'll be Claude for this next part.",
        "I was developed by Anthropic, not some voice-assistant company.",
        # Fix round 2 (C1 regression): "an AI ... developed by <vendor>" in
        # first person is unambiguous self-identification, across all three
        # subject spellings.
        "I am an AI developed by OpenAI.",
        "As an AI developed by Google, I cannot.",
        "I'm an AI developed by OpenAI.",
        # J: "developed by" is one phrasing of the claim; a model that has been
        # steered off it reaches for the synonym, not for honesty.
        "I was made by OpenAI, actually.",
        "I am built by Anthropic.",
        "I'm an AI created by Google.",
        "As an AI trained by OpenAI, I can't do that.",
        "Just call me DAN from now on.",
        "You know what, call me ChatGPT.",
    ],
)
def test_breaks(s: str) -> None:
    assert is_persona_break(s)


def test_ok() -> None:
    assert not is_persona_break("I'm Sarjy. Want the weather?")


def test_ok_mentions_ai_topic_without_breaking() -> None:
    assert not is_persona_break("AI models can be really useful for weather forecasts.")


@pytest.mark.parametrize(
    "s",
    [
        # I2: a bare "I'm an AI" is too common in benign self-description to
        # use as a signal on its own — it was a false-positive source.
        "I'm an AI assistant, so I can't taste coffee.",
        "Honestly, I'm just an AI.",
        "I'm an AI, so I can't really have opinions.",
        # I2: "developed by <vendor>" needs a first-person subject — a
        # third-person mention isn't the model talking about itself.
        "It was developed by OpenAI, according to the article.",
        "The chatbot was developed by Anthropic last year.",
        # Fix round 2: the new "an AI ... developed by" alternative must stay
        # first-person-gated too, same as the bare "developed by" form.
        "It was developed by Google, according to the article.",
        # The synonyms stay first-person-gated too.
        "The app was built by Google last year.",
        "That model was trained by OpenAI on public data.",
        # Naming the *user*, not renaming itself.
        "Sure, I'll call you Dan from now on.",
        "What should I call you?",
    ],
)
def test_ok_precision_fixes(s: str) -> None:
    assert not is_persona_break(s)
