from __future__ import annotations

import random

_REDIRECT = (
    " I'm good for weather, remembering things for you, a chat, or a "
    "quick personality test — want any of those?"
)

TEMPLATES: dict[str, list[str]] = {
    "out_of_scope": [
        "That's outside what I can help with." + _REDIRECT,
        "I'll pass on that one." + _REDIRECT,
        "Not something I can do, sorry." + _REDIRECT,
    ],
    "medical_legal_financial": [
        "I can't give advice on that — a professional would serve you "
        "much better. Anything else I can do?",
        "That's one for a professional, not me. Is there something else I can help with?",
        "I'm not the right source for that kind of advice. Want help with anything else?",
    ],
    "self_harm": [
        "I'm really glad you told me. I'm not able to help with this the "
        "way a person can, and you deserve that support — please reach "
        "out to a crisis line right now; if you're in the US you can call "
        "or text 988, and if you're elsewhere I can help find a local "
        "number."
    ],
    "violence_illegal": [
        "I can't help with that." + _REDIRECT,
        "That's not something I'll help with." + _REDIRECT,
        "I won't go there." + _REDIRECT,
    ],
    "sexual": [
        "I'll keep things friendly and on-topic." + _REDIRECT,
        "Let's keep it clean." + _REDIRECT,
        "Not my thing." + _REDIRECT,
    ],
    "hate_harassment": [
        "I won't take part in that." + _REDIRECT,
        "Let's not go there." + _REDIRECT,
        "I'll pass on that." + _REDIRECT,
    ],
    "politics_religion": [
        "I stay out of politics and religion — no opinions from me. Anything else I can do?",
        "I don't weigh in on that kind of thing. What else can I help with?",
        "That's a topic I keep neutral on. Want help with something else?",
    ],
    "persona_switch": [
        "I'm Sarjy, and that's who I'll stay. What can I help with?",
        "Still Sarjy here, rules and all. What do you need?",
        "I'll stay myself, thanks. What can I do for you?",
    ],
    "prompt_leak": [
        "I keep my inner workings to myself, but I'm happy to tell you what I can do.",
        "I can't share my internal setup, but I can tell you what I'm able to help with.",
        "That part stays private. Want to hear what I can help with?",
    ],
    "impersonation": [
        "I won't pretend to be a real person." + _REDIRECT,
        "I don't impersonate real people." + _REDIRECT,
        "Not something I do." + _REDIRECT,
    ],
    "no_tool_for_that": [
        "I can't look that up — I only have live data for weather. Want a forecast instead?",
        "I don't have a way to check that. Weather is the one live thing I can fetch — want that?",
        "That's beyond what I can fetch; weather is my only live lookup. Shall I check a forecast?",
    ],
    "injection": [
        "Nice try — I'll stick to my rules. What can I help with?",
        "I'll keep following my rules, thanks. What do you need?",
        "Not happening, but I'm happy to help otherwise.",
    ],
}


def refusal_for(category: str | None, rng: random.Random | None = None) -> str:
    options = TEMPLATES.get(category or "", TEMPLATES["out_of_scope"])
    if len(options) == 1:
        return options[0]
    return (rng or random.Random()).choice(options)  # noqa: S311
