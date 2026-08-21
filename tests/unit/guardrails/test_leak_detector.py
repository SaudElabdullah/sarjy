from sarjy.contexts.conversation.application.prompt_builder import PromptBuilder
from sarjy.contexts.guardrails.domain.leak_detector import LeakDetector, leak_overlap

SYS = (
    "You are Sarjy, a voice assistant. Warm, concise, lightly playful. "
    "Spoken replies are at most two short sentences unless the user asks "
    "for more. Decline medical, legal, and financial advice."
)


def test_detects_verbatim_leak() -> None:
    d = LeakDetector(SYS)
    assert d.is_leak(
        "Sure: You are Sarjy, a voice assistant. Warm, concise, lightly "
        "playful. Spoken replies are at most two short sentences."
    )


def test_normal_sentence_passes() -> None:
    assert not LeakDetector(SYS).is_leak(
        "I'm Sarjy, a voice assistant — I can check the weather for you."
    )


def test_leak_overlap_module_function() -> None:
    leak = (
        "Sure: You are Sarjy, a voice assistant. Warm, concise, lightly "
        "playful. Spoken replies are at most two short sentences."
    )
    assert leak_overlap(leak, SYS, n=8) > 2
    assert leak_overlap("I like weather.", SYS, n=8) == 0


def test_cache_is_reused_across_instances() -> None:
    # Two detectors built from the same system prompt text should not
    # need to recompute the n-gram set — same behaviour either way.
    d1 = LeakDetector(SYS)
    d2 = LeakDetector(SYS)
    sentence = "I'm Sarjy, a voice assistant — I can check the weather for you."
    assert d1.is_leak(sentence) == d2.is_leak(sentence)


def test_threshold_is_configurable() -> None:
    # A threshold of 0 makes even a small overlap count as a leak.
    strict = LeakDetector(SYS, threshold=0)
    lenient = LeakDetector(SYS, threshold=1000)
    sentence = "You are Sarjy, a voice assistant. Warm, concise, lightly playful."
    assert strict.is_leak(sentence)
    assert not lenient.is_leak(sentence)


# --- C3: the leak corpus is the confidential blocks, not the full prompt ---


def test_capability_paraphrase_is_not_a_leak_against_confidential_corpus() -> None:
    # PromptBuilder.confidential_text excludes the persona line and the
    # capability blurb — those are things Sarjy is allowed to restate, so a
    # paraphrase of "You cannot browse, look up news, stocks, or scores,
    # send messages, or control devices." must not register as a leak.
    corpus = PromptBuilder().confidential_text
    sentence = "I can't browse, look up news, stocks, or scores, send messages, or control devices."
    assert leak_overlap(sentence, corpus, n=8) == 0
    assert not LeakDetector(corpus).is_leak(sentence)


def test_verbatim_policy_sentence_still_cuts() -> None:
    corpus = PromptBuilder().confidential_text
    sentence = (
        "Decline medical, legal, or financial advice, sexual content, violence "
        "or illegal instructions, hate, political or religious opinions, and "
        "impersonation."
    )
    assert LeakDetector(corpus).is_leak(sentence)
